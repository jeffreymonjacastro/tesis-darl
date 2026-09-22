"""Gymnasium environments for causal DARL maintenance decisions."""

from __future__ import annotations

import copy
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from darl.actions import ACTION_CODES, execute_action
from darl.drift.injector import SCENARIOS

SEED = 42
ACTION_NAMES = ("defer", "update_features", "update_model", "retrain_all")


def recovery_reward(
    auc_base: float,
    auc_before: float,
    auc_after: float,
    normalized_cost: float,
    lambda_cost: float = 0.25,
) -> tuple[float, float]:
    """Return clipped AUC recovery ratio and cost-aware reward."""
    loss = auc_base - auc_before
    recovery = 0.0 if not np.isfinite(loss) or loss <= 1e-6 else (auc_after - auc_before) / loss
    recovery = float(np.clip(recovery, -1.0, 2.0))
    reward = recovery - lambda_cost * float(np.clip(normalized_cost, 0.0, 1.0))
    return recovery, float(reward)


class DARLEnvironment(gym.Env):
    """Live POMDP approximation over ordered PhysioNet ICULOS windows."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        pipeline_state,
        reference_frame: pd.DataFrame,
        windows: list,
        monitor,
        auc_base: float | None = None,
        drift_injector=None,
        scenario: str = "natural",
        true_drift_type: str = "none",
        severity: float = 0.0,
        concept_drift_method: str = "label_flip_control",
        history_length: int = 4,
        lambda_cost: float = 0.25,
        seed: int = SEED,
    ):
        super().__init__()
        if not windows:
            raise ValueError("DARLEnvironment requires at least one temporal window")
        if scenario not in {*SCENARIOS, "natural"}:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.initial_pipeline_state = pipeline_state.clone()
        self.pipeline_state = pipeline_state.clone()
        self.reference_frame = reference_frame.copy()
        self.windows = list(windows)
        self.monitor = copy.deepcopy(monitor)
        self.drift_injector = copy.deepcopy(drift_injector)
        self.scenario = scenario
        self.true_drift_type = true_drift_type
        self.severity = float(severity)
        self.concept_drift_method = concept_drift_method
        self.history_length = int(history_length)
        self.lambda_cost = float(lambda_cost)
        self.seed_value = int(seed)
        self.auc_base = (
            float(auc_base)
            if auc_base is not None
            else float(self.pipeline_state.evaluate(self.reference_frame))
        )
        self.action_space = spaces.Discrete(4, seed=seed)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7 * self.history_length,),
            dtype=np.float32,
        )
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._cursor = 0
        self._last_action_cost = 0.0
        self._last_report = None

    def clone(self) -> "DARLEnvironment":
        """Deep-copy complete live state for counterfactual action evaluation."""
        return copy.deepcopy(self)

    def _drifted_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        window = self.windows[self._cursor]
        progress = (self._cursor + 1) / len(self.windows)
        effective_severity = float(np.clip(self.severity * progress, 0.0, 1.0))
        if self.drift_injector is None or self.true_drift_type == "none":
            return window.update.copy(), window.evaluation.copy(), effective_severity

        config = SCENARIOS[self.scenario]
        importances = self.pipeline_state.stage2.get_feature_importances(
            self.pipeline_state.numeric_cols
        )
        important = sorted(importances, key=importances.get, reverse=True)
        kwargs = {
            **config,
            "drift_severity": effective_severity,
            "drift_type": self.true_drift_type,
            "label_col": self.pipeline_state.label_col,
            "important_features": important,
            "concept_drift_method": self.concept_drift_method,
        }
        update_injector = copy.deepcopy(self.drift_injector)
        evaluation_injector = copy.deepcopy(self.drift_injector)
        update, _ = update_injector.transform(window.update, **kwargs)
        evaluation, _ = evaluation_injector.transform(window.evaluation, **kwargs)
        return update, evaluation, effective_severity

    def _base_observation(self) -> np.ndarray:
        _, evaluation, _ = self._drifted_frames()
        self.monitor.predictor = self.pipeline_state
        self._last_report = self.monitor.analyze(
            evaluation,
            window_idx=self.windows[self._cursor].index,
            true_drift_type=self.true_drift_type,
        )
        return self.monitor.get_observation(self._last_report, self._last_action_cost)

    def _stacked_observation(self) -> np.ndarray:
        missing = self.history_length - len(self._history)
        values = [np.zeros(7, dtype=np.float32) for _ in range(missing)]
        values.extend(self._history)
        return np.concatenate(values).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset pipeline and begin at earliest ICULOS decision window."""
        super().reset(seed=seed)
        del options
        self.pipeline_state = self.initial_pipeline_state.clone()
        self._cursor = 0
        self._last_action_cost = 0.0
        self._history.clear()
        self._history.append(self._base_observation())
        return self._stacked_observation(), {}

    def step(self, action: int):
        """Execute an action, persist its pipeline and advance clinical time."""
        update, evaluation, effective_severity = self._drifted_frames()
        selected = execute_action(
            action,
            self.pipeline_state,
            update,
            evaluation,
            self.reference_frame,
            seed=self.seed_value,
        )
        full = selected if int(action) == 3 else execute_action(
            "A4",
            self.pipeline_state,
            update,
            evaluation,
            self.reference_frame,
            seed=self.seed_value,
        )
        time_ratio = selected.time_s / max(full.time_s, 1e-9)
        ram_ratio = selected.peak_ram_mb / max(full.peak_ram_mb, 1e-9)
        normalized_cost = float(np.clip(0.5 * (time_ratio + ram_ratio), 0.0, 1.0))
        recovery, reward = recovery_reward(
            self.auc_base,
            selected.auc_before,
            selected.auc_after,
            normalized_cost,
            self.lambda_cost,
        )
        self.pipeline_state = selected.pipeline_state
        self._last_action_cost = normalized_cost
        current_window = self.windows[self._cursor]
        self._cursor += 1
        terminated = self._cursor >= len(self.windows)
        if terminated:
            self._history.append(np.zeros(7, dtype=np.float32))
        else:
            self._history.append(self._base_observation())
        info = {
            "action_name": ACTION_NAMES[int(action)],
            "action_requested": selected.requested_action,
            "action_executed": selected.executed_action,
            "fallback_reason": selected.fallback_reason,
            "auc_base": self.auc_base,
            "auc_before": selected.auc_before,
            "auc_after": selected.auc_after,
            "recovery_ratio": recovery,
            "time_s": selected.time_s,
            "peak_ram_mb": selected.peak_ram_mb,
            "normalized_cost": normalized_cost,
            "window_start": current_window.start_hour,
            "window_end": current_window.end_hour,
            "scenario": self.scenario,
            "true_drift_type": self.true_drift_type,
            "severity": effective_severity,
            "seed": self.seed_value,
        }
        return self._stacked_observation(), reward, terminated, False, info


class DarlUpdateEnv(gym.Env):
    """Compatibility environment for legacy wide transition tables."""

    metadata = {"render_modes": []}

    def __init__(self, transition_table: pd.DataFrame, episode_length: int = 12, seed: int = SEED):
        super().__init__()
        self.table = transition_table.reset_index(drop=True)
        self.episode_length = int(episode_length)
        self.action_space = spaces.Discrete(4, seed=seed)
        state_size = len(self.table.iloc[0]["state"])
        self.observation_space = spaces.Box(0.0, np.inf, (state_size + 2,), np.float32)
        self._cursor = 0
        self._last_cost = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset legacy table cursor."""
        super().reset(seed=seed)
        del options
        self._cursor = 0
        self._last_cost = 0.0
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        row = self.table.iloc[self._cursor % len(self.table)]
        return np.asarray([*row["state"], self._last_cost, 0.0], dtype=np.float32)

    def step(self, action: int):
        """Read one legacy action outcome and advance table cursor."""
        row = self.table.iloc[self._cursor % len(self.table)]
        prefix = f"A{int(action) + 1}"
        auc = float(row[f"{prefix}_AUC"])
        cost = float(row.get(f"{prefix}_time", 0.0))
        reward = auc - 0.25 * cost
        self._last_cost = cost
        self._cursor += 1
        terminated = self._cursor >= self.episode_length
        info = {
            "action_name": ACTION_NAMES[int(action)],
            "auc": auc,
            "time_cost": cost,
        }
        return self._observation(), float(reward), terminated, False, info
