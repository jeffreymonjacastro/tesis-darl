"""Gymnasium environment for DARL selective update decisions."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

SEED = 42
ACTION_NAMES = ("defer", "update_features", "update_model", "retrain_all")
OBSERVATION_COLUMNS = (
    "severity",
    "psi_mean",
    "ks_mean",
    "c2st_score",
    "delta_auc",
    "delta_f1",
    "has_covariate_signal",
    "has_concept_signal",
)


class DarlUpdateEnv(gym.Env):
    """Practical POMDP approximation for DARL maintenance actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: pd.DataFrame,
        episode_length: int = 12,
        lambda_auc: float = 1.0,
        lambda_cost: float = 0.25,
        seed: int = SEED,
    ):
        super().__init__()
        self.scenarios = scenarios.reset_index(drop=True).copy()
        self.episode_length = int(episode_length)
        self.lambda_auc = float(lambda_auc)
        self.lambda_cost = float(lambda_cost)
        self.seed_value = int(seed)

        self.action_space = spaces.Discrete(len(ACTION_NAMES), seed=seed)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(OBSERVATION_COLUMNS) + 2,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._order = np.arange(len(self.scenarios))
        self._cursor = 0
        self._step_count = 0
        self._last_time_cost = 0.0
        self._last_ram_cost = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Start a new episode and return the first observation."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(len(self.scenarios))
        self._cursor = 0
        self._step_count = 0
        self._last_time_cost = 0.0
        self._last_ram_cost = 0.0
        return self._observation(), {}

    def step(self, action: int):
        """Apply a maintenance action and return reward plus next observation."""
        action_id = int(action)
        row = self._current_row()
        action_name = ACTION_NAMES[action_id]

        reward, time_cost, ram_cost = self._reward(row, action_name)
        info = {
            "action_name": action_name,
            "drift_type": row["drift_type"],
            "severity_label": row["severity_label"],
            "auc_recovery": float(row[f"{action_name}_auc_recovery"]),
            "time_cost": time_cost,
            "ram_cost": ram_cost,
        }

        self._last_time_cost = time_cost
        self._last_ram_cost = ram_cost
        self._step_count += 1
        self._cursor = (self._cursor + 1) % len(self.scenarios)

        terminated = self._step_count >= self.episode_length
        return self._observation(), float(reward), terminated, False, info

    def _current_row(self) -> pd.Series:
        return self.scenarios.iloc[int(self._order[self._cursor])]

    def _observation(self) -> np.ndarray:
        row = self._current_row()
        values = [float(row[col]) for col in OBSERVATION_COLUMNS]
        values.extend([self._last_time_cost, self._last_ram_cost])
        return np.asarray(values, dtype=np.float32)

    def _reward(self, row: pd.Series, action_name: str) -> tuple[float, float, float]:
        auc_recovery = float(row[f"{action_name}_auc_recovery"])
        time_cost = float(row[f"{action_name}_time_cost"])
        ram_cost = float(row[f"{action_name}_ram_cost"])
        relative_cost = 0.5 * (time_cost + ram_cost)
        reward = self.lambda_auc * auc_recovery - self.lambda_cost * relative_cost
        return reward, time_cost, ram_cost
