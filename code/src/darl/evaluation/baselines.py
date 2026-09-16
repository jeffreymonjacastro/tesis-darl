"""Baseline policies for cost-aware DARL evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


class RandomPolicy:
    """Choose each maintenance action uniformly."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation, candidates=None) -> int:
        """Return a reproducible random action."""
        del observation, candidates
        return int(self.rng.integers(0, 4))


class AlwaysDeferPolicy:
    """Always choose A1."""

    def select_action(self, observation, candidates=None) -> int:
        """Return A1."""
        del observation, candidates
        return 0


class FixedActionPolicy:
    """Always choose one configured action."""

    def __init__(self, action: int):
        if action not in range(4):
            raise ValueError("action must be in [0, 3]")
        self.action = int(action)

    def select_action(self, observation, candidates=None) -> int:
        """Return configured action."""
        del observation, candidates
        return self.action


class ThresholdPolicy:
    """Map calibrated drift probabilities to selective updates."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def select_action(self, observation, candidates=None) -> int:
        """Choose A4/A3/A2/A1 from latest seven-value observation."""
        del candidates
        current = np.asarray(observation, dtype=float)[-7:]
        p_covariate, p_concept, p_both = current[:3]
        if p_both >= self.threshold:
            return 3
        if p_concept >= self.threshold:
            return 2
        if p_covariate >= self.threshold:
            return 1
        return 0


class EmpiricalTablePolicy:
    """Estimate action rewards by nearest neighbors in empirical transitions."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = int(n_neighbors)
        self.models: dict[int, KNeighborsRegressor] = {}

    def fit(self, transitions: pd.DataFrame) -> "EmpiricalTablePolicy":
        """Fit one local reward estimator per action."""
        for action, group in transitions.groupby("action"):
            neighbors = min(self.n_neighbors, len(group))
            model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
            model.fit(np.stack(group["observation"]), group["reward"].to_numpy())
            self.models[int(action)] = model
        return self

    def select_action(self, observation, candidates=None) -> int:
        """Choose action with largest locally estimated reward."""
        del candidates
        if len(self.models) != 4:
            raise RuntimeError("EmpiricalTablePolicy must be fitted with all actions")
        sample = np.asarray(observation, dtype=float).reshape(1, -1)
        estimates = {action: model.predict(sample)[0] for action, model in self.models.items()}
        return max(estimates, key=estimates.get)


class OraclePolicy:
    """Evaluation-only ceiling that observes all counterfactual rewards."""

    def select_action(self, observation, candidates=None) -> int:
        """Return action with maximum realized reward."""
        del observation
        if candidates is None or candidates.empty:
            raise ValueError("OraclePolicy requires counterfactual candidates")
        return int(candidates.loc[candidates["reward"].idxmax(), "action"])


class DQNPolicy:
    """Adapter exposing a trained DQN agent as an evaluation policy."""

    def __init__(self, agent):
        self.agent = agent

    def select_action(self, observation, candidates=None) -> int:
        """Return deterministic DQN action without counterfactual leakage."""
        del candidates
        return self.agent.select_action(observation, epsilon=0.0)


def run_baseline_always_a1(transition_table: pd.DataFrame) -> dict:
    """Compatibility summary for legacy A1-wide tables."""
    return {
        "mean_auc": transition_table["A1_AUC"].mean(),
        "mean_time": transition_table["A1_time"].mean(),
    }


def run_baseline_always_a4(transition_table: pd.DataFrame) -> dict:
    """Compatibility summary for legacy A4-wide tables."""
    return {
        "mean_auc": transition_table["A4_AUC"].mean(),
        "mean_time": transition_table["A4_time"].mean(),
    }


def run_baseline_reactive(
    transition_table: pd.DataFrame,
    c2st_threshold: float = 0.55,
) -> dict:
    """Compatibility C2ST rule for legacy wide tables."""
    actions = np.where(
        transition_table["state"].map(lambda state: state[0]) > c2st_threshold,
        2,
        1,
    )
    aucs = [row[f"A{action}_AUC"] for action, (_, row) in zip(actions, transition_table.iterrows())]
    times = [row[f"A{action}_time"] for action, (_, row) in zip(actions, transition_table.iterrows())]
    return {"mean_auc": float(np.mean(aucs)), "mean_time": float(np.mean(times))}


def run_baseline_random(transition_table: pd.DataFrame, seed: int = 42) -> dict:
    """Compatibility random policy for legacy wide tables."""
    rng = np.random.default_rng(seed)
    actions = rng.integers(1, 5, size=len(transition_table))
    aucs = [row[f"A{action}_AUC"] for action, (_, row) in zip(actions, transition_table.iterrows())]
    times = [row[f"A{action}_time"] for action, (_, row) in zip(actions, transition_table.iterrows())]
    return {"mean_auc": float(np.mean(aucs)), "mean_time": float(np.mean(times))}
