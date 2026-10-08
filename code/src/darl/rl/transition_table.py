"""Empirical counterfactual transition collection from live DARL state."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TransitionTableBuilder:
    """Collect all four action outcomes at each visited live state."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def build(
        self,
        env,
        episode_id: str = "episode-0",
        behavior: str = "random",
    ) -> pd.DataFrame:
        """Build long-form transitions while one behavior branch advances."""
        if behavior not in {"random", "defer", "oracle"}:
            raise ValueError(f"Unknown behavior policy: {behavior}")
        rng = np.random.default_rng(self.seed)
        observation, _ = env.reset(seed=self.seed)
        rows: list[dict] = []
        terminated = False
        step = 0
        while not terminated:
            counterfactuals = []
            for action in range(4):
                branch = env.clone()
                next_observation, reward, branch_done, _, info = branch.step(action)
                counterfactuals.append((action, reward, branch))
                rows.append(
                    {
                        "episode_id": episode_id,
                        "step": step,
                        "observation": observation.tolist(),
                        "action": action,
                        "action_requested": info["action_requested"],
                        "action_executed": info["action_executed"],
                        "reward": reward,
                        "next_observation": next_observation.tolist(),
                        "terminated": branch_done,
                        **{key: value for key, value in info.items() if key != "action_name"},
                    }
                )
            if behavior == "random":
                selected_action = int(rng.integers(0, 4))
            elif behavior == "defer":
                selected_action = 0
            else:
                selected_action = max(counterfactuals, key=lambda item: item[1])[0]
            next_observation, _, terminated, _, _ = env.step(selected_action)
            observation = next_observation
            step += 1
        return pd.DataFrame(rows)


def get_reward(auc: float, time_cost: float, lambda_cost: float = 0.05) -> float:
    """Compatibility reward helper for legacy wide tables."""
    return float(auc - lambda_cost * time_cost)


def build_transition_row(
    state_vector,
    df_drifted_target,
    df_drifted_train,
    df_reference_train,
    qt,
    imputer,
    scaler,
    model,
    vitals,
    numeric_cols,
    label_col,
    threshold,
    seed,
) -> dict:
    """Compatibility builder for one legacy wide-table row."""
    from darl.actions import (
        run_a1_xgb,
        run_a2_with_contract,
        run_a3_xgb,
        run_a4_xgb,
    )

    calls = [
        lambda: run_a1_xgb(
            df_drifted_target, qt, imputer, scaler, model, vitals, numeric_cols,
            label_col, threshold
        ),
        lambda: run_a2_with_contract(
            df_drifted_target, df_drifted_train, df_reference_train, qt, imputer,
            scaler, model, vitals, numeric_cols, label_col, threshold, seed
        ),
        lambda: run_a3_xgb(
            df_drifted_target, df_drifted_train, qt, imputer, scaler, vitals,
            numeric_cols, label_col, threshold, seed
        ),
        lambda: run_a4_xgb(
            df_drifted_target, df_drifted_train, vitals, numeric_cols, label_col,
            threshold, seed
        ),
    ]
    row = {"state": state_vector}
    for index, call in enumerate(calls, start=1):
        metrics, elapsed, peak_ram = call()
        row[f"A{index}_AUC"] = metrics["auc"]
        row[f"A{index}_time"] = elapsed
        row[f"A{index}_ram"] = peak_ram
    return row
