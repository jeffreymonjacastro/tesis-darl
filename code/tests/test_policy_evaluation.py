"""Matched policy-evaluation tests for long-form empirical transitions."""

import numpy as np
import pandas as pd

from darl.evaluation import AlwaysDeferPolicy, OraclePolicy, evaluate_policies


def _transitions() -> pd.DataFrame:
    rows = []
    observation = np.zeros(28, dtype=np.float32).tolist()
    for episode in ("patient-group-a", "patient-group-b"):
        for step in range(2):
            for action in range(4):
                rows.append(
                    {
                        "episode_id": episode,
                        "step": step,
                        "observation": observation,
                        "action": action,
                        "action_requested": f"A{action + 1}",
                        "action_executed": f"A{action + 1}",
                        "reward": float(action),
                        "auc_before": 0.5,
                        "auc_after": 0.5 + action / 10,
                        "recovery_ratio": float(action),
                        "time_s": float(action),
                        "peak_ram_mb": float(action),
                    }
                )
    return pd.DataFrame(rows)


def test_policies_share_episodes_and_oracle_has_zero_regret():
    summary, steps = evaluate_policies(
        _transitions(),
        {"defer": AlwaysDeferPolicy(), "oracle": OraclePolicy()},
    )
    oracle = summary.set_index("policy").loc["oracle"]
    assert oracle["optimal_action_rate"] == 1.0
    assert oracle["mean_regret"] == 0.0
    assert steps.groupby("policy").size().nunique() == 1
