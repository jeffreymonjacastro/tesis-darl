"""Policy comparison and decision-quality metrics for DARL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate predictive, decision and computational policy metrics."""

    policy: str
    mean_reward: float
    mean_episode_reward: float
    mean_auc_before: float
    mean_auc_after: float
    mean_recovery_ratio: float
    total_time_s: float
    peak_ram_mb: float
    optimal_action_rate: float
    mean_regret: float
    full_retrainings_avoided: float
    reward_ci_low: float
    reward_ci_high: float
    seed: int = 42


def bootstrap_mean_ci(
    values,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return deterministic percentile bootstrap interval for a mean."""
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def evaluate_transition_policy(
    transitions: pd.DataFrame,
    policy,
    policy_name: str,
    seed: int = 42,
) -> tuple[EvaluationResult, pd.DataFrame]:
    """Evaluate one policy against matched counterfactual transition groups."""
    selected_rows: list[pd.Series] = []
    regrets: list[float] = []
    optimal: list[float] = []
    for _, candidates in transitions.groupby(["episode_id", "step"], sort=False):
        observation = candidates.iloc[0]["observation"]
        action = int(policy.select_action(observation, candidates))
        matches = candidates[candidates["action"] == action]
        if matches.empty:
            raise ValueError(f"No counterfactual transition for action {action}")
        selected = matches.iloc[0]
        best_reward = float(candidates["reward"].max())
        selected_rows.append(selected)
        regrets.append(best_reward - float(selected["reward"]))
        optimal.append(float(np.isclose(float(selected["reward"]), best_reward)))
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    episode_rewards = selected.groupby("episode_id")["reward"].sum()
    ci_low, ci_high = bootstrap_mean_ci(episode_rewards, seed=seed)
    executed_a4 = (selected["action_executed"] == "A4").mean()
    result = EvaluationResult(
        policy=policy_name,
        mean_reward=float(selected["reward"].mean()),
        mean_episode_reward=float(episode_rewards.mean()),
        mean_auc_before=float(selected["auc_before"].mean()),
        mean_auc_after=float(selected["auc_after"].mean()),
        mean_recovery_ratio=float(selected["recovery_ratio"].mean()),
        total_time_s=float(selected["time_s"].sum()),
        peak_ram_mb=float(selected["peak_ram_mb"].max()),
        optimal_action_rate=float(np.mean(optimal)),
        mean_regret=float(np.mean(regrets)),
        full_retrainings_avoided=float(1.0 - executed_a4),
        reward_ci_low=ci_low,
        reward_ci_high=ci_high,
        seed=seed,
    )
    selected["policy"] = policy_name
    selected["regret"] = regrets
    selected["is_optimal"] = optimal
    return result, selected


def evaluate_policies(
    transitions: pd.DataFrame,
    policies: dict[str, object],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate policies on identical episodes and return summaries plus steps."""
    summaries = []
    steps = []
    for name, policy in policies.items():
        result, selected = evaluate_transition_policy(transitions, policy, name, seed)
        summaries.append(result.__dict__)
        steps.append(selected)
    return pd.DataFrame(summaries), pd.concat(steps, ignore_index=True)


def action_distribution(selected_steps: pd.DataFrame) -> pd.DataFrame:
    """Return normalized requested-action distribution by policy."""
    return (
        pd.crosstab(
            selected_steps["policy"],
            selected_steps["action_requested"],
            normalize="index",
        )
        .reindex(columns=["A1", "A2", "A3", "A4"], fill_value=0.0)
        .reset_index()
    )


def calculate_accumulated_reward(
    df_results: pd.DataFrame,
    lambda_cost: float = 0.05,
) -> float:
    """Return accumulated reward from new or legacy evaluation frames."""
    if "reward" in df_results:
        return float(df_results["reward"].sum())
    return float(df_results["auc"].sum() - lambda_cost * df_results["time_cost"].sum())


def compare_policies(
    darl_results: pd.DataFrame,
    transition_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compatibility comparison used by the original wide-table tests."""
    from darl.evaluation.baselines import (
        run_baseline_always_a1,
        run_baseline_always_a4,
        run_baseline_random,
        run_baseline_reactive,
    )

    rows = [
        {
            "Policy": "DARL (DQN)",
            "Mean AUC": darl_results["auc"].mean(),
            "Mean Time": darl_results["time_cost"].mean(),
        }
    ]
    for name, result in [
        ("Always A1 (Ignore)", run_baseline_always_a1(transition_table)),
        ("Always A4 (Retrain)", run_baseline_always_a4(transition_table)),
        ("Reactive (C2ST > 0.55 -> A2)", run_baseline_reactive(transition_table)),
        ("Random", run_baseline_random(transition_table)),
    ]:
        rows.append(
            {"Policy": name, "Mean AUC": result["mean_auc"], "Mean Time": result["mean_time"]}
        )
    return pd.DataFrame(rows)
