"""Training and evaluation helpers for the DARL SB3 PPO demo."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from darl.rl.env import ACTION_NAMES, DarlUpdateEnv
from darl.rl.scenarios import SEED, make_synthetic_scenarios


def train_ppo(env: DarlUpdateEnv, total_timesteps: int = 3000, seed: int = SEED) -> PPO:
    """Train PPO on the DARL update environment."""
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        n_steps=64,
        batch_size=32,
        n_epochs=5,
        gamma=0.95,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def training_progress(
    env: DarlUpdateEnv,
    checkpoints: tuple[int, ...] = (0, 500, 1000, 1500, 2000, 2500, 3000),
    eval_episodes: int = 5,
    seed: int = SEED,
) -> tuple[PPO, pd.DataFrame]:
    """Train PPO incrementally and return checkpoint-level evaluation metrics."""
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        n_steps=64,
        batch_size=32,
        n_epochs=5,
        gamma=0.95,
    )

    rows: list[dict[str, float | int]] = []
    previous = 0
    for checkpoint in checkpoints:
        if checkpoint < previous:
            raise ValueError("checkpoints must be sorted in ascending order.")
        delta = checkpoint - previous
        if delta:
            model.learn(total_timesteps=delta, reset_num_timesteps=False)

        eval_df = evaluate_policy(model, env, n_episodes=eval_episodes)
        action_share = (
            eval_df["action"]
            .value_counts(normalize=True)
            .reindex(ACTION_NAMES, fill_value=0.0)
        )
        episode_reward = eval_df.groupby("episode")["reward"].sum()

        rows.append(
            {
                "timesteps": checkpoint,
                "mean_reward": float(eval_df["reward"].mean()),
                "std_reward": float(eval_df["reward"].std(ddof=0)),
                "mean_episode_reward": float(episode_reward.mean()),
                "std_episode_reward": float(episode_reward.std(ddof=0)),
                "best_action_share": float(action_share.max()),
                "defer_share": float(action_share["defer"]),
                "update_features_share": float(action_share["update_features"]),
                "update_model_share": float(action_share["update_model"]),
                "retrain_all_share": float(action_share["retrain_all"]),
            }
        )
        previous = checkpoint

    return model, pd.DataFrame(rows)


def evaluate_policy(
    model: PPO,
    env: DarlUpdateEnv,
    n_episodes: int = 10,
) -> pd.DataFrame:
    """Run deterministic PPO actions and return one row per environment step."""
    rows: list[dict[str, float | int | str | bool]] = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=SEED + episode)
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            done = bool(terminated or truncated)
            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "action": info["action_name"],
                    "action_id": ACTION_NAMES.index(info["action_name"]),
                    "drift_type": info["drift_type"],
                    "severity": info["severity_label"],
                    "reward": reward,
                    "auc_recovery": info["auc_recovery"],
                    "time_cost": info["time_cost"],
                    "ram_cost": info["ram_cost"],
                    "done": done,
                }
            )
            step += 1

    return pd.DataFrame(rows)


def episode_reward_summary(eval_df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Summarize total reward per evaluation episode."""
    summary = (
        eval_df.groupby("episode", as_index=False)["reward"]
        .sum()
        .rename(columns={"reward": "episode_reward"})
    )
    summary["rolling_mean"] = summary["episode_reward"].rolling(
        window=rolling_window,
        min_periods=1,
    ).mean()
    return summary


def step_reward_summary(eval_df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """Summarize reward by step across evaluation episodes."""
    summary = (
        eval_df.groupby("step")["reward"]
        .agg(
            mean_reward="mean",
            std_reward=lambda values: values.std(ddof=0),
            count="count",
        )
        .reset_index()
    )
    summary["sem_reward"] = summary["std_reward"] / np.sqrt(summary["count"])
    summary["smooth_reward"] = summary["mean_reward"].rolling(
        window=rolling_window,
        center=True,
        min_periods=1,
    ).mean()
    summary["cumulative_mean_reward"] = summary["mean_reward"].cumsum()
    return summary


def normalized_action_distribution(eval_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Return action shares normalized within a context column."""
    if by not in eval_df.columns:
        raise KeyError(f"{by!r} is not a column in eval_df.")

    distribution = pd.crosstab(eval_df[by], eval_df["action"], normalize="index")
    distribution = distribution.reindex(columns=ACTION_NAMES, fill_value=0.0)
    return distribution.reset_index()


def reward_context_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize reward, AUC recovery, and cost by drift context and action."""
    df = eval_df.copy()
    df["relative_cost"] = 0.5 * (df["time_cost"] + df["ram_cost"])
    return (
        df.groupby(["drift_type", "severity", "action"], as_index=False)
        .agg(
            n=("reward", "size"),
            mean_reward=("reward", "mean"),
            mean_auc_recovery=("auc_recovery", "mean"),
            mean_relative_cost=("relative_cost", "mean"),
        )
        .sort_values(["drift_type", "severity", "mean_reward"], ascending=[True, True, False])
    )


def smoke_test(total_timesteps: int = 512) -> pd.DataFrame:
    """Validate the env, train a small PPO model, and print a compact report."""
    scenarios = make_synthetic_scenarios(n_per_type=12, seed=SEED)
    env = DarlUpdateEnv(scenarios, episode_length=8, seed=SEED)
    check_env(env, warn=True)

    model = train_ppo(env, total_timesteps=total_timesteps, seed=SEED)
    results = evaluate_policy(model, env, n_episodes=2)

    print(f"scenarios_shape={scenarios.shape}")
    print(f"mean_reward={results['reward'].mean():.4f}")
    print("action_counts=")
    print(results["action"].value_counts().sort_index().to_string())
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DARL SB3 PPO helper")
    parser.add_argument("--smoke", action="store_true", help="Run a quick PPO smoke test.")
    parser.add_argument("--timesteps", type=int, default=512, help="Smoke-test PPO timesteps.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke:
        smoke_test(total_timesteps=args.timesteps)
