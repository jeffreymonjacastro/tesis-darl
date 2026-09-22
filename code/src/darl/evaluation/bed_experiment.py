"""Reproducible A-to-B hourly ICU experiment and policy comparisons."""

from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from darl.data.bed_stream import BedConfig, SCENARIOS, VITALS, inject_scenario, make_bed_schedule, patient_partitions
from darl.rl.bed_env import BedDAREnvironment, BedPipeline, prediction_metrics
from darl.rl.course_dqn import CourseDQN


def run_episode(schedule: pd.DataFrame, model: BedPipeline, policy: str, *, agent: CourseDQN | None = None, days: int = 21, seed: int = 42, train: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Replay one scenario with a selected policy and delayed online learning."""
    env = BedDAREnvironment(schedule, model, days=days)
    observation, _ = env.reset()
    rng = np.random.default_rng(seed)
    for _ in range(max(0, days - 1)):
        if policy == "dqn":
            if agent is None:
                raise ValueError("DQN policy requires an agent")
            action = agent.select(observation, greedy=not train)
        elif policy == "a1":
            action = 0
        elif policy == "a4":
            action = 3
        elif policy == "random":
            action = int(rng.integers(4))
        elif policy == "threshold":
            action = 3 if observation[-7] > 0.12 else 0
        else:
            raise ValueError(f"Unknown policy: {policy}")
        observation, _, done, _, info = env.step(action)
        if train and agent is not None:
            agent.learn(info["matured"])
        if done:
            break
    if train and agent is not None:
        agent.learn(env.flush_rewards())
        agent.end_episode()
    predictions = pd.concat(env.hourly_predictions, ignore_index=True) if env.hourly_predictions else pd.DataFrame()
    return env.finish(), pd.DataFrame(env.actions), predictions


def run_experiment(parquet_path: str | Path, output_dir: str | Path, *, version: str = "v2", device: str = "cuda", seed: int = 42) -> dict:
    """Train on A, tune DQN on B-train/val, and evaluate reserved B-test."""
    if version not in {"v1", "v2"}:
        raise ValueError("version must be v1 or v2")
    data_path = Path(parquet_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "run_summary.json"
    progress_path.write_text(json.dumps({"status": "running", "stage": "load_inputs", "version": version}), encoding="utf-8")
    frame = pd.read_parquet(data_path, columns=["patient_id", "source_set", "ICULOS", "SepsisLabel", *VITALS])
    partition = patient_partitions(frame, seed=seed)
    a_train = frame.loc[frame.patient_id.isin(partition["a_train"])].copy()
    if version == "v1":
        sampled = np.random.default_rng(seed).choice(sorted(partition["a_train"]), size=500, replace=False)
        a_train = a_train.loc[a_train.patient_id.isin(sampled)]
    model = BedPipeline.fit_initial(a_train, n_estimators=40 if version == "v1" else 80, device=device, seed=seed)
    progress_path.write_text(json.dumps({"status": "running", "stage": "train_dqn", "version": version}), encoding="utf-8")
    a_metrics = {}
    for split in ("a_val", "a_test"):
        rows = frame.loc[frame.patient_id.isin(partition[split])].copy()
        rows["probability"] = model.predict(rows)
        a_metrics[split] = prediction_metrics(rows)
    agent = CourseDQN(device=device)
    days = 3 if version == "v1" else 21
    beds = 200
    train_scenarios = ("clean", "combined") if version == "v1" else SCENARIOS
    train_repeats = 1 if version == "v1" else 3
    history = []
    candidates = []
    for repeat in range(train_repeats):
        schedule = make_bed_schedule(frame, partition["b_train_update"], partition["b_train_eval"], BedConfig(beds=beds, days=days, seed=seed + repeat))
        for scenario in train_scenarios:
            shifted = inject_scenario(schedule, scenario)
            metrics, actions, _ = run_episode(shifted, model, "dqn", agent=agent, days=days, seed=seed + repeat, train=True)
            history.append({"partition": "b_train", "seed": seed + repeat, "scenario": scenario, **metrics})
        candidates.append(copy.deepcopy(agent))
    validation = []
    progress_path.write_text(json.dumps({"status": "running", "stage": "validation", "n_training_episodes": len(history)}), encoding="utf-8")
    val_schedule = make_bed_schedule(frame, partition["b_val_update"], partition["b_val_eval"], BedConfig(beds=beds, days=days, seed=seed))
    for checkpoint, candidate in enumerate(candidates, start=1):
        for scenario in train_scenarios:
            metrics, _, _ = run_episode(inject_scenario(val_schedule, scenario), model, "dqn", agent=candidate, days=days)
            validation.append({"partition": "b_val", "checkpoint": checkpoint, "scenario": scenario, **metrics})
    validation_table = pd.DataFrame(validation)
    chosen_checkpoint = int(validation_table.groupby("checkpoint")["return"].mean().idxmax())
    selected_agent = candidates[chosen_checkpoint - 1]
    # Final test is untouched during policy training and hyperparameter selection.
    results = []
    progress_path.write_text(json.dumps({"status": "running", "stage": "final_test", "n_training_episodes": len(history)}), encoding="utf-8")
    action_rows = []
    policies = ("dqn", "a1", "a4", "random", "threshold")
    seeds = [seed] if version == "v1" else [seed + i for i in range(5)]
    for test_seed in seeds:
        schedule = make_bed_schedule(frame, partition["b_test_update"], partition["b_test_eval"], BedConfig(beds=beds, days=days, seed=test_seed))
        for scenario in SCENARIOS:
            shifted = inject_scenario(schedule, scenario)
            for policy in policies:
                metrics, actions, predictions = run_episode(shifted, model, policy, agent=selected_agent if policy == "dqn" else None, days=days, seed=test_seed)
                results.append({"seed": test_seed, "scenario": scenario, "policy": policy, **metrics})
                actions.insert(0, "seed", test_seed)
                actions.insert(1, "scenario", scenario)
                actions.insert(2, "policy", policy)
                action_rows.append(actions)
                if policy == "dqn":
                    predictions.to_parquet(output / f"predictions_{scenario}_{test_seed}.parquet", index=False)
            pd.DataFrame(results).to_csv(output / "final_metrics.partial.csv", index=False)
            progress_path.write_text(json.dumps({"status": "running", "stage": "final_test", "completed_runs": len(results)}), encoding="utf-8")
    result_table = pd.DataFrame(results)
    result_table.to_csv(output / "final_metrics.csv", index=False)
    pd.concat(action_rows, ignore_index=True).to_csv(output / "actions.csv", index=False)
    pd.DataFrame(history + validation).to_csv(output / "learning_history.csv", index=False)
    aggregate = result_table.groupby(["scenario", "policy"])[["auprc_hour", "auprc_patient", "auroc_hour", "physionet_utility_observed_segment", "return", "n_updates", "time_s", "drift_detection_day"]].mean().round(6).reset_index()
    aggregate_rows = aggregate.astype(object).where(pd.notna(aggregate), None).to_dict(orient="records")
    summary = {
        "status": "complete", "version": version, "device": device,
        "n_input_rows": len(frame), "n_input_patients": frame.patient_id.nunique(),
        "n_beds": beds, "days": days, "test_seeds": seeds,
        "partitions": {key: len(ids) for key, ids in partition.items()},
        "a_metrics": a_metrics, "n_training_episodes": len(history),
        "n_test_runs": len(results), "dqn_updates": agent.updates,
        "selected_checkpoint": chosen_checkpoint,
        "test_aggregate": aggregate_rows,
        "files": ["final_metrics.csv", "actions.csv", "learning_history.csv"],
        "limitations": ["Simulated days are not calendar dates", "Last observed row is not a confirmed discharge", "Synthetic label drift has no clinical interpretation"],
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
