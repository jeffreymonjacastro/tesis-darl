"""High-level orchestration for the reproducible PhysioNet DARL PoC."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from darl.data import (
    PatientSplit,
    ReferenceFrames,
    TemporalWindow,
    load_physionet_temporal,
    make_reference_frames,
    make_temporal_windows,
    model_feature_columns,
    split_patients,
)
from darl.drift import DriftInjector
from darl.monitoring import DriftMonitor, psi_numeric
from darl.pipeline import PipelineState, XGBStage2, apply_stage1, fit_stage1
from darl.rl import DARLEnvironment, TransitionTableBuilder

SEED = 42
DEFAULT_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]


@dataclass
class PhysioNetPoC:
    """Prepared data, pipeline, monitor and injector for DARL experiments."""

    data: pd.DataFrame
    split: PatientSplit
    reference: ReferenceFrames
    windows: list[TemporalWindow]
    feature_columns: list[str]
    vitals: list[str]
    pipeline_state: PipelineState
    monitor: DriftMonitor
    drift_injector: DriftInjector
    auc_base: float


def prepare_physionet_poc(
    max_patients: int | None = 5000,
    max_windows: int | None = None,
    n_bootstrap: int = 100,
    n_estimators: int = 200,
    seed: int = SEED,
) -> PhysioNetPoC:
    """Load PhysioNet and fit reference pipeline plus calibrated monitor."""
    data = load_physionet_temporal(max_patients=max_patients, seed=seed)
    split = split_patients(data, seed=seed)
    reference = make_reference_frames(data, split)
    windows = make_temporal_windows(data, split)
    if max_windows is not None:
        windows = windows[:max_windows]
    if not windows:
        raise ValueError("No usable temporal decision windows were produced")
    train_min = max(2, int(0.10 * len(reference.train)))
    evaluation_min = max(2, int(0.10 * len(reference.evaluation)))
    features = [
        column
        for column in model_feature_columns(data)
        if reference.train[column].notna().sum() >= train_min
        and reference.evaluation[column].notna().sum() >= evaluation_min
    ]
    vitals = [column for column in DEFAULT_VITALS if column in features]
    stage1 = fit_stage1(reference.train, vitals, features, seed)
    transformed = apply_stage1(reference.train, *stage1, vitals, features)
    stage2 = XGBStage2(n_estimators=n_estimators, seed=seed).fit(
        transformed[features], reference.train["SepsisLabel"].to_numpy()
    )
    state = PipelineState(stage1, stage2, vitals, features)
    auc_base = state.evaluate(reference.evaluation)
    monitor = DriftMonitor(
        reference.evaluation,
        features,
        predictor=state,
        seed=seed,
    )
    monitor.calibrate(n_bootstrap=n_bootstrap)
    injector = DriftInjector(random_state=seed).fit(
        reference.train,
        numeric_cols=features,
    )
    return PhysioNetPoC(
        data,
        split,
        reference,
        windows,
        features,
        vitals,
        state,
        monitor,
        injector,
        auc_base,
    )


def make_live_environment(
    poc: PhysioNetPoC,
    scenario: str = "natural",
    true_drift_type: str = "none",
    severity: float = 0.0,
    seed: int = SEED,
) -> DARLEnvironment:
    """Create a live causal environment from prepared PoC components."""
    return DARLEnvironment(
        poc.pipeline_state,
        poc.reference.train,
        poc.windows,
        poc.monitor,
        auc_base=poc.auc_base,
        drift_injector=poc.drift_injector,
        scenario=scenario,
        true_drift_type=true_drift_type,
        severity=severity,
        seed=seed,
    )


def collect_transitions(
    poc: PhysioNetPoC,
    scenario: str = "control",
    true_drift_type: str = "both",
    severity: float = 0.6,
    behavior: str = "random",
    seed: int = SEED,
) -> pd.DataFrame:
    """Collect counterfactual transitions from one real sequential episode."""
    env = make_live_environment(poc, scenario, true_drift_type, severity, seed)
    episode_id = f"{scenario}-{true_drift_type}-{severity:.2f}-{seed}"
    return TransitionTableBuilder(seed).build(env, episode_id, behavior)


def temporal_drift_summary(poc: PhysioNetPoC) -> pd.DataFrame:
    """Summarize natural change across ICULOS evaluation windows."""
    rows = []
    reference = poc.reference.evaluation
    for window in poc.windows:
        psi_values = [
            psi_numeric(reference[column], window.evaluation[column])
            for column in poc.feature_columns
        ]
        rows.append(
            {
                "window_start": window.start_hour,
                "window_end": window.end_hour,
                "n_samples": len(window.evaluation),
                "prevalence": float(window.evaluation["SepsisLabel"].mean()),
                "mean_psi": float(sum(psi_values) / len(psi_values)),
            }
        )
    return pd.DataFrame(rows)
