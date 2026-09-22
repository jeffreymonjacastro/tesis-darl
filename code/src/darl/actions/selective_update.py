"""Selective maintenance actions for the DARL two-stage pipeline."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from darl.pipeline.compatibility import CompatibilityReport, check_stage1_compatibility
from darl.pipeline.logreg_stage import apply_stage1, fit_stage1, make_logreg
from darl.pipeline.xgb_stage import XGBStage2

ACTION_CODES = ("A1", "A2", "A3", "A4")


@dataclass
class ActionResult:
    """Outcome, cost and deployed state resulting from one maintenance action."""

    requested_action: str
    executed_action: str
    pipeline_state: object
    auc_before: float
    auc_after: float
    time_s: float
    peak_ram_mb: float
    fallback_reason: str | None = None
    compatibility: CompatibilityReport | None = None


def eval_metrics(
    model,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate AUC, AUPR and F1 at a fixed threshold."""
    probabilities = model.predict_proba(x)[:, 1]
    auc = float("nan") if np.unique(y).size < 2 else float(roc_auc_score(y, probabilities))
    return {
        "auc": auc,
        "aupr": float(average_precision_score(y, probabilities)),
        "f1": float(f1_score(y, probabilities >= threshold, zero_division=0)),
    }


def fit_reference_quantile_map(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    cols: list[str],
    n_quantiles: int = 501,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Learn a marginal source-to-reference quantile map per column."""
    probabilities = np.linspace(0.0, 1.0, n_quantiles)
    mapping: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column in cols:
        if column not in source_df or column not in reference_df:
            continue
        source = source_df[column].dropna().to_numpy(dtype=float)
        reference = reference_df[column].dropna().to_numpy(dtype=float)
        if len(source) < 2 or len(reference) < 2:
            continue
        source_quantiles = np.quantile(source, probabilities)
        reference_quantiles = np.quantile(reference, probabilities)
        source_quantiles, indices = np.unique(source_quantiles, return_index=True)
        if len(source_quantiles) >= 2:
            mapping[column] = (source_quantiles, reference_quantiles[indices])
    return mapping


def apply_reference_quantile_map(
    df: pd.DataFrame,
    qmap: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Map drifted columns back to reference marginal quantiles."""
    output = df.copy()
    for column, (source, reference) in qmap.items():
        values = output[column].to_numpy(dtype=float, copy=True)
        valid = np.isfinite(values)
        values[valid] = np.interp(
            values[valid], source, reference, left=reference[0], right=reference[-1]
        )
        output[column] = values
    return output


def fit_reference_location_scale_map(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    cols: list[str],
    *,
    robust: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    """Learn location-scale corrections; robust mode uses median and IQR."""
    mapping: dict[str, tuple[float, float, float, float]] = {}
    for column in cols:
        if column not in source_df or column not in reference_df:
            continue
        source = source_df[column].dropna().to_numpy(dtype=float)
        reference = reference_df[column].dropna().to_numpy(dtype=float)
        if len(source) < 2 or len(reference) < 2:
            continue
        if robust:
            source_location = float(np.median(source))
            reference_location = float(np.median(reference))
            source_scale = float(np.subtract(*np.percentile(source, [75, 25])))
            reference_scale = float(np.subtract(*np.percentile(reference, [75, 25])))
        else:
            source_location = float(np.mean(source))
            reference_location = float(np.mean(reference))
            source_scale = float(np.std(source))
            reference_scale = float(np.std(reference))
        if source_scale > np.finfo(float).eps and reference_scale > 0:
            mapping[column] = (
                source_location,
                source_scale,
                reference_location,
                reference_scale,
            )
    return mapping


def apply_reference_location_scale_map(
    df: pd.DataFrame,
    maps: dict[str, tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Apply fitted location-scale corrections while preserving missing values."""
    output = df.copy()
    for column, values_map in maps.items():
        source_location, source_scale, reference_location, reference_scale = values_map
        values = output[column].to_numpy(dtype=float, copy=True)
        valid = np.isfinite(values)
        values[valid] = (
            (values[valid] - source_location)
            / source_scale
            * reference_scale
            + reference_location
        )
        output[column] = values
    return output


def measure_execution(function: Callable, *args, **kwargs):
    """Return function result, elapsed seconds and Python peak allocated RAM."""
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = function(*args, **kwargs)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, float(elapsed), float(peak / (1024 * 1024))


def _new_xgb_like(model: XGBStage2, seed: int) -> XGBStage2:
    return XGBStage2(
        n_estimators=model.n_estimators,
        max_depth=model.max_depth,
        learning_rate=model.learning_rate,
        seed=seed,
        n_jobs=model.n_jobs,
        device=model.device,
    )


def execute_action(
    action: int | str,
    pipeline_state,
    update_window: pd.DataFrame,
    eval_window: pd.DataFrame,
    reference_frame: pd.DataFrame,
    mean_threshold: float = 0.5,
    max_threshold: float = 0.5,
    fallback: str = "A4",
    seed: int = 42,
) -> ActionResult:
    """Execute one causal maintenance action and return a new pipeline state."""
    requested = ACTION_CODES[int(action)] if isinstance(action, (int, np.integer)) else str(action).upper()
    if requested not in ACTION_CODES:
        raise ValueError(f"Unknown action: {action}")
    if fallback not in {"A3", "A4"}:
        raise ValueError("fallback must be A3 or A4")

    before_auc = pipeline_state.evaluate(eval_window)
    candidate = pipeline_state.clone()
    compatibility: CompatibilityReport | None = None
    executed = requested
    fallback_reason: str | None = None

    def apply_selected() -> None:
        nonlocal candidate, compatibility, executed, fallback_reason
        if requested == "A1":
            return
        if requested == "A2":
            adapter = fit_reference_location_scale_map(
                update_window,
                reference_frame,
                candidate.vitals,
                robust=True,
            )
            mapped_update = apply_reference_location_scale_map(update_window, adapter)
            compatibility = check_stage1_compatibility(
                candidate.stage1,
                candidate.stage1,
                mapped_update,
                candidate.vitals,
                candidate.numeric_cols,
                threshold=mean_threshold,
                max_threshold=max_threshold,
                reference_sample=reference_frame,
            )
            if compatibility.is_compatible:
                candidate.adapter_map = adapter
                candidate.stage1_version += 1
                return
            executed = fallback
            fallback_reason = (
                "A2 contract failed: mean="
                f"{compatibility.distribution_divergence:.4f}, "
                f"max={compatibility.max_feature_divergence:.4f}"
            )

        if requested == "A3" or executed == "A3":
            transformed = candidate.transform(update_window)
            model = _new_xgb_like(candidate.stage2, seed)
            model.fit(
                transformed[candidate.numeric_cols],
                update_window[candidate.label_col].to_numpy(),
            )
            candidate.stage2 = model
            candidate.stage2_version += 1
            return

        if requested == "A4" or executed == "A4":
            candidate.stage1 = fit_stage1(
                update_window,
                candidate.vitals,
                candidate.numeric_cols,
                seed,
            )
            candidate.adapter_map = {}
            transformed = apply_stage1(
                update_window,
                *candidate.stage1,
                candidate.vitals,
                candidate.numeric_cols,
            )
            model = _new_xgb_like(candidate.stage2, seed)
            model.fit(
                transformed[candidate.numeric_cols],
                update_window[candidate.label_col].to_numpy(),
            )
            candidate.stage2 = model
            candidate.stage1_version += 1
            candidate.stage2_version += 1

    _, elapsed, peak_ram = measure_execution(apply_selected)
    after_auc = candidate.evaluate(eval_window)
    return ActionResult(
        requested_action=requested,
        executed_action=executed,
        pipeline_state=candidate,
        auc_before=float(before_auc),
        auc_after=float(after_auc),
        time_s=elapsed,
        peak_ram_mb=peak_ram,
        fallback_reason=fallback_reason,
        compatibility=compatibility,
    )


def run_a1_xgb(
    df_drifted_target,
    qt,
    imputer,
    scaler,
    model,
    vitals,
    numeric_cols,
    label_col,
    threshold,
):
    """Evaluate frozen Stage 1 and XGBoost Stage 2."""
    def run():
        transformed = apply_stage1(
            df_drifted_target, qt, imputer, scaler, vitals, numeric_cols
        )
        return eval_metrics(
            model,
            transformed[numeric_cols].to_numpy(),
            df_drifted_target[label_col].to_numpy(),
            threshold,
        )

    return measure_execution(run)


def run_a2_xgb(
    df_drifted_target,
    df_drifted_train,
    df_reference_train,
    qt_ref,
    imputer_ref,
    scaler_ref,
    model,
    vitals,
    numeric_cols,
    label_col,
    threshold,
    seed=42,
):
    """Apply robust median-IQR A2 and keep both fitted stages frozen."""
    del seed

    def run():
        mapping = fit_reference_location_scale_map(
            df_drifted_train, df_reference_train, vitals, robust=True
        )
        mapped = apply_reference_location_scale_map(df_drifted_target, mapping)
        transformed = apply_stage1(
            mapped, qt_ref, imputer_ref, scaler_ref, vitals, numeric_cols
        )
        return eval_metrics(
            model,
            transformed[numeric_cols].to_numpy(),
            df_drifted_target[label_col].to_numpy(),
            threshold,
        )

    return measure_execution(run)


def run_a3_xgb(
    df_drifted_target,
    df_drifted_train,
    qt_ref,
    imputer_ref,
    scaler_ref,
    vitals,
    numeric_cols,
    label_col,
    threshold,
    seed=42,
):
    """Keep Stage 1 frozen and retrain XGBoost."""
    def run():
        train = apply_stage1(
            df_drifted_train, qt_ref, imputer_ref, scaler_ref, vitals, numeric_cols
        )
        model = XGBStage2(seed=seed).fit(
            train[numeric_cols], df_drifted_train[label_col].to_numpy()
        )
        target = apply_stage1(
            df_drifted_target, qt_ref, imputer_ref, scaler_ref, vitals, numeric_cols
        )
        return eval_metrics(
            model,
            target[numeric_cols].to_numpy(),
            df_drifted_target[label_col].to_numpy(),
            threshold,
        )

    return measure_execution(run)


def run_a4_xgb(
    df_drifted_target,
    df_drifted_train,
    vitals,
    numeric_cols,
    label_col,
    threshold,
    seed=42,
):
    """Refit Stage 1 and retrain XGBoost."""
    def run():
        stage1 = fit_stage1(df_drifted_train, vitals, numeric_cols, seed)
        train = apply_stage1(
            df_drifted_train, *stage1, vitals, numeric_cols
        )
        model = XGBStage2(seed=seed).fit(
            train[numeric_cols], df_drifted_train[label_col].to_numpy()
        )
        target = apply_stage1(
            df_drifted_target, *stage1, vitals, numeric_cols
        )
        return eval_metrics(
            model,
            target[numeric_cols].to_numpy(),
            df_drifted_target[label_col].to_numpy(),
            threshold,
        )

    return measure_execution(run)


def run_a2_with_contract(
    df_drifted_target,
    df_drifted_train,
    df_reference_train,
    qt_ref,
    imputer_ref,
    scaler_ref,
    model,
    vitals,
    numeric_cols,
    label_col,
    threshold,
    seed=42,
    fallback="RETRAIN_ALL",
):
    """Run robust A2 only when its formal Wasserstein contract passes."""
    mapping = fit_reference_location_scale_map(
        df_drifted_train, df_reference_train, vitals, robust=True
    )
    mapped_train = apply_reference_location_scale_map(df_drifted_train, mapping)
    stage1 = (qt_ref, imputer_ref, scaler_ref)
    report = check_stage1_compatibility(
        stage1,
        stage1,
        mapped_train,
        vitals,
        numeric_cols,
        reference_sample=df_reference_train,
    )
    if report.is_compatible:
        return run_a2_xgb(
            df_drifted_target,
            df_drifted_train,
            df_reference_train,
            qt_ref,
            imputer_ref,
            scaler_ref,
            model,
            vitals,
            numeric_cols,
            label_col,
            threshold,
            seed,
        )
    if fallback == "RETRAIN_ALL":
        return run_a4_xgb(
            df_drifted_target,
            df_drifted_train,
            vitals,
            numeric_cols,
            label_col,
            threshold,
            seed,
        )
    if fallback == "RETRAIN_MODEL":
        return run_a3_xgb(
            df_drifted_target,
            df_drifted_train,
            qt_ref,
            imputer_ref,
            scaler_ref,
            vitals,
            numeric_cols,
            label_col,
            threshold,
            seed,
        )
    raise ValueError(f"Unknown fallback: {fallback}")


def run_a2_corrective(
    df_drifted_target,
    df_drifted_train,
    df_reference_train,
    qt_ref,
    imputer_ref,
    scaler_ref,
    model,
    vitals,
    numeric_cols,
    label_col,
    threshold,
):
    """Compatibility implementation of quantile-mapping A2."""
    started = time.perf_counter()
    mapping = fit_reference_quantile_map(
        df_drifted_train, df_reference_train, vitals
    )
    corrected = apply_reference_quantile_map(df_drifted_target, mapping)
    elapsed = time.perf_counter() - started
    transformed = apply_stage1(
        corrected, qt_ref, imputer_ref, scaler_ref, vitals, numeric_cols
    )
    metrics = eval_metrics(
        model,
        transformed[numeric_cols].to_numpy(),
        df_drifted_target[label_col].to_numpy(),
        threshold,
    )
    return metrics, elapsed


def run_a1(*args, **kwargs):
    """Compatibility alias for frozen logistic-regression evaluation."""
    return _run_legacy_logreg("A1", *args, **kwargs)


def run_a2(*args, **kwargs):
    """Compatibility alias for Stage 1 logistic-regression refit."""
    return _run_legacy_logreg("A2", *args, **kwargs)


def run_a3(*args, **kwargs):
    """Compatibility alias for Stage 2 logistic-regression refit."""
    return _run_legacy_logreg("A3", *args, **kwargs)


def run_a4(*args, **kwargs):
    """Compatibility alias for full logistic-regression refit."""
    return _run_legacy_logreg("A4", *args, **kwargs)


def _run_legacy_logreg(action: str, *args, **kwargs):
    """Retain notebook compatibility for pre-XGBoost action functions."""
    if action == "A1":
        target, qt, imputer, scaler, model, vitals, columns, label, threshold = args
        transformed = apply_stage1(target, qt, imputer, scaler, vitals, columns)
        return eval_metrics(model, transformed[columns].to_numpy(), target[label], threshold), 0.0
    if action == "A2":
        target, train, model, vitals, columns, label, threshold = args[:7]
        seed = kwargs.get("seed", args[7] if len(args) > 7 else 42)
        started = time.perf_counter()
        stage1 = fit_stage1(train, vitals, columns, seed)
        elapsed = time.perf_counter() - started
        transformed = apply_stage1(target, *stage1, vitals, columns)
        return eval_metrics(model, transformed[columns], target[label], threshold), elapsed
    target, train, *rest = args
    if action == "A3":
        qt, imputer, scaler, vitals, columns, label, threshold = rest[:7]
        seed = kwargs.get("seed", rest[7] if len(rest) > 7 else 42)
        stage1 = (qt, imputer, scaler)
    else:
        vitals, columns, label, threshold = rest[:4]
        seed = kwargs.get("seed", rest[4] if len(rest) > 4 else 42)
        stage1 = fit_stage1(train, vitals, columns, seed)
    train_transformed = apply_stage1(train, *stage1, vitals, columns)
    model = make_logreg(seed)
    started = time.perf_counter()
    model.fit(train_transformed[columns], train[label])
    elapsed = time.perf_counter() - started
    target_transformed = apply_stage1(target, *stage1, vitals, columns)
    return eval_metrics(model, target_transformed[columns], target[label], threshold), elapsed
