from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from darl.pipeline import apply_stage1, fit_stage1, make_logreg


def eval_metrics(model, x: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    """Evaluate AUC, AUPR and F1 at a fixed threshold."""
    y_prob = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, y_prob)
    aupr = average_precision_score(y, y_prob)
    f1 = f1_score(y, (y_prob >= threshold).astype(int))
    return {"auc": auc, "aupr": aupr, "f1": f1}


def fit_reference_quantile_map(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    cols: list[str],
    n_quantiles: int = 501,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Learn a marginal source-to-reference quantile map per numeric column."""
    probs = np.linspace(0.0, 1.0, n_quantiles)
    qmap = {}

    for col in cols:
        if col not in source_df.columns or col not in reference_df.columns:
            continue

        src = source_df[col].dropna().to_numpy(dtype=float)
        ref = reference_df[col].dropna().to_numpy(dtype=float)
        if len(src) < 2 or len(ref) < 2:
            continue

        src_q = np.quantile(src, probs)
        ref_q = np.quantile(ref, probs)
        src_q, idx = np.unique(src_q, return_index=True)
        ref_q = ref_q[idx]
        if len(src_q) >= 2:
            qmap[col] = (src_q, ref_q)

    return qmap


def apply_reference_quantile_map(
    df: pd.DataFrame,
    qmap: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Map drifted columns back to the marginal reference scale."""
    out = df.copy()

    for col, (src_q, ref_q) in qmap.items():
        values = out[col].to_numpy(dtype=float, copy=True)
        mask = np.isfinite(values)
        values[mask] = np.interp(
            values[mask], src_q, ref_q, left=ref_q[0], right=ref_q[-1]
        )
        out[col] = values

    return out


def fit_reference_location_scale_map(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    cols: list[str],
    *,
    robust: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    """Learn source-to-reference location and scale corrections per column.

    When ``robust`` is false, each correction matches mean and standard
    deviation.  When true, it matches median and interquartile range (IQR),
    which is less sensitive to Beta-injected extreme values.  The fitted
    statistics use no labels and do not alter the predictive model.
    """
    maps: dict[str, tuple[float, float, float, float]] = {}

    for col in cols:
        if col not in source_df.columns or col not in reference_df.columns:
            continue

        source = source_df[col].dropna().to_numpy(dtype=float)
        reference = reference_df[col].dropna().to_numpy(dtype=float)
        if len(source) < 2 or len(reference) < 2:
            continue

        if robust:
            source_location = float(np.median(source))
            reference_location = float(np.median(reference))
            source_scale = float(np.subtract(*np.percentile(source, [75, 25])))
            reference_scale = float(
                np.subtract(*np.percentile(reference, [75, 25]))
            )
        else:
            source_location = float(np.mean(source))
            reference_location = float(np.mean(reference))
            source_scale = float(np.std(source))
            reference_scale = float(np.std(reference))

        if source_scale > np.finfo(float).eps and reference_scale > 0:
            maps[col] = (
                source_location,
                source_scale,
                reference_location,
                reference_scale,
            )

    return maps


def apply_reference_location_scale_map(
    df: pd.DataFrame,
    maps: dict[str, tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Align mapped columns to reference location and scale, preserving NaNs."""
    out = df.copy()

    for col, (
        source_location,
        source_scale,
        reference_location,
        reference_scale,
    ) in maps.items():
        values = out[col].to_numpy(dtype=float, copy=True)
        mask = np.isfinite(values)
        values[mask] = (
            (values[mask] - source_location)
            / source_scale
            * reference_scale
            + reference_location
        )
        out[col] = values

    return out


def run_a1(
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
    """A1: keep the original Stage 1 and Stage 2 frozen."""
    df_t = apply_stage1(df_drifted_target, qt, imputer, scaler, vitals, numeric_cols)
    x = df_t[numeric_cols].values
    y = df_drifted_target[label_col].values
    metrics = eval_metrics(model, x, y, threshold)
    return metrics, 0.0


def run_a2(
    df_drifted_target,
    df_drifted_train,
    model,
    vitals,
    numeric_cols,
    label_col,
    threshold,
    seed=42,
):
    """A2: refit Stage 1 on drifted data and keep Stage 2 frozen."""
    t0 = time.perf_counter()
    qt_new, imputer_new, scaler_new = fit_stage1(
        df_drifted_train, vitals, numeric_cols, seed
    )
    t_fit = time.perf_counter() - t0

    df_t = apply_stage1(
        df_drifted_target, qt_new, imputer_new, scaler_new, vitals, numeric_cols
    )
    x = df_t[numeric_cols].values
    y = df_drifted_target[label_col].values
    metrics = eval_metrics(model, x, y, threshold)
    return metrics, t_fit


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
    """A2c: correct drift toward reference, then use frozen Stage 1 and Stage 2."""
    t0 = time.perf_counter()
    qmap = fit_reference_quantile_map(df_drifted_train, df_reference_train, vitals)
    df_corrected_target = apply_reference_quantile_map(df_drifted_target, qmap)
    t_update = time.perf_counter() - t0

    df_t = apply_stage1(
        df_corrected_target, qt_ref, imputer_ref, scaler_ref, vitals, numeric_cols
    )
    x = df_t[numeric_cols].values
    y = df_drifted_target[label_col].values
    metrics = eval_metrics(model, x, y, threshold)
    return metrics, t_update


def run_a3(
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
    """A3: keep Stage 1 frozen and retrain Logistic Regression."""
    df_tr_t = apply_stage1(
        df_drifted_train, qt_ref, imputer_ref, scaler_ref, vitals, numeric_cols
    )
    x_new = df_tr_t[numeric_cols].values
    y_new = df_drifted_train[label_col].values

    model_new = make_logreg(seed)
    t0 = time.perf_counter()
    model_new.fit(x_new, y_new)
    t_fit = time.perf_counter() - t0

    df_t = apply_stage1(
        df_drifted_target, qt_ref, imputer_ref, scaler_ref, vitals, numeric_cols
    )
    x = df_t[numeric_cols].values
    y = df_drifted_target[label_col].values
    metrics = eval_metrics(model_new, x, y, threshold)
    return metrics, t_fit


def run_a4(
    df_drifted_target,
    df_drifted_train,
    vitals,
    numeric_cols,
    label_col,
    threshold,
    seed=42,
):
    """A4: refit Stage 1 and retrain Logistic Regression."""
    t0_stage1 = time.perf_counter()
    qt_new, imputer_new, scaler_new = fit_stage1(
        df_drifted_train, vitals, numeric_cols, seed
    )
    t_stage1 = time.perf_counter() - t0_stage1

    df_tr_t = apply_stage1(
        df_drifted_train, qt_new, imputer_new, scaler_new, vitals, numeric_cols
    )
    x_new = df_tr_t[numeric_cols].values
    y_new = df_drifted_train[label_col].values

    model_new = make_logreg(seed)
    t0_model = time.perf_counter()
    model_new.fit(x_new, y_new)
    t_model = time.perf_counter() - t0_model

    df_t = apply_stage1(
        df_drifted_target, qt_new, imputer_new, scaler_new, vitals, numeric_cols
    )
    x = df_t[numeric_cols].values
    y = df_drifted_target[label_col].values
    metrics = eval_metrics(model_new, x, y, threshold)
    return metrics, t_stage1 + t_model
