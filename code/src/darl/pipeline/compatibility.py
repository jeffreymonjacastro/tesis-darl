"""Formal distribution contract between Stage 1 representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from darl.pipeline.logreg_stage import apply_stage1


@dataclass(frozen=True)
class CompatibilityReport:
    """Wasserstein contract result for two Stage 1 representations."""

    is_compatible: bool
    dimension_match: bool
    distribution_divergence: float
    max_feature_divergence: float
    per_feature_divergence: dict[str, float]
    mismatches: list[str]


def check_stage1_compatibility(
    stage1_old: tuple[Any, Any, Any],
    stage1_new: tuple[Any, Any, Any],
    X_sample: pd.DataFrame,
    vitals: list[str],
    numeric_cols: list[str],
    threshold: float = 0.5,
    max_threshold: float | None = None,
    reference_sample: pd.DataFrame | None = None,
) -> CompatibilityReport:
    """Compare old/reference and new/candidate Stage 1 output distributions."""
    max_allowed = threshold if max_threshold is None else max_threshold
    old_input = X_sample if reference_sample is None else reference_sample
    try:
        old_frame = apply_stage1(old_input, *stage1_old, vitals, numeric_cols)
        new_frame = apply_stage1(X_sample, *stage1_new, vitals, numeric_cols)
    except Exception as exc:
        return CompatibilityReport(
            False,
            False,
            float("inf"),
            float("inf"),
            {},
            [f"Stage 1 transformation failed: {exc}"],
        )

    if old_frame.shape[1] != new_frame.shape[1]:
        return CompatibilityReport(
            False,
            False,
            float("inf"),
            float("inf"),
            {},
            ["Stage 1 output dimensions differ"],
        )

    divergences: dict[str, float] = {}
    for column in numeric_cols:
        old_values = old_frame[column].dropna().to_numpy(dtype=float)
        new_values = new_frame[column].dropna().to_numpy(dtype=float)
        divergence = (
            float("inf")
            if len(old_values) == 0 or len(new_values) == 0
            else float(wasserstein_distance(old_values, new_values))
        )
        divergences[column] = divergence

    mean_divergence = float(np.mean(list(divergences.values())))
    max_divergence = float(np.max(list(divergences.values())))
    mismatches = [name for name, value in divergences.items() if value > max_allowed]
    compatible = mean_divergence <= threshold and max_divergence <= max_allowed
    return CompatibilityReport(
        compatible,
        True,
        mean_divergence,
        max_divergence,
        divergences,
        mismatches,
    )
