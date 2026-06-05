"""
drift_framework/drift/injector.py

Synthetic drift injection for tabular ML experiments.

Supports covariate shift (changes to P(X)), concept drift (changes to P(Y|X)),
or both simultaneously, with configurable severity and temporal scope.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 42

# Severity multipliers for covariate shift (added to feature mean in std units)
_SEVERITY_COVARIATE: dict[str, float] = {
    "low": 0.5,
    "medium": 1.5,
    "high": 3.0,
}

# Label flip probabilities for concept drift (capped at 45%)
_SEVERITY_CONCEPT: dict[str, float] = {
    "low": 0.08,
    "medium": 0.25,
    "high": 0.45,
}


class DriftInjector:
    """Injects synthetic drift into a dataset split.

    Supports three drift modes:
    - ``"covariate"``: shifts numeric feature distributions by severity × std.
    - ``"concept"``: randomly flips labels with probability proportional to severity.
    - ``"both"``: applies covariate shift and label flipping simultaneously.

    Args:
        drift_type: one of ``"covariate"``, ``"concept"``, ``"both"``.
        severity: one of ``"low"``, ``"medium"``, ``"high"``.
        start_idx: integer row index at which drift begins.
        duration: number of rows affected. ``None`` means permanent drift
                  (all rows from ``start_idx`` onward are affected).
        random_state: integer seed for reproducibility.
    """

    def __init__(
        self,
        drift_type: Literal["covariate", "concept", "both"],
        severity: Literal["low", "medium", "high"],
        start_idx: int = 0,
        duration: Optional[int] = None,
        random_state: int = SEED,
    ):
        if drift_type not in ("covariate", "concept", "both"):
            raise ValueError(
                f"drift_type must be 'covariate', 'concept', or 'both'; got '{drift_type}'"
            )
        if severity not in _SEVERITY_COVARIATE:
            raise ValueError(
                f"severity must be 'low', 'medium', or 'high'; got '{severity}'"
            )

        self.drift_type = drift_type
        self.severity = severity
        self.start_idx = start_idx
        self.duration = duration
        self.random_state = random_state

        self._covariate_multiplier = _SEVERITY_COVARIATE[severity]
        self._flip_prob = _SEVERITY_CONCEPT[severity]
        self._num_features: list[str] = []
        self._feature_stds: Optional[dict[str, float]] = None

    def fit(self, X_ref: pd.DataFrame, num_features: list[str]) -> "DriftInjector":
        """Compute reference statistics needed for drift injection.

        Must be called with the training (reference) split before ``inject()``.

        Args:
            X_ref: reference (training) feature DataFrame.
            num_features: list of numeric column names to be shifted.

        Returns:
            self
        """
        self._num_features = [c for c in num_features if c in X_ref.columns]
        self._feature_stds = {
            col: float(X_ref[col].std()) for col in self._num_features
        }
        return self

    def inject(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply drift to the specified row range of (X, y).

        Args:
            X: feature DataFrame (will not be mutated).
            y: label Series (will not be mutated).

        Returns:
            ``(X_drifted, y_drifted)``: copies with drift applied to affected rows.

        Raises:
            RuntimeError: if ``fit()`` has not been called yet.
        """
        if self._feature_stds is None:
            raise RuntimeError("Call fit() with reference data before inject().")

        rng = np.random.default_rng(self.random_state)

        X_out = X.copy()
        y_out = y.copy()

        end_idx = (
            min(self.start_idx + self.duration, len(X))
            if self.duration is not None
            else len(X)
        )
        affected_mask = np.zeros(len(X), dtype=bool)
        affected_mask[self.start_idx : end_idx] = True

        if self.drift_type in ("covariate", "both"):
            X_out = self._apply_covariate_shift(X_out, affected_mask)

        if self.drift_type in ("concept", "both"):
            y_out = self._apply_concept_drift(y_out, affected_mask, rng)

        return X_out, y_out

    def _apply_covariate_shift(
        self,
        X: pd.DataFrame,
        mask: np.ndarray,
    ) -> pd.DataFrame:
        """Shift numeric features by severity × std for affected rows."""
        for col in self._num_features:
            shift = self._covariate_multiplier * self._feature_stds[col]
            X.loc[mask, col] = X.loc[mask, col] + shift
        return X

    def _apply_concept_drift(
        self,
        y: pd.Series,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> pd.Series:
        """Randomly flip labels for affected rows with probability = flip_prob."""
        n_affected = int(mask.sum())
        flip_flags = rng.random(n_affected) < self._flip_prob
        affected_idxs = np.where(mask)[0]
        flip_idxs = affected_idxs[flip_flags]
        y_arr = y.values.copy()
        y_arr[flip_idxs] = 1 - y_arr[flip_idxs]  # binary 0/1 labels only
        return pd.Series(y_arr, index=y.index, name=y.name)


if __name__ == "__main__":
    from code.drift_framework.data.loader import load_dataset

    b = load_dataset("adult")

    injector = DriftInjector(drift_type="both", severity="high", start_idx=0)
    injector.fit(b.X_ref, b.num_features)
    X_d, y_d = injector.inject(b.X_post, b.y_post)

    first_num = b.num_features[0]
    print(f"Feature '{first_num}':")
    print(f"  Original mean : {b.X_post[first_num].mean():.4f}")
    print(f"  Drifted  mean : {X_d[first_num].mean():.4f}")
    print(f"Original label rate : {b.y_post.mean():.4f}")
    print(f"Drifted  label rate : {y_d.mean():.4f}")
