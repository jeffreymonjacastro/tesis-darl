"""
darl.drift.concept_injector
----------------------------
ConceptDriftInjector: structured concept drift via logit-shift amplification
of a reference logistic regression model, plus a symmetric label-flip
"noise_control" mechanism kept only as a negative control.

Usage
-----
from darl.drift import ConceptDriftInjector

inj = ConceptDriftInjector(random_state=42)
inj.fit(df_train, target_vars=VITALS, label_col="SepsisLabel")
df_drift, metadata = inj.transform(df_target, severity=0.5, mechanism="logit_shift")
summary = ConceptDriftInjector.summary(metadata)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from darl.types.types import _ConceptMeta

EPS = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ConceptDriftInjector:
    """
    Fits a reference logistic regression P(Y|X) on train data and injects
    concept drift on target data by amplifying (`logit_shift`) or discarding
    (`noise_control`) that learned relationship. Never modifies X.

    Parameters
    ----------
    random_state : int
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.target_vars: list[str] = []
        self.label_col: str | None = None
        self.coef_: dict[str, float] = {}
        self.intercept_: float = 0.0
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None

    # ─── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df_train: pd.DataFrame,
        target_vars: list[str],
        label_col: str,
    ) -> "ConceptDriftInjector":
        """Fit a logistic regression on standardized `target_vars` -> `label_col`."""
        self.target_vars = list(target_vars)
        self.label_col = label_col

        clean = df_train[self.target_vars + [label_col]].dropna()
        X = clean[self.target_vars]
        y = clean[label_col]

        self._mean = X.mean()
        self._std = X.std().replace(0.0, EPS)

        Z = (X - self._mean) / self._std

        model = LogisticRegression(max_iter=1000)
        model.fit(Z, y)

        self.coef_ = dict(zip(self.target_vars, model.coef_.ravel().tolist()))
        self.intercept_ = float(model.intercept_[0])
        return self

    # ─── Reference model ──────────────────────────────────────────────────────

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Missing vitals are treated as population-mean (zero contribution to the logit).
        return ((df[self.target_vars] - self._mean) / self._std).fillna(0.0)

    def _logit0(self, df: pd.DataFrame) -> np.ndarray:
        Z = self._standardize(df)
        coefs = pd.Series(self.coef_)
        return (self.intercept_ + Z.mul(coefs).sum(axis=1)).to_numpy()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(Y=1|X) under the frozen reference model (X is never drifted)."""
        return _sigmoid(self._logit0(df))

    # ─── Transform ────────────────────────────────────────────────────────────

    def transform(
        self,
        df_target: pd.DataFrame,
        severity: float,
        mechanism: str = "logit_shift",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Apply concept drift to `df_target[label_col]`. X is never modified.

        Parameters
        ----------
        df_target : DataFrame to perturb.
        severity : float — drift severity (0 = no drift).
        mechanism : "logit_shift" | "noise_control"
        """
        if self.label_col is None:
            raise RuntimeError("ConceptDriftInjector must be fit() before transform().")

        df = df_target.copy()
        y = df[self.label_col].to_numpy(dtype=float)
        before_prev = float(y.mean())
        n = len(df)

        logit0 = self._logit0(df)

        if mechanism == "logit_shift":
            delta = logit0 - self.intercept_
            logit_new = logit0 + severity * delta
            p0 = _sigmoid(logit0)
            p_new = _sigmoid(logit_new)
            p_flip = np.abs(p_new - p0)
            resample_mask = self._rng.binomial(1, p_flip).astype(bool)
            resampled_labels = self._rng.binomial(1, p_new)
            y_drift = np.where(resample_mask, resampled_labels, y)
        elif mechanism == "noise_control":
            p_flip_scalar = 0.45 * severity
            p_flip = np.full(n, p_flip_scalar)
            flip_mask = self._rng.binomial(1, p_flip_scalar, n).astype(bool)
            y_drift = np.where(flip_mask, 1 - y, y)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism!r}")

        df[self.label_col] = y_drift.astype(df_target[self.label_col].dtype)

        after_prev = float(y_drift.mean())
        metadata = {
            self.label_col: _ConceptMeta(
                col=self.label_col,
                mechanism=mechanism,
                drift_severity=severity,
                p_flip_mean=float(np.mean(p_flip)),
                n_flipped=int(np.sum(y_drift != y)),
                before_prevalence=before_prev,
                after_prevalence=after_prev,
            )
        }
        return df, metadata

    # ─── Summary ──────────────────────────────────────────────────────────────

    @staticmethod
    def summary(metadata: dict[str, _ConceptMeta]) -> pd.DataFrame:
        """Return a tidy DataFrame with per-label concept-drift metrics."""
        rows = []
        for col, meta in metadata.items():
            row: dict[str, Any] = {
                "variable": col,
                "mechanism": meta.mechanism,
                "severity": meta.drift_severity,
                "p_flip_mean": meta.p_flip_mean,
                "n_flipped": meta.n_flipped,
                "before_prev": meta.before_prevalence,
                "after_prev": meta.after_prevalence,
                **meta.extra,
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("variable")
