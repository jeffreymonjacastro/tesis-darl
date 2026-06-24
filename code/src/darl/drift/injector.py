"""
darl.drift.injector
-------------------
DriftInjector: synthetic covariate shift via Beta mixture (numeric)
and categorical marginal resampling.

Usage
-----
from darl.drift import DriftInjector

inj = DriftInjector(random_state=42)
inj.fit(df_train, numeric_cols, categorical_cols)
df_drift, metadata = inj.transform(df_target, alpha=0.3)
summary = inj.summary(metadata)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from darl.evaluation import (
    ks_stat,
    psi_numeric,
    psi_categorical,
    js_divergence,
    hellinger,
    chi2_test,
)

EPS = 1e-6

# ─── Beta target presets ───────────────────────────────────────────────────────
BETA_TARGETS = {
    "high": (5.0, 2.0),
    "low": (2.0, 5.0),
    "central": (8.0, 8.0),
    "extreme": (0.5, 0.5),
}


@dataclass
class _NumericMeta:
    col: str
    col_min: float
    col_max: float
    alpha_0: float  # Beta original shape a
    beta_0: float  # Beta original shape b
    direction: str  # high/low/central/extreme
    alpha_q: float  # Beta target shape a
    beta_q: float  # Beta target shape b
    drift_severity: float  # α mixture weight
    extra: dict = field(default_factory=dict)  # KS, PSI after transform


@dataclass
class _CatMeta:
    col: str
    p0: pd.Series  # original distribution (proportions)
    q: pd.Series  # target distribution (proportions)
    p_alpha: pd.Series  # mixed distribution
    drift_severity: float
    strategy: str
    extra: dict = field(default_factory=dict)  # PSI, JS, Hellinger, chi2


@dataclass
class _LabelMeta:
    col: str
    drift_severity: float
    p_flip: float
    before_prevalence: float
    after_prevalence: float
    extra: dict = field(default_factory=dict)


class DriftInjector:
    """
    Fits on train data, applies Beta-mixture numeric drift and
    categorical marginal-resampling drift on target data.

    Parameters
    ----------
    random_state : int
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._numeric_meta: dict[str, _NumericMeta] = {}
        self._cat_meta_fit: dict[str, pd.Series] = {}  # p0 per col
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []

    # ─── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df_train: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str] | None = None,
    ) -> "DriftInjector":
        """Compute min/max + Beta params from train split."""
        self._numeric_cols = list(numeric_cols)
        self._categorical_cols = list(categorical_cols or [])

        for col in self._numeric_cols:
            clean = df_train[col].dropna()
            col_min, col_max = float(clean.min()), float(clean.max())
            z = np.clip((clean - col_min) / (col_max - col_min + EPS), EPS, 1 - EPS)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a0, b0, _, _ = stats.beta.fit(z, floc=0, fscale=1)

            self._numeric_meta[col] = _NumericMeta(
                col=col,
                col_min=col_min,
                col_max=col_max,
                alpha_0=a0 if a0 else 1.0,
                beta_0=b0 if b0 else 1.0,
                direction="",
                alpha_q=0.0,
                beta_q=0.0,
                drift_severity=0.0,
            )

        for col in self._categorical_cols:
            self._cat_meta_fit[col] = df_train[col].value_counts(
                normalize=True, dropna=True
            )

        return self

    # ─── Transform ────────────────────────────────────────────────────────────

    def transform(
        self,
        df_target: pd.DataFrame,
        drift_severity: float,
        numeric_drift_config: dict[str, str] | None = None,
        categorical_drift_config: dict[str, dict] | None = None,
        label_col: str | None = None,
        drift_type: str = "covariate",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Apply synthetic drift.

        Parameters
        ----------
        df_target : DataFrame to perturb.
        drift_severity : float in [0, 1] — drift severity.
        numeric_drift_config : {col: direction} where direction ∈
            {"high", "low", "central", "extreme"}.  Defaults to "high".
        categorical_drift_config : {col: config_dict}.
            Each config_dict may have keys:
              strategy: "uniform" | "manual_boost" | "rare" | "swap_top"
              increase: {cat: multiplier}   (manual_boost only)
              decrease: {cat: multiplier}   (manual_boost only)
        label_col : str | None
            The binary label column name to apply label flipping to.
        drift_type : "covariate" | "concept" | "both"
            The type of drift to inject.
        """
        numeric_drift_config = numeric_drift_config or {}
        categorical_drift_config = categorical_drift_config or {}

        df = df_target.copy()
        metadata: dict[str, Any] = {}

        # ── concept / label shift (label flipping) ──
        if (
            (drift_type in ("concept", "both"))
            and label_col
            and (label_col in df.columns)
        ):
            p_flip = 0.45 * drift_severity
            before_prev = float(df[label_col].mean())

            n = len(df)
            mask = self._rng.binomial(1, p_flip, n).astype(bool)
            df.loc[mask, label_col] = 1 - df.loc[mask, label_col]

            after_prev = float(df[label_col].mean())
            metadata[label_col] = _LabelMeta(
                col=label_col,
                drift_severity=drift_severity,
                p_flip=p_flip,
                before_prevalence=before_prev,
                after_prevalence=after_prev,
                extra={"psi": 0.0},
            )

        # ── numeric ──
        if drift_type in ("covariate", "both"):
            for col in self._numeric_cols:
                if col not in df.columns:
                    continue
                meta = self._numeric_meta[col]
                direction = numeric_drift_config.get(col, "high")
                aq, bq = BETA_TARGETS[direction]
                meta.direction, meta.alpha_q, meta.beta_q, meta.drift_severity = (
                    direction,
                    aq,
                    bq,
                    drift_severity,
                )

                before = df[col].copy()
                df[col] = self._apply_numeric(df[col], meta, drift_severity)
                after = df[col].copy()

                meta.extra = {
                    **ks_stat(before.dropna(), after.dropna()),
                    "psi": psi_numeric(before.dropna(), after.dropna()),
                }
                metadata[col] = meta

        # ── categorical — resample rows ──
        if (
            (drift_type in ("covariate", "both"))
            and self._categorical_cols
            and categorical_drift_config
        ):
            df, cat_metas = self._apply_categorical(
                df, drift_severity, categorical_drift_config, df_target
            )
            metadata.update(cat_metas)

        return df, metadata

    # ─── Numeric internals ────────────────────────────────────────────────────

    def _apply_numeric(
        self,
        series: pd.Series,
        meta: _NumericMeta,
        drift_severity: float,
    ) -> pd.Series:
        out = series.copy()
        mask = out.notna()
        z = np.clip(
            (out[mask] - meta.col_min) / (meta.col_max - meta.col_min + EPS),
            EPS,
            1 - EPS,
        )
        n = mask.sum()
        # Bernoulli mask
        m = self._rng.binomial(1, drift_severity, n).astype(bool)
        z_tilde = self._rng.beta(meta.alpha_q, meta.beta_q, n)
        z_drift = np.where(m, z_tilde, z)
        # back to original scale
        out[mask] = z_drift * (meta.col_max - meta.col_min) + meta.col_min
        return out

    # ─── Categorical internals ────────────────────────────────────────────────

    def _apply_categorical(
        self,
        df: pd.DataFrame,
        drift_severity: float,
        drift_config: dict[str, dict],
        df_original: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, _CatMeta]]:
        """Row-resample with combined importance weights across all cat cols."""
        metas: dict[str, _CatMeta] = {}
        weights = np.ones(len(df))

        for col in self._categorical_cols:
            if col not in drift_config or col not in df.columns:
                continue
            p0 = self._cat_meta_fit[col]
            q = self._build_q(p0, drift_config[col])
            p_alpha = (1 - drift_severity) * p0 + drift_severity * q
            p_alpha /= p_alpha.sum()

            # per-row importance ratio  p_α(x) / p_0(x)
            row_cats = df[col].astype(str)
            p0_map = p0.rename(index=str)
            pa_map = p_alpha.rename(index=str)
            w_col = row_cats.map(pa_map).fillna(EPS) / row_cats.map(p0_map).fillna(EPS)
            weights *= np.asarray(w_col.values, dtype=np.float64)

            metas[col] = _CatMeta(
                col=col,
                p0=p0,
                q=q,
                p_alpha=p_alpha,
                drift_severity=drift_severity,
                strategy=drift_config[col].get("strategy", ""),
            )

        # clip + normalise weights
        w_lo, w_hi = np.percentile(weights, 5), np.percentile(weights, 95)
        weights = np.clip(weights, w_lo if w_lo > 0 else EPS, w_hi)
        weights = weights / weights.sum()

        df_resampled = df.sample(
            n=len(df), replace=True, weights=weights, random_state=self.random_state
        ).reset_index(drop=True)

        # compute post-resample metrics
        for col, meta in metas.items():
            p_after = df_resampled[col].value_counts(normalize=True, dropna=True)
            meta.extra = {
                "psi": psi_categorical(meta.p0, p_after),
                "js": js_divergence(meta.p0, p_after),
                "hellinger": hellinger(meta.p0, p_after),
                **chi2_test(
                    meta.p0 * len(df_original),
                    p_after * len(df_resampled),
                ),
            }
        return df_resampled, metas

    @staticmethod
    def _build_q(p0: pd.Series, cfg: dict) -> pd.Series:
        """Build target distribution q from config."""
        strategy = cfg.get("strategy", "uniform")
        q = p0.copy()

        if strategy == "uniform":
            q[:] = 1.0 / len(q)

        elif strategy == "manual_boost":
            for cat, mult in cfg.get("increase", {}).items():
                if cat in q.index:
                    q[cat] *= mult
            for cat, mult in cfg.get("decrease", {}).items():
                if cat in q.index:
                    q[cat] *= mult

        elif strategy == "rare":
            # invert: give more weight to rare categories
            q = 1.0 / (p0 + EPS)

        elif strategy == "swap_top":
            # swap top-1 and top-2
            sorted_idx = p0.sort_values(ascending=False).index
            if len(sorted_idx) >= 2:
                q[sorted_idx[0]], q[sorted_idx[1]] = (
                    p0[sorted_idx[1]],
                    p0[sorted_idx[0]],
                )

        q = q.clip(lower=EPS)
        return q / q.sum()

    # ─── Summary ──────────────────────────────────────────────────────────────

    @staticmethod
    def summary(metadata: dict[str, Any]) -> pd.DataFrame:
        """Return a tidy DataFrame with per-variable drift metrics."""
        rows = []
        for col, meta in metadata.items():
            row: dict[str, Any] = {"variable": col}
            if isinstance(meta, _NumericMeta):
                row.update(
                    {
                        "type": "numeric",
                        "drift_severity": meta.drift_severity,
                        "direction": meta.direction,
                        "beta_orig": f"({meta.alpha_0:.2f}, {meta.beta_0:.2f})",
                        "beta_target": f"({meta.alpha_q:.2f}, {meta.beta_q:.2f})",
                        **meta.extra,
                    }
                )
            elif isinstance(meta, _CatMeta):
                row.update(
                    {
                        "type": "categorical",
                        "drift_severity": meta.drift_severity,
                        "strategy": meta.strategy,
                        **meta.extra,
                    }
                )
            elif isinstance(meta, _LabelMeta):
                row.update(
                    {
                        "type": "label",
                        "drift_severity": meta.drift_severity,
                        "strategy": f"flip (p={meta.p_flip:.2f})",
                        "before_prev": f"{meta.before_prevalence:.3f}",
                        "after_prev": f"{meta.after_prevalence:.3f}",
                        **meta.extra,
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("variable")
