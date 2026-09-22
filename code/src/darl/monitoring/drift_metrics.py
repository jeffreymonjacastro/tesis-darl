"""
darl.monitoring.drift_metrics
------------------------------
Drift metrics for monitoring data drift.
"""

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-10


# ─── Numeric ──────────────────────────────────────────────────────────────────


def ks_stat(before: pd.Series, after: pd.Series) -> dict:
    """Two-sample KS test."""
    clean_before = before.dropna()
    clean_after = after.dropna()
    if clean_before.empty or clean_after.empty:
        return {"ks_stat": 0.0, "ks_pval": 1.0}
    stat, pval = stats.ks_2samp(clean_before, clean_after)
    return {"ks_stat": stat, "ks_pval": pval}


def psi_numeric(before: pd.Series, after: pd.Series, n_bins: int = 10) -> float:
    """PSI using reference quantile bins with open extreme intervals."""
    b_clean = before.dropna().to_numpy(dtype=float)
    a_clean = after.dropna().to_numpy(dtype=float)
    if len(b_clean) == 0 or len(a_clean) == 0:
        return 0.0
    bins = np.unique(np.quantile(b_clean, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(bins) < 2:
        return 0.0
    bins = bins.astype(float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    p0 = np.histogram(b_clean, bins=bins)[0] / len(b_clean) + EPS
    p1 = np.histogram(a_clean, bins=bins)[0] / len(a_clean) + EPS
    p0, p1 = p0 / p0.sum(), p1 / p1.sum()
    return float(np.sum((p1 - p0) * np.log(p1 / p0)))


# ─── Categorical ──────────────────────────────────────────────────────────────


def _align(p0: pd.Series, p1: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align two frequency series to same index, fill missing with EPS."""
    idx = p0.index.union(p1.index)
    a = np.asarray(
        p0.reindex(idx, fill_value=0).to_numpy(dtype=np.float64), dtype=np.float64
    ) + float(EPS)
    b = np.asarray(
        p1.reindex(idx, fill_value=0).to_numpy(dtype=np.float64), dtype=np.float64
    ) + float(EPS)
    return a / a.sum(), b / b.sum()


def psi_categorical(p0: pd.Series, p1: pd.Series) -> float:
    """PSI for categorical distributions."""
    a, b = _align(p0, p1)
    return float(np.sum((b - a) * np.log(b / a)))


def js_divergence(p0: pd.Series, p1: pd.Series) -> float:
    """Jensen-Shannon divergence (base-2, bounded [0,1])."""
    a, b = _align(p0, p1)
    m = 0.5 * (a + b)
    return float(0.5 * np.sum(a * np.log2(a / m)) + 0.5 * np.sum(b * np.log2(b / m)))


def hellinger(p0: pd.Series, p1: pd.Series) -> float:
    """Hellinger distance, bounded [0,1]."""
    a, b = _align(p0, p1)
    return float(np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / np.sqrt(2))


def chi2_test(before: pd.Series, after: pd.Series) -> dict:
    """Chi-square test comparing observed counts before vs after."""
    cats = before.dropna().index.union(after.dropna().index)
    obs_b = np.array([before.get(c, 0) for c in cats], dtype=float)
    obs_a = np.array([after.get(c, 0) for c in cats], dtype=float)
    # need counts, not proportions
    stat, pval = stats.chisquare(obs_a, f_exp=obs_b + EPS)
    return {"chi2_stat": stat, "chi2_pval": pval}
