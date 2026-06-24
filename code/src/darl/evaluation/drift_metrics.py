"""
darl.evaluation.drift_metrics
------------------------------
Drift metrics for numeric and categorical variables.
"""
import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-10


# ─── Numeric ──────────────────────────────────────────────────────────────────

def ks_stat(before: pd.Series, after: pd.Series) -> dict:
    """Two-sample KS test."""
    stat, pval = stats.ks_2samp(before.dropna(), after.dropna())
    return {"ks_stat": stat, "ks_pval": pval}


def psi_numeric(before: pd.Series, after: pd.Series, n_bins: int = 10) -> float:
    """PSI over equal-width bins computed from *before* distribution."""
    lo, hi = before.min(), before.max()
    bins = np.linspace(lo, hi, n_bins + 1)
    p0 = np.histogram(before.dropna(), bins=bins)[0] / len(before.dropna()) + EPS
    p1 = np.histogram(after.dropna(), bins=bins)[0] / len(after.dropna()) + EPS
    p0, p1 = p0 / p0.sum(), p1 / p1.sum()
    return float(np.sum((p1 - p0) * np.log(p1 / p0)))


# ─── Categorical ──────────────────────────────────────────────────────────────

def _align(p0: pd.Series, p1: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align two frequency series to same index, fill missing with EPS."""
    idx = p0.index.union(p1.index)
    a = np.asarray(p0.reindex(idx, fill_value=0).to_numpy(dtype=np.float64), dtype=np.float64) + float(EPS)
    b = np.asarray(p1.reindex(idx, fill_value=0).to_numpy(dtype=np.float64), dtype=np.float64) + float(EPS)
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
