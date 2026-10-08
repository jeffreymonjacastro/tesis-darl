"""
darl.visualization.drift_plots
-------------------------------
Before/after visualisations for numeric and categorical drift.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.figure import Figure

# ─── Numeric ──────────────────────────────────────────────────────────────────


def plot_numeric_drift(
    before: pd.Series,
    after: pd.Series,
    col: str,
    bins: int = 60,
    figsize: tuple = (10, 4),
) -> Figure:
    """Histogram overlay: before vs after numeric drift."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        before.dropna(),
        bins=bins,
        density=True,
        alpha=0.55,
        edgecolor="none",
        label="Antes del drift",
        color="#1f77b4",
    )
    ax.hist(
        after.dropna(),
        bins=bins,
        density=True,
        alpha=0.55,
        edgecolor="none",
        label="Después del drift",
        color="#ff7f0e",
    )
    ax.set_title(f"{col} — distribución antes/después", fontweight="bold")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_numeric_drift_grid(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cols: list[str],
    ncols: int = 2,
    bins: int = 50,
    figsize_per: tuple = (5, 3),
) -> Figure:
    """Grid of before/after histograms for all numeric cols."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
    )
    axes = np.array(axes).flatten()
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.hist(
            df_before[col].dropna(),
            bins=bins,
            density=True,
            alpha=0.55,
            edgecolor="none",
            label="Antes",
            color="#1f77b4",
        )
        ax.hist(
            df_after[col].dropna(),
            bins=bins,
            density=True,
            alpha=0.55,
            edgecolor="none",
            label="Después",
            color="#ff7f0e",
        )
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Drift numérico — antes vs después", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ─── Categorical ──────────────────────────────────────────────────────────────


def plot_categorical_drift(
    p_before: pd.Series,
    p_after: pd.Series,
    col: str,
    top_k: int = 15,
    figsize: tuple = (10, 5),
) -> Figure:
    """Horizontal bar chart overlay for categorical drift."""
    cats = (p_before + p_after).nlargest(top_k).index
    pb = p_before.reindex(cats, fill_value=0)
    pa = p_after.reindex(cats, fill_value=0)
    pb_vals = np.asarray(pb.to_numpy(dtype=float), dtype=float)
    pa_vals = np.asarray(pa.to_numpy(dtype=float), dtype=float)

    y = np.arange(len(cats))
    h = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        y + h / 2,
        pb_vals,
        h,
        label="Antes del drift",
        color="#1f77b4",
        alpha=0.8,
    )
    ax.barh(
        y - h / 2,
        pa_vals,
        h,
        label="Después del drift",
        color="#ff7f0e",
        alpha=0.8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(cats.astype(str), fontsize=9)
    ax.set_xlabel("Proporción")
    ax.set_title(f"{col} — distribución antes/después (top {top_k})", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_categorical_drift_grid(
    metas: dict,
    top_k: int = 12,
    figsize_per: tuple = (8, 4),
) -> list[Figure]:
    """One figure per categorical variable with before/after overlay."""
    figs = []
    for col, meta in metas.items():
        p_after = meta.extra.get("p_after")
        if p_after is None:
            continue
        fig = plot_categorical_drift(
            meta.p0, p_after, col, top_k=top_k, figsize=figsize_per
        )
        figs.append(fig)
    return figs
