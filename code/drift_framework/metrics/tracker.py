"""
drift_framework/metrics/tracker.py

Tracks performance and compute metrics across experiment runs.

Records AUC-ROC, F1-score, delta AUC, retraining time (seconds), and peak RAM
usage (MB). Results can be exported to CSV.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class ExperimentRecord:
    """One row of experiment results."""

    dataset: str
    model: str
    drift_type: str
    severity: str
    start_idx: int
    auc_before: float
    auc_after: float
    delta_auc: float
    f1_before: float
    f1_after: float
    retrain_time_sec: float
    ram_mb: float


class MetricsTracker:
    """Records per-run metrics across multiple experiment configurations.

    Usage::

        tracker = MetricsTracker()
        with timed_ram_block() as m:
            pipeline.fit(X, y)
        tracker.record(
            dataset="adult", model="xgboost", drift_type="covariate",
            severity="high", start_idx=0,
            y_true_before=y_pre, y_prob_before=proba_pre,
            y_true_after=y_post, y_prob_after=proba_post,
            retrain_time_sec=m["elapsed_sec"], ram_mb=m["peak_ram_mb"],
        )
        tracker.to_csv("results/results.csv")
    """

    def __init__(self):
        self._records: List[ExperimentRecord] = []

    def record(
        self,
        dataset: str,
        model: str,
        drift_type: str,
        severity: str,
        start_idx: int,
        y_true_before: pd.Series,
        y_prob_before: np.ndarray,
        y_true_after: pd.Series,
        y_prob_after: np.ndarray,
        retrain_time_sec: float,
        ram_mb: float,
    ) -> ExperimentRecord:
        """Compute and store metrics for one experiment run.

        Args:
            dataset: dataset name (e.g. ``"adult"``).
            model: model type (e.g. ``"xgboost"``).
            drift_type: injected drift type (e.g. ``"covariate"``).
            severity: drift severity (e.g. ``"high"``).
            start_idx: row index where drift was injected.
            y_true_before: ground-truth labels on the pre-drift split.
            y_prob_before: predicted probabilities (positive class) on the pre-drift split.
            y_true_after: ground-truth labels after drift injection.
            y_prob_after: predicted probabilities on the drifted post-drift split.
            retrain_time_sec: elapsed seconds for any retraining.
            ram_mb: peak RAM usage in MB during retraining.

        Returns:
            The ExperimentRecord that was stored.
        """
        auc_before = roc_auc_score(y_true_before, y_prob_before)
        auc_after = roc_auc_score(y_true_after, y_prob_after)
        delta_auc = auc_before - auc_after

        f1_before = f1_score(y_true_before, (y_prob_before >= 0.5).astype(int))
        f1_after = f1_score(y_true_after, (y_prob_after >= 0.5).astype(int))

        rec = ExperimentRecord(
            dataset=dataset,
            model=model,
            drift_type=drift_type,
            severity=severity,
            start_idx=start_idx,
            auc_before=auc_before,
            auc_after=auc_after,
            delta_auc=delta_auc,
            f1_before=f1_before,
            f1_after=f1_after,
            retrain_time_sec=retrain_time_sec,
            ram_mb=ram_mb,
        )
        self._records.append(rec)
        print(
            f"[MetricsTracker] {dataset}/{model}/{drift_type}/{severity}: "
            f"AUC {auc_before:.4f} -> {auc_after:.4f}  (delta={delta_auc:+.4f})  "
            f"time={retrain_time_sec:.2f}s  RAM={ram_mb:.1f}MB"
        )
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        """Return all recorded experiments as a pandas DataFrame."""
        return pd.DataFrame([vars(r) for r in self._records])

    def to_csv(self, path: str) -> None:
        """Write all recorded experiments to a CSV file.

        Args:
            path: file path for the output CSV.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[MetricsTracker] Saved {len(df)} records to {path}")


@contextmanager
def timed_ram_block() -> Generator[dict, None, None]:
    """Context manager that measures wall-clock time and peak RAM usage.

    Yields a dict that is populated on exit with:
      - ``"elapsed_sec"``: float seconds elapsed.
      - ``"peak_ram_mb"``: float peak memory in MB (via tracemalloc).

    Usage::

        with timed_ram_block() as m:
            model.fit(X, y)
        print(m["elapsed_sec"], m["peak_ram_mb"])
    """
    result: dict = {}
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - t0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["elapsed_sec"] = elapsed
        result["peak_ram_mb"] = peak / (1024 ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_true = pd.Series(rng.integers(0, 2, 200))
    y_prob = rng.random(200)

    tracker = MetricsTracker()
    with timed_ram_block() as m:
        time.sleep(0.01)  # simulate work

    tracker.record(
        dataset="adult", model="xgboost",
        drift_type="covariate", severity="high", start_idx=0,
        y_true_before=y_true[:100], y_prob_before=y_prob[:100],
        y_true_after=y_true[100:], y_prob_after=y_prob[100:],
        retrain_time_sec=m["elapsed_sec"], ram_mb=m["peak_ram_mb"],
    )
    print(tracker.to_dataframe().to_string())
