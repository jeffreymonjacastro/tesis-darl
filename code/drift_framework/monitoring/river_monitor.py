"""
drift_framework/monitoring/river_monitor.py

Online drift detection using River's ADWIN algorithm.

Simulates a streaming evaluation: feeds the model's binary prediction error
(correct=0, incorrect=1) row-by-row to ADWIN and logs the index at which
drift is flagged.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from river import drift, stream


class RiverMonitor:
    """Online drift detector using ADWIN on the model's prediction error stream.

    ADWIN (Adaptive Windowing) maintains a shrinking window and detects when
    the mean error rate shifts significantly between sub-windows.

    Args:
        delta: ADWIN's confidence parameter. Smaller = more sensitive.
               Recommended range: 0.002–0.1.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta

    def detect(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
    ) -> dict:
        """Stream (X, y) through the model and detect error-rate drift with ADWIN.

        Predictions are computed in batch upfront to avoid per-row model overhead;
        the error stream is then fed one step at a time into ADWIN.

        Args:
            X: feature DataFrame to stream.
            y: true labels (0/1) for the stream.
            predict_fn: callable that accepts a DataFrame and returns a 1-D
                        array of 0/1 predictions (e.g. ``pipeline.predict``).

        Returns:
            dict with keys:

            - ``"drift_indices"``: list of row indices where drift was flagged.
            - ``"n_drift_detections"``: int count of detections.
            - ``"first_drift_index"``: int index of first detection, or ``None``.
        """
        adwin = drift.ADWIN(delta=self.delta)
        drift_indices: list[int] = []

        y_arr = y.values
        # Batch-predict all rows upfront (much faster than one-by-one for sklearn/XGBoost)
        y_pred_all = predict_fn(X)

        for i, (_x_dict, _y_val) in enumerate(stream.iter_pandas(X, y)):
            error = int(y_pred_all[i] != y_arr[i])
            adwin.update(error)
            if adwin.drift_detected:
                drift_indices.append(i)
                print(f"[RiverMonitor] ADWIN drift flagged at index {i}")

        result = {
            "drift_indices": drift_indices,
            "n_drift_detections": len(drift_indices),
            "first_drift_index": drift_indices[0] if drift_indices else None,
        }

        print(
            f"[RiverMonitor] Total ADWIN detections: {len(drift_indices)} | "
            f"First at index: {result['first_drift_index']}"
        )
        return result


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    n = 500
    X_test = pd.DataFrame({"f": rng.normal(0, 1, n)})
    y_test = pd.Series(rng.integers(0, 2, n))

    def fake_predict(X: pd.DataFrame) -> np.ndarray:
        # Perfect predictions (0% error) for first 250 rows,
        # then purely random predictions (~50% error) for remaining 250.
        preds = y_test.values.copy()          # mirror true labels → 0% error
        preds[250:] = rng.integers(0, 2, 250) # random after index 250 → ~50% error
        return preds

    monitor = RiverMonitor(delta=0.002)
    result = monitor.detect(X_test, y_test, fake_predict)
    print("First drift at:", result["first_drift_index"])
