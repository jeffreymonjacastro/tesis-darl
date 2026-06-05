"""
drift_framework/monitoring/evidently_monitor.py

Batch drift detection using Evidently (v0.4+).

Computes per-feature KS test statistic and PSI between reference and post-drift
windows, and exports results as a JSON-serialisable dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

# Evidently 0.7+ legacy API (DataDriftPreset was moved to evidently.legacy in 0.7)
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


class EvidentlyMonitor:
    """Computes batch data drift between a reference and current dataset.

    Uses Evidently's DataDriftPreset, which applies KS test for numeric
    features and chi-squared for categoricals by default.

    Args:
        num_features: list of numeric column names.
        cat_features: list of categorical column names.
        stattest: statistical test override (e.g. ``"ks"``, ``"psi"``).
                  Applies to all features when set; otherwise Evidently
                  picks per-type defaults.
    """

    def __init__(
        self,
        num_features: list[str],
        cat_features: list[str],
        stattest: Optional[str] = "ks",
    ):
        self.num_features = num_features
        self.cat_features = cat_features
        self.stattest = stattest

        self._column_mapping = ColumnMapping(
            target=None,
            numerical_features=num_features,
            categorical_features=cat_features,
        )

    def detect(
        self,
        X_reference: pd.DataFrame,
        X_current: pd.DataFrame,
        export_path: Optional[str] = None,
    ) -> dict:
        """Run drift detection and return results.

        Args:
            X_reference: reference (training) feature DataFrame.
            X_current: current (post-drift) feature DataFrame.
            export_path: if provided, write JSON results to this path.

        Returns:
            dict with keys:

            - ``"dataset_drift"``: bool, True if overall drift is detected.
            - ``"n_drifted_features"``: int count of drifted features.
            - ``"drift_by_feature"``: dict mapping feature name to
              ``{drift_detected, stattest, stat_value, p_value}``.
        """
        preset_kwargs = {}
        if self.stattest is not None:
            preset_kwargs["stattest"] = self.stattest

        report = Report(metrics=[DataDriftPreset(**preset_kwargs)])
        report.run(
            reference_data=X_reference,
            current_data=X_current,
            column_mapping=self._column_mapping,
        )

        raw = report.as_dict()

        # Evidently 0.7+: DataDriftPreset produces two metric entries.
        # metrics[0] = DatasetDriftMetric (dataset-level summary)
        # metrics[1] = DataDriftTable     (per-column breakdown with drift_by_columns)
        summary_result = raw["metrics"][0]["result"]
        table_result = raw["metrics"][1]["result"] if len(raw["metrics"]) > 1 else summary_result

        dataset_drift = summary_result.get("dataset_drift", False)
        n_drifted = summary_result.get("number_of_drifted_columns", 0)
        drift_by_col = table_result.get("drift_by_columns", {})

        drift_by_feature: dict = {}
        for col_name, col_data in drift_by_col.items():
            drift_by_feature[col_name] = {
                "drift_detected": col_data.get("drift_detected", False),
                "stattest": col_data.get("stattest_name", self.stattest),
                "stat_value": col_data.get("drift_score", None),  # Evidently 0.7 uses drift_score
                "p_value": col_data.get("p_value", None),
            }

        result = {
            "dataset_drift": dataset_drift,
            "n_drifted_features": n_drifted,
            "drift_by_feature": drift_by_feature,
        }

        if export_path is not None:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"[EvidentlyMonitor] Drift report saved to {export_path}")

        drifted = [k for k, v in drift_by_feature.items() if v["drift_detected"]]
        print(
            f"[EvidentlyMonitor] Dataset drift: {dataset_drift} | "
            f"Drifted features: {n_drifted} | "
            f"Examples: {drifted[:5]}"
        )

        return result


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "a": rng.normal(0, 1, 500),
        "b": rng.normal(5, 2, 500),
    })
    # Feature 'a' is shifted by 3 std; 'b' is unchanged
    cur = pd.DataFrame({
        "a": rng.normal(3, 1, 200),
        "b": rng.normal(5, 2, 200),
    })

    monitor = EvidentlyMonitor(num_features=["a", "b"], cat_features=[])
    result = monitor.detect(ref, cur, export_path="results/test_evidently.json")
    drifted = [k for k, v in result["drift_by_feature"].items() if v["drift_detected"]]
    print("Drifted features:", drifted)
