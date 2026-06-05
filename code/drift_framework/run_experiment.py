"""
drift_framework/run_experiment.py

Main experiment runner for the drift-aware selective updating framework.

Steps per run:
  1. Load dataset
  2. Train baseline pipeline (no drift)
  3. Inject drift using DriftInjector
  4. Evaluate model performance on post-drift data (no retraining)
  5. Run Evidently batch drift detection (reference vs. drifted post-drift)
  6. Run River ADWIN online drift detection (streaming over drifted post-drift)
  7. Log all metrics to results/experiment_results.csv

Usage::

    python -m drift_framework.run_experiment \\
        --dataset adult --model xgboost \\
        --drift_type covariate --severity high --start_idx 0

    python drift_framework/run_experiment.py  # uses defaults
"""

from __future__ import annotations

import argparse
from pathlib import Path

from code.drift_framework.data.loader import load_dataset
from code.drift_framework.drift.injector import DriftInjector
from code.drift_framework.metrics.tracker import MetricsTracker, timed_ram_block
from code.drift_framework.monitoring.evidently_monitor import EvidentlyMonitor
from code.drift_framework.monitoring.river_monitor import RiverMonitor
from code.drift_framework.pipeline.two_stage import TwoStagePipeline

RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "experiment_results.csv"


def run_experiment(
    dataset_name: str = "adult",
    model_type: str = "xgboost",
    drift_type: str = "covariate",
    severity: str = "high",
    start_idx: int = 0,
    cache_dir: str = "tableshift_cache",
) -> dict:
    """Run one complete experiment: load → train → inject → detect → record.

    Args:
        dataset_name: ``"adult"`` or ``"diabetes_readmission"``.
        model_type: ``"logreg"`` or ``"xgboost"``.
        drift_type: ``"covariate"``, ``"concept"``, or ``"both"``.
        severity: ``"low"``, ``"medium"``, or ``"high"``.
        start_idx: row index in the post-drift split where injection begins.
        cache_dir: tableshift local data cache directory.

    Returns:
        dict with all logged metrics for this experiment run.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    bundle = load_dataset(dataset_name, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # 2. Train baseline pipeline on reference split (no drift)
    # ------------------------------------------------------------------
    print(f"\n--- Training baseline pipeline ({model_type}) ---")
    pipe = TwoStagePipeline(
        model_type=model_type,
        num_features=bundle.num_features,
        cat_features=bundle.cat_features,
    )

    with timed_ram_block() as train_metrics:
        pipe.fit(bundle.X_ref, bundle.y_ref)

    # ------------------------------------------------------------------
    # 3. Evaluate on pre-drift test split (baseline performance)
    # ------------------------------------------------------------------
    proba_pre = pipe.predict_proba(bundle.X_pre)[:, 1]

    # ------------------------------------------------------------------
    # 4. Inject drift into post-drift split
    # ------------------------------------------------------------------
    print(f"\n--- Injecting drift: type={drift_type}, severity={severity} ---")
    injector = DriftInjector(
        drift_type=drift_type,
        severity=severity,
        start_idx=start_idx,
    )
    injector.fit(bundle.X_ref, bundle.num_features)
    X_post_drifted, y_post_drifted = injector.inject(bundle.X_post, bundle.y_post)

    # ------------------------------------------------------------------
    # 5. Evaluate (no retraining) on drifted post-drift data
    # ------------------------------------------------------------------
    proba_post = pipe.predict_proba(X_post_drifted)[:, 1]

    # ------------------------------------------------------------------
    # 6. Evidently batch drift detection
    # ------------------------------------------------------------------
    print("\n--- Running Evidently batch drift detection ---")
    ev_monitor = EvidentlyMonitor(
        num_features=bundle.num_features,
        cat_features=bundle.cat_features,
    )
    ev_result = ev_monitor.detect(
        X_reference=bundle.X_ref,
        X_current=X_post_drifted,
        export_path=str(
            RESULTS_DIR / f"evidently_{dataset_name}_{drift_type}_{severity}.json"
        ),
    )

    # ------------------------------------------------------------------
    # 7. River ADWIN online drift detection
    # ------------------------------------------------------------------
    print("\n--- Running River ADWIN online drift detection ---")
    river_monitor = RiverMonitor(delta=0.002)
    river_result = river_monitor.detect(
        X=X_post_drifted,
        y=y_post_drifted,
        predict_fn=pipe.predict,
    )

    # ------------------------------------------------------------------
    # 8. Record metrics
    # ------------------------------------------------------------------
    tracker = MetricsTracker()
    tracker.record(
        dataset=dataset_name,
        model=model_type,
        drift_type=drift_type,
        severity=severity,
        start_idx=start_idx,
        y_true_before=bundle.y_pre,
        y_prob_before=proba_pre,
        y_true_after=y_post_drifted,
        y_prob_after=proba_post,
        retrain_time_sec=train_metrics["elapsed_sec"],
        ram_mb=train_metrics["peak_ram_mb"],
    )

    # ------------------------------------------------------------------
    # 9. Append to CSV (append mode; write header only on first run)
    # ------------------------------------------------------------------
    df = tracker.to_dataframe()
    df["evidently_dataset_drift"] = ev_result["dataset_drift"]
    df["evidently_n_drifted_features"] = ev_result["n_drifted_features"]
    df["river_first_drift_index"] = river_result["first_drift_index"]
    df["river_n_detections"] = river_result["n_drift_detections"]

    write_header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)
    print(f"\nResults appended to {RESULTS_CSV}")

    return df.to_dict(orient="records")[0]


def main() -> None:
    """Parse CLI arguments and run the experiment."""
    parser = argparse.ArgumentParser(
        description="Drift-aware selective updating experiment runner"
    )
    parser.add_argument(
        "--dataset",
        default="adult",
        choices=["adult", "diabetes_readmission"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["logreg", "xgboost"],
        help="Predictive model for Stage 2.",
    )
    parser.add_argument(
        "--drift_type",
        default="covariate",
        choices=["covariate", "concept", "both"],
        help="Type of synthetic drift to inject.",
    )
    parser.add_argument(
        "--severity",
        default="high",
        choices=["low", "medium", "high"],
        help="Drift severity.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Row index in the post-drift split where injection starts.",
    )
    parser.add_argument(
        "--cache_dir",
        default="tableshift_cache",
        help="Local directory for tableshift data cache.",
    )
    args = parser.parse_args()

    result = run_experiment(
        dataset_name=args.dataset,
        model_type=args.model,
        drift_type=args.drift_type,
        severity=args.severity,
        start_idx=args.start_idx,
        cache_dir=args.cache_dir,
    )

    print("\n=== Experiment complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
