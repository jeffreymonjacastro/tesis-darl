"""Calibrated monitor that constructs DARL's observable belief proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from darl.monitoring.c2st import c2st_score
from darl.monitoring.drift_metrics import ks_stat, psi_categorical, psi_numeric
from darl.monitoring.drift_report import DriftReport


@dataclass(frozen=True)
class CalibrationResult:
    """Empirical H0 distributions and 95th-percentile thresholds."""

    thresholds: dict[str, float]
    distributions: dict[str, np.ndarray]


class DriftMonitor:
    """Calibrate and summarize covariate and predictive drift evidence."""

    observation_schema = (
        "p_covariate",
        "p_concept",
        "p_both",
        "severity",
        "confidence",
        "delta_auc",
        "last_action_cost",
    )

    def __init__(
        self,
        reference_df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str] | None = None,
        label_col: str = "SepsisLabel",
        predictor=None,
        seed: int = 42,
    ):
        self.reference_df = reference_df.copy()
        self.numeric_cols = [column for column in numeric_cols if column in reference_df]
        self.categorical_cols = [
            column for column in (categorical_cols or []) if column in reference_df
        ]
        self.label_col = label_col
        self.predictor = predictor
        self.seed = int(seed)
        self.calibration_: CalibrationResult | None = None
        self._reference_auc, self._reference_logloss = self._predictive_metrics(
            self.reference_df
        )

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.numeric_cols + self.categorical_cols]

    def _probabilities(self, frame: pd.DataFrame) -> np.ndarray | None:
        if self.predictor is None:
            return None
        if hasattr(self.predictor, "predict_proba"):
            return np.asarray(self.predictor.predict_proba(frame))[:, 1]
        return np.asarray(self.predictor(frame), dtype=float)

    def _predictive_metrics(self, frame: pd.DataFrame) -> tuple[float, float]:
        if self.label_col not in frame or frame.empty:
            return float("nan"), float("nan")
        probabilities = self._probabilities(frame)
        if probabilities is None:
            return float("nan"), float("nan")
        labels = frame[self.label_col].to_numpy(dtype=int)
        auc = (
            float("nan")
            if np.unique(labels).size < 2
            else float(roc_auc_score(labels, probabilities))
        )
        return auc, float(log_loss(labels, probabilities, labels=[0, 1]))

    def _raw_metrics(self, reference: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
        psi_values = [
            psi_numeric(reference[column], target[column]) for column in self.numeric_cols
        ]
        ks_values = [
            ks_stat(reference[column], target[column])["ks_stat"]
            for column in self.numeric_cols
        ]
        return {
            "psi": float(np.mean(psi_values)) if psi_values else 0.0,
            "ks": float(np.mean(ks_values)) if ks_values else 0.0,
            "c2st": c2st_score(
                self._feature_frame(reference),
                self._feature_frame(target),
                n_splits=3,
                seed=self.seed,
            ),
        }

    def calibrate(self, n_bootstrap: int = 100) -> CalibrationResult:
        """Estimate H0 distributions by comparing bootstrap reference samples."""
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least two")
        rng = np.random.default_rng(self.seed)
        values = {"psi": [], "ks": [], "c2st": [], "auc_loss": [], "logloss_gain": []}
        sample_size = max(4, len(self.reference_df) // 2)
        for _ in range(n_bootstrap):
            first = self.reference_df.iloc[
                rng.integers(0, len(self.reference_df), sample_size)
            ].reset_index(drop=True)
            second = self.reference_df.iloc[
                rng.integers(0, len(self.reference_df), sample_size)
            ].reset_index(drop=True)
            raw = self._raw_metrics(first, second)
            for name in ("psi", "ks", "c2st"):
                values[name].append(raw[name])
            auc, loss = self._predictive_metrics(second)
            values["auc_loss"].append(
                0.0
                if np.isnan(auc) or np.isnan(self._reference_auc)
                else max(0.0, self._reference_auc - auc)
            )
            values["logloss_gain"].append(
                0.0
                if np.isnan(loss) or np.isnan(self._reference_logloss)
                else max(0.0, loss - self._reference_logloss)
            )
        distributions = {
            name: np.asarray(metric_values, dtype=float)
            for name, metric_values in values.items()
        }
        thresholds = {
            name: max(float(np.quantile(metric_values, 0.95)), np.finfo(float).eps)
            for name, metric_values in distributions.items()
        }
        self.calibration_ = CalibrationResult(thresholds, distributions)
        return self.calibration_

    def _empirical_probability(self, name: str, value: float) -> float:
        if self.calibration_ is None:
            raise RuntimeError("DriftMonitor.calibrate must be called before analyze")
        distribution = self.calibration_.distributions[name]
        return float((np.count_nonzero(distribution <= value) + 1) / (len(distribution) + 1))

    def analyze(
        self,
        target_df: pd.DataFrame,
        window_idx: int = 0,
        true_drift_type: str | None = None,
    ) -> DriftReport:
        """Measure a closed target window and return calibrated drift evidence."""
        raw = self._raw_metrics(self.reference_df, target_df)
        psi_scores = {
            column: psi_numeric(self.reference_df[column], target_df[column])
            for column in self.numeric_cols
        }
        ks_scores = {
            column: ks_stat(self.reference_df[column], target_df[column])["ks_pval"]
            for column in self.numeric_cols
        }
        categorical_psi = {}
        for column in self.categorical_cols:
            reference_dist = self.reference_df[column].value_counts(normalize=True)
            target_dist = target_df[column].value_counts(normalize=True)
            categorical_psi[column] = psi_categorical(reference_dist, target_dist)

        auc, loss = self._predictive_metrics(target_df)
        delta_auc = (
            0.0
            if np.isnan(auc) or np.isnan(self._reference_auc)
            else float(auc - self._reference_auc)
        )
        delta_logloss = (
            0.0
            if np.isnan(loss) or np.isnan(self._reference_logloss)
            else float(loss - self._reference_logloss)
        )
        covariate_evidence = max(
            self._empirical_probability("psi", raw["psi"]),
            self._empirical_probability("ks", raw["ks"]),
            self._empirical_probability("c2st", raw["c2st"]),
        )
        concept_evidence = max(
            self._empirical_probability("auc_loss", max(0.0, -delta_auc)),
            self._empirical_probability("logloss_gain", max(0.0, delta_logloss)),
        )
        p_covariate = covariate_evidence * (1.0 - concept_evidence)
        p_concept = (1.0 - covariate_evidence) * concept_evidence
        p_both = covariate_evidence * concept_evidence
        p_none = (1.0 - covariate_evidence) * (1.0 - concept_evidence)
        return DriftReport(
            p_covariate=float(p_covariate),
            p_concept=float(p_concept),
            p_both=float(p_both),
            severity_score=float((covariate_evidence + concept_evidence) / 2.0),
            confidence=float(max(p_none, p_covariate, p_concept, p_both)),
            delta_auc=float(np.clip(delta_auc, -1.0, 1.0)),
            delta_logloss=delta_logloss,
            psi_scores=psi_scores,
            ks_scores=ks_scores,
            c2st_auc=raw["c2st"],
            true_drift_type=true_drift_type,
            window_idx=window_idx,
            n_samples=len(target_df),
            metadata={"categorical_psi": categorical_psi},
        )

    def measure(self, target_df: pd.DataFrame, window_idx: int = 0) -> DriftReport:
        """Compatibility alias that calibrates lazily then analyzes a window."""
        if self.calibration_ is None:
            self.calibrate(n_bootstrap=10)
        return self.analyze(target_df, window_idx=window_idx)

    def get_observation(
        self,
        report: DriftReport,
        last_action_cost: float = 0.0,
    ) -> np.ndarray:
        """Return the seven-value observable state; hidden truth is excluded."""
        return np.asarray(
            [
                report.p_covariate,
                report.p_concept,
                report.p_both,
                report.severity_score,
                report.confidence,
                report.delta_auc,
                float(np.clip(last_action_cost, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
