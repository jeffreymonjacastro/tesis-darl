"""Daily DARL decisions over an hourly, patient-disjoint ICU replay."""

from __future__ import annotations

import copy
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from darl.actions.selective_update import apply_reference_location_scale_map, fit_reference_location_scale_map
from darl.data.bed_stream import VITALS, patient_weights
from darl.pipeline.logreg_stage import apply_stage1, fit_stage1
from darl.pipeline.xgb_stage import XGBStage2


def physionet_utility(frame: pd.DataFrame, threshold: float = 0.5) -> float:
    """Compute the 2019 challenge's normalized utility on observed patient segments."""
    if frame.empty:
        return float("nan")

    def utility(labels: np.ndarray, decisions: np.ndarray) -> float:
        onset = int(np.flatnonzero(labels)[0]) + 6 if labels.any() else np.inf
        result = 0.0
        for t, alarm in enumerate(decisions):
            if t > onset + 3:
                continue
            relative = t - onset
            if np.isfinite(onset):
                if alarm:
                    result += max((relative + 12) / 6, -0.05) if relative <= -6 else (3 - relative) / 9
                elif relative > -6:
                    result += -2 * (relative + 6) / 9
            elif alarm:
                result -= 0.05
        return result

    observed = optimum = inaction = 0.0
    for _, patient in frame.sort_values(["patient_id", "ICULOS"]).groupby("patient_id"):
        labels = patient.SepsisLabel.to_numpy(dtype=int)
        decisions = patient.probability.to_numpy() >= threshold
        best = np.zeros(len(labels), dtype=bool)
        if labels.any():
            onset = int(np.flatnonzero(labels)[0]) + 6
            best[max(0, onset - 12):min(len(labels), onset + 4)] = True
        observed += utility(labels, decisions)
        optimum += utility(labels, best)
        inaction += utility(labels, np.zeros(len(labels), dtype=bool))
    return float((observed - inaction) / (optimum - inaction)) if optimum != inaction else float("nan")


def prediction_metrics(frame: pd.DataFrame, *, include_utility: bool = True) -> dict[str, float]:
    """Return hourly and equal-patient AUPRC/AUROC with explicit undefined cases."""
    if frame.empty:
        return {"auprc_hour": float("nan"), "auroc_hour": float("nan"), "auprc_patient": float("nan"), "auroc_patient": float("nan"), "prevalence": float("nan")}
    y = frame.SepsisLabel.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    weights = patient_weights(frame)
    result = {"prevalence": float(y.mean())}
    for suffix, weight in (("hour", None), ("patient", weights)):
        result[f"auprc_{suffix}"] = float(average_precision_score(y, p, sample_weight=weight)) if y.sum() else float("nan")
        result[f"auroc_{suffix}"] = float(roc_auc_score(y, p, sample_weight=weight)) if len(np.unique(y)) == 2 else float("nan")
    if include_utility:
        result["physionet_utility_observed_segment"] = physionet_utility(frame)
    return result


@dataclass
class BedPipeline:
    """Seven-vital preprocessing and XGBoost with an optional robust A2 adapter."""

    stage1: tuple
    stage2: XGBStage2
    adapter: dict
    reference: pd.DataFrame
    seed: int = 42

    @classmethod
    def fit_initial(cls, train: pd.DataFrame, *, n_estimators: int = 80, device: str = "cpu", seed: int = 42) -> "BedPipeline":
        """Fit the A-only model with inverse-observed-stay weights."""
        stage1 = fit_stage1(train, VITALS, VITALS, seed)
        x = apply_stage1(train, *stage1, VITALS, VITALS)[VITALS]
        model = XGBStage2(n_estimators=n_estimators, max_depth=4, seed=seed, device=device)
        model.fit(x, train.SepsisLabel.to_numpy(), sample_weight=patient_weights(train))
        return cls(stage1, model, {}, train[VITALS].copy(), seed)

    def clone(self) -> "BedPipeline":
        """Copy the deployed state before trying an action."""
        return copy.deepcopy(self)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply fixed-width seven-vital adapter and preprocessing."""
        corrected = apply_reference_location_scale_map(frame[VITALS], self.adapter)
        return apply_stage1(corrected, *self.stage1, VITALS, VITALS)[VITALS]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return sepsis probabilities from already observed vital rows only."""
        return self.stage2.predict_proba(self.transform(frame))[:, 1]

    def act(self, requested: int, update: pd.DataFrame) -> tuple["BedPipeline", int, str | None, float, float]:
        """Run A1–A4 on revealed update labels; A2 keeps XGBoost identical."""
        if requested not in range(4):
            raise ValueError("Action must be 0..3")
        candidate = self.clone()
        executed = requested
        reason = None
        if requested >= 2 and (update.empty or update.SepsisLabel.nunique() < 2):
            executed, reason = 0, "insufficient_revealed_classes"
        if requested == 1 and update.empty:
            executed, reason = 0, "no_revealed_update_rows"
        tracemalloc.start()
        started = time.perf_counter()
        try:
            if executed == 1:
                adapter = fit_reference_location_scale_map(update, self.reference, VITALS, robust=True)
                # A2 is defined only if every observed, nonconstant vital can be mapped.
                needed = [v for v in VITALS if update[v].notna().sum() >= 2 and self.reference[v].notna().sum() >= 2]
                if not all(v in adapter for v in needed):
                    executed, reason = 0, "a2_mapping_contract_failed"
                else:
                    candidate.adapter = adapter
            if executed == 3:
                candidate.stage1 = fit_stage1(update, VITALS, VITALS, self.seed)
                candidate.adapter = {}
            if executed in (2, 3):
                model = XGBStage2(
                    n_estimators=self.stage2.n_estimators,
                    max_depth=self.stage2.max_depth,
                    learning_rate=self.stage2.learning_rate,
                    seed=self.seed,
                    device=self.stage2.device,
                )
                model.fit(candidate.transform(update), update.SepsisLabel.to_numpy(), sample_weight=patient_weights(update))
                candidate.stage2 = model
            elapsed = time.perf_counter() - started
            peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        finally:
            tracemalloc.stop()
        return candidate, executed, reason, elapsed, peak_mb


class BedDAREnvironment:
    """Expose daily observations while withholding B labels for one full day."""

    def __init__(self, schedule: pd.DataFrame, initial: BedPipeline, *, days: int = 21, cost_lambda: float = 0.25):
        self.schedule = schedule
        self.initial = initial
        self.days = days
        self.cost_lambda = cost_lambda
        self.cost = (0.0, 0.1, 0.6, 1.0)
        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset deployed model and pending outcomes for a fresh episode."""
        self.pipeline = self.initial.clone()
        self.day = 1
        self.history: list[np.ndarray] = []
        self.pending: list[dict] = []
        self.predictions: list[pd.DataFrame] = []
        self.hourly_predictions: list[pd.DataFrame] = []
        self.actions: list[dict] = []
        first_day = self.schedule.loc[self.schedule.day.eq(1)].copy()
        if not first_day.empty:
            first_day["probability"] = self.pipeline.predict(first_day)
            self.hourly_predictions.append(first_day)
            first_eval = first_day.loc[first_day.role.eq("eval")]
            if not first_eval.empty:
                self.predictions.append(first_eval)
        return self.observation(), {}

    def _day_rows(self, day: int, role: str) -> pd.DataFrame:
        return self.schedule.loc[self.schedule.day.eq(day) & self.schedule.role.eq(role)]

    def revealed_update(self) -> pd.DataFrame:
        """Return only adaptation rows whose labels have completed 24h delay."""
        return self.schedule.loc[self.schedule.role.eq("update") & self.schedule.day.le(self.day - 1)]

    def observation(self) -> np.ndarray:
        """Encode four recent days of drift, missingness and delayed performance."""
        rows = self.schedule.loc[self.schedule.day.eq(self.day)]
        clean_baseline = self.schedule.loc[self.schedule.day.lt(self.day) & self.schedule.day.le(7), VITALS]
        reference = clean_baseline if not clean_baseline.empty else self.initial.reference
        if rows.empty:
            vital_shift = 0.0
            missing = 1.0
            mean_probability = 0.0
        else:
            difference = (rows[VITALS].mean() - reference[VITALS].mean()).abs()
            scale = reference[VITALS].std().clip(lower=1e-6)
            vital_shift = float(np.nanmean(np.minimum((difference / scale).to_numpy(), 10)))
            missing = float(rows[VITALS].isna().mean().mean())
            mean_probability = float(np.nanmean(self.pipeline.predict(rows)))
        revealed_parts = [part for part in self.predictions if int(part.day.iloc[0]) <= self.day - 1]
        known_eval = revealed_parts[-1] if revealed_parts else pd.DataFrame()
        known_metrics = prediction_metrics(known_eval, include_utility=False) if not known_eval.empty else {}
        block = np.array([
            vital_shift / 10, missing, mean_probability,
            float(known_metrics.get("prevalence", 0) or 0),
            float(known_metrics.get("auprc_hour", 0) or 0),
            min(len(self.revealed_update()) / 10000, 1),
            self.cost[self.actions[-1]["executed"]] if self.actions else 0,
        ], dtype=np.float32)
        block = np.nan_to_num(block, nan=0.0)
        if not self.history or len(self.history) < self.day:
            self.history.append(block)
        else:
            self.history[-1] = block
        previous = self.history[-4:]
        return np.concatenate([np.zeros(7, dtype=np.float32)] * (4 - len(previous)) + previous)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply maintenance at day close; reveal prior rewards only after delay."""
        if self.day >= self.days:
            raise RuntimeError("Episode already finished")
        day = self.day
        current_observation = self.observation()
        update = self.revealed_update()
        old = self.pipeline
        new, executed, reason, elapsed, peak = old.act(action, update)
        self.pipeline = new
        clean_parts = [part for part in self.predictions if int(part.day.iloc[0]) <= min(day - 2, 7)]
        baseline_prevalence = float(np.mean([part.SepsisLabel.mean() for part in clean_parts])) if clean_parts else float("nan")
        recent_parts = [part for part in self.predictions if int(part.day.iloc[0]) == day - 1]
        recent_prevalence = float(recent_parts[0].SepsisLabel.mean()) if recent_parts else float("nan")
        concept_score = abs(recent_prevalence - baseline_prevalence) / max(baseline_prevalence, 0.01) if np.isfinite(baseline_prevalence) and np.isfinite(recent_prevalence) else float("nan")
        record = {"day": day, "requested": int(action), "executed": executed, "fallback_reason": reason, "time_s": elapsed, "peak_ram_mb": peak, "n_revealed_update": len(update), "drift_score": float(current_observation[-7]), "concept_score": concept_score}
        self.actions.append(record)
        next_day = day + 1
        if next_day <= self.days:
            all_rows = self.schedule.loc[self.schedule.day.eq(next_day)].copy()
            if not all_rows.empty:
                all_rows["probability"] = new.predict(all_rows)
                self.hourly_predictions.append(all_rows)
            evaluation = all_rows.loc[all_rows.role.eq("eval")].copy()
            if not evaluation.empty:
                baseline = old.predict(evaluation)
                self.predictions.append(evaluation)
                y = evaluation.SepsisLabel.to_numpy()
                gain = float(average_precision_score(y, evaluation.probability) - average_precision_score(y, baseline)) if y.sum() else 0.0
                record["auprc_evaluable"] = bool(y.sum())
                record["reward"] = gain - self.cost_lambda * self.cost[executed]
                record["reward_reveal_day"] = day + 2
                self.pending.append({"reveal_day": day + 2, "observation": current_observation, "action": action, "reward": record["reward"], "next_observation": None, "terminated": next_day == self.days})
            else:
                record.update(auprc_evaluable=False, reward=-self.cost_lambda * self.cost[executed], reward_reveal_day=day + 2)
        self.day += 1
        done = self.day >= self.days
        next_observation = np.zeros(28, dtype=np.float32) if done else self.observation()
        for pending in self.pending:
            if pending["next_observation"] is None:
                pending["next_observation"] = next_observation
        matured = [p for p in self.pending if p["reveal_day"] <= self.day]
        self.pending = [p for p in self.pending if p["reveal_day"] > self.day]
        return next_observation, 0.0, done, False, {"day": day, "matured": matured, **record}

    def flush_rewards(self) -> list[dict]:
        """Reveal the final day's labels after its 24-hour post-horizon delay."""
        if self.day < self.days:
            raise RuntimeError("Cannot flush before the simulated horizon ends")
        matured = self.pending
        self.pending = []
        return matured

    def finish(self) -> dict:
        """Return episode metrics after all labels are revealed offline."""
        pred = pd.concat(self.predictions, ignore_index=True) if self.predictions else pd.DataFrame()
        metrics = prediction_metrics(pred)
        metrics.update({"n_predictions": len(pred), "n_hourly_all_beds": sum(len(part) for part in self.hourly_predictions), "n_patients": pred.patient_id.nunique() if not pred.empty else 0, "n_updates": sum(a["executed"] in (2, 3) for a in self.actions), "return": float(sum(a.get("reward", 0) for a in self.actions)), "time_s": float(sum(a["time_s"] for a in self.actions)), "peak_ram_mb": float(max((a["peak_ram_mb"] for a in self.actions), default=0))})
        covariate_days = [a["day"] for a in self.actions if a["drift_score"] > 0.12 and a["day"] >= 8]
        concept_days = [a["day"] for a in self.actions if a["concept_score"] > 0.75 and a["day"] >= 9]
        metrics["covariate_detection_day"] = min(covariate_days) if covariate_days else float("nan")
        metrics["concept_detection_day"] = min(concept_days) if concept_days else float("nan")
        detections = covariate_days + concept_days
        metrics["drift_detection_day"] = min(detections) if detections else float("nan")
        return metrics
