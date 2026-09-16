"""Persistent two-stage pipeline state used by sequential DARL actions."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from darl.pipeline.logreg_stage import apply_stage1
from darl.pipeline.xgb_stage import XGBStage2


@dataclass
class PipelineState:
    """Mutable deployed pipeline plus version and adaptation metadata."""

    stage1: tuple[Any, Any, Any]
    stage2: XGBStage2
    vitals: list[str]
    numeric_cols: list[str]
    label_col: str = "SepsisLabel"
    threshold: float = 0.5
    adapter_map: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    stage1_version: int = 0
    stage2_version: int = 0

    def clone(self) -> "PipelineState":
        """Deep-copy state so counterfactual actions remain independent."""
        return copy.deepcopy(self)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the persistent A2 adapter followed by deployed Stage 1."""
        from darl.actions.selective_update import apply_reference_location_scale_map

        adapted = apply_reference_location_scale_map(frame, self.adapter_map)
        return apply_stage1(
            adapted,
            *self.stage1,
            self.vitals,
            self.numeric_cols,
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities from a raw feature frame."""
        transformed = self.transform(frame)
        return self.stage2.predict_proba(transformed[self.numeric_cols])

    def evaluate(self, frame: pd.DataFrame) -> float:
        """Evaluate deployed pipeline ROC AUC on a labeled frame."""
        transformed = self.transform(frame)
        labels = frame[self.label_col].to_numpy()
        return self.stage2.evaluate(transformed[self.numeric_cols], labels)
