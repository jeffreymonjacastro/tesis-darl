"""Decoupled XGBoost stage for the DARL two-stage pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


class XGBStage2:
    """Sklearn-compatible, independently replaceable XGBoost classifier."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.model: XGBClassifier | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> "XGBStage2":
        """Fit the binary classifier and retain optional feature names."""
        labels = np.asarray(y)
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2:
            raise ValueError("XGBStage2 requires both binary classes")
        count_by_class = dict(zip(classes.tolist(), counts.tolist()))
        scale_pos_weight = count_by_class.get(0, 1) / max(count_by_class.get(1, 1), 1)
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=float(scale_pos_weight),
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X, labels)
        return self

    def _require_model(self) -> XGBClassifier:
        if self.model is None:
            raise ValueError("XGBStage2 has not been fitted")
        return self.model

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return class probabilities with shape ``(n_samples, 2)``."""
        return self._require_model().predict_proba(X)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        return self._require_model().predict(X)

    def evaluate(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> float:
        """Return ROC AUC, or NaN when the evaluation frame has one class."""
        labels = np.asarray(y)
        if np.unique(labels).size < 2:
            return float("nan")
        return float(roc_auc_score(labels, self.predict_proba(X)[:, 1]))

    def get_feature_importances(
        self,
        feature_names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Return feature importance values keyed by supplied or fitted names."""
        importances = self._require_model().feature_importances_
        names = list(feature_names) if feature_names is not None else self.feature_names_
        if names is None:
            names = [f"f{index}" for index in range(len(importances))]
        if len(names) != len(importances):
            raise ValueError("feature_names length does not match fitted features")
        return {name: float(value) for name, value in zip(names, importances)}

    def save(self, path: str | Path) -> None:
        """Persist the fitted XGBoost model."""
        self._require_model().save_model(Path(path))

    def load(self, path: str | Path) -> "XGBStage2":
        """Load an XGBoost model into this stage and return ``self``."""
        self.model = XGBClassifier()
        self.model.load_model(Path(path))
        return self
