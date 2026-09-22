"""Predictive metrics for DARL model comparisons."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def evaluate_auc(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute ROC AUC from a pipeline's positive-class probabilities."""
    probabilities = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, probabilities))
