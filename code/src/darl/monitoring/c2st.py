"""Classifier two-sample testing for tabular drift monitoring."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def c2st_score(
    df_reference: pd.DataFrame,
    df_target: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Return cross-validated logistic C2ST ROC AUC."""
    if df_reference.empty or df_target.empty:
        return 0.5
    common = [column for column in df_reference if column in df_target]
    combined = pd.concat(
        [df_reference[common], df_target[common]], ignore_index=True
    )
    usable = [column for column in common if combined[column].notna().any()]
    if not usable:
        return 0.5
    reference = df_reference[usable].copy()
    target = df_target[usable].copy()
    features = pd.concat([reference, target], ignore_index=True)
    labels = np.r_[np.zeros(len(reference)), np.ones(len(target))]
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    minimum_class = int(np.bincount(labels.astype(int)).min())
    folds = min(n_splits, minimum_class)
    if folds < 2:
        return 0.5
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_index, test_index in splitter.split(features, labels):
        estimator = make_pipeline(
            transformer,
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            estimator.fit(features.iloc[train_index], labels[train_index])
            probabilities = estimator.predict_proba(features.iloc[test_index])[:, 1]
        scores.append(float(roc_auc_score(labels[test_index], probabilities)))
    return float(np.mean(scores))
