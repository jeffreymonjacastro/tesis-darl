"""
drift_framework/pipeline/two_stage.py

A two-stage sklearn-compatible ML pipeline with independently refittable stages.

Stage 1 — Preprocessing:
  - QuantileTransformer  : skewed numeric features (|skewness| > skew_threshold)
  - TargetEncoder        : high-cardinality categorical features (unique > high_card_thresh)
  - StandardScaler       : remaining numeric features

Stage 2 — Predictive Model:
  - LogisticRegression (model_type="logreg")  — baseline
  - XGBClassifier      (model_type="xgboost") — main model
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier

SEED = 42


class Stage1Preprocessor:
    """Fit-and-transform preprocessor for tabular features.

    Applies QuantileTransformer to skewed numeric features, TargetEncoder to
    high-cardinality categoricals, StandardScaler to remaining numerics, and
    integer label encoding to low-cardinality categoricals.

    Args:
        num_features: list of numeric column names.
        cat_features: list of categorical column names.
        skew_threshold: absolute skewness above which QuantileTransformer is used.
        high_card_thresh: unique-value count above which TargetEncoder is used.
        random_state: global random seed.
    """

    def __init__(
        self,
        num_features: list[str],
        cat_features: list[str],
        skew_threshold: float = 1.0,
        high_card_thresh: int = 10,
        random_state: int = SEED,
    ):
        self.num_features = num_features
        self.cat_features = cat_features
        self.skew_threshold = skew_threshold
        self.high_card_thresh = high_card_thresh
        self.random_state = random_state

        self._skewed_cols: list[str] = []
        self._normal_cols: list[str] = []
        self._high_card_cols: list[str] = []
        self._low_card_cols: list[str] = []

        self._qt: Optional[QuantileTransformer] = None
        self._ss: Optional[StandardScaler] = None
        self._te: Optional[TargetEncoder] = None
        self._label_maps: dict[str, dict] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Stage1Preprocessor":
        """Fit all transformers on training data.

        Args:
            X: raw feature DataFrame.
            y: binary target Series (required for TargetEncoder).

        Returns:
            self
        """
        self._skewed_cols = []
        self._normal_cols = []
        for col in self.num_features:
            if col in X.columns:
                sk = X[col].dropna().skew()
                if abs(sk) > self.skew_threshold:
                    self._skewed_cols.append(col)
                else:
                    self._normal_cols.append(col)

        self._high_card_cols = []
        self._low_card_cols = []
        for col in self.cat_features:
            if col in X.columns:
                if X[col].nunique() > self.high_card_thresh:
                    self._high_card_cols.append(col)
                else:
                    self._low_card_cols.append(col)

        if self._skewed_cols:
            self._qt = QuantileTransformer(
                output_distribution="normal",
                random_state=self.random_state,
            )
            self._qt.fit(X[self._skewed_cols].fillna(0))

        if self._normal_cols:
            self._ss = StandardScaler()
            self._ss.fit(X[self._normal_cols].fillna(0))

        if self._high_card_cols:
            self._te = TargetEncoder(cols=self._high_card_cols, smoothing=1.0)
            self._te.fit(X[self._high_card_cols].astype(str), y)

        for col in self._low_card_cols:
            uniques = sorted(X[col].dropna().astype(str).unique())
            self._label_maps[col] = {v: i for i, v in enumerate(uniques)}

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers.

        Args:
            X: raw feature DataFrame (must have same columns as training data).

        Returns:
            Fully numeric DataFrame ready for Stage 2.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        parts: list[pd.DataFrame] = []

        if self._skewed_cols:
            qt_arr = self._qt.transform(X[self._skewed_cols].fillna(0))
            parts.append(pd.DataFrame(qt_arr, columns=self._skewed_cols, index=X.index))

        if self._normal_cols:
            ss_arr = self._ss.transform(X[self._normal_cols].fillna(0))
            parts.append(pd.DataFrame(ss_arr, columns=self._normal_cols, index=X.index))

        if self._high_card_cols:
            te_df = self._te.transform(X[self._high_card_cols].astype(str))
            te_df.index = X.index
            parts.append(te_df)

        for col in self._low_card_cols:
            encoded = (
                X[col].astype(str).map(self._label_maps[col]).fillna(-1).astype(int)
            )
            parts.append(encoded.rename(col).to_frame())

        if not parts:
            return pd.DataFrame(index=X.index)

        return pd.concat(parts, axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit then transform in one step."""
        return self.fit(X, y).transform(X)


class TwoStagePipeline:
    """A two-stage ML pipeline with independently refittable stages.

    Stage 1 applies feature preprocessing; Stage 2 applies the predictive model.
    Either stage can be refit independently to simulate selective updating after
    covariate shift (refit Stage 1) or concept drift (refit Stage 2).

    Args:
        model_type: ``"logreg"`` for LogisticRegression or ``"xgboost"`` for XGBClassifier.
        num_features: list of numeric column names.
        cat_features: list of categorical column names.
        skew_threshold: passed to Stage1Preprocessor.
        high_card_thresh: passed to Stage1Preprocessor.
        random_state: global random seed.
    """

    def __init__(
        self,
        model_type: Literal["logreg", "xgboost"] = "xgboost",
        num_features: list[str] | None = None,
        cat_features: list[str] | None = None,
        skew_threshold: float = 1.0,
        high_card_thresh: int = 10,
        random_state: int = SEED,
    ):
        self.model_type = model_type
        self.num_features = num_features or []
        self.cat_features = cat_features or []
        self.random_state = random_state

        self.stage1 = Stage1Preprocessor(
            num_features=self.num_features,
            cat_features=self.cat_features,
            skew_threshold=skew_threshold,
            high_card_thresh=high_card_thresh,
            random_state=random_state,
        )
        self.stage2 = self._build_model(model_type, random_state)
        self._stage1_fitted = False
        self._stage2_fitted = False

    @staticmethod
    def _build_model(model_type: str, random_state: int):
        """Instantiate the Stage 2 predictive model."""
        if model_type == "logreg":
            return LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "xgboost":
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Use 'logreg' or 'xgboost'."
            )

    def fit_stage1(self, X: pd.DataFrame, y: pd.Series) -> "TwoStagePipeline":
        """Fit (or re-fit) Stage 1 preprocessing only.

        Args:
            X: raw training features.
            y: training labels (required for TargetEncoder).

        Returns:
            self
        """
        self.stage1.fit(X, y)
        self._stage1_fitted = True
        return self

    def fit_stage2(self, X: pd.DataFrame, y: pd.Series) -> "TwoStagePipeline":
        """Fit (or re-fit) Stage 2 model only, using already-fitted Stage 1.

        Args:
            X: raw training features (Stage 1 transform applied internally).
            y: training labels.

        Returns:
            self
        """
        if not self._stage1_fitted:
            raise RuntimeError("Fit Stage 1 first via fit_stage1().")
        X_transformed = self.stage1.transform(X)
        self.stage2.fit(X_transformed, y)
        self._stage2_fitted = True
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TwoStagePipeline":
        """Fit both stages sequentially (full pipeline fit).

        Args:
            X: raw training features.
            y: training labels.

        Returns:
            self
        """
        self.fit_stage1(X, y)
        self.fit_stage2(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: raw features.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        if not (self._stage1_fitted and self._stage2_fitted):
            raise RuntimeError("Pipeline must be fully fitted before predict_proba().")
        X_transformed = self.stage1.transform(X)
        return self.stage2.predict_proba(X_transformed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions.

        Args:
            X: raw features.

        Returns:
            Array of shape (n_samples,) with 0/1 predictions.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    from code.drift_framework.data.loader import load_dataset

    b = load_dataset("adult")
    pipe = TwoStagePipeline(
        model_type="xgboost",
        num_features=b.num_features,
        cat_features=b.cat_features,
    )
    pipe.fit(b.X_ref, b.y_ref)
    proba = pipe.predict_proba(b.X_pre)
    print(f"predict_proba shape: {proba.shape}")
    print(f"Sample probabilities (positive class): {proba[:3, 1]}")
