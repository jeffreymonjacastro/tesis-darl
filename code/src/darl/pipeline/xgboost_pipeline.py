"""XGBoost pipeline factory for mixed tabular data."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


def make_xgb_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Build a fresh imputation, encoding, and class-weighted XGBoost pipeline."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    )
    categorical_transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )

    class_counts = pd.Series(y).value_counts()
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 0)
    if negative_count == 0 or positive_count == 0:
        raise ValueError("y debe contener las clases binarias 0 y 1.")

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=negative_count / positive_count,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])
