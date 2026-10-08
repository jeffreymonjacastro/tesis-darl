"""Reusable preparation helpers for tabular DARL experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_model_frame(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy with selected columns coerced to numeric values."""
    prepared = df.copy()
    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column].astype(object), errors="coerce"
            )

    for column in prepared.select_dtypes(exclude="number").columns:
        values = prepared[column].astype(object)
        prepared[column] = values.where(pd.notna(values), np.nan)
    return prepared


def select_training_sample(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int | None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a reproducible stratified sample or the complete training set."""
    if len(X) != len(y):
        raise ValueError("X e y deben tener el mismo número de filas.")
    if max_rows is None or len(X) <= max_rows:
        return X.copy(), y.copy()
    if max_rows < 2:
        raise ValueError("max_rows debe ser al menos 2 o None.")

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        stratify=y,
        random_state=random_state,
    )
    return X_sample, y_sample
