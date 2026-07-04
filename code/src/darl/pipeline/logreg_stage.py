from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler


def make_logreg(seed: int = 42) -> LogisticRegression:
    """Create the fixed Logistic Regression model used as Stage 2."""
    return LogisticRegression(
        solver="saga",
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        n_jobs=-1,
    )


def apply_qt(
    df: pd.DataFrame,
    qt: QuantileTransformer,
    vitals: list[str],
) -> pd.DataFrame:
    """Apply the fitted QuantileTransformer to vital-sign columns."""
    out = df.copy()
    vitals_present = [col for col in vitals if col in out.columns]
    out[vitals_present] = qt.transform(out[vitals_present])
    return out


def apply_stage1(
    df: pd.DataFrame,
    qt: QuantileTransformer,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    vitals: list[str],
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Apply QT over vitals, then impute and scale numeric columns."""
    out = apply_qt(df, qt, vitals)
    x_imp = imputer.transform(out[numeric_cols])
    out[numeric_cols] = scaler.transform(x_imp)
    return out


def fit_stage1(
    df_fit: pd.DataFrame,
    vitals: list[str],
    numeric_cols: list[str],
    seed: int = 42,
) -> tuple[QuantileTransformer, SimpleImputer, StandardScaler]:
    """Fit QT on vitals, then median imputer and scaler on numeric columns."""
    qt = QuantileTransformer(
        n_quantiles=500,
        output_distribution="normal",
        random_state=seed,
    )
    vitals_present = [col for col in vitals if col in df_fit.columns]
    qt.fit(df_fit[vitals_present])

    df_qt = apply_qt(df_fit, qt, vitals)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_qt[numeric_cols])

    scaler = StandardScaler()
    x_imp = imputer.transform(df_qt[numeric_cols])
    scaler.fit(x_imp)
    return qt, imputer, scaler
