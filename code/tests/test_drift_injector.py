import numpy as np
import pandas as pd
from darl.drift.injector import DriftInjector

def test_drift_injector_selection_strategies():
    np.random.seed(42)
    df = pd.DataFrame({
        'HR': np.random.normal(80, 10, 1000),
        'SBP': np.random.normal(120, 15, 1000),
        'other1': np.random.normal(0, 1, 1000),
        'other2': np.random.normal(0, 1, 1000),
        'label': np.random.randint(0, 2, 1000)
    })

    injector = DriftInjector(random_state=42)
    numeric_cols = ['HR', 'SBP', 'other1', 'other2']
    injector.fit(df, numeric_cols=numeric_cols)

    # Test 'domain' strategy
    df_drift, meta = injector.transform(
        df, drift_severity=1.0, drift_type="covariate",
        selection_strategy="domain"
    )
    # domain selects vitals (HR, SBP)
    assert meta['HR'].drift_severity == 1.0
    assert meta['SBP'].drift_severity == 1.0
    assert meta['other1'].drift_severity == 0.0
    assert meta['other2'].drift_severity == 0.0

    # Test 'important' strategy
    df_drift2, meta2 = injector.transform(
        df, drift_severity=1.0, drift_type="covariate",
        selection_strategy="important",
        important_features=['other1', 'HR'],
        feature_fraction=0.5 # 2 features out of 4
    )
    assert meta2['other1'].drift_severity == 1.0
    assert meta2['HR'].drift_severity == 1.0
    assert meta2['SBP'].drift_severity == 0.0

    # Test 'random' strategy
    df_drift3, meta3 = injector.transform(
        df, drift_severity=1.0, drift_type="covariate",
        selection_strategy="random",
        feature_fraction=0.5
    )
    drifted_count = sum(1 for c in numeric_cols if meta3[c].drift_severity > 0)
    assert drifted_count == 2
