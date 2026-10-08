import numpy as np
import pandas as pd
from darl.monitoring.monitor import DriftMonitor
from darl.monitoring.c2st import c2st_score
from darl.monitoring.drift_metrics import psi_numeric, psi_categorical

def test_c2st_score():
    np.random.seed(42)
    # No drift
    df1 = pd.DataFrame({'a': np.random.randn(100)})
    df2 = pd.DataFrame({'a': np.random.randn(100)})
    score = c2st_score(df1, df2)
    assert abs(score - 0.5) < 0.1

    # High drift
    df3 = pd.DataFrame({'a': np.random.randn(100) + 5})
    score_drift = c2st_score(df1, df3)
    assert score_drift > 0.8

def test_psi_numeric():
    np.random.seed(42)
    s1 = pd.Series(np.random.randn(1000))
    s2 = pd.Series(np.random.randn(1000) + 0.1)

    psi = psi_numeric(s1, s2)
    assert isinstance(psi, float)
    assert psi >= 0

def test_drift_monitor():
    np.random.seed(42)
    df_ref = pd.DataFrame({
        'n1': np.random.randn(1000),
        'c1': np.random.choice(['A', 'B'], 1000)
    })

    df_tgt = pd.DataFrame({
        'n1': np.random.randn(500) + 2.0,
        'c1': np.random.choice(['A', 'B'], 500, p=[0.2, 0.8])
    })

    monitor = DriftMonitor(df_ref, numeric_cols=['n1'], categorical_cols=['c1'])
    report = monitor.measure(df_tgt, window_idx=1)

    assert report.window_idx == 1
    assert report.c2st_auc > 0.7
    assert 'n1' in report.feature_psi
    assert report.feature_psi['n1'] > 0.1
    assert 'c1' in report.cat_feature_psi
    assert report.cat_feature_psi['c1'] > 0.1

    state = report.get_state_vector(important_features=['n1', 'c1'])
    assert len(state) == 4
