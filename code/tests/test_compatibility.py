import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from darl.pipeline.compatibility import check_stage1_compatibility, CompatibilityReport

def test_check_stage1_compatibility():
    # Setup dummy data and transformers
    np.random.seed(42)
    X = pd.DataFrame({
        'v1': np.random.randn(100),
        'n1': np.random.randn(100)
    })
    vitals = ['v1']
    numeric = ['v1', 'n1']

    qt = QuantileTransformer(n_quantiles=10, random_state=42)
    qt.fit(X[vitals])
    imp = SimpleImputer()
    imp.fit(X)
    scaler = StandardScaler()
    scaler.fit(X)

    stage1 = (qt, imp, scaler)

    report = check_stage1_compatibility(
        stage1_old=stage1,
        stage1_new=stage1,
        X_sample=X,
        vitals=vitals,
        numeric_cols=numeric,
        threshold=0.5
    )

    assert isinstance(report, CompatibilityReport)
    assert report.is_compatible
    assert report.dimension_match
    # Exact same pipeline should yield 0 divergence
    assert report.distribution_divergence < 1e-6
