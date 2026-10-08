import numpy as np
from darl.pipeline.xgb_stage import XGBStage2

def test_xgb_stage2_fit_predict():
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    model = XGBStage2(n_estimators=10)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 100
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)

    importances = model.get_feature_importances()
    assert len(importances) == 5
