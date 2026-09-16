import pandas as pd
import numpy as np
from darl.evaluation.baselines import (
    run_baseline_always_a1,
    run_baseline_always_a4,
    run_baseline_reactive,
    run_baseline_random
)
from darl.evaluation.eval_metrics import compare_policies

def test_baselines():
    transition_table = pd.DataFrame({
        "state": [[0.6] for _ in range(10)], # C2ST > 0.55
        "A1_AUC": np.full(10, 0.5),
        "A1_time": np.full(10, 0.1),
        "A2_AUC": np.full(10, 0.6),
        "A2_time": np.full(10, 0.5),
        "A3_AUC": np.full(10, 0.7),
        "A3_time": np.full(10, 1.0),
        "A4_AUC": np.full(10, 0.8),
        "A4_time": np.full(10, 2.0),
    })

    # Always A1
    b1 = run_baseline_always_a1(transition_table)
    assert b1["mean_auc"] == 0.5
    assert b1["mean_time"] == 0.1

    # Always A4
    b4 = run_baseline_always_a4(transition_table)
    assert b4["mean_auc"] == 0.8
    assert b4["mean_time"] == 2.0

    # Reactive
    b_react = run_baseline_reactive(transition_table, c2st_threshold=0.55)
    # Since C2ST=0.6 > 0.55, it should always choose A2
    assert abs(b_react["mean_auc"] - 0.6) < 1e-6
    assert abs(b_react["mean_time"] - 0.5) < 1e-6

    # Random
    b_rand = run_baseline_random(transition_table)
    assert 0.5 <= b_rand["mean_auc"] <= 0.8
    assert 0.1 <= b_rand["mean_time"] <= 2.0

def test_compare_policies():
    transition_table = pd.DataFrame({
        "state": [[0.5] for _ in range(10)],
        "A1_AUC": np.full(10, 0.5),
        "A1_time": np.full(10, 0.1),
        "A2_AUC": np.full(10, 0.6),
        "A2_time": np.full(10, 0.5),
        "A3_AUC": np.full(10, 0.7),
        "A3_time": np.full(10, 1.0),
        "A4_AUC": np.full(10, 0.8),
        "A4_time": np.full(10, 2.0),
    })

    darl_results = pd.DataFrame({
        "auc": np.full(10, 0.75),
        "time_cost": np.full(10, 1.2)
    })

    df_cmp = compare_policies(darl_results, transition_table)
    assert len(df_cmp) == 5
    assert list(df_cmp["Policy"].values) == [
        "DARL (DQN)", "Always A1 (Ignore)", "Always A4 (Retrain)",
        "Reactive (C2ST > 0.55 -> A2)", "Random"
    ]
