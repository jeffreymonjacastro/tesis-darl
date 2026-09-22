"""Calibrated observation and causal live-environment tests."""

import numpy as np
import pandas as pd

from darl.data.window import TemporalWindow
from darl.monitoring import DriftMonitor
from darl.pipeline import PipelineState, XGBStage2, apply_stage1, fit_stage1
from darl.rl import DARLEnvironment, TransitionTableBuilder, recovery_reward


def _live_env():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "HR": rng.normal(80, 5, 120),
            "Glucose": rng.normal(110, 10, 120),
            "SepsisLabel": np.tile([0, 1], 60),
        }
    )
    update = reference.copy()
    evaluation = reference.copy()
    evaluation["HR"] += 8
    stage1 = fit_stage1(reference, ["HR"], ["HR", "Glucose"])
    transformed = apply_stage1(reference, *stage1, ["HR"], ["HR", "Glucose"])
    model = XGBStage2(n_estimators=6, max_depth=2, n_jobs=1).fit(
        transformed[["HR", "Glucose"]], reference["SepsisLabel"]
    )
    state = PipelineState(stage1, model, ["HR"], ["HR", "Glucose"])
    monitor = DriftMonitor(reference, ["HR", "Glucose"], predictor=state)
    monitor.calibrate(n_bootstrap=3)
    windows = [
        TemporalWindow(0, 7, 9, update, evaluation),
        TemporalWindow(1, 10, 12, update, evaluation),
    ]
    return DARLEnvironment(state, reference, windows, monitor)


def test_hidden_truth_is_not_observation():
    env = _live_env()
    observation, _ = env.reset()
    assert observation.shape == (28,)
    assert env._last_report.true_drift_type == "none"
    assert observation[-7:].shape == (7,)


def test_action_changes_next_observation_and_pipeline():
    env = _live_env()
    env.reset()
    defer = env.clone()
    retrain = env.clone()
    defer_observation, *_ = defer.step(0)
    retrain_observation, *_ = retrain.step(3)
    assert not np.allclose(defer_observation, retrain_observation)
    assert defer.pipeline_state.stage2_version == 0
    assert retrain.pipeline_state.stage2_version == 1


def test_transition_table_has_four_counterfactual_actions():
    table = TransitionTableBuilder().build(_live_env(), behavior="defer")
    assert len(table) == 8
    assert set(table["action"]) == {0, 1, 2, 3}
    assert table["next_observation"].map(len).eq(28).all()


def test_reward_formula_and_no_loss_case():
    recovery, reward = recovery_reward(0.8, 0.6, 0.7, 0.4)
    assert np.isclose(recovery, 0.5)
    assert np.isclose(reward, 0.4)
    recovery, reward = recovery_reward(0.8, 0.81, 0.82, 0.4)
    assert recovery == 0.0
    assert np.isclose(reward, -0.1)
