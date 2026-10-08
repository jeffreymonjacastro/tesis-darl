"""Persistent pipeline action and formal contract tests."""

import numpy as np
import pandas as pd

from darl.actions import execute_action
from darl.pipeline import PipelineState, XGBStage2, apply_stage1, fit_stage1


def _pipeline_fixture():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "HR": rng.normal(80, 8, 160),
            "Glucose": rng.normal(110, 15, 160),
            "SepsisLabel": np.tile([0, 1], 80),
        }
    )
    stage1 = fit_stage1(reference, ["HR"], ["HR", "Glucose"])
    transformed = apply_stage1(reference, *stage1, ["HR"], ["HR", "Glucose"])
    stage2 = XGBStage2(n_estimators=8, max_depth=2, n_jobs=1).fit(
        transformed[["HR", "Glucose"]], reference["SepsisLabel"]
    )
    return reference, PipelineState(stage1, stage2, ["HR"], ["HR", "Glucose"])


def test_a2_passes_contract_and_persists_adapter():
    reference, state = _pipeline_fixture()
    result = execute_action("A2", state, reference, reference, reference)
    assert result.executed_action == "A2"
    assert result.compatibility is not None and result.compatibility.is_compatible
    assert result.pipeline_state.adapter_map
    assert not state.adapter_map


def test_a2_severe_unmapped_drift_falls_back_to_a4():
    reference, state = _pipeline_fixture()
    drifted = reference.copy()
    drifted["Glucose"] *= 50
    result = execute_action(
        "A2",
        state,
        drifted,
        drifted,
        reference,
        mean_threshold=0.01,
        max_threshold=0.01,
    )
    assert result.executed_action == "A4"
    assert result.fallback_reason
    assert result.pipeline_state.stage2_version == 1
