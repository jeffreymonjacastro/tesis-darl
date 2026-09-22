"""Causal checks for patient-disjoint ICU replay and daily actions."""

import numpy as np
import pandas as pd

from darl.data.bed_stream import BedConfig, VITALS, inject_scenario, make_bed_schedule, patient_partitions
from darl.rl.bed_env import BedDAREnvironment, BedPipeline


def _frame(n=80):
    rows = []
    for site in ("A", "B"):
        for patient in range(n):
            positive = patient % 4 == 0
            for hour in range(3 + patient % 4):
                row = {"source_set": site, "patient_id": f"{site}_{patient}", "ICULOS": hour + 2, "SepsisLabel": int(positive and hour >= 2)}
                row.update({v: float((patient % 7) * 2 + hour + i) for i, v in enumerate(VITALS)})
                rows.append(row)
    return pd.DataFrame(rows)


def test_patient_splits_and_bed_clock():
    frame = _frame()
    parts = patient_partitions(frame)
    all_ids = [pid for ids in parts.values() for pid in ids]
    assert len(all_ids) == len(set(all_ids))
    schedule = make_bed_schedule(frame, parts["b_test_update"], parts["b_test_eval"], BedConfig(beds=4, days=3))
    assert schedule.groupby("global_hour").bed.nunique().max() <= 4
    assert schedule.groupby("patient_id").ICULOS.diff().dropna().eq(1).all()
    assert schedule.groupby("patient_id").global_hour.diff().dropna().eq(1).all()
    assert schedule.groupby("patient_id").ICULOS.min().eq(2).all()
    assert not (set(schedule.loc[schedule.role.eq("update"), "patient_id"]) & set(schedule.loc[schedule.role.eq("eval"), "patient_id"]))


def test_delayed_labels_and_frozen_a2_model():
    frame = _frame()
    parts = patient_partitions(frame)
    train = frame.loc[frame.patient_id.isin(parts["a_train"])]
    model = BedPipeline.fit_initial(train, n_estimators=5)
    assert model.stage2.feature_names_ == VITALS
    schedule = make_bed_schedule(frame, parts["b_train_update"], parts["b_train_eval"], BedConfig(beds=8, days=3))
    env = BedDAREnvironment(schedule, model, days=3)
    assert env.revealed_update().empty
    assert len(env.hourly_predictions[0]) == len(schedule.loc[schedule.day.eq(1)])
    _, _, _, _, info = env.step(2)
    assert info["executed"] == 0
    assert not info["matured"]
    assert len(env.hourly_predictions[1]) == len(schedule.loc[schedule.day.eq(2)])
    assert env.revealed_update().day.max() == 1
    _, _, done, _, info = env.step(1)
    assert done
    assert info["executed"] in (0, 1)
    assert all(item["reveal_day"] <= env.day for item in info["matured"])
    assert all(item["reveal_day"] == 4 for item in env.flush_rewards())
    if info["executed"] == 1:
        assert env.pipeline.stage2.model.get_booster().save_raw() == model.stage2.model.get_booster().save_raw()


def test_synthetic_label_shift_is_monotone_and_begins_day_eight():
    frame = _frame()
    parts = patient_partitions(frame)
    schedule = make_bed_schedule(frame, parts["b_train_update"], parts["b_train_eval"], BedConfig(beds=8, days=9))
    shifted = inject_scenario(schedule, "combined")
    assert shifted.loc[shifted.day.le(7), VITALS + ["SepsisLabel"]].equals(schedule.loc[schedule.day.le(7), VITALS + ["SepsisLabel"]])
    assert (shifted.SepsisLabel >= shifted.original_label).all()
