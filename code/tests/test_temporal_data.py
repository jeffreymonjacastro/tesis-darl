"""Temporal PhysioNet loading and patient-isolation tests."""

import numpy as np
import pandas as pd

from darl.data import (
    load_physionet_temporal,
    make_reference_frames,
    make_temporal_windows,
    split_patients,
)


def _temporal_frame(n_patients: int = 20) -> pd.DataFrame:
    rows = []
    for patient in range(n_patients):
        for hour in range(1, 13):
            rows.append(
                {
                    "patient_id": f"p{patient:03d}",
                    "source_set": "a" if patient % 2 == 0 else "b",
                    "ICULOS": hour,
                    "HR": 70 + patient + hour,
                    "SepsisLabel": int(patient % 4 == 0 and hour >= 7),
                }
            )
    return pd.DataFrame(rows)


def test_load_physionet_preserves_patient_and_time(tmp_path):
    root = tmp_path / "training"
    source = root / "training_setA"
    source.mkdir(parents=True)
    for patient in (1, 2):
        pd.DataFrame(
            {
                "HR": [70 + patient, 72 + patient],
                "HospAdmTime": [-1.0, -1.0],
                "ICULOS": [1, 2],
                "SepsisLabel": [0, patient % 2],
            }
        ).to_csv(source / f"p{patient:06d}.psv", sep="|", index=False)
    data = load_physionet_temporal(root)
    assert data["patient_id"].nunique() == 2
    assert data.groupby("patient_id")["ICULOS"].apply(lambda x: x.is_monotonic_increasing).all()


def test_patient_split_and_windows_have_no_leakage():
    data = _temporal_frame()
    split = split_patients(data)
    split.validate()
    reference = make_reference_frames(data, split)
    windows = make_temporal_windows(data, split, start_hour=7, end_hour=12)
    assert len(windows) == 2
    assert set(reference.train["patient_id"]) <= split.train_ids
    for window in windows:
        assert set(window.update["patient_id"]).isdisjoint(window.evaluation["patient_id"])
        assert window.update["ICULOS"].between(window.start_hour, window.end_hour).all()
