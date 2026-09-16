"""Dataset loaders, including patient-preserving PhysioNet ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from darl.utils import find_project_root

from tableshift import get_dataset
from tableshift.core.features import PreprocessorConfig

SEED = 42
PHYSIONET_RELATIVE_ROOT = Path(
    "data/raw/tableshift_cache/physionet.org/files/" "challenge-2019/1.0.0/training"
)
TEMPORAL_METADATA_COLUMNS = {
    "patient_id",
    "source_set",
    "set",
    "ICULOS",
    "time_bin",
}


@dataclass(frozen=True)
class PatientSplit:
    """Disjoint patient identifiers used by each experimental role."""

    train_ids: frozenset[str]
    update_ids: frozenset[str]
    evaluation_ids: frozenset[str]

    def validate(self) -> None:
        """Raise when any patient occurs in more than one partition."""
        if self.train_ids & self.update_ids:
            raise ValueError("train and update patients overlap")
        if self.train_ids & self.evaluation_ids:
            raise ValueError("train and evaluation patients overlap")
        if self.update_ids & self.evaluation_ids:
            raise ValueError("update and evaluation patients overlap")


def passthrough_preprocessor() -> Any:
    """Return a TableShift preprocessor that preserves original columns."""

    return PreprocessorConfig(
        categorical_features="passthrough",
        numeric_features="passthrough",
        dropna="all",
    )


def load_dataset(dataset_name: str):
    """Load a conventional TableShift dataset by name."""
    cache_dir = find_project_root() / "data" / "raw" / "tableshift_cache"

    return get_dataset(
        name=dataset_name,
        cache_dir=str(cache_dir),
        preprocessor_config=passthrough_preprocessor(),
    )


def _physionet_files(data_root: Path) -> list[Path]:
    files = sorted(data_root.glob("training_set*/*.psv"))
    if not files:
        raise FileNotFoundError(
            f"No PhysioNet .psv files found under {data_root}. "
            "Download TableShift physionet first."
        )
    return files


def load_physionet_temporal(
    data_root: str | Path | None = None,
    max_patients: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Load hourly PhysioNet rows while preserving patient and ICULOS order.

    ``ICULOS`` is relative clinical time since ICU admission, not a global
    calendar timestamp. Patient identifiers are derived from source filenames.
    """
    root = (
        Path(data_root)
        if data_root is not None
        else find_project_root() / PHYSIONET_RELATIVE_ROOT
    )
    files = _physionet_files(root)
    if max_patients is not None:
        if max_patients <= 0:
            raise ValueError("max_patients must be positive")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            len(files), size=min(max_patients, len(files)), replace=False
        )
        files = [files[index] for index in sorted(chosen)]

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, sep="|")
        source_set = path.parent.name.removeprefix("training_set").lower()
        patient_id = f"{source_set}_{path.stem}"
        frame.insert(0, "patient_id", patient_id)
        frame.insert(1, "source_set", source_set)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    validate_physionet_temporal(data)
    return data.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)


def validate_physionet_temporal(data: pd.DataFrame) -> None:
    """Validate required fields and strict within-patient ICULOS ordering."""
    required = {"patient_id", "source_set", "ICULOS", "SepsisLabel"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing PhysioNet columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Temporal identifiers and label must not contain nulls")
    for patient_id, group in data.groupby("patient_id", sort=False):
        times = group["ICULOS"].to_numpy(dtype=float)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"ICULOS is not strictly increasing for {patient_id}")


def split_patients(
    data: pd.DataFrame,
    train_fraction: float = 0.6,
    update_fraction: float = 0.2,
    seed: int = SEED,
) -> PatientSplit:
    """Split patient identifiers, stratified by source and ever-sepsis label."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 < update_fraction < 1 - train_fraction:
        raise ValueError("update_fraction leaves no evaluation patients")

    patients = data.groupby("patient_id", as_index=False).agg(
        source_set=("source_set", "first"), ever_sepsis=("SepsisLabel", "max")
    )
    strata = (
        patients["source_set"].astype(str) + "_" + patients["ever_sepsis"].astype(str)
    )
    try:
        train, remainder = train_test_split(
            patients,
            train_size=train_fraction,
            random_state=seed,
            stratify=strata,
        )
        remainder_strata = (
            remainder["source_set"].astype(str)
            + "_"
            + remainder["ever_sepsis"].astype(str)
        )
        update, evaluation = train_test_split(
            remainder,
            train_size=update_fraction / (1 - train_fraction),
            random_state=seed,
            stratify=remainder_strata,
        )
    except ValueError:
        train, remainder = train_test_split(
            patients, train_size=train_fraction, random_state=seed
        )
        update, evaluation = train_test_split(
            remainder,
            train_size=update_fraction / (1 - train_fraction),
            random_state=seed,
        )

    result = PatientSplit(
        train_ids=frozenset(train["patient_id"]),
        update_ids=frozenset(update["patient_id"]),
        evaluation_ids=frozenset(evaluation["patient_id"]),
    )
    result.validate()
    return result


def model_feature_columns(
    data: pd.DataFrame,
    label_col: str = "SepsisLabel",
) -> list[str]:
    """Return numeric predictors while excluding label and temporal metadata."""
    excluded = TEMPORAL_METADATA_COLUMNS | {label_col}
    return [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
