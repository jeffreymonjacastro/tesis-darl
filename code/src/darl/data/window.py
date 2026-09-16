"""Patient-safe temporal windows for PhysioNet Challenge 2019."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalWindow:
    """A closed ICULOS cohort split into adaptation and evaluation patients."""

    index: int
    start_hour: int
    end_hour: int
    update: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class ReferenceFrames:
    """Reference-period frames for base fitting, adaptation and evaluation."""

    train: pd.DataFrame
    update: pd.DataFrame
    evaluation: pd.DataFrame


def _last_patient_record(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one last observation per patient from an ICULOS interval."""
    if frame.empty:
        return frame.copy()
    return (
        frame.sort_values(["patient_id", "ICULOS"])
        .groupby("patient_id", as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def make_reference_frames(
    data: pd.DataFrame, split, end_hour: int = 6
) -> ReferenceFrames:
    """Build reference frames from hours 1 through ``end_hour``."""
    reference = data[data["ICULOS"].between(1, end_hour, inclusive="both")]

    def select(patient_ids: frozenset[str]) -> pd.DataFrame:
        return _last_patient_record(
            reference[reference["patient_id"].isin(patient_ids)]
        )

    return ReferenceFrames(
        train=select(split.train_ids),
        update=select(split.update_ids),
        evaluation=select(split.evaluation_ids),
    )


def make_temporal_windows(
    data: pd.DataFrame,
    split,
    window_hours: int = 3,
    start_hour: int = 7,
    end_hour: int = 200,
) -> list[TemporalWindow]:
    """Create ordered ICULOS windows without sharing patients across cohorts."""
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if start_hour > end_hour:
        raise ValueError("start_hour must not exceed end_hour")

    windows: list[TemporalWindow] = []
    for index, lower in enumerate(range(start_hour, end_hour + 1, window_hours)):
        upper = min(lower + window_hours - 1, end_hour)
        current = data[data["ICULOS"].between(lower, upper, inclusive="both")]
        update = _last_patient_record(
            current[current["patient_id"].isin(split.update_ids)]
        )
        evaluation = _last_patient_record(
            current[current["patient_id"].isin(split.evaluation_ids)]
        )
        if not update.empty and not evaluation.empty:
            windows.append(
                TemporalWindow(
                    index=index,
                    start_hour=lower,
                    end_hour=upper,
                    update=update,
                    evaluation=evaluation,
                )
            )
    return windows


def create_sequential_windows(
    df: pd.DataFrame,
    window_size: int = 1000,
) -> list[pd.DataFrame]:
    """Compatibility helper returning complete non-overlapping row windows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    return [
        df.iloc[start : start + window_size].copy()
        for start in range(0, len(df), window_size)
        if len(df.iloc[start : start + window_size]) == window_size
    ]
