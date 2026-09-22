"""Patient-disjoint splits and a deterministic 200-bed PhysioNet replay clock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
SCENARIOS = ("clean", "measurement", "label", "combined")


def patient_partitions(frame: pd.DataFrame, seed: int = 42) -> dict[str, set[str]]:
    """Split sites into disjoint A predictor and B policy/role partitions."""
    patients = frame.groupby(["source_set", "patient_id"], sort=False)["SepsisLabel"].max().reset_index()

    def split(ids: np.ndarray, labels: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        stratify = labels if len(np.unique(labels)) > 1 and min(np.bincount(labels.astype(int))) >= 2 else None
        return train_test_split(ids, test_size=fraction, random_state=seed, stratify=stratify)

    source = patients.source_set.astype(str).str.upper()
    a = patients[source == "A"]
    b = patients[source == "B"]
    a_train, a_rest = split(a.patient_id.to_numpy(), a.SepsisLabel.to_numpy(), 0.30)
    a_rest_y = a.set_index("patient_id").loc[a_rest, "SepsisLabel"].to_numpy()
    a_val, a_test = split(a_rest, a_rest_y, 0.50)
    b_train, b_rest = split(b.patient_id.to_numpy(), b.SepsisLabel.to_numpy(), 0.40)
    b_rest_y = b.set_index("patient_id").loc[b_rest, "SepsisLabel"].to_numpy()
    b_val, b_test = split(b_rest, b_rest_y, 0.50)
    result = {"a_train": set(a_train), "a_val": set(a_val), "a_test": set(a_test)}
    for name, ids in (("train", b_train), ("val", b_val), ("test", b_test)):
        labels = b.set_index("patient_id").loc[ids, "SepsisLabel"].to_numpy()
        update, evaluation = split(ids, labels, 0.50)
        result[f"b_{name}_update"] = set(update)
        result[f"b_{name}_eval"] = set(evaluation)
    seen: set[str] = set()
    for ids in result.values():
        if seen & ids:
            raise AssertionError("A patient appears in multiple partitions")
        seen |= ids
    return result


def patient_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each patient total weight one regardless of observed stay length."""
    return (1.0 / frame.groupby("patient_id")["patient_id"].transform("size")).to_numpy(dtype=float)


@dataclass(frozen=True)
class BedConfig:
    """Fixed replay horizon and ICU capacity."""

    beds: int = 200
    days: int = 21
    seed: int = 42


def make_bed_schedule(frame: pd.DataFrame, update_ids: set[str], eval_ids: set[str], config: BedConfig) -> pd.DataFrame:
    """Replay each patient's observed rows once; fill a vacated bed next hour."""
    if config.beds < 2 or config.days < 1:
        raise ValueError("At least two beds and one day are required")
    if update_ids & eval_ids:
        raise ValueError("Update and evaluation patients must be disjoint")
    selected = frame[frame.patient_id.isin(update_ids | eval_ids)].copy()
    selected.sort_values(["patient_id", "ICULOS"], inplace=True)
    histories = {pid: part.index.to_numpy() for pid, part in selected.groupby("patient_id", sort=False)}
    rng = np.random.default_rng(config.seed)
    queues = {"update": list(rng.permutation(sorted(update_ids))), "eval": list(rng.permutation(sorted(eval_ids)))}
    positions = {role: 0 for role in queues}
    slots: list[tuple[str, int, np.ndarray] | None] = [None] * config.beds
    row_indices: list[int] = []
    hours: list[int] = []
    beds: list[int] = []
    roles: list[str] = []
    for hour in range(1, config.days * 24 + 1):
        for bed in range(config.beds):
            role = "update" if bed < config.beds // 2 else "eval"
            slot = slots[bed]
            if slot is None or slot[1] >= len(slot[2]):
                if positions[role] >= len(queues[role]):
                    slots[bed] = None
                    continue
                pid = queues[role][positions[role]]
                positions[role] += 1
                slot = (pid, 0, histories[pid])
            pid, index, history = slot
            row_indices.append(int(history[index]))
            hours.append(hour)
            beds.append(bed)
            roles.append(role)
            slots[bed] = (pid, index + 1, history)
    result = selected.loc[row_indices].reset_index(drop=True)
    result["global_hour"] = hours
    result["day"] = (np.asarray(hours) - 1) // 24 + 1
    result["bed"] = beds
    result["role"] = roles
    return result


def inject_scenario(schedule: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Apply gradual post-day-7 measurement and patient-coherent onset shifts."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    out = schedule.copy()
    out["original_label"] = out.SepsisLabel.astype(int)
    out["severity"] = np.clip((out.day.to_numpy() - 7) / 7, 0.0, 1.0)
    if scenario in {"measurement", "combined"}:
        for column, magnitude in (("HR", 12.0), ("SBP", -10.0), ("MAP", -7.0), ("Resp", 3.0)):
            out[column] = out[column] + magnitude * out.severity
    if scenario in {"label", "combined"}:
        onset = out.loc[out.original_label.eq(1)].groupby("patient_id").ICULOS.min()
        onset_at_row = out.patient_id.map(onset)
        shift = np.floor(6 * out.severity)
        synthetic = onset_at_row.notna() & out.day.ge(8) & out.ICULOS.ge(onset_at_row - shift)
        out["SepsisLabel"] = (out.original_label.eq(1) | synthetic).astype(int)
    return out
