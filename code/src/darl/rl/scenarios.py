"""Synthetic DARL scenarios for quick PPO experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

DRIFT_TYPES = ("covariate", "concept", "both")
SEVERITIES = {
    "low": 0.25,
    "medium": 0.60,
    "high": 0.90,
}


def make_synthetic_scenarios(n_per_type: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Return deterministic synthetic drift/update scenarios for DARL PPO.

    The values are shaped so the preferred action follows the thesis intuition:
    covariate drift favors feature updates, concept drift favors model updates,
    combined drift favors full retraining, and low severity often favors defer.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []

    for drift_type in DRIFT_TYPES:
        for severity_label, severity in SEVERITIES.items():
            for _ in range(n_per_type):
                noise = float(rng.normal(0.0, 0.01))
                is_cov = drift_type in {"covariate", "both"}
                is_concept = drift_type in {"concept", "both"}

                auc_drop = {
                    "covariate": 0.17,
                    "concept": 0.30,
                    "both": 0.70,
                }[drift_type] * severity + noise
                auc_drop = float(np.clip(auc_drop, 0.01, 0.75))

                rows.append(
                    {
                        "drift_type": drift_type,
                        "severity_label": severity_label,
                        "severity": severity,
                        "psi_mean": float(np.clip((0.28 if is_cov else 0.05) * severity + rng.normal(0, 0.015), 0, 1)),
                        "ks_mean": float(np.clip((0.34 if is_cov else 0.06) * severity + rng.normal(0, 0.015), 0, 1)),
                        "c2st_score": float(np.clip((0.20 if is_cov else 0.08) * severity + rng.normal(0, 0.01), 0, 1)),
                        "delta_auc": auc_drop,
                        "delta_f1": float(np.clip(auc_drop * 0.8 + rng.normal(0, 0.01), 0, 1)),
                        "defer_auc_recovery": 0.0,
                        "update_features_auc_recovery": _recovery(auc_drop, drift_type, "features"),
                        "update_model_auc_recovery": _recovery(auc_drop, drift_type, "model"),
                        "retrain_all_auc_recovery": _recovery(auc_drop, drift_type, "all"),
                        "defer_time_cost": 0.0,
                        "defer_ram_cost": 0.0,
                        "update_features_time_cost": 0.28,
                        "update_features_ram_cost": 0.22,
                        "update_model_time_cost": 0.45,
                        "update_model_ram_cost": 0.35,
                        "retrain_all_time_cost": 1.0,
                        "retrain_all_ram_cost": 1.0,
                        "has_covariate_signal": float(is_cov),
                        "has_concept_signal": float(is_concept),
                    }
                )

    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _recovery(auc_drop: float, drift_type: str, action: str) -> float:
    """Approximate AUC recovery for a synthetic maintenance action."""
    factors = {
        "covariate": {"features": 0.82, "model": 0.25, "all": 0.90},
        "concept": {"features": 0.18, "model": 0.82, "all": 0.90},
        "both": {"features": 0.25, "model": 0.30, "all": 0.95},
    }
    return float(auc_drop * factors[drift_type][action])
