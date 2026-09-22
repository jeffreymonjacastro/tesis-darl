"""Ensure slide figures render from versioned experiment artifacts."""

import pandas as pd

from darl.visualization.bed_figures import make_presentation_figures


def test_presentation_figures_export_png_and_svg(tmp_path):
    policies = ("dqn", "a1", "a4", "random", "threshold")
    scenarios = ("clean", "measurement", "label", "combined")
    pd.DataFrame(
        {"scenario": scenario, "policy": policy, "auprc_hour": 0.1}
        for scenario in scenarios for policy in policies
    ).to_csv(tmp_path / "final_metrics.csv", index=False)
    pd.DataFrame(
        {"scenario": scenario, "policy": policy, "seed": 42, "day": day, "reward": 0.0, "executed": 0}
        for scenario in scenarios for policy in policies for day in (1, 2)
    ).to_csv(tmp_path / "actions.csv", index=False)
    pd.DataFrame(
        {"global_hour": hour, "day": (hour - 1) // 24 + 1, "bed": bed, "patient_id": f"p{bed}-{(hour - 1) // 24}", "ICULOS": (hour - 1) % 24 + 1, "probability": 0.1, "SepsisLabel": 0, "severity": 0.0}
        for hour in range(1, 49) for bed in range(2)
    ).to_parquet(tmp_path / "predictions_clean_42.parquet", index=False)
    figures = make_presentation_figures(tmp_path)
    assert len(figures) == 6
    assert all((tmp_path / "figures" / f"{index:02d}_{name}.svg").is_file() for index, name in enumerate(("camas_y_recambio", "rotacion_por_cama", "calendario_drift", "auprc_politicas", "recompensa_acumulada", "decisiones_dqn"), start=1))
