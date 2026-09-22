"""Presentation-ready figures for the simulated DARL ICU experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"dqn": "#1B4965", "a1": "#8AA6B5", "a4": "#E3A24F", "random": "#A77B8A", "threshold": "#6F8F72"}
ACTION_COLORS = ["#8AA6B5", "#4682A9", "#E3A24F", "#9B6370"]
SCENARIOS = ["clean", "measurement", "label", "combined"]
LABELS = {"clean": "B sin inyección", "measurement": "Medición", "label": "Etiqueta", "combined": "Combinado"}


def _save(fig: plt.Figure, folder: Path, name: str) -> None:
    """Export raster and editable vector formats at slide-friendly dimensions."""
    fig.savefig(folder / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_presentation_figures(output_dir: str | Path, *, seed: int = 42) -> list[str]:
    """Build six source-backed figures from downloaded or remote run artifacts."""
    root = Path(output_dir)
    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(root / "final_metrics.csv")
    actions = pd.read_csv(root / "actions.csv")
    predicted = pd.read_parquet(root / f"predictions_clean_{seed}.parquet")
    files: list[str] = []

    # The denominator is occupied beds, not unique patients during a day.
    hourly = predicted.groupby("global_hour").bed.nunique()
    first_seen = predicted.groupby("patient_id").global_hour.min()
    arrivals = first_seen.groupby((first_seen - 1) // 24 + 1).size()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.plot(hourly.index / 24, hourly.values, color=COLORS["dqn"], linewidth=2.4, label="Camas ocupadas por hora")
    ax.set(xlabel="Día simulado", ylabel="Camas ocupadas", title="UCI simulada: ocupación horaria y recambio de pacientes")
    ax.set_ylim(0, 210)
    ax.axhline(200, color="#333333", linestyle="--", linewidth=1, label="Capacidad: 200 camas")
    ax2 = ax.twinx()
    ax2.bar(arrivals.index, arrivals.values, width=0.6, color=COLORS["a4"], alpha=0.45, label="Ingresos al simulador por día")
    ax2.set_ylabel("Pacientes nuevos")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.09, 0.02, "Fin de registro ≠ alta confirmada; ICULOS es tiempo relativo al paciente.", fontsize=10, color="#555555")
    _save(fig, folder, "01_camas_y_recambio")
    files.append("01_camas_y_recambio")

    selected_beds = sorted(predicted.bed.unique())[:12]
    segments = predicted.loc[predicted.bed.isin(selected_beds) & predicted.global_hour.le(120)]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for position, bed in enumerate(selected_beds):
        bed_rows = segments.loc[segments.bed.eq(bed)]
        for index, (_, patient) in enumerate(bed_rows.groupby("patient_id", sort=False)):
            start = int(patient.global_hour.min())
            end = int(patient.global_hour.max())
            ax.broken_barh([((start - 1) / 24, (end - start + 1) / 24)], (position - 0.38, 0.76), facecolors=("#1B4965" if index % 2 == 0 else "#E3A24F"), edgecolors="white", linewidth=0.8)
    ax.set(xlim=(0, min(5, int(predicted.day.max()))), ylim=(-0.7, len(selected_beds) - 0.3), xlabel="Día simulado", ylabel="Cama UCI", title="Cada bloque es un paciente: reemplazo después de su última hora observada")
    ax.set_yticks(range(len(selected_beds)), labels=[str(bed) for bed in selected_beds])
    ax.set_xticks(np.arange(0, min(5, int(predicted.day.max())) + 1))
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.09, 0.02, "Los colores alternan pacientes dentro de una cama; los días son el reloj del simulador, no fechas clínicas.", fontsize=10, color="#555555")
    _save(fig, folder, "02_rotacion_por_cama")
    files.append("02_rotacion_por_cama")

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    severity = predicted.groupby("day").severity.mean() if "severity" in predicted else pd.Series(0.0, index=sorted(predicted.day.unique()))
    days = np.arange(1, int(predicted.day.max()) + 1)
    planned = np.clip((days - 7) / 7, 0, 1)
    ax.plot(days, planned, color=COLORS["a4"], marker="o", linewidth=2.8, label="Severidad programada")
    if max(days) >= 8:
        ax.axvspan(8, min(14, max(days)), color="#F5DDBB", alpha=0.5, label="Rampa de inyección")
        ax.axvline(8, color="#5A5A5A", linestyle="--", linewidth=1.3)
    ax.set(xlim=(1, max(days)), ylim=(-0.04, 1.08), xlabel="Día simulado", ylabel="Severidad relativa", title="Diseño de escenarios: siete días limpios y cambio gradual desde el día 8")
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.09, 0.02, "La intensidad es control experimental; no representa progresión clínica real.", fontsize=10, color="#555555")
    _save(fig, folder, "03_calendario_drift")
    files.append("03_calendario_drift")

    comparison = metrics.groupby(["scenario", "policy"]).auprc_hour.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(SCENARIOS))
    policies = ["dqn", "a1", "a4", "random", "threshold"]
    width = 0.16
    for index, policy in enumerate(policies):
        subset = comparison.loc[comparison.policy.eq(policy)].set_index("scenario").reindex(SCENARIOS)
        yerr = subset["std"].fillna(0).to_numpy()
        ax.bar(x + (index - 2) * width, subset["mean"], width=width, color=COLORS[policy], label=policy.upper() if policy in {"dqn", "a1", "a4"} else policy.capitalize(), yerr=yerr, capsize=3, error_kw={"linewidth": 1})
    ax.set(xticks=x, xticklabels=[LABELS[s] for s in SCENARIOS], ylabel="AUPRC horaria en B-test", title="Predicción en B-test: DARL frente a cuatro políticas comparables")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.09, 0.02, "Barras: media entre semillas; líneas: ±1 desviación estándar (si hay ≥2 semillas).", fontsize=10, color="#555555")
    _save(fig, folder, "04_auprc_politicas")
    files.append("04_auprc_politicas")

    combined = actions.loc[actions.scenario.eq("combined")].copy()
    combined.sort_values(["policy", "seed", "day"], inplace=True)
    combined["cumulative_reward"] = combined.groupby(["policy", "seed"]).reward.cumsum()
    reward_mean = combined.groupby(["policy", "day"]).cumulative_reward.mean().reset_index()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for policy in policies:
        part = reward_mean.loc[reward_mean.policy.eq(policy)]
        ax.plot(part.day, part.cumulative_reward, label=policy.upper() if policy in {"dqn", "a1", "a4"} else policy.capitalize(), color=COLORS[policy], linewidth=2.5)
    if combined.day.max() >= 8:
        ax.axvspan(8, min(14, combined.day.max()), color="#F5DDBB", alpha=0.45)
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set(xlabel="Día simulado", ylabel="Recompensa acumulada media", title="Costo y ganancia predictiva: recompensa acumulada en escenario combinado")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.09, 0.02, "Reward = ganancia AUPRC en pacientes de evaluación − 0,25 × costo de acción.", fontsize=10, color="#555555")
    _save(fig, folder, "05_recompensa_acumulada")
    files.append("05_recompensa_acumulada")

    dqn = actions.loc[actions.scenario.eq("combined") & actions.policy.eq("dqn")]
    counts = pd.crosstab(dqn.day, dqn.executed).reindex(columns=range(4), fill_value=0)
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    bottom = np.zeros(len(counts))
    for action in range(4):
        values = counts[action].to_numpy()
        ax.bar(counts.index, values, bottom=bottom, color=ACTION_COLORS[action], label=f"A{action + 1}", width=0.8)
        bottom += values
    if dqn.day.max() >= 8:
        ax.axvspan(7.5, min(14.5, dqn.day.max() + 0.5), color="#F5DDBB", alpha=0.35)
    ax.set(xlabel="Día simulado", ylabel="Número de semillas", title="Decisiones ejecutadas por DARL en el escenario combinado")
    ax.set_xticks(counts.index)
    ax.legend(title="Acción", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.09, 0.02, "A1 diferir · A2 corregir vitales · A3 actualizar XGBoost · A4 actualizar ambas etapas.", fontsize=10, color="#555555")
    _save(fig, folder, "06_decisiones_dqn")
    files.append("06_decisiones_dqn")
    return [str(folder / f"{name}.png") for name in files]
