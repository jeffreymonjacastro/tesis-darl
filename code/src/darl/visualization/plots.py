import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

from darl.utils import find_project_root
from darl.utils.beta_dist import beta_distribution


def plot_beta_distributions(
    data: pd.DataFrame,
    columns: list[str],
    nrows: int = 3,
    ncols: int = 2,
    figsize: tuple[int, int] = (14, 16),
    bins: int = 60,
    title: str = "Ajuste de Distribuciones Beta a Signos Vitales (PhysioNet)",
):
    """
    Grafica los histogramas reales y las curvas PDF Beta teóricas ajustadas
    para múltiples columnas de un DataFrame.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # Descartar ejes sobrantes
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    for i, col in enumerate(columns):
        ax = axes[i]

        α, β, _, data_scaled, col_min, col_max = beta_distribution(
            data[col].dropna(), EPS=1e-6
        )

        # Histograma de datos reales
        ax.hist(
            data_scaled,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#1f77b4",
            edgecolor="black",
            label="Datos reales (escalados)",
        )

        # PDF Beta teórica
        x = np.linspace(0, 1, 300)
        pdf = stats.beta.pdf(x, α, β)
        ax.plot(x, pdf, "r-", lw=2.5, label=f"Ajuste Beta(α={α:.2f}, β={β:.2f})")

        ax.set_title(
            f"{col} | Rango: [{col_min:.1f}, {col_max:.1f}]",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Valor escalado [0, 1]", fontsize=10)
        ax.set_ylabel("Densidad", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()
    return fig


def save_plot_to_figures(fig, filename: str, dpi: int = 300):
    """
    Guarda una figura de matplotlib (fig) en la carpeta outputs/figures/<filename>.
    Si no se especifica extensión, por defecto se guarda como .png.
    """
    project_root = find_project_root()
    figures_dir = project_root / "outputs" / "figures"

    # Asegurar que la carpeta de destino existe
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Asegurar extensión
    if not any(
        filename.lower().endswith(ext)
        for ext in [".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps"]
    ):
        filename += ".png"

    save_path = figures_dir / filename

    # Guardar la figura
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Figura exportada y guardada en: {save_path}")
    return save_path


def plot_temporal_drift(window_summary: pd.DataFrame):
    """Plot sample count, prevalence and mean PSI across ICULOS windows."""
    required = {"window_end", "n_samples", "prevalence", "mean_psi"}
    missing = required - set(window_summary.columns)
    if missing:
        raise ValueError(f"Missing temporal summary columns: {sorted(missing)}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    series = [
        ("n_samples", "Pacientes", "#4c78a8"),
        ("prevalence", "Prevalencia de sepsis", "#f58518"),
        ("mean_psi", "PSI medio", "#54a24b"),
    ]
    for axis, (column, title, color) in zip(axes, series):
        axis.plot(window_summary["window_end"], window_summary[column], marker="o", color=color)
        axis.set_title(title)
        axis.set_xlabel("ICULOS final (horas)")
        axis.grid(alpha=0.25)
    fig.suptitle("Evolución temporal clínica en PhysioNet")
    fig.tight_layout()
    return fig


def plot_training_loss(history: pd.DataFrame):
    """Plot custom DQN Bellman loss by offline update."""
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history["update"], history["loss"], color="#4c78a8")
    axis.set(title="Entrenamiento DQN", xlabel="Actualización", ylabel="Huber loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_policy_comparison(summary: pd.DataFrame):
    """Compare episode reward and regret across maintenance policies."""
    ordered = summary.sort_values("mean_episode_reward", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(ordered["policy"], ordered["mean_episode_reward"], color="#4c78a8")
    axes[0].set_title("Reward episódico medio")
    axes[1].bar(ordered["policy"], ordered["mean_regret"], color="#e45756")
    axes[1].set_title("Regret medio")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
