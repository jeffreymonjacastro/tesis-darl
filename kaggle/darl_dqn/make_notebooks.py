"""Generate versioned Kaggle notebooks from one reviewed experiment template."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
SOURCES = ["jeffreyamc/physionet-sepsis-processed", "jeffreymonjacastro/darl-package"]


def write_version(version: str) -> None:
    """Write a small-cell notebook and matching T4 kernel metadata."""
    folder = ROOT / version / "input"
    folder.mkdir(parents=True, exist_ok=True)
    source_root = ROOT.parents[1] / "code" / "src" / "darl"
    source_hashes = {
        "data/bed_stream.py": hashlib.sha256((source_root / "data" / "bed_stream.py").read_bytes()).hexdigest(),
        "evaluation/bed_experiment.py": hashlib.sha256((source_root / "evaluation" / "bed_experiment.py").read_bytes()).hexdigest(),
        "rl/bed_env.py": hashlib.sha256((source_root / "rl" / "bed_env.py").read_bytes()).hexdigest(),
        "visualization/bed_figures.py": hashlib.sha256((source_root / "visualization" / "bed_figures.py").read_bytes()).hexdigest(),
    }
    cells = [
        nbformat.v4.new_markdown_cell(
            f"# DARL PhysioNet A → B — {version}\n\n"
            "A: XGBoost con pacientes disjuntos 70/15/15. B: UCI simulada de 200 camas, "
            "predicción cada hora y decisión al cierre de cada día. ICULOS es relativo al paciente, "
            "no una fecha. Las etiquetas de B llegan con 24 horas de retraso."
        ),
        nbformat.v4.new_markdown_cell("## 1. Importaciones y semilla\nSe usa la semilla 42 en particiones, colas y DQN."),
        nbformat.v4.new_code_cell("import json, sys, zipfile, shutil\nfrom pathlib import Path\nimport numpy as np\nimport torch\nnp.random.seed(42)\ntorch.manual_seed(42)"),
        nbformat.v4.new_markdown_cell("## 2. Resolver los dos datasets\nEl paquete puede venir como directorio o como `darl.zip`."),
        nbformat.v4.new_code_cell(
            "mounted = Path('/kaggle/input')\n"
            "data_candidates = list(mounted.rglob('physionet.parquet'))\n"
            "print('Montajes:', [p.name for p in mounted.iterdir()])\n"
            "assert len(data_candidates) == 1, data_candidates\n"
            "data_path = data_candidates[0]\n"
            "package_candidates = list(mounted.rglob('darl.zip'))\n"
            "directory_candidates = [p for p in mounted.rglob('bed_stream.py') if p.parent.name == 'data']\n"
            "assert package_candidates or directory_candidates, 'darl package not mounted'\n"
            "package_root = package_candidates[0].parent if package_candidates else directory_candidates[0].parent.parent\n"
            "if (package_root / 'darl').is_dir():\n"
            "    sys.path.insert(0, str(package_root))\n"
            "elif (package_root / 'data' / 'bed_stream.py').is_file():\n"
            "    shutil.copytree(package_root, Path('/kaggle/temp/darl'), dirs_exist_ok=True)\n"
            "    sys.path.insert(0, '/kaggle/temp')\n"
            "else:\n"
            "    archive = package_root / 'darl.zip'\n"
            "    assert archive.is_file(), list(package_root.iterdir())\n"
            "    destination = Path('/kaggle/temp/darl')\n"
            "    destination.mkdir(exist_ok=True)\n"
            "    with zipfile.ZipFile(archive) as zipped:\n"
            "        zipped.extractall(destination)\n"
            "    sys.path.insert(0, '/kaggle/temp')\n"
            "from darl.evaluation.bed_experiment import run_experiment\n"
            "import hashlib, darl\n"
            f"expected_hashes = {source_hashes!r}\n"
            "package_dir = Path(next(iter(darl.__path__)))\n"
            "for relative, expected in expected_hashes.items():\n"
            "    actual = hashlib.sha256((package_dir / relative).read_bytes()).hexdigest()\n"
            "    assert actual == expected, f'Paquete Kaggle desactualizado: {relative}'\n"
            "assert torch.cuda.is_available(), 'T4/CUDA no disponible'\n"
            "print('CUDA:', torch.cuda.get_device_name(0), '| parquet:', data_path.name)"
        ),
        nbformat.v4.new_markdown_cell(
            "## 3. DQN del laboratorio\nRed Q de dos capas ocultas de 128 ReLU, replay FIFO, "
            "ε-greedy, Adam y copia rígida de la red objetivo cada 200 actualizaciones. "
            "Estado DARL de 28 componentes y cuatro acciones. Una transición entra al replay "
            "solo cuando se conoce la etiqueta que determina su recompensa."
        ),
        nbformat.v4.new_code_cell(
            "from darl.rl.course_dqn import CourseDQNConfig\n"
            "cfg = CourseDQNConfig()\n"
            "print({'state_dim': cfg.state_dim, 'actions': cfg.action_dim, 'hidden': cfg.hidden, "
            "'target_interval': cfg.target_interval, 'device': 'cuda'})"
        ),
        nbformat.v4.new_markdown_cell("## 4. Experimento\nLa versión corta verifica integración; la completa evalúa cuatro escenarios y cinco secuencias finales."),
        nbformat.v4.new_code_cell(
            f"output_dir = Path('/kaggle/working')\n"
            f"summary = run_experiment(data_path, output_dir, version='{version}', device='cuda', seed=42)\n"
            "print({key: summary[key] for key in ('status', 'version', 'n_test_runs', 'dqn_updates')})"
        ),
        nbformat.v4.new_markdown_cell("## 5. Verificación de artefactos\nLas métricas finales usan exclusivamente pacientes B reservados para prueba."),
        nbformat.v4.new_code_cell(
            "assert summary['status'] == 'complete'\n"
            "assert (output_dir / 'run_summary.json').is_file()\n"
            "assert (output_dir / 'final_metrics.csv').is_file()\n"
            "print('Archivos:', sorted(path.name for path in output_dir.iterdir() if path.is_file()))"
        ),
        nbformat.v4.new_markdown_cell("## 6. Figuras para presentación\nOcupación, calendario del drift, AUPRC, recompensa y acciones del DQN. Se exportan PNG y SVG."),
        nbformat.v4.new_code_cell(
            "from darl.visualization.bed_figures import make_presentation_figures\n"
            "figures = make_presentation_figures(output_dir, seed=42)\n"
            "summary['presentation_figures'] = [str(Path(item).relative_to(output_dir)) for item in figures]\n"
            "(output_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')\n"
            "print('Figuras:', [Path(item).name for item in figures])"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.write(notebook, folder / "main.ipynb")
    metadata = {
        "id": f"jeffreymonjacastro/darl-physionet-a-to-b-{version}",
        "title": f"DARL PhysioNet A to B {version}",
        "code_file": "main.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": SOURCES,
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    for name in ("v1", "v2"):
        write_version(name)
