## Reglas vigentes del monorepo DARL

Este repositorio se llama `tesis-darl` y se organiza como monorepo para la tesis
en LaTeX y el entorno experimental en Python de DARL: Drift-Aware
Reinforcement Learning for selective updating of two-stage tabular ML pipelines.

La estructura autorizada del proyecto es:

```text
tesis-darl/
├── AGENTS.md
├── README.md
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .agents/
│   ├── README.md
│   └── skills/
│       ├── README.md
│       ├── latex-thesis/
│       │   └── README.md
│       ├── formula-explanation/
│       │   └── README.md
│       ├── literature-review/
│       │   └── README.md
│       ├── experiment-workflow/
│       │   └── README.md
│       ├── results-to-latex/
│       │   └── README.md
│       ├── python-style/
│       │   └── README.md
│       └── citation-rules/
│           └── README.md
├── thesis/
│   ├── README.md
│   ├── chapters/
│   │   └── README.md
│   ├── figures/
│   │   ├── README.md
│   │   ├── manual/
│   │   │   └── README.md
│   │   └── generated/
│   │       └── README.md
│   ├── tables/
│   │   ├── README.md
│   │   └── generated/
│   │       └── README.md
│   ├── styles/
│   │   └── README.md
│   └── build/
│       └── README.md
├── code/
│   ├── README.md
│   ├── src/
│   │   ├── README.md
│   │   └── darl/
│   │       ├── README.md
│   │       ├── data/
│   │       │   └── README.md
│   │       ├── drift/
│   │       │   └── README.md
│   │       ├── pipeline/
│   │       │   └── README.md
│   │       ├── actions/
│   │       │   └── README.md
│   │       ├── rl/
│   │       │   └── README.md
│   │       ├── evaluation/
│   │       │   └── README.md
│   │       └── visualization/
│   │           └── README.md
│   ├── experiments/
│   │   ├── README.md
│   │   └── configs/
│   │       └── README.md
│   ├── notebooks/
│   │   └── README.md
│   └── tests/
│       └── README.md
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── README.md
│   ├── interim/
│   │   └── README.md
│   ├── processed/
│   │   └── README.md
│   └── external/
│       └── README.md
├── outputs/
│   ├── README.md
│   ├── models/
│   │   └── README.md
│   ├── metrics/
│   │   └── README.md
│   ├── logs/
│   │   └── README.md
│   ├── figures/
│   │   └── README.md
│   └── tables/
│       └── README.md
├── literature/
│   ├── README.md
│   ├── papers/
│   │   └── README.md
│   ├── notes/
│   │   └── README.md
│   └── bibtex/
│       └── README.md
└── scripts/
    └── README.md
```

Reglas de trabajo para Codex:

- `thesis/` contiene la tesis en LaTeX. Mantener la redacción académica en
  español y no editar manualmente archivos generados.
- Cuando se agreguen fórmulas en la tesis, usar el patrón: fórmula, `Donde:`,
  explicación de símbolos y explicación conceptual.
- `code/` contiene el paquete Python y experimentos. El código Python debe ser
  modular, reproducible y documentado antes de ejecutar experimentos grandes.
- `data/` contiene datasets y no debe versionarse salvo sus `README.md`.
- `outputs/` contiene resultados generados y no debe versionarse salvo sus
  `README.md`.
- Los resultados que entren a la tesis deben generarse primero en `outputs/` y
  luego exportarse a `thesis/figures/generated/` o
  `thesis/tables/generated/`.
- `literature/` contiene papers, notas y BibTeX. No subir PDFs con restricciones
  de licencia si no corresponde.
- `.agents/skills/` contendrá skills futuras de Codex. Cada skill real deberá
  tener un `SKILL.md`, pero no crear esos archivos hasta que se pida
  explícitamente.
- No subir datasets pesados, modelos entrenados, cachés, logs, notebooks de
  checkpoint, builds de LaTeX ni resultados temporales.
- Antes de borrar o mover contenido existente, revisar el estado de Git y
  preservar cambios del usuario.
- En la fase de inicialización del repositorio solo se crean carpetas,
  `README.md` de orientación y placeholders mínimos permitidos.

## Contexto experimental futuro conservado

You are helping me build the experimental environment for my CS thesis:
"Drift-Aware Selective Updating of Two-Stage Tabular ML Pipelines:
A Diagnostic Framework and Empirical Evaluation."

## Context

The thesis evaluates a framework that detects whether performance
degradation in a two-stage ML pipeline (preprocessing + predictive model)
is caused by covariate shift (change in P(X)) or concept drift
(change in P(Y|X)), and recommends which stage to update selectively
instead of retraining the full pipeline.

## What to build

Set up a modular Python project with the following components:

---

### 1. Data Loading — TableShift

- Install and configure the `tableshift` library
- Load two datasets: `adult` (Income) and `hospital_readmission`
- Split each into: reference set (train), pre-drift test, post-drift test
- Print shape, class balance, and feature types for each dataset

---

### 2. Two-Stage Pipeline

Build a sklearn-compatible pipeline with two explicit stages:

**Stage 1 — Preprocessing:**

- QuantileTransformer for skewed numeric features
- TargetEncoder for high-cardinality categoricals
- StandardScaler for remaining numeric features

**Stage 2 — Predictive Model:**

- Model A: LogisticRegression (baseline)
- Model B: XGBClassifier (main model)

Each stage must be independently refittable without touching the other.

---

### 3. Synthetic Drift Injector

Create a `DriftInjector` class with these parameters:

- `drift_type`: "covariate" | "concept" | "both"
- `severity`: "low" | "medium" | "high"
  (map to numeric multipliers: 0.5, 1.5, 3.0)
- `start_idx`: index in the data stream where drift begins
- `duration`: None = permanent abrupt drift, int = temporary spike

Behaviors:

- Covariate shift: shift numeric feature distributions by
  severity × std of each feature
- Concept drift: randomly flip labels with probability
  proportional to severity (cap at 45%)
- Both: apply both transformations simultaneously

---

### 4. Metrics Tracker

Create a `MetricsTracker` class that records for each experiment run:

**Performance metrics:**

- AUC-ROC (primary)
- F1-score
- Δ AUC (drop from baseline)

**Compute metrics:**

- Retraining time (seconds) per strategy
- Peak RAM usage (MB) using tracemalloc or memory_profiler

---

### 5. Drift Detection — Evidently + River

Integrate two detection approaches:

**Evidently (batch monitoring):**

- Use `DataDriftPreset` to compute per-feature KS test and PSI
  between reference and post-drift windows
- Export results as JSON for programmatic use

**River (online/streaming detection):**

- Use `river.drift.ADWIN` on the model's prediction error stream
- Use `river.stream.iter_pandas` to simulate streaming from DataFrames
- Log the index at which drift is flagged

---

### 6. Experiment Runner

Create a script `run_experiment.py` that:

1. Loads a dataset
2. Trains the baseline pipeline (no drift)
3. Injects drift using DriftInjector with specified parameters
4. Evaluates model performance on post-drift data
5. Runs Evidently and River detection
6. Logs all metrics to a results CSV with columns:
   dataset, model, drift_type, severity, start_idx,
   auc_before, auc_after, delta_auc, retrain_time_sec, ram_mb

---

### Project structure

drift_framework/
├── data/
│ └── loader.py # TableShift wrappers
├── pipeline/
│ └── two_stage.py # Stage 1 + Stage 2, independently refittable
├── drift/
│ └── injector.py # DriftInjector class
├── monitoring/
│ ├── evidently_monitor.py
│ └── river_monitor.py
├── metrics/
│ └── tracker.py # MetricsTracker class
├── run_experiment.py # Main experiment runner
└── requirements.txt

---

### Requirements

- Python 3.10+
- tableshift, scikit-learn, xgboost, category_encoders
- evidently, river
- pandas, numpy, tracemalloc
- All random seeds fixed to 42 for reproducibility
- Add brief docstrings to every class and public method

Start by creating the project structure and requirements.txt,
then implement each module in the order listed above.
After each module, run a quick sanity check (print shapes,
sample output, or a small unit test) before moving to the next.
