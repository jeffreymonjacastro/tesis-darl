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
├── .temp/
├── pyproject.toml
├── uv.lock
├── .agents/
│   └── skills/
│       ├── kaggle/
│       ├── context7-mcp/
│       ├── latex-thesis/
│       ├── formula-explanation/
│       ├── experiment-workflow/
│       ├── results-to-latex/
│       ├── python-style/
│       ├── tikz-diagrams/
│       └── citation-rules/
├── thesis/
│   ├── chapters/
│   ├── figures/
│   │   ├── manual/
│   │   └── generated/
│   ├── tables/
│   │   └── generated/
│   ├── styles/
│   └── build/
├── code/
│   ├── src/
│   │   ├── darl/
│   │   │   ├── data/
│   │   │   ├── drift/
│   │   │   ├── pipeline/
│   │   │   ├── actions/
│   │   │   ├── rl/
│   │   │   ├── evaluation/
│   │   │   └── visualization/
│   │   └── notebooks/
│   ├── experiments/
│   │   └── configs/
│   └── tests/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── data_profiling/
│   ├── models/
│   ├── metrics/
│   ├── logs/
│   ├── figures/
│   └── tables/
└── literature/
    ├── papers/
    ├── notes/
    └── bibtex/
```

Reglas de trabajo:

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
- Cualquier script temporal o de prueba debe ir en `.temp/` y no versionarse. No subir scripts de prueba a `code/src/darl/`.

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
