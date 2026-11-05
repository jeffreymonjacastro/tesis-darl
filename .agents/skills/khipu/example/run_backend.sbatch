#!/bin/bash
#SBATCH --job-name=qwen-backend
#SBATCH --account=pregrado
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-qwen-backend-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

printf 'Job %s | Nodo %s | Inicio %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"

test -x .venv/bin/python || {
    echo "No se puede ejecutar $PWD/.venv/bin/python. Ejecuta bash prepare_backend.sh en el nodo de acceso." >&2
    exit 1
}
.venv/bin/python --version
.venv/bin/python -c 'import uvicorn; from main import app; from model_runtime import resolve_checkpoint; print(resolve_checkpoint()[1])' || {
    echo "Revisa las dependencias, el intérprete compartido y models/. Prepara con bash prepare_backend.sh." >&2
    exit 1
}

# Un solo proceso carga una copia del modelo en la GPU compartida.
exec .venv/bin/python -u -m uvicorn main:app \
    --host 0.0.0.0 --port 8000 --workers 1
