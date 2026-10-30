#!/bin/bash
#SBATCH --job-name=darl-ddqn-v1
#SBATCH --account=pregrado
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=.tmp/darl-ddqn-v1-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
input=khipu/dqn-dummy/v1/input
export PATH="$PWD/$input/.venv/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONHASHSEED=42 MPLBACKEND=Agg
export DARL_DUMMY_MODE="${DARL_DUMMY_MODE:-full}"
export DARL_DUMMY_OUTPUT="${DARL_DUMMY_OUTPUT:-khipu/dqn-dummy/v1/outputs/$DARL_DUMMY_MODE}"
export DARL_DUMMY_BUDGET_SECONDS="${DARL_DUMMY_BUDGET_SECONDS:-27000}"
export DARL_DUMMY_PARQUET="$PWD/data/processed/physionet/physionet.parquet"
mkdir -p "$DARL_DUMMY_OUTPUT"
"$input/.venv/bin/python" -m pip freeze > "$DARL_DUMMY_OUTPUT/environment.txt" 2>/dev/null || \
    "$HOME/.local/bin/uv" pip freeze --python "$input/.venv/bin/python" > "$DARL_DUMMY_OUTPUT/environment.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$DARL_DUMMY_OUTPUT/gpu.txt"
# The batch shell executes Python directly; srun steps with shards fail on this cluster.
"$input/.venv/bin/python" -u - <<'PY' &
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
notebook = nbformat.read('khipu/dqn-dummy/v1/input/main.ipynb', as_version=4)
client = NotebookClient(notebook, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}})
try:
    client.execute()
finally:
    nbformat.write(notebook, Path(os.environ['DARL_DUMMY_OUTPUT'])/'executed.ipynb')
PY
runner=$!
# The 7.5h software budget is the primary stop. USR1 provides a final chance to yield.
trap 'pkill -USR1 -P "$runner" || true' USR1
set +e
wait "$runner"
result=$?
if kill -0 "$runner" 2>/dev/null; then
    wait "$runner"
    result=$?
fi
exit "$result"
