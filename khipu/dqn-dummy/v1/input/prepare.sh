#!/bin/bash
# Run on the access node: environment preparation only, no experimental fitting.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
input=khipu/dqn-dummy/v1/input
uv_bin="${HOME}/.local/bin/uv"
if [[ ! -x "$input/.venv/bin/python" ]]; then
    "$uv_bin" venv --python 3.13 "$input/.venv"
fi
"$uv_bin" pip sync --python "$input/.venv/bin/python" "$input/requirements.lock"
"$input/.venv/bin/python" -c 'import torch, xgboost, pandas, nbclient; print(torch.__version__, xgboost.__version__, pandas.__version__)'
