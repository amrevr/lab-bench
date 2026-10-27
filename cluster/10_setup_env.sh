#!/usr/bin/env bash
# Step 2. Builds a user-space virtualenv (no sudo). Edit the module line to what 00_probe.sh showed.
#   PYMODULE="python/3.11" CUDAMODULE="cuda/12.4" VLLM_SPEC="vllm" ./10_setup_env.sh
# No internet on the nodes? Build a wheelhouse elsewhere with make_wheelhouse.sh, scp it, then:
#   WHEELHOUSE=~/wheelhouse ./10_setup_env.sh
set -euo pipefail
VENV="${VENV:-$HOME/labbench-venv}"
[ -n "${PYMODULE:-}" ]   && module load "$PYMODULE"
[ -n "${CUDAMODULE:-}" ] && module load "$CUDAMODULE"
python3 --version
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PIPARGS=""
[ -n "${WHEELHOUSE:-}" ] && PIPARGS="--no-index --find-links $WHEELHOUSE"
python3 -m pip install -q --upgrade pip $PIPARGS || true
python3 -m pip install $PIPARGS "${VLLM_SPEC:-vllm}"
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(i, p.name, round(p.total_memory / 2**30, 1), "GiB", "cc", p.major, p.minor)
PY
echo "env ready: source $VENV/bin/activate"
