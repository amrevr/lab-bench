#!/usr/bin/env bash
# Step 3. Serves a model with vLLM across all visible GPUs. V100 = no bf16, no FP8, no FlashAttention-2.
#   MODEL=/path/or/hf-id [TP=4] [MAXLEN=8192] ./20_serve_glimmer.sh
set -euo pipefail
source "${VENV:-$HOME/labbench-venv}/bin/activate"
: "${MODEL:?set MODEL to the weights directory or HF id}"
export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-$PWD/results/torch_profile}"
mkdir -p results "$VLLM_TORCH_PROFILER_DIR"
exec vllm serve "$MODEL" --served-model-name "${NAME:-glimmer}" \
  --dtype float16 --tensor-parallel-size "${TP:-4}" --max-model-len "${MAXLEN:-8192}" \
  --limit-mm-per-prompt "image=${NIMG:-4}" --gpu-memory-utilization "${GPUMEM:-0.90}" \
  --enforce-eager --port "${PORT:-8000}" ${EXTRA:-} 2>&1 | tee results/vllm_server.log
