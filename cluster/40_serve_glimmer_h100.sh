#!/usr/bin/env bash
# Glimmer 30B FP8 on the dsp 2x H100 (shared). The command that ran (pid 361921, 2026-10-31) plus, with DFLASH=1,
# Meta's DFlash drafter (4 draft tokens: the recipe's 15 is slower under load) and async scheduling. Run on dsp:  [DFLASH=1] bash 40_serve_glimmer_h100.sh
set -euo pipefail
ENV=/data/vermar/envs/glimmer-vllm
export HF_HOME=/data/vermar/lab-bench-glimmer/hf
# no nvcc on dsp: keep FlashInfer/DeepGEMM JIT paths off
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
       VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
SPEC=() LOG=serve.log MEM=0.30
if [ "${DFLASH:-0}" = 1 ]; then
  SPEC=(--speculative-config '{"method":"dflash","model":"meta-models/Muse-Glimmer-30B-assistant","num_speculative_tokens":'"${NSPEC:-4}"'}'
        --async-scheduling)
  LOG=serve_dflash.log MEM=0.44     # drafter weights + its KV need a few GB more per GPU
fi
setsid nohup "$ENV/bin/vllm" serve RedHatAI/Muse-Glimmer-30B-FP8-block --served-model-name glimmer \
  --host 127.0.0.1 --port 18080 --generation-config auto --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
  --max-model-len 16384 --limit-mm-per-prompt '{"image":4}' --gpu-memory-utilization "${GPUMEM:-$MEM}" \
  "${SPEC[@]}" > "/data/vermar/lab-bench-glimmer/$LOG" 2>&1 &
echo "started pid $!; ready when the log shows 'Application startup complete' (~2.5 min)"
