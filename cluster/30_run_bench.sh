#!/usr/bin/env bash
# Step 4. With the server from step 3 running (other terminal / background), benchmark + sample GPUs.
set -euo pipefail
mkdir -p results
python3 gpu_monitor.py --out results/gpu.csv --interval 0.5 -- \
  python3 bench_glimmer.py --url "http://localhost:${PORT:-8000}/v1" --model "${NAME:-glimmer}" \
  --frames "frames/*.jpg" --nframes "${NFRAMES:-4}" --runs "${RUNS:-20}" --warmup 3 \
  --out results/glimmer.json ${PROFILE:+--profile}
nvidia-smi > results/nvidia_smi_after.txt 2>&1 || true
tar czf lab-bench-results.tgz results && echo "scp this back: $PWD/lab-bench-results.tgz"
