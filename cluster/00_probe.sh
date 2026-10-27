#!/usr/bin/env bash
# Step 1. Collects facts about the cluster; changes nothing. Run on the login node AND, if GPUs only
# exist on compute nodes, inside an allocation (srun/salloc/qsub, whatever your site uses).
# Send results/probe.txt back.
mkdir -p results; out=results/probe.txt
{
echo "## host";   hostname; date; uname -a
echo "## python"; for p in python python3 python3.9 python3.10 python3.11 python3.12; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"; done
echo "## pip";    python3 -m pip --version 2>&1 | head -1
echo "## gpu";    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv 2>&1 || nvidia-smi 2>&1 | head -15
echo "## nvlink"; nvidia-smi topo -m 2>&1 | head -12
echo "## cuda";   command -v nvcc && nvcc --version | tail -2
echo "## modules (python/cuda/conda/gcc/torch/vllm)"
if type module >/dev/null 2>&1; then
  for k in python anaconda conda miniconda cuda cudnn nccl gcc torch pytorch vllm apptainer singularity; do echo "-- $k"; module -t avail $k 2>&1 | head -15; done
else echo "no module command in this shell"; fi
echo "## containers"; command -v apptainer singularity docker podman 2>&1
echo "## internet";   for u in https://pypi.org https://huggingface.co https://files.pythonhosted.org; do echo "$u -> $(curl -s -m 8 -o /dev/null -w '%{http_code}' $u 2>&1)"; done
echo "## disk";       df -h ~ /tmp 2>&1 | head -5; echo "quota:"; quota -s 2>&1 | head -5
echo "## scheduler"; command -v sbatch qsub bsub 2>&1; echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "## ram/cpu";    free -g | head -2; nproc
} 2>&1 | tee "$out"
echo; echo "Wrote $out"
