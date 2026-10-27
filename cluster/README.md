# Lab Bench on a V100 cluster

Measures Glimmer latency (issue #14) on 4x V100-SXM2 and profiles the GPUs. Everything except step 2 is
stdlib-only Python (3.6+) plus `nvidia-smi`, so it runs on Python 3.9 with no installs.

## Is Python 3.9 a deal breaker?

Not for the harness. For vLLM it may be: recent vLLM wants Python 3.10+, but your cluster uses Lmod, so
`module avail python` (or anaconda / miniconda) likely offers a newer one. Step 1 checks that. A venv in your
home directory needs no sudo. If the nodes have no internet, build a wheelhouse on another machine.

V100 caveats: compute capability 7.0, so no bf16 (use fp16), no FP8, no FlashAttention-2. Newer vLLM
releases may not support Volta at all; if the install works but the server fails on the attention backend, try
an older vLLM (for example `VLLM_SPEC="vllm==0.8.5"`). This has NOT been tested on a V100 (the build
container has no GPU), and whether a given vLLM version supports the Glimmer architecture is unknown until
the weights are in hand.

Memory: Glimmer 30B in fp16 is about 60 GB of weights. 4x 32 GB cards fit with TP=4; 4x 16 GB cards do not.

## Run order

```
scp -r cluster/ you@cluster:~/lab-bench-cluster && cd ~/lab-bench-cluster
./00_probe.sh                       # send results/probe.txt back; stop here if you want me to pick the path
PYMODULE=python/3.11 CUDAMODULE=cuda/12.4 ./10_setup_env.sh      # module names from the probe
MODEL=/path/to/glimmer ./20_serve_glimmer.sh                      # keep running
PROFILE=1 ./30_run_bench.sh                                       # in another shell: bench + gpu.csv
```

Output: `lab-bench-results.tgz` with `glimmer.json` (TTFT/total p50/p90, sample output), `gpu.csv`
(utilisation/memory/power/clocks every 0.5 s), `vllm_server.log`, and a vLLM torch profile trace in
`results/torch_profile/` (open in https://ui.perfetto.dev).

Dry run without a GPU: `python3 mock_server.py &` then
`python3 bench_glimmer.py --model mock --frames "frames/*.jpg" --runs 3`.

`frames/` holds four keyframes (about 1 s apart) from CaptainCook4D clip 12_43 (Apache-2.0), 640 px wide.
