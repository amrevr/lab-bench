# Running Lab Bench

## Credentials

Keys come from environment variables only and are never committed (`.env` is gitignored):

| Variable | Used for |
|---|---|
| `META_API_KEY` | SAM 3.1 and Muse Wonder (Meta Model API; base URL `META_API_BASE`, default `https://api.meta.ai/v1`) |
| `BASETEN_API_KEY` | Muse Spark Open, backend `spark` |
| `GLIMMER_API_KEY` | optional; the self-hosted vLLM server needs none unless you configure one |
| `GLIMMER_BASE_URL` | default `http://localhost:8000/v1` |

In the dev sandbox the keys are injected by the proxy. Use `urllib`/`curl`-style clients: the OpenAI-SDK-style
client bypassed the proxy and got a 401 (issue #14). Without `META_API_KEY`, Wonder returns the aligner's own
message.

## Tests, offline eval (no API calls, no cost)

```
python3 -m pip install -r backend/requirements.txt     # pydantic, imageio-ffmpeg
cd backend && python3 -m unittest discover -s tests    # or scripts/check.sh from the repo root
python3 -m labbench.eval                               # synthetic self-test, 4 synthetic error clips
python3 -m labbench.eval --root <dir> --manifest <manifest.json>
python3 -m labbench.eval --cc4d ../fixtures/recipe12/cc4d_error_annotations.json --sessions <dir of *.session.json> \
    --held-out 12_48,12_51,12_9,12_13,12_16 [--realign ../fixtures/recipe12/protocol.json]
    # timed scoring vs CaptainCook4D step times: strict (right step, kind, on time) and lenient side by side,
    # false alarms mid-clip vs at end, EgoPER EDA and omission metrics, recording-level TPR/FPR/balanced F1
python3 ../scripts/calibrate.py                        # hit vs false alarm per kind, suggests thresholds
```

`eval --manifest` needs `--root` to hold `protocol.json` and `<clip>/events.json`. It cannot parse the later
`manifest_v2` format (errors are dicts there); held-out 2 and 3 were scored with a scorer copy.

## Replay (real API calls, costs money)

Needs `data/clips/<id>.mp4` and keys (where to get clips, and one stored session: [SETUP.md](SETUP.md)). From `backend/`:

```
python -m labbench.replay --clip ../data/clips/12_43.mp4 --protocol ../fixtures/recipe12/protocol.json \
    --backend spark --start 0 --end 200 --pace realtime --max-spend 3 --finish --out ../out/12_43.session.json
```

- `--backend spark|glimmer` (CLI default `spark`; the `Vision` class default is `glimmer`).
- `--pace realtime` waits for each window to close; `fast` submits immediately.
- `--max-spend` is required: it refuses if the estimate is over the cap and aborts if billed frames would exceed
  it. SAM costs $0.0002 per frame, 150 frames per window at 10 fps (3 concepts x 50-frame minimum).
- `--fps 10 --height 480` are the defaults. `--finish` calls every undone step skipped at `--end`.
- The session JSON has events, verdicts (with `closed_t`, `latency_s`, `coach`) and a `run` block with cost and
  failure counts (`vision_failed`, `sam_failed`, `errors`).
- Check the run block: failed and idle windows are counted separately.

Smoke test of one chunk against the real SAM + Spark APIs: `python3 scripts/hosted_smoke.py --clip ... --workdir ...`.

## Overlay

`python3 scripts/serve_overlay.py --clip 12_43` runs the synthetic fixtures and serves them. For a stored replay
session, serve the repo root statically and pass URLs:

```
python3 -m http.server 8765
# http://localhost:8765/overlay/?session=/out/12_43.session.json&video=/data/clips/12_43.mp4&frames=/out/frames
```

## Glimmer on the cluster (2x H100, shared)

What was done (issue #14): Muse Glimmer 30B, FP8-block, tensor parallel 2, vLLM 0.32, conda env under
`/data/vermar/` (no docker, no sudo), 16384 context, about 26 GB per GPU (32 GB with the drafter); startup takes about 2.5 min (58 s of it
torch.compile; CUDA graphs are on). The command that ran is `cluster/40_serve_glimmer_h100.sh` (run it on the GPU cluster);
`DFLASH=1` adds Meta's DFlash drafter (`meta-models/Muse-Glimmer-30B-assistant`, 4 draft tokens; `NSPEC` overrides)
and `--async-scheduling`: p50 1.8 s instead of 3.6 s per call at real-time pace, for about 6 GB more per GPU
(docs/measurements.md). Stop it by PID: `ps -u vermar -o pid,cmd | grep "[b]in/vllm serve"`.

The cluster has no `nvcc`, so the FlashInfer/DeepGEMM JIT paths crash; the switches in the script avoid them (the
sampler one was the TP=2 crash, `VLLM_USE_DEEP_GEMM=0` stops the FP8 crash; keep all of them). Requests must pass
`chat_template_kwargs: {"reasoning_strength": "low"}` (the client does this for `glimmer`); vLLM ignores
`reasoning_effort`, and default reasoning takes about 18 s and is often cut off.

Forward from the sandbox (only when the backend is `glimmer`):

```
ssh -N -L 8000:127.0.0.1:18080 <gpu-host>
curl -s localhost:8000/v1/models        # check
```

Stop the server: `kill <API server PID>` (only processes owned by vermar).

Shared-GPU etiquette: another user's jobs hold about 42 GB per GPU (swinging about 10 GB), which leaves about
15 GiB per GPU for them while Glimmer runs. Keep the server up only while measuring or recording, stop it
afterwards, kill only your own PIDs, and stop it at once if someone complains. Latency in the pipeline on the
shared machine was worse than in isolation (p50 15.8 s vs about 7.5 s).
