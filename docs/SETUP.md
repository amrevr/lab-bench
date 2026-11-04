# Setting up Lab Bench (for organizers and judges)

Four levels. Each one adds a requirement, so stop at whichever answers your question. Run every command from the
repo root unless the step says otherwise.

| Level | What you get | Needs | Cost |
|---|---|---|---|
| 1. Offline checks | unit tests, synthetic end-to-end eval, fixtures valid | Python 3.10+ | free, ~5 s |
| 2. Stored session | the overlay on a real recorded run; offline re-scoring against CaptainCook4D | + internet (two downloads) | free |
| 3. Hosted replay | the full pipeline live on any CaptainCook4D clip, through Muse Spark Open | + `BASETEN_API_KEY`, the clips | about $0.03 per 2.5 s of video |
| 4. Self-hosted | the same pipeline with Muse Glimmer 30B on your own GPUs; video never leaves the machine | + 2x H100 (80 GB), vLLM | GPU time |

You can also skip the reading: open the repo in Claude Code, Codex or another coding agent and ask it to *"run the
lab-bench tests"*. The skill in [`.claude/skills/lab-bench/SKILL.md`](../.claude/skills/lab-bench/SKILL.md)
(also at `.agents/skills/lab-bench/`, and linked from `AGENTS.md`) walks the agent through levels 1-2 and reports back.

## 1. Offline checks

```
git clone https://github.com/amrevr/lab-bench && cd lab-bench
python3 -m venv .venv && . .venv/bin/activate        # optional
python3 -m pip install -r backend/requirements.txt   # pydantic, imageio-ffmpeg (bundles its own ffmpeg)
scripts/check.sh
```

`scripts/check.sh` runs the unit tests, a synthetic end-to-end eval (four clips with one injected error each plus
one clean clip; the expected result is `caught 4/4, false alarms 0`) and parses every fixture. It makes no network
calls and needs no keys. It ends with `ALL OFFLINE CHECKS PASSED` and exits non-zero on any failure. GitHub Actions runs the
same script on every push.

## 2. A stored real session

This is one recorded replay of CaptainCook4D clip 12_119 (recipe 12, tomato salad) made with the hosted backend.

```
mkdir -p out/sessions data/clips
curl -L -o out/sessions/12_119.session.json https://files.catbox.moe/x062ss.json
curl -L -o data/clips/12_119.mp4 https://files.catbox.moe/xe5dwu.mp4     # 150 MB; CaptainCook4D, Apache-2.0

# re-run the current aligner on the stored events and score against the dataset's timed annotations
cd backend && python3 -m labbench.eval --cc4d ../fixtures/recipe12/cc4d_error_annotations.json \
    --sessions ../out/sessions --realign ../fixtures/recipe12/protocol.json && cd ..

# overlay: video, step checklist, verdicts
python3 -m http.server 8765
# open http://localhost:8765/overlay/?session=/out/sessions/12_119.session.json&video=/data/clips/12_119.mp4
```

12_119 is a recording with no annotated errors, so the score lines read 0/0. The parts worth reading are the
false-alarm counts and the single end-of-clip question. Clips without a stored session are listed as "not scored".

## 3. Hosted replay (Muse Spark Open on Baseten)

```
cp .env.example .env            # fill in BASETEN_API_KEY (and META_API_KEY for Muse Wonder coaching text)
set -a && . ./.env && set +a
cd backend
python3 -m labbench.replay --clip ../data/clips/12_119.mp4 --protocol ../fixtures/recipe12/protocol.json \
    --backend spark --end 120 --pace realtime --max-spend 0 --finish --out ../out/12_119.live.session.json
```

- `--end` is the clip time in seconds to stop at. `--pace fast` submits windows without waiting for real time.
- `--max-spend` caps billed SAM frames. SAM is off by default, so 0 is fine. Spark calls are billed by Baseten
  separately.
- Without `META_API_KEY` the coach line falls back to the aligner's own wording and nothing else changes.
- Other clips: download CaptainCook4D videos with the dataset's official downloader
  (https://github.com/CaptainCook4D/downloader) into `data/clips/<recording id>.mp4`. The recipes with committed
  protocols and annotations are 1, 3, 8, 10, 12 and 27 (`fixtures/recipe<N>/`). Recordings `<N>_*` belong to
  recipe N.
- To score a batch, put the `*.session.json` files in one directory and run the `labbench.eval --cc4d ...` line
  from level 2 with that recipe's fixtures.
- Each recipe's visibility calibration (which steps the vision model can see at all) comes from its expert run:
  `python3 scripts/calibrate_visibility.py --protocol fixtures/recipe27/protocol.json --expert-session out/27_13.session.json --out out/recipe27.protocol.json`.
  The expert clip is `source_clip` in each protocol. Calibration differs per backend, so the numbers in
  `docs/measurements.md` use a protocol calibrated for that backend.

## 4. Self-hosted Muse Glimmer 30B (2x H100)

vLLM 0.32 with the FP8-block checkpoint, tensor parallel 2, about 26 GB per GPU (32 GB with the drafter). The
script `cluster/40_serve_glimmer_h100.sh` contains the exact command that was measured. Set its paths for your
machine:

```
VLLM_ENV=/path/to/venv-with-vllm HF_HOME=/path/to/hf-cache LOGDIR=/path/to/logs \
    DFLASH=1 bash cluster/40_serve_glimmer_h100.sh        # ready after "Application startup complete" (~2.5 min)
curl -s localhost:18080/v1/models
export GLIMMER_BASE_URL=http://localhost:18080/v1
cd backend && python3 -m labbench.replay ... --backend glimmer      # same flags as level 3
```

- `DFLASH=1` adds Meta's DFlash drafter, which halved per-call latency in our runs (3.6 s to 1.8 s median).
- Keep the `VLLM_*=0` switches in the script. They avoid JIT paths that crash on machines without `nvcc`.
- The server binds to 127.0.0.1:18080. From another machine, use an SSH forward (`ssh -N -L 8000:127.0.0.1:18080 <gpu-host>`)
  and leave `GLIMMER_BASE_URL` at its default `http://localhost:8000/v1`.

## What the numbers mean

The README results and `docs/measurements.md` come from one run per clip. Repeat runs vary by about 4 strict catches
per recipe. The session files behind each table (other than 12_119) are not in the repo, so to reproduce a row,
re-run level 3 or 4 on that recipe's clips.
