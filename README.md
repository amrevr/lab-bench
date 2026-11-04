# Lab Bench Copilot

[![tests](https://github.com/amrevr/lab-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/amrevr/lab-bench/actions/workflows/tests.yml)

Watches someone carry out a hands-on procedure and speaks up when a step goes wrong. It learns the procedure from
its written steps and one expert recording, then watches the novice's video in 3-second windows, ticks steps off,
asks a quick question when unsure ("Did you add the salt?"), and flags skipped steps, wrong order, wrong
ingredients and visible mishaps such as spills. Cooking stands in for lab work: everything is evaluated on
CaptainCook4D recipes against the dataset's own timed error annotations.

Built for the Meta Global AI Developer Hackathon, Indie track, by team *Your Unified Interpreter Recorder*.

**Organizers and judges:** [docs/SETUP.md](docs/SETUP.md) goes from a 5-second offline check to a self-hosted run.
Or ask your coding agent (Claude Code, Codex, ...) to *"run the lab-bench tests"*: it picks up
[the bundled skill](.claude/skills/lab-bench/SKILL.md) through `AGENTS.md`.

## How it works

```
novice video -> chunker: 3 s windows, 0.5 s overlap, 4 keyframes each, several windows in flight
  -> Muse Glimmer 30B (self-hosted, vLLM FP8 on 2x H100, DFlash drafter): keyframes -> JSON events
       {action, obj, target, technique, issue (mishap), conf}
  -> aligner (deterministic): events vs the procedure's prerequisite graph -> verdicts
       on_track | ask ("Quick check: did you ...?") | flag (skipped, wrong order, wrong item, mishap)
  -> Muse Wonder: rewrites each verdict as a short coaching line, asynchronously (never delays a flag)
  -> overlay: video, step checklist, flags, the keyframes behind each verdict
```

**Learning a new procedure without labels.** `scripts/make_protocol.py` turns a CaptainCook4D task graph into a
protocol: Glimmer names each step from its text (action, object, target, verb synonyms, how the step looks on
camera), and the graph's edges become prerequisites. `scripts/calibrate_visibility.py` replays the expert recording
and marks steps Glimmer never sees as not visible, so the system doesn't ask about steps it cannot perceive.

**Ask before alarming.** A verdict needs a confident event (>= 0.6) or the same evidence seen again; otherwise it
becomes a question or is dropped. Mishap flags need confidence >= 0.8 or a repeat.

## Meta models

| Model | Role | Where it runs |
|---|---|---|
| Muse Glimmer 30B + `Muse-Glimmer-30B-assistant` drafter | vision events per window; writes protocols from step text | self-hosted, vLLM, tensor parallel 2, FP8, 2x H100 |
| Muse Wonder | coaching text | Meta Model API |
| Muse Spark Open | alternate vision backend (`--backend spark`) | hosted on Baseten |
| SAM 3.1 | object tracks; optional (`--sam`, `--sam-url`), off by default | Meta Model API or self-hosted |
| V-JEPA 2 ViT-L | evaluated as features for error detection and action recognition; not adopted | self-hosted |

## Results (measured, one run per clip)

**What a user would experience: hosted vs self-hosted.** Both backends ran the same pipeline, aligner and scorer.

| | Hosted: Muse Spark Open (Baseten) | Self-hosted: Muse Glimmer 30B (2x H100) |
|---|---|---|
| Live, end to end (12_41 at real-time pace) | keeps up: alerts ~9 s after the moment (median), max 17 s | 1.8 s per call on dedicated GPUs (4.1 s p90); on a shared, saturated machine calls took ~15 s and alerts fell 1.5-4 min behind |
| Recipe 27 (never tuned on): errors caught on time / lenient | **15/45** / **24/45** | 9/45 / 16/45 |
| Error recordings with a catch | 8/9 | 8/9 |
| Mid-clip false alarms | 39 | **20** |
| Video leaves your machines | yes (hosted API) | no |

Spark gives more and earlier catches and is the live option; Glimmer gives fewer false alarms and keeps video
local. The drafter halved Glimmer's per-call latency (3.6 s -> 1.8 s median at real-time pace). One run per
backend; details in `docs/measurements.md`.

**Accuracy** against CaptainCook4D's timed error annotations. Strict = right step and compatible verdict, during
the step or within 30 s; lenient = any warning on that step. The expert recording is never scored.

| Recipe | Role | Errors caught strict / lenient | Error recordings caught | Normal recordings with a mid-clip alarm |
|---|---|---|---|---|
| 12 tomato salad | tuning | 7/24 / 12/24 | 4/4 | 2/7 |
| 12 | held-out clips | 0/7 / 2/7 | 0/2 | 1/3 |
| 8 hot chocolate | development | 15/41 / 28/41 | 9/9 | 2/6 |
| 10 pinwheels | development | 19/91 / 32/91 | 7/8 | 3/3 |
| 1 egg sandwich | first clean test, later informed a fix | 18/77 / 40/77 | 11/13 | 2/4 |
| 3 mug pizza | clean test | 9/43 / 21/43 | 5/7 | 5/5 |
| 27 cheese pimiento | clean test of the current code | 12/45 / 16/45 | 8/9 | 5/5 |

What this means: it catches at least one error in most recordings that have one, mostly skipped steps, wrong order
and mishaps. It does not reliably catch technique, quantity or temperature errors (most aren't visible as
steps). Precision on unseen recipes is the main weakness: about 2-5 false alarms per recording (recipe 27: 1.9; recipe 3: 4.6). Accuracy depends
strongly on how visible a recipe's steps are to the vision model. Repeat runs vary by about 4 strict catches.
Full table, method and caveats: [docs/measurements.md](docs/measurements.md).

**Tried, measured, dropped:** SAM tracks in the loop (no verdict changed, real-time headroom lost), V-JEPA 2
features (error AUC ~0.55; a cross-recipe action probe below the majority baseline), a Glimmer "second look"
investigator (confirms nearly every alert: AUC 0.48), skipping Glimmer's reasoning, 8 frames per window, protocol
text in the prompt, ensembling runs.

## Run it

Step by step, with a stored session to try for free: [docs/SETUP.md](docs/SETUP.md).

```
python3 -m pip install -r backend/requirements.txt      # pydantic, imageio-ffmpeg
scripts/check.sh                                        # offline: unit tests, synthetic eval, fixtures
cd backend

# Glimmer server (2x H100): DFLASH=1 ../cluster/40_serve_glimmer_h100.sh ; GLIMMER_BASE_URL defaults to localhost:8000
python3 -m labbench.replay --clip ../data/clips/12_41.mp4 --protocol ../fixtures/recipe12/protocol.json \
    --backend glimmer --end 300 --pace realtime --max-spend 0 --finish --out /tmp/12_41.session.json
mkdir -p ../out && cp /tmp/12_41.session.json ../out/ && cd .. && python3 -m http.server 8765
# open http://localhost:8765/overlay/?session=/out/12_41.session.json&video=/data/clips/12_41.mp4
# (then cd backend again for the commands below)

# protocol for any CaptainCook4D recipe, then score sessions
python3 ../scripts/make_protocol.py --task-graph <graph.json> --step-desc <step_idx_description.json> \
    --annotations <complete_step_annotations.json> --recipe 8 --out ../fixtures/recipe8/protocol.json
python3 -m labbench.eval --cc4d ../fixtures/recipe8/cc4d_error_annotations.json --sessions <dir> \
    --realign ../fixtures/recipe8/protocol.json
```

Keys come from environment variables only (`META_API_KEY`, `BASETEN_API_KEY`; see `.env.example`). Details:
[docs/RUNNING.md](docs/RUNNING.md). Clips: CaptainCook4D (Apache-2.0), downloaded into `data/clips/<id>.mp4`; the
dataset's error annotations are used only to score, never as pipeline input.

## Layout

| Path | What |
|---|---|
| `backend/labbench/pipeline/` | `chunker`, `vision` (Glimmer/Spark client), `align` (step tracker and verdicts), `wonder`, `sam` |
| `backend/labbench/replay.py` | run a clip through the pipeline at real-time or fast pace, write a session |
| `backend/labbench/eval.py` | timed scorer against CaptainCook4D annotations |
| `scripts/` | protocol generation, visibility calibration, overlay server, `check.sh` (offline checks, also run by CI) |
| `fixtures/recipe*/` | protocols and error annotations for recipes 1, 3, 8, 10, 12, 27 |
| `overlay/index.html` | coaching overlay (static page reading a session JSON) |
| `cluster/` | Glimmer serving scripts and benchmarks for the H100 machine |
| `docs/` | `SETUP.md` (organizer guide), `RUNNING.md` (details), `measurements.md`, `SUBMISSION.md`, `DEMO.md` |
| `.claude/skills/lab-bench/`, `AGENTS.md` | agent skill: set up, test and demo the repo through any coding agent |
