# Lab Bench Copilot

Learn a procedure from one expert video plus its written steps, then watch a novice video in short chunks and
try to flag skipped steps, wrong order and wrong ingredient. Technique, quantity and spill errors are not reliably
detected. Latency: roughly 10-15 s from the mistake (6-10 s median from chunk close to verdict on hosted APIs),
near-live, not real time. When unsure it asks ("Quick check") instead of alarming. Built for the Meta Global AI Developer Hackathon, Indie track.

## Pipeline

```
novice clip -> chunker (3 s windows, 0.5 s overlap, H.264 + 4 keyframes, chunks processed concurrently)
  -> SAM 3.1 (hosted, Meta Model API; self-hosted SAM not built)  hand, tomato, plate (one request each, in parallel),
     stitched across chunks by IoU at the same timestamp in the overlap
  -> vision model  tracks + keyframes -> timestamped events + conf; VISION_BACKEND=glimmer (default,
     self-hosted on the H100s, localhost:8000; the replay CLI defaults to spark) or spark (Muse Spark Open on Baseten)
  -> step alignment (deterministic) + Muse Wonder  deviation verdicts, coaching text, ask-vs-flag
  -> overlay  video + protocol checklist + flags + source-frame refs
```

Wonder never sees raw video. Fallback if Glimmer is weak: SAM tracks plus motion/position rules.
Not used on purpose: Muse Image.

## Layout

| Path | What |
|---|---|
| `backend/labbench/schema.py` | Event, Step, Protocol, Verdict (with evidence); flag/ask thresholds per kind (0.75 / 0.40; technique and wrong-item 0.85 / 0.50); false-alarm control in `align.py`: `STRONG` 0.8 (an event at least this confident backs a verdict alone; weaker ones must be seen again, same action/object, in another chunk within `CORROBORATE_GAP` 1.0 s, same technique for a technique verdict, or the verdict is parked and dropped), `ORDER_GAP` 3.0 s, `SKIP_AFTER` 2; a lone event only counts as a step if it shows the step's target/technique (or is seen in a second window) and the step's `after` prerequisites were attempted; unordered `group` peers are not evidence of a skip; an unknown noun is a wrong item only if strong and repeated; ask verdicts are worded as questions |
| `backend/labbench/pipeline/` | `chunker`, `sam`, `vision`, `align`, `wonder` |
| `backend/labbench/run.py`, `eval.py`, `answerkey.py` | run a clip; score verdicts against annotations; `--manifest` scores real keys per step and category |
| `overlay/index.html` | coaching overlay (static page, reads a session JSON) |
| `fixtures/` | `recipe12/protocol.json` (real step ids), `synthetic/` self-test events |
| `scripts/` | fixture generator, overlay server, `calibrate.py` (suggest thresholds), `hosted_smoke.py` (real SAM + Spark call, costs money) |
| `data/` | gitignored: manifest and clips |

## Setup

```
python3 -m pip install -r backend/requirements.txt   # pydantic only; ffmpeg on PATH
cp .env.example .env                                  # keys (META_API_KEY, BASETEN_API_KEY): env only, never commit
cd backend && python3 -m unittest discover -s tests
python3 -m labbench.eval                              # synthetic self-test: expect 4/4 caught
python3 ../scripts/calibrate.py                       # per-verdict hit / false alarm, suggested thresholds
python3 ../scripts/serve_overlay.py --clip 12_43      # open the printed URL
```

Real clips (CaptainCook4D recipe 12, Apache-2.0): manifest `https://files.catbox.moe/dda4z0.json`; download the
`url` of each clip into `data/clips/<id>.mp4`. The dataset's error annotations are used only as the answer key
for scoring, never as pipeline input.

## Status and results

Both vision backends run in the pipeline (Spark Open hosted on Baseten: most runs; Glimmer 30B self-hosted on
2x H100: 12_26 first 80 s and 12_17). Measured on CaptainCook4D recipe 12, one run per clip, small samples:
a fresh set of 3 clean clips gave 0 false alarms with the final aligner; an earlier fresh set found 1 genuine
detection of 9 annotated errors (before the last fixes); there are no fresh error clips for the final code, so
there is no recall claim. Technique, quantity and spill errors are not detected. Details and caveats:

- [docs/SUBMISSION.md](docs/SUBMISSION.md) write-up draft, results and limitations
- [docs/DEMO.md](docs/DEMO.md) demo video storyboard, commands, fallback
- [docs/RUNNING.md](docs/RUNNING.md) tests, replay, eval, Glimmer server, credentials

Progress, decisions and test logs are on the `ledger` branch (`docs/ledger/`) and in the GitHub issues (#7 has the
real-footage results). Thresholds and gate parameters were tuned on the development clips.
