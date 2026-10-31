# Lab Bench Copilot

Learn a procedure from one expert video plus its written steps, then watch a novice video in short chunks and
flag skipped steps, wrong order, wrong technique or wrong ingredient. Latency claim: about 10 s, near-live, on
hosted APIs (3 s chunks; 5 s is not reachable with this chunking, see issue #10). When unsure it asks
("Quick check") instead of alarming. Built for the Meta Global AI Developer Hackathon, Indie track.

## Pipeline

```
novice clip -> chunker (3 s windows, 0.5 s overlap, H.264 + 3-4 keyframes, chunks processed concurrently)
  -> SAM 3.1 (Meta Model API / self-hosted)  hand, tomato, plate (one request each, in parallel),
     stitched across chunks by IoU at the same timestamp in the overlap
  -> vision model  tracks + keyframes -> timestamped events + conf; VISION_BACKEND=glimmer (default,
     self-hosted on the H100s, localhost:8000) or spark (Muse Spark Open on Baseten)
  -> step alignment (deterministic) + Muse Wonder  deviation verdicts, coaching text, ask-vs-flag
  -> overlay  video + protocol checklist + flags + source-frame refs
```

Wonder never sees raw video. Fallback if Glimmer is weak: SAM tracks plus motion/position rules.
Not used on purpose: Muse Image.

## Layout

| Path | What |
|---|---|
| `backend/labbench/schema.py` | Event, Step, Protocol, Verdict (with evidence); flag/ask thresholds per kind (0.75 / 0.40; technique and wrong-item 0.85 / 0.50); false-alarm control in `align.py`: `STRONG` 0.8 (an event at least this confident backs a verdict alone; weaker ones must be seen again, same action/object, in another chunk within `CORROBORATE_GAP` 1.0 s, same technique for a technique verdict, or the verdict is parked and dropped), `ORDER_GAP` 3.0 s, `SKIP_AFTER` 2 |
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

## Status

See the `ledger` branch (`docs/ledger/`) for progress, decisions and test results. `MetaSam` and the Spark backend are implemented and
smoke-tested on one chunk; the Glimmer backend is untested until the cluster server is up. Thresholds were tuned
on a handful of clips (synthetic so far), so no general recall claim.
