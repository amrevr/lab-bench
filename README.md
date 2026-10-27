# Lab Bench Copilot

Learn a procedure from one expert video plus its written steps, then watch a novice video in short chunks and
flag skipped steps, wrong order, wrong technique or wrong ingredient within seconds. When unsure it asks
("Quick check") instead of alarming. Built for the Meta Global AI Developer Hackathon, Indie track.

## Pipeline

```
novice clip -> chunker (2-5 s H.264 + 3-4 keyframes)
  -> SAM 3.1 (Meta Model API / self-hosted)  hands + objects, stitched across chunks by us
  -> Muse Glimmer 30B (vLLM, own H100 or Baseten)  tracks + keyframes -> timestamped events + conf
  -> step alignment (deterministic) + Muse Wonder  deviation verdicts, coaching text, ask-vs-flag
  -> overlay  video + protocol checklist + flags + source-frame refs
```

Wonder never sees raw video. Fallback if Glimmer is weak: SAM tracks plus motion/position rules.
Not used on purpose: Muse Image, Muse Spark Open.

## Layout

| Path | What |
|---|---|
| `backend/labbench/schema.py` | Event, Step, Protocol, Verdict; flag/ask thresholds (0.75 / 0.40) |
| `backend/labbench/pipeline/` | `chunker`, `sam`, `glimmer`, `align`, `wonder` |
| `backend/labbench/run.py`, `eval.py`, `answerkey.py` | run a clip; score verdicts against annotations |
| `overlay/index.html` | coaching overlay (static page, reads a session JSON) |
| `fixtures/` | `recipe12/protocol.json` (real step ids), `synthetic/` self-test events |
| `scripts/` | fixture generator, overlay server |
| `data/` | gitignored: manifest and clips |

## Setup

```
python3 -m pip install -r backend/requirements.txt   # pydantic only; ffmpeg on PATH
cp .env.example .env                                  # keys: never commit
cd backend && python3 -m unittest discover -s tests
python3 -m labbench.eval                              # synthetic self-test: expect 4/4 caught
python3 ../scripts/serve_overlay.py --clip 12_43      # open the printed URL
```

Real clips (CaptainCook4D recipe 12, Apache-2.0): manifest `https://files.catbox.moe/dda4z0.json`; download the
`url` of each clip into `data/clips/<id>.mp4`. The dataset's error annotations are used only as the answer key
for scoring, never as pipeline input.

## Status

See the `ledger` branch (`docs/ledger/`) for progress, decisions and test results. Perception on real clips
is waiting on the Meta API key / GPU; until then the pipeline runs on synthetic events.
