"""Real-API smoke test (costs money, not part of the unit tests).
python scripts/hosted_smoke.py --clip /path/expert.mp4 --workdir /path/empty-dir [--start 120]

Cuts a 3 s H.264 faststart chunk, runs MetaSam for hand/tomato/plate (needs META_API_KEY) and then the
Spark vision backend on 4 keyframes (needs BASETEN_API_KEY). Keys come from the environment only.
"""
import argparse
import os
import pathlib
import sys
import time

import imageio_ffmpeg

os.environ.setdefault("FFMPEG", imageio_ffmpeg.get_ffmpeg_exe())   # static build, no system ffmpeg needed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))
from labbench.pipeline import chunker  # noqa: E402
from labbench.pipeline.sam import CONCEPTS, MetaSam  # noqa: E402
from labbench.pipeline.vision import Vision  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clip", type=pathlib.Path, required=True)
ap.add_argument("--workdir", type=pathlib.Path, required=True)
ap.add_argument("--start", type=float, default=120.0)
ap.add_argument("--length", type=float, default=3.0)
a = ap.parse_args()

chunk = chunker.cut_chunk(a.clip, a.start, a.start + a.length, a.workdir / "chunk.mp4", height=720)
print(f"chunk {chunk.stat().st_size / 1024:.0f} KB, {a.start}-{a.start + a.length}s")

sam = MetaSam()
t = time.time()
tracks = sam.track(chunk, list(CONCEPTS), a.start)
print(f"SAM wall {time.time() - t:.1f}s (3 concepts in parallel)")
for c in CONCEPTS:
    mine = [x for x in tracks if x.label == c]
    print(f"  {c}: {len(mine)} objects, {sum(len(x.boxes) for x in mine)} boxes, "
          f"conf {min((x.conf for x in mine), default=0):.2f}-{max((x.conf for x in mine), default=0):.2f}")

kf = [(a.start + t0, p) for t0, p in chunker.keyframes(chunk, 0, a.length, a.workdir / "kf", n=4)]
t = time.time()
evs = Vision("spark").events(kf, tracks, a.start, a.start + a.length, clip=a.clip.stem)
print(f"Spark wall {time.time() - t:.1f}s, {len(evs)} events")
for e in evs:
    print(f"  {e.t_start:.1f}-{e.t_end:.1f} {e.action} {e.obj} -> {e.target} {e.technique} conf {e.conf} sam {e.sam_conf} tracks {e.track_ids}")
