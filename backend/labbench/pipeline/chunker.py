"""Split a clip into short H.264 chunks (faststart) and pull keyframes. Needs ffmpeg (or set FFMPEG)."""
from __future__ import annotations

import os
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path: pathlib.Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def windows(total: float, length: float = 3.0, overlap: float = 0.5):
    """Yield (start, end) windows covering [0, total)."""
    t = 0.0
    while t < total - 0.25:
        yield t, min(t + length, total)
        t += length - overlap


def process_chunks(total: float, fn, workers: int = 3, length: float = 3.0, overlap: float = 0.5) -> list:
    """Run fn(start, end) on every window concurrently; results come back in window order.

    One SAM stream is slower than real time (~5.5 s for a 2 s chunk), so chunks must overlap in
    flight or the pipeline falls behind. Workers pipeline whole chunks; MetaSam parallelizes concepts.
    """
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, a, b) for a, b in windows(total, length, overlap)]
        return [f.result() for f in futs]


def cut_chunk(src: pathlib.Path, start: float, end: float, dst: pathlib.Path, height: int = 480) -> pathlib.Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([FFMPEG, "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(src), "-an",
          "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
          "-movflags", "+faststart", str(dst)])
    return dst


def keyframes(src: pathlib.Path, start: float, end: float, out_dir: pathlib.Path, n: int = 4,
              width: int = 640) -> list[tuple[float, pathlib.Path]]:
    """n evenly spaced JPEG keyframes inside [start, end). Returns (timestamp, path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        p = out_dir / f"{t:08.2f}.jpg"
        _run([FFMPEG, "-y", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
              "-vf", f"scale={width}:-2", "-q:v", "4", str(p)])
        frames.append((t, p))
    return frames
