"""SAM 3.1 perception: per-chunk tracks, stitched across chunks by us.

The hosted API takes a whole clip per request, one noun phrase per request, and object ids do not
carry across requests, so each chunk gets fresh ids and `link_tracks` re-links them to the previous
chunk by box overlap. `NullSam` lets the rest of the pipeline run without a key.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

MAX_VIDEO_BYTES = 50 * 1024 * 1024       # API limit (file_above_max_size)
CONCEPTS = ("hand", "tomato", "plate")   # not "bowl": finds nothing on a brown plate (#14)


@dataclass
class Track:
    id: str                      # chunk-local id as returned by SAM
    label: str                   # noun phrase it was prompted with: hand, tomato, knife ...
    boxes: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    # (t, x1, y1, x2, y2) normalized 0..1, t in clip time (s)
    conf: float = 1.0            # mean SAM detection confidence


class Sam(Protocol):
    def track(self, chunk: pathlib.Path, prompts: list[str], t0: float) -> list[Track]: ...


class NullSam:
    def track(self, chunk, prompts, t0):
        return []


def parse_stream(text: str, label: str, t0: float = 0.0, fps: float = 30.0) -> list[Track]:
    """Parse concatenated deltas: `<Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|><|mask;..|>,...` per frame.

    Boxes are in source pixels (normalized here with w, h); ids are stable only within one response.
    """
    tracks: dict[str, Track] = {}
    confs: dict[str, list[float]] = {}
    parts = re.split(r"<(\d+)f>", text)          # ['', frame, objects, frame, objects, ...]
    for frame, objs in zip(parts[1::2], parts[2::2]):
        t = t0 + int(frame) / fps
        for oid, fields in re.findall(r"([^\s,<>|]+)<\|box;([^|]*)\|>", objs):
            f = dict(kv.split("=", 1) for kv in fields.split(";") if "=" in kv)
            w, h = float(f["w"]), float(f["h"])
            tr = tracks.setdefault(oid, Track(f"{label}:{oid}", label))
            tr.boxes.append((t, float(f["x1"]) / w, float(f["y1"]) / h, float(f["x2"]) / w, float(f["y2"]) / h))
            confs.setdefault(oid, []).append(float(f.get("c", 1.0)))
    for oid, tr in tracks.items():
        tr.conf = sum(confs[oid]) / len(confs[oid])
    return list(tracks.values())


class MetaSam:
    """Hosted SAM 3.1 over urllib (the OpenAI SDK bypasses the sandbox proxy). Keys come from env only.

    shortcut: `fps` is assumed (the stream carries frame numbers only), set it to the chunk's real
    frame rate; ffprobe it if sources vary.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str = "sam-3.1",
                 fps: float = 30.0, timeout: float = 120):
        self.api_key = api_key or os.environ.get("META_API_KEY", "")
        self.base_url = (base_url or os.environ.get("META_API_BASE", "https://api.meta.ai/v1")).rstrip("/")
        self.model, self.fps, self.timeout, self.backoff = model, fps, timeout, 1.5
        self.frames = 0                   # billed frames so far (usage.video_frames_processed), all requests
        self._lock = threading.Lock()

    def _request(self, data_url: str, concept: str) -> str:
        body = json.dumps({
            "model": self.model, "stream": True,
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
            "input": [{"role": "user", "content": [
                {"type": "input_video", "video_url": data_url},
                {"type": "input_text", "text": concept}]}]}).encode()
        req = urllib.request.Request(f"{self.base_url}/responses", body, {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        out = []
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                ev = json.loads(line[5:])
                if ev.get("type") == "response.output_text.delta":
                    out.append(ev["delta"])
                elif ev.get("type") == "response.completed":
                    with self._lock:
                        self.frames += ev["response"].get("usage", {}).get("video_frames_processed", 0)
                elif ev.get("type") in ("error", "response.failed", "response.error"):
                    raise RuntimeError(f"SAM request failed for {concept!r}: {json.dumps(ev)[:300]}")
        return "".join(out)       # empty stream = no match, not an error

    def _retry(self, data_url: str, concept: str) -> str:
        try:
            return self._request(data_url, concept)
        except Exception:                      # 503s are common: one retry after a backoff
            time.sleep(self.backoff)
            return self._request(data_url, concept)

    def track(self, chunk, prompts, t0, errors: list[str] | None = None):
        """Tracks for every concept that answered; a concept that fails twice adds a line to `errors`
        (an empty stream is "not found", a failure is not the same thing)."""
        chunk = pathlib.Path(chunk)
        if chunk.stat().st_size > MAX_VIDEO_BYTES:
            raise ValueError(f"{chunk} is over the 50 MiB API limit")
        data_url = "data:video/mp4;base64," + base64.b64encode(chunk.read_bytes()).decode()

        def one(c):
            try:
                return parse_stream(self._retry(data_url, c), c, t0, self.fps)
            except Exception as x:       # a bad request or a malformed frame skips this concept only
                (errors if errors is not None else []).append(f"{c}: {type(x).__name__}: {x}"[:200])
                return []

        with ThreadPoolExecutor(max(1, len(prompts))) as ex:      # one request per concept, in parallel
            return [t for tracks in ex.map(one, prompts) for t in tracks]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _overlap_iou(p: Track, n: Track, tol: float) -> float:
    """Mean IoU over the timestamps where both tracks have a box (within tol seconds); 0 if none."""
    scores = []
    for b in n.boxes:
        q = min(p.boxes, key=lambda x: abs(x[0] - b[0]))
        if abs(q[0] - b[0]) <= tol:
            scores.append(iou(q[1:], b[1:]))
    return sum(scores) / len(scores) if scores else 0.0


def link_tracks(prev: list[Track], new: list[Track], min_iou: float = 0.3, tol: float = 0.05) -> dict[str, str]:
    """Map each new track id to a persistent id: the matching previous track's id, else its own.

    Chunks overlap, so both tracks have boxes at the same timestamps inside the overlap window;
    compare those (a fast hand moves too far between the previous chunk's last box and the new
    chunk's first), same label only, greedily by best IoU.
    """
    pairs = []
    for n in new:
        for p in prev:
            if p.label == n.label and p.boxes and n.boxes:
                pairs.append((_overlap_iou(p, n, tol), p.id, n.id))
    mapping, used = {}, set()
    for score, pid, nid in sorted(pairs, reverse=True):
        if score >= min_iou and pid not in used and nid not in mapping:
            mapping[nid] = pid
            used.add(pid)
    for n in new:
        mapping.setdefault(n.id, n.id)
    return mapping
