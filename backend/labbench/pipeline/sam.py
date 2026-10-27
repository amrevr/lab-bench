"""SAM 3.1 perception: per-chunk tracks, stitched across chunks by us.

The hosted API takes a whole clip per request and track ids do not carry across requests, so each
chunk gets fresh ids and `link_tracks` re-links them to the previous chunk by box overlap.
`MetaSam` is a stub until the API key lands (~Oct 29); the request shape must be checked against the
Meta Model API docs then. `NullSam` lets the rest of the pipeline run without it.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Track:
    id: str                      # chunk-local id as returned by SAM
    label: str                   # noun phrase it was prompted with: hand, tomato, knife ...
    boxes: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    # (t, x1, y1, x2, y2) normalized 0..1


class Sam(Protocol):
    def track(self, chunk: pathlib.Path, prompts: list[str], t0: float) -> list[Track]: ...


class NullSam:
    def track(self, chunk, prompts, t0):
        return []


class MetaSam:
    def __init__(self, api_key: str, base_url: str, model: str = "sam-3.1"):
        self.api_key, self.base_url, self.model = api_key, base_url, model

    def track(self, chunk, prompts, t0):
        raise NotImplementedError("Wire to Meta Model API when the key arrives (issue #5/#14)")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tracks(prev: list[Track], new: list[Track], min_iou: float = 0.3) -> dict[str, str]:
    """Map each new track id to a persistent id: the matching previous track's id, else its own.

    Compares the last box of a previous track to the first box of a new one with the same label,
    greedily by best IoU. Chunks overlap, so the boxes refer to nearly the same moment.
    """
    pairs = []
    for n in new:
        if not n.boxes:
            continue
        for p in prev:
            if p.label == n.label and p.boxes:
                pairs.append((iou(p.boxes[-1][1:], n.boxes[0][1:]), p.id, n.id))
    mapping, used = {}, set()
    for score, pid, nid in sorted(pairs, reverse=True):
        if score >= min_iou and pid not in used and nid not in mapping:
            mapping[nid] = pid
            used.add(pid)
    for n in new:
        mapping.setdefault(n.id, n.id)
    return mapping
