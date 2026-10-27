"""Contract between perception (Glimmer/rules) and reasoning (alignment + Wonder).

Event    : what happened in a chunk, from perception.
Step     : one protocol step, learned from the expert run + written steps.
Verdict  : the coach's judgment about a step for the novice.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Confidence thresholds on a *deviation*. Above FLAG we say it plainly, between ASK and
# FLAG we ask the novice to confirm instead of alarming, below ASK we stay silent.
FLAG_THRESHOLD = 0.75
ASK_THRESHOLD = 0.40


class Event(BaseModel):
    t_start: float
    t_end: float
    action: str                      # pick, cut, place, pour, sprinkle, mix, ...
    obj: str                         # ingredient or tool acted on
    target: Optional[str] = None     # destination (plate, bowl, ...)
    technique: Optional[str] = None  # slice, tear, saw, ...
    conf: float = Field(ge=0, le=1, default=0.8)
    track_ids: list[str] = []
    frame_refs: list[float] = []     # timestamps (s) of source keyframes


class Step(BaseModel):
    id: int                          # dataset step id; execution order is list position
    text: str
    action: str
    obj: str
    target: Optional[str] = None
    technique: Optional[str] = None
    optional: bool = False


class Protocol(BaseModel):
    name: str
    source_clip: str
    steps: list[Step]


class VerdictKind(str, Enum):
    ON_TRACK = "on_track"
    SKIPPED = "skipped_step"
    WRONG_ORDER = "wrong_order"
    WRONG_TECHNIQUE = "wrong_technique"
    WRONG_ITEM = "wrong_item"


class Verdict(BaseModel):
    t: float                         # when the coach speaks (clip time, s)
    step_id: int
    kind: VerdictKind
    conf: float
    mode: str                        # "flag" | "ask" | "ok"
    message: str
    frame_refs: list[float] = []


def mode_for(kind: VerdictKind, conf: float) -> str:
    if kind == VerdictKind.ON_TRACK:
        return "ok"
    if conf >= FLAG_THRESHOLD:
        return "flag"
    if conf >= ASK_THRESHOLD:
        return "ask"
    return "ok"
