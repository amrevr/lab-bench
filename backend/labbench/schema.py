"""Contract between perception (Glimmer/rules) and reasoning (alignment + Wonder).

Event    : what happened in a chunk, from perception.
Step     : one protocol step, learned from the expert run + written steps.
Verdict  : the coach's judgment about a step for the novice.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

# Confidence thresholds on a *deviation*. Above FLAG we say it plainly, between ASK and
# FLAG we ask the novice to confirm instead of alarming, below ASK we stay silent.
# Technique and wrong-item verdicts need a higher bar (issue #9). Tuned on 4 novice clips only.
FLAG_THRESHOLD = 0.75
ASK_THRESHOLD = 0.40
STRICT = (0.85, 0.50)


class Event(BaseModel):
    t_start: float
    t_end: float
    action: str                      # pick, cut, place, pour, sprinkle, mix, ...
    obj: str                         # ingredient or tool acted on
    target: Optional[str] = None     # destination (plate, bowl, ...)
    technique: Optional[str] = None  # slice, tear, saw, ...
    issue: Optional[str] = None      # a visible mishap in how it was done: spill, drop, overflow, pile-up
    conf: float = Field(ge=0, le=1, default=0.8)       # the vision model's confidence
    sam_conf: Optional[float] = None                   # SAM detection confidence, kept separate
    track_ids: list[str] = []        # cross-chunk track ids (see sam.link_tracks)
    frame_refs: list[float] = []     # timestamps (s) of source keyframes
    clip: Optional[str] = None
    chunk: Optional[list[float]] = None                # [start, end] of the chunk it came from


class Step(BaseModel):
    id: int                          # dataset step id; execution order is list position
    label: str                       # short step name (was `text`)
    recipe_text: Optional[str] = None
    narration: Optional[str] = None  # null until the expert narration source is decided (#15)
    qty: Optional[str] = None        # "1/2 in", "1/4 tsp"; not visible to the pipeline, shown only
    source_frames: list[float] = []  # expert-clip timestamps (s)
    action: str
    aliases: list[str] = []          # other verbs a viewer may use for this action (fill: pour, add)
    seen_as: list[dict] = []         # other ways it looks on camera: {action, target} (microwave: place -> microwave)
    obj: str
    target: Optional[str] = None
    technique: Optional[str] = None
    optional: bool = False
    visible: bool = True             # False: never recognized in the expert run, so never asked about as skipped
    group: Optional[str] = None      # steps sharing a group may be done in any order (e.g. seasonings)
    after: list[int] = []            # step ids that must have been attempted first; an event earlier than that is not this step

    @model_validator(mode="before")
    @classmethod
    def _old_text(cls, d):
        if isinstance(d, dict) and "text" in d and "label" not in d:
            d = {**d, "label": d["text"]}
        return d


class Protocol(BaseModel):
    name: str
    source_clip: str
    steps: list[Step]
    partial_order: bool = False      # True: `after` is the full prerequisite graph (generated protocols); order = reachability


class VerdictKind(str, Enum):
    ON_TRACK = "on_track"
    SKIPPED = "skipped_step"
    WRONG_ORDER = "wrong_order"
    WRONG_TECHNIQUE = "wrong_technique"
    WRONG_ITEM = "wrong_item"


class Evidence(BaseModel):
    clip: Optional[str] = None
    t_start: float
    t_end: float
    frame_t: list[float] = []        # keyframe timestamps; the overlay shows these images
    event_idx: Optional[int] = None  # index into the session's events


class Verdict(BaseModel):
    t: float                         # when the coach speaks (clip time, s)
    source: Optional[str] = None     # "issue": raised from a reported mishap, not a technique word
    step_id: int
    kind: VerdictKind
    conf: float
    mode: str                        # "flag" | "ask" | "ok"
    message: str
    evidence: list[Evidence] = []
    coach: Optional[str] = None      # Muse Wonder wording; arrives after the verdict, never gates it
    closed_t: Optional[float] = None   # replay: video time (s) the chunk behind this verdict closed
    emitted_at: Optional[float] = None  # replay: wall-clock (epoch s) the aligner emitted it
    latency_s: Optional[float] = None   # replay: emitted_at minus the wall time the chunk was fed

    @model_validator(mode="before")
    @classmethod
    def _old_frame_refs(cls, d):
        if isinstance(d, dict) and "frame_refs" in d:
            d = dict(d)
            refs = d.pop("frame_refs")
            if refs and "evidence" not in d:
                d["evidence"] = [{"t_start": min(refs), "t_end": max(refs), "frame_t": refs}]
        return d


# (flag, ask) per kind; kinds not listed use the defaults.
THRESHOLDS = {VerdictKind.WRONG_TECHNIQUE: STRICT, VerdictKind.WRONG_ITEM: STRICT}


def mode_for(kind: VerdictKind, conf: float, thresholds: dict = THRESHOLDS) -> str:
    if kind == VerdictKind.ON_TRACK:
        return "ok"
    flag, ask = thresholds.get(kind, (FLAG_THRESHOLD, ASK_THRESHOLD))
    if conf >= flag:
        return "flag"
    if conf >= ask:
        return "ask"
    return "ok"
