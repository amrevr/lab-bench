"""Deterministic step alignment and deviation judgment.

This is the local judge. Muse Wonder (wonder.py) re-words the findings into coaching text and
may overrule low-confidence ones; if no API key is present this module alone runs the demo.
"""
from __future__ import annotations

from ..schema import Event, Protocol, Step, Verdict, VerdictKind, mode_for


def _same(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and a.lower().strip() == b.lower().strip()


class Aligner:
    """Streaming: feed events in time order, collect verdicts.

    A step the novice attempts (even wrongly) counts as attempted, so one mistake yields one
    verdict instead of a cascade. A gap in the sequence is only called "skipped" once two later
    steps have happened without it; if the missing step shows up after a later one, it is an
    order error on that later step. Whatever is still missing at the end of the clip is skipped.
    """

    SKIP_AFTER = 2

    def __init__(self, protocol: Protocol):
        self.steps = protocol.steps
        self.pos = {s.id: i for i, s in enumerate(protocol.steps)}   # step ids need not be sorted
        self.done: dict[int, float] = {}       # step id -> time attempted
        self.verdicts: list[Verdict] = []
        self._reported: set[tuple[VerdictKind, int]] = set()

    def _find(self, ev: Event) -> tuple[Step | None, str]:
        """Return (step, how) where how is exact | technique | item | none."""
        for s in self.steps:
            if s.action == ev.action and _same(s.obj, ev.obj) and (
                s.target is None or ev.target is None or _same(s.target, ev.target)
            ) and s.id not in self.done:
                if s.technique and ev.technique and not _same(s.technique, ev.technique):
                    return s, "technique"
                return s, "exact"
        # Same action, same destination, different object -> wrong item for the next such step.
        for s in self.steps:
            if s.action == ev.action and not _same(s.obj, ev.obj) and s.id not in self.done and (
                s.target is None or _same(s.target, ev.target)
            ) and not any(o.action == ev.action and _same(o.obj, ev.obj) for o in self.steps):
                return s, "item"
        return None, "none"

    def _emit(self, ev_t: float, kind: VerdictKind, sid: int, msg: str, conf: float,
              refs: list[float]) -> Verdict | None:
        if (kind, sid) in self._reported:
            return None
        v = Verdict(t=ev_t, step_id=sid, kind=kind, conf=round(conf, 2),
                    mode=mode_for(kind, conf), message=msg, frame_refs=refs)
        if v.mode == "ok":
            return None
        self._reported.add((kind, sid))
        self.verdicts.append(v)
        return v

    def feed(self, ev: Event) -> list[Verdict]:
        step, how = self._find(ev)
        if step is None:
            return []
        out: list[Verdict | None] = []
        t, refs = ev.t_end, ev.frame_refs
        if how == "technique":
            out.append(self._emit(t, VerdictKind.WRONG_TECHNIQUE, step.id,
                f"Step {step.id}: the expert used '{step.technique}' on the {step.obj}; "
                f"this looks like '{ev.technique}'.", ev.conf, refs))
        elif how == "item":
            out.append(self._emit(t, VerdictKind.WRONG_ITEM, step.id,
                f"Step {step.id} calls for {step.obj}, but this looks like {ev.obj}.", ev.conf, refs))
        if ev.conf >= 0.5:
            later = sorted((d for d in self.done if self.pos[d] > self.pos[step.id]), key=self.pos.get)
            if later:
                first = next(s for s in self.steps if s.id == later[0])
                out.append(self._emit(t, VerdictKind.WRONG_ORDER, first.id,
                    f"'{first.text}' (step {first.id}) came before step {step.id} "
                    f"('{step.text}'); the expert did them the other way round.", ev.conf, refs))
            self.done.setdefault(step.id, t)
            for m in self.steps:
                if self.pos[m.id] < self.pos[step.id] and m.id not in self.done and not m.optional:
                    seen_after = sum(1 for d in self.done if self.pos[d] > self.pos[m.id])
                    if seen_after >= self.SKIP_AFTER:
                        out.append(self._emit(t, VerdictKind.SKIPPED, m.id,
                            f"Step {m.id} ('{m.text}') looks skipped.", ev.conf, refs))
        return [v for v in out if v]

    def finish(self, t_end: float) -> list[Verdict]:
        """End of clip: anything still undone and not optional was skipped."""
        out = []
        for s in self.steps:
            if s.id not in self.done and not s.optional:
                out.append(self._emit(t_end, VerdictKind.SKIPPED, s.id,
                    f"Step {s.id} ('{s.text}') was never done.", 0.8, []))
        return [v for v in out if v]
