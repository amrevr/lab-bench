"""Deterministic step alignment and deviation judgment.

This is the local judge. Muse Wonder (wonder.py) re-words the findings into coaching text and
may overrule low-confidence ones; if no API key is present this module alone runs the demo.
"""
from __future__ import annotations

from ..schema import Event, Protocol, Step, Verdict, VerdictKind, mode_for


def _same(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and a.lower().strip() == b.lower().strip()


class Aligner:
    """Streaming: feed events in time order, collect verdicts."""

    def __init__(self, protocol: Protocol):
        self.steps = protocol.steps
        self.done: dict[int, float] = {}       # step id -> time completed
        self.verdicts: list[Verdict] = []
        self._announced_skip: set[int] = set()

    def _find(self, ev: Event) -> tuple[Step | None, str]:
        """Return (step, how) where how is exact | technique | item | none."""
        for s in self.steps:
            if s.action == ev.action and _same(s.obj, ev.obj):
                if s.technique and ev.technique and not _same(s.technique, ev.technique):
                    return s, "technique"
                return s, "exact"
        # Same action on the same destination but a different object -> wrong item.
        for s in self.steps:
            if s.action == ev.action and not _same(s.obj, ev.obj) and (
                s.target is None or _same(s.target, ev.target)
            ) and s.id not in self.done:
                return s, "item"
        return None, "none"

    def feed(self, ev: Event) -> list[Verdict]:
        out: list[Verdict] = []
        step, how = self._find(ev)
        if step is None:
            return out
        t = ev.t_end

        def emit(kind: VerdictKind, sid: int, msg: str, conf: float) -> None:
            v = Verdict(t=t, step_id=sid, kind=kind, conf=round(conf, 2),
                        mode=mode_for(kind, conf), message=msg, frame_refs=ev.frame_refs)
            if v.mode != "ok":
                out.append(v)

        if how == "technique":
            emit(VerdictKind.WRONG_TECHNIQUE, step.id,
                 f"Step {step.id}: expert used '{step.technique}' on the {step.obj}, "
                 f"this looks like '{ev.technique}'.", ev.conf)
        elif how == "item":
            emit(VerdictKind.WRONG_ITEM, step.id,
                 f"Step {step.id} calls for {step.obj}, but this looks like {ev.obj}.", ev.conf)
        else:
            earlier = [s for s in self.steps if s.id < step.id and s.id not in self.done and not s.optional]
            if step.id in self.done:
                pass  # repeat of a finished step is not a deviation
            elif earlier:
                missing = earlier[0]
                if len(earlier) == 1 and ev.conf >= 0.5:
                    # Ambiguous between skipped and out-of-order until the missing step shows up
                    # later; we report skipped now and upgrade to wrong_order if it does.
                    emit(VerdictKind.WRONG_ORDER, step.id,
                         f"Step {missing.id} ('{missing.text}') hasn't happened yet, but you've "
                         f"started step {step.id}. Should that go first?", ev.conf * 0.9)
                else:
                    for m in earlier:
                        if m.id not in self._announced_skip:
                            self._announced_skip.add(m.id)
                            emit(VerdictKind.SKIPPED, m.id,
                                 f"Step {m.id} ('{m.text}') looks skipped.", ev.conf)
            if how == "exact" and ev.conf >= 0.5:
                self.done.setdefault(step.id, t)
        self.verdicts.extend(out)
        return out

    def finish(self, t_end: float) -> list[Verdict]:
        """End of clip: anything still undone and not optional was skipped."""
        out = []
        for s in self.steps:
            if s.id not in self.done and not s.optional and s.id not in self._announced_skip:
                self._announced_skip.add(s.id)
                out.append(Verdict(t=t_end, step_id=s.id, kind=VerdictKind.SKIPPED, conf=0.8,
                                   mode="flag", message=f"Step {s.id} ('{s.text}') was never done."))
        self.verdicts.extend(out)
        return out
