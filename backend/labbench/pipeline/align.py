"""Deterministic step alignment and deviation judgment.

This is the local judge. Muse Wonder (wonder.py) re-words the findings into coaching text and
may overrule low-confidence ones; if no API key is present this module alone runs the demo.
"""
from __future__ import annotations

from ..schema import THRESHOLDS, Event, Evidence, Protocol, Step, Verdict, VerdictKind, mode_for


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

    def __init__(self, protocol: Protocol, thresholds: dict = THRESHOLDS):
        self.thresholds, self._n = thresholds, 0
        self.steps = protocol.steps
        self.pos = {s.id: i for i, s in enumerate(protocol.steps)}   # step ids need not be sorted
        self.done: dict[int, tuple[int, Event]] = {}   # step id -> (event idx, event) that attempted it
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

    def _emit(self, t: float, kind: VerdictKind, sid: int, msg: str, contrib: list[tuple[int, Event]],
              conf: float | None = None) -> Verdict | None:
        """contrib: (event_idx, event) pairs behind the verdict; conf defaults to their minimum."""
        if (kind, sid) in self._reported:
            return None
        conf = min(e.conf for _, e in contrib) if conf is None else conf
        ev = [Evidence(clip=e.clip, t_start=e.t_start, t_end=e.t_end, frame_t=e.frame_refs, event_idx=i)
              for i, e in contrib]
        v = Verdict(t=t, step_id=sid, kind=kind, conf=round(conf, 2),
                    mode=mode_for(kind, conf, self.thresholds), message=msg, evidence=ev)
        if v.mode == "ok" and kind != VerdictKind.ON_TRACK:
            return None
        self._reported.add((kind, sid))
        self.verdicts.append(v)
        return v

    def feed(self, ev: Event) -> list[Verdict]:
        """Events get their index in feed order; Verdict evidence refers to it."""
        cur = (self._n, ev)
        self._n += 1
        step, how = self._find(ev)
        if step is None:
            return []
        out: list[Verdict | None] = []
        t = ev.t_end
        if how == "technique":
            out.append(self._emit(t, VerdictKind.WRONG_TECHNIQUE, step.id,
                f"Step {step.id}: the expert used '{step.technique}' on the {step.obj}; "
                f"this looks like '{ev.technique}'.", [cur]))
        elif how == "item":
            out.append(self._emit(t, VerdictKind.WRONG_ITEM, step.id,
                f"Step {step.id} calls for {step.obj}, but this looks like {ev.obj}.", [cur]))
        if ev.conf >= 0.5:
            later = sorted((d for d in self.done if self.pos[d] > self.pos[step.id]), key=self.pos.get)
            if later:
                first = next(s for s in self.steps if s.id == later[0])
                out.append(self._emit(t, VerdictKind.WRONG_ORDER, first.id,
                    f"'{first.label}' (step {first.id}) came before step {step.id} "
                    f"('{step.label}'); the expert did them the other way round.", [self.done[first.id], cur]))
            fresh = step.id not in self.done
            self.done.setdefault(step.id, cur)
            for m in self.steps:
                if self.pos[m.id] < self.pos[step.id] and m.id not in self.done and not m.optional:
                    after = [c for d, c in self.done.items() if self.pos[d] > self.pos[m.id]]
                    if len(after) >= self.SKIP_AFTER:
                        out.append(self._emit(t, VerdictKind.SKIPPED, m.id,
                            f"Step {m.id} ('{m.label}') looks skipped.", after))
            if how == "exact" and fresh and not any(out):
                out.append(self._emit(t, VerdictKind.ON_TRACK, step.id, f"Step {step.id} ('{step.label}') done.", [cur]))
        return [v for v in out if v]

    def finish(self, t_end: float) -> list[Verdict]:
        """End of clip: anything still undone and not optional was skipped.

        Confidence is the weakest event seen in the clip: any of them could have been the missing step.
        """
        seen = list(self.done.values())
        conf = min((e.conf for _, e in seen), default=0.5)
        out = []
        for s in self.steps:
            if s.id not in self.done and not s.optional:
                out.append(self._emit(t_end, VerdictKind.SKIPPED, s.id,
                    f"Step {s.id} ('{s.label}') was never done.", seen[-1:], conf))
        return [v for v in out if v]
