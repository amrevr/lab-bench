"""Deterministic step alignment and deviation judgment.

This is the local judge. Muse Wonder (wonder.py) re-words the findings into coaching text and
may overrule low-confidence ones; if no API key is present this module alone runs the demo.
"""
from __future__ import annotations

from ..schema import THRESHOLDS, Event, Evidence, Protocol, Step, Verdict, VerdictKind, mode_for


SYNONYMS = {"tray": "plate", "platter": "plate", "dish": "plate"}   # a cutting board is not a serving plate
ORDER_GAP = 3.0    # s (one chunk): events closer than this in time are not ordered, window timing is that coarse
STRONG = 0.8       # an event this confident backs a verdict alone; weaker ones need a second window (false asks maxed at 0.72)
CORROBORATE_GAP = 1.0   # s: a same action/object event in another window this close (chunks overlap 0.5 s) corroborates


def _win(e: Event) -> tuple:
    return tuple(e.chunk) if e.chunk else (e.t_start,)


def _norm(x: str) -> str:
    x = x.lower().strip()
    return SYNONYMS.get(x, x)


def _same(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and _norm(a) == _norm(b)


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
        self.names = {_norm(n) for s in self.steps for n in (s.obj, s.target) if n}
        self.pos = {s.id: i for i, s in enumerate(protocol.steps)}   # step ids need not be sorted
        self.done: dict[int, tuple[int, Event]] = {}   # step id -> (event idx, event) that attempted it
        self.verdicts: list[Verdict] = []
        self._reported: set[tuple[VerdictKind, int]] = set()
        self.seen: list[Event] = []
        self.parked: dict[tuple[VerdictKind, int], tuple] = {}   # verdicts waiting for weak evidence to be corroborated

    def _find(self, ev: Event) -> tuple[Step | None, str]:
        """Return (step, how) where how is exact | technique | item | none.

        Action + object identify the step. The target is soft: any destination the recipe itself names
        (a step's target or object, synonyms folded) is accepted and only breaks ties between steps;
        a place the recipe never mentions (fridge, cutting board) means this is some other action."""
        cands = [s for s in self.steps if (ev.target is None or s.target is None or _norm(ev.target) in self.names)
                 and s.action == ev.action and _same(s.obj, ev.obj) and s.id not in self.done]
        if cands:
            s = next((c for c in cands if _same(c.target, ev.target)), cands[0])
            if s.technique and ev.technique and not _same(s.technique, ev.technique):
                return s, "technique"
            return s, "exact"
        # An attempted step is judged again for technique: the first event is often the setup, the method shows later.
        for s in self.steps:
            if s.id in self.done and s.technique and ev.technique and s.action == ev.action and _same(s.obj, ev.obj) \
                    and not _same(s.technique, ev.technique):
                return s, "technique"
        # Same action, same destination, different object -> wrong item for the next such step.
        for s in self.steps:
            if s.action == ev.action and not _same(s.obj, ev.obj) and s.id not in self.done and (
                s.target is None or _same(s.target, ev.target)
            ) and not any(o.action == ev.action and _same(o.obj, ev.obj) for o in self.steps):
                return s, "item"
        return None, "none"

    def _backed(self, kind: VerdictKind, ev: Event) -> bool:
        """Strong, or seen again (same action/object, compatible target, same technique for a technique verdict) in another window."""
        return ev.conf >= STRONG or any(
            o is not ev and _win(o) != _win(ev) and o.action == ev.action and _same(o.obj, ev.obj)
            and (o.target is None or ev.target is None or _same(o.target, ev.target))
            and (kind != VerdictKind.WRONG_TECHNIQUE or _same(o.technique, ev.technique))
            and max(o.t_start - ev.t_end, ev.t_start - o.t_end) <= CORROBORATE_GAP for o in self.seen)

    def _emit(self, t: float, kind: VerdictKind, sid: int, msg: str, contrib: list[tuple[int, Event]],
              conf: float | None = None, gated: bool = True) -> Verdict | None:
        """contrib: (event_idx, event) pairs behind the verdict; conf defaults to their minimum.

        A verdict resting on a weak event nobody corroborates yet is parked, not spoken (_release)."""
        if (kind, sid) in self._reported:
            return None
        if gated and not all(self._backed(kind, e) for _, e in contrib):
            self.parked[(kind, sid)] = (kind, sid, msg, contrib, conf)
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

    def _release(self, t: float) -> list[Verdict]:
        """Speak parked verdicts whose evidence is now corroborated; withdraw skips of steps that have since happened."""
        out = []
        for key, (kind, sid, msg, contrib, conf) in list(self.parked.items()):
            if kind == VerdictKind.SKIPPED and sid in self.done:
                del self.parked[key]
            elif all(self._backed(kind, e) for _, e in contrib):
                del self.parked[key]
                out.append(self._emit(t, kind, sid, msg, contrib, conf, gated=False))
        return [v for v in out if v]

    def feed(self, ev: Event) -> list[Verdict]:
        """Events get their index in feed order; Verdict evidence refers to it."""
        self.seen.append(ev)
        return self._feed(ev) + self._release(ev.t_end)

    def _feed(self, ev: Event) -> list[Verdict]:
        cur = (self._n, ev)
        self._n += 1
        step, how = self._find(ev)
        if step is None:
            return []
        out: list[Verdict | None] = []
        t = ev.t_end
        if how == "technique" and step.id in self.done:
            return [v for v in [self._emit(t, VerdictKind.WRONG_TECHNIQUE, step.id,
                f"Step {step.id}: the expert used '{step.technique}' on the {step.obj}; "
                f"this looks like '{ev.technique}'.", [cur])] if v]
        if how == "technique":
            out.append(self._emit(t, VerdictKind.WRONG_TECHNIQUE, step.id,
                f"Step {step.id}: the expert used '{step.technique}' on the {step.obj}; "
                f"this looks like '{ev.technique}'.", [cur]))
        elif how == "item":
            out.append(self._emit(t, VerdictKind.WRONG_ITEM, step.id,
                f"Step {step.id} calls for {step.obj}, but this looks like {ev.obj}.", [cur]))
        if ev.conf >= 0.5:
            later = sorted((d for d in self.done if self.pos[d] > self.pos[step.id]
                            and not (step.group and step.group == self.steps[self.pos[d]].group)
                            and ev.t_start - self.done[d][1].t_end >= ORDER_GAP), key=self.pos.get)
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
                    f"Step {s.id} ('{s.label}') was never done.", seen[-1:], conf, gated=False))
        return [v for v in out if v]
