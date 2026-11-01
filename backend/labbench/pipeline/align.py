"""Deterministic step alignment and deviation judgment.

This is the local judge. Muse Wonder (wonder.py) re-words the findings into coaching text and
may overrule low-confidence ones; if no API key is present this module alone runs the demo.
"""
from __future__ import annotations

from ..schema import THRESHOLDS, Event, Evidence, Protocol, Step, Verdict, VerdictKind, mode_for


SYNONYMS = {"tray": "plate", "platter": "plate", "dish": "plate"}   # a cutting board is not a serving plate
ORDER_GAP = 3.0    # s (one chunk): events closer than this in time are not ordered, window timing is that coarse
STRONG = 0.6       # an event this confident backs a verdict alone; weaker ones need a second window. 0.6 on all 17 recipe-12 clips (Glimmer): same catches and false alarms as 0.8, warnings up to 5 min earlier; 0.5 adds a false alarm
MIN_CONF = 0.5     # events below this confidence are ignored for ordering/attempts
ITEM_NEEDS_REPEAT = True   # an unknown-noun (wrong item) event must also be corroborated
CORROBORATE_GAP = 1.0   # s: a same action/object event in another window this close (chunks overlap 0.5 s) corroborates
ACTION_SYNONYMS = {"season": "sprinkle", "garnish": "sprinkle", "drizzle": "pour"}
SPRINKLE_ACTIONS = {"sprinkle", "pour"}   # the obj of these is the ingredient; the vision model often names what it lands on
SURFACES = {"plate", "board", "cutting board", "tray", "platter", "dish", "bowl", "table", "counter"}


def _win(e: Event) -> tuple:
    return tuple(e.chunk) if e.chunk else (e.t_start,)


def _norm(x: str) -> str:
    x = x.lower().strip()
    return SYNONYMS.get(x, x)


def _words(x: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in _norm(x).split()}


def _same(a: str | None, b: str | None) -> bool:
    """Same thing: equal, or one name's words contain the other's ("tomato slices" ~ "tomato", "black pepper" ~
    "pepper"), so generated protocols and the vision model need not word things identically.
    shortcut: a modifier also matches ("olives" ~ "olive oil"); fine while a recipe has no two such items, else
    compare head nouns."""
    if not a or not b:
        return False
    if _norm(a) == _norm(b):
        return True
    wa, wb = _words(a), _words(b)
    return wa <= wb or wb <= wa


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
        self.ingredients = {(s.action, _norm(s.obj)) for s in self.steps}
        self.pos = {s.id: i for i, s in enumerate(protocol.steps)}   # step ids need not be sorted
        self.partial = protocol.partial_order
        self.anc: dict[int, set[int]] = {}         # step -> every step that must come before it (transitive `after`)
        for st in self.steps:
            todo, seen = list(st.after), set()
            while todo:
                a = todo.pop()
                if a not in seen and a in self.pos:
                    seen.add(a)
                    todo += self.steps[self.pos[a]].after
            self.anc[st.id] = seen
        self.done: dict[int, tuple[int, Event]] = {}   # step id -> (event idx, event) that attempted it
        self.verdicts: list[Verdict] = []
        self._reported: set[tuple[VerdictKind, int]] = set()
        self.seen: list[Event] = []
        self.parked: dict[tuple[VerdictKind, int], tuple] = {}   # verdicts waiting for weak evidence to be corroborated

    def _normalise(self, ev: Event) -> Event:
        """Fold action synonyms. A sprinkle/pour whose obj is a surface or something the recipe never sprinkles or pours
        (the tomato, the plate) names the destination: make it the target and leave the ingredient unknown."""
        ev = ev.model_copy(update={"action": ACTION_SYNONYMS.get(ev.action.lower().strip(), ev.action.lower().strip())})
        o = _norm(ev.obj)
        if ev.action in SPRINKLE_ACTIONS and not any(a == ev.action and _same(i, o) for a, i in self.ingredients) and (o in SURFACES or self._known(o)):
            ev.obj, ev.target = "", ev.target or o
        return ev

    def _corroborated(self, ev: Event, same_technique: bool = False) -> bool:
        """The same action/object (compatible target; same technique if asked) was seen in another window within CORROBORATE_GAP."""
        return any(o is not ev and _win(o) != _win(ev) and o.action == ev.action and _same(o.obj, ev.obj)
                   and (not same_technique or _same(o.technique, ev.technique))
                   and (o.target is None or ev.target is None or _same(o.target, ev.target))
                   and max(o.t_start - ev.t_end, ev.t_start - o.t_end) <= CORROBORATE_GAP for o in self.seen)

    def _consistent(self, step: Step, ev: Event) -> bool:
        """One event counts as the step only if it shows what the step defines (target, technique), or is seen again."""
        return bool((ev.target or not step.target) and (ev.technique or not step.technique)) or self._corroborated(ev)

    def _find(self, ev: Event) -> tuple[Step | None, str]:
        """Return (step, how) where how is exact | technique | item | none.

        Action + object identify the step. The target is soft: any destination the recipe itself names
        (a step's target or object, synonyms folded) is accepted and only breaks ties between steps;
        a place the recipe never mentions (fridge, cutting board) means this is some other action."""
        if not ev.obj:
            return None, "none"
        cands = [s for s in self.steps if (ev.target is None or s.target is None or self._known(ev.target))
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
            ) and not self._known(ev.obj):     # a protocol object in another role is not a wrong item
                return s, "item"
        return None, "none"

    def _known(self, x: str) -> bool:
        """A name the recipe itself uses (as any step's object or target), matched leniently like _same."""
        return any(_same(x, n) for n in self.names)

    def _follows(self, d: int, s: int) -> bool:
        """Must step d come after step s? A partial-order protocol answers from its prerequisite graph; a hand-written
        list answers by position, except within a group (steps that may be done in any order)."""
        if self.partial:
            return s in self.anc[d]
        sd, ss = self.steps[self.pos[d]], self.steps[self.pos[s]]
        return self.pos[d] > self.pos[s] and not (ss.group and ss.group == sd.group)

    def _backed(self, kind: VerdictKind, ev: Event) -> bool:
        """Strong, or seen again (same action/object, compatible target, same technique for a technique verdict) in another window."""
        return ev.conf >= STRONG or self._corroborated(ev, kind == VerdictKind.WRONG_TECHNIQUE)

    def _emit(self, t: float, kind: VerdictKind, sid: int, msg: tuple[str, str], contrib: list[tuple[int, Event]],
              conf: float | None = None, gated: bool = True) -> Verdict | None:
        """msg: (statement for a flag, confirming question for an ask).
        contrib: (event_idx, event) pairs behind the verdict; conf defaults to their minimum.

        A verdict resting on a weak event nobody corroborates yet is parked, not spoken (_release)."""
        if (kind, sid) in self._reported:
            return None
        if gated and not all(self._backed(kind, e) for _, e in contrib):
            self.parked[(kind, sid)] = (kind, sid, msg, contrib, conf)
            return None
        conf = min(e.conf for _, e in contrib) if conf is None else conf
        ev = [Evidence(clip=e.clip, t_start=e.t_start, t_end=e.t_end, frame_t=e.frame_refs, event_idx=i)
              for i, e in contrib]
        mode = mode_for(kind, conf, self.thresholds)
        v = Verdict(t=t, step_id=sid, kind=kind, conf=round(conf, 2), mode=mode,
                    message=msg[mode == "ask"], evidence=ev)
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
        ev = self._normalise(ev)
        self.seen.append(ev)
        return self._feed(ev) + self._release(ev.t_end)

    def _issue(self, ev: Event, cur) -> list[Verdict]:
        """A visible mishap (spill, pile-up, ...) is a technique error on the step being done, even one already done."""
        if not ev.issue or ev.issue.strip().lower() in ("", "none", "null", "no") or ev.conf < MIN_CONF:
            return []
        cands = [s for s in self.steps if s.action == ev.action and _same(s.obj, ev.obj)]
        s = next((c for c in cands if c.id not in self.done), cands[0] if cands else None)
        if s is None:
            return []
        v = self._emit(ev.t_end, VerdictKind.WRONG_TECHNIQUE, s.id,
                       (f"Step {s.id} ('{s.label}'): that looked like a {ev.issue}.",
                        f"Step {s.id} ('{s.label}'): was there a {ev.issue}?"), [cur])
        return [v] if v else []

    def _feed(self, ev: Event) -> list[Verdict]:
        cur = (self._n, ev)
        self._n += 1
        return self._issue(ev, cur) + self._feed_step(ev, cur)

    def _feed_step(self, ev: Event, cur) -> list[Verdict]:
        step, how = self._find(ev)
        if step is None or not self._consistent(step, ev):
            return []
        ready = all(a in self.done for a in step.after)   # else attempted anyway: the missing prerequisite is what gets flagged
        if how == "item" and not (ev.conf >= STRONG and (not ITEM_NEEDS_REPEAT or self._corroborated(ev))):   # an unknown noun is noise until it is strong and repeated
            return []
        out: list[Verdict | None] = []
        t = ev.t_end
        if how == "technique" and step.id in self.done:
            return [v for v in [self._emit(t, VerdictKind.WRONG_TECHNIQUE, step.id,
                (f"Step {step.id}: the expert used '{step.technique}' on the {step.obj}; this looks like '{ev.technique}'.",
                 f"Step {step.id}: did you {ev.technique} the {step.obj}? The expert used '{step.technique}'."), [cur])] if v]
        if how == "technique":
            out.append(self._emit(t, VerdictKind.WRONG_TECHNIQUE, step.id,
                (f"Step {step.id}: the expert used '{step.technique}' on the {step.obj}; this looks like '{ev.technique}'.",
                 f"Step {step.id}: did you {ev.technique} the {step.obj}? The expert used '{step.technique}'."), [cur]))
        elif how == "item":
            out.append(self._emit(t, VerdictKind.WRONG_ITEM, step.id,
                (f"Step {step.id} calls for {step.obj}, but this looks like {ev.obj}.",
                 f"Step {step.id} calls for {step.obj}; did you use {ev.obj}?"), [cur]))
        if ev.conf >= MIN_CONF:
            if step.id not in self.done:   # a step done before its prerequisite was only an earlier, different action: not this step
                for d in [d for d in self.done if step.id in self.steps[self.pos[d]].after]:
                    del self.done[d]
            later = sorted((d for d in self.done if self._follows(d, step.id)
                            and ev.t_start - self.done[d][1].t_end >= ORDER_GAP), key=self.pos.get)
            if later:
                first = next(s for s in self.steps if s.id == later[0])
                out.append(self._emit(t, VerdictKind.WRONG_ORDER, first.id,
                    (f"'{first.label}' (step {first.id}) came before step {step.id} "
                     f"('{step.label}'); the expert did them the other way round.",
                     f"Did you do '{first.label}' (step {first.id}) before '{step.label}' (step {step.id})? "
                     "The expert did them the other way round."), [self.done[first.id], cur]))
            fresh = step.id not in self.done
            self.done.setdefault(step.id, cur)
            for m in self.steps:
                if self._follows(step.id, m.id) and m.id not in self.done and not m.optional:
                    after = [c for d, c in self.done.items() if self._follows(d, m.id)]   # an unordered peer is no evidence
                    if len(after) >= self.SKIP_AFTER:
                        out.append(self._emit(t, VerdictKind.SKIPPED, m.id,
                            (f"Step {m.id} ('{m.label}') looks skipped.", f"Did you do '{m.label}' (step {m.id})?"), after))
            if how == "exact" and fresh and ready and not any(out):
                out.append(self._emit(t, VerdictKind.ON_TRACK, step.id, (f"Step {step.id} ('{step.label}') done.",) * 2, [cur]))
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
                    (f"Step {s.id} ('{s.label}') was never done.", f"Did you do '{s.label}' (step {s.id})?"), seen[-1:], conf, gated=False))
        return [v for v in out if v]
