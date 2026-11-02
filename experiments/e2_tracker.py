"""E2: belief tracker over the protocol's precedence graph, drop-in for pipeline.align.Aligner.

State: b[i] = P(step i done) = noisy-OR of the confidences of events assigned to it (corroboration accumulates).
An event goes to the best-matching step: undone and prerequisite-ready steps are preferred (soft prior), nothing is
ever un-done, so evidence is never erased.  With done(i) = b[i] >= TAU:
  wrong_order(d): d is done and a step p it must follow becomes done >= ORDER_GAP later (per violated d, or once per
                  event with per_edge=False, like Aligner)
  skipped(p):     p is not done while its dependants carry soft count sum(b_d) >= K (noisy-OR is the confidence);
                  anything still not done at finish() is skipped.
Technique / wrong-item / issue verdicts are inherited unchanged from Aligner.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from labbench.pipeline.align import Aligner, CONTINUE_GAP, ORDER_GAP, STRONG, MIN_CONF
from labbench.schema import Event, Protocol, VerdictKind, THRESHOLDS


class Tracker(Aligner):
    def __init__(self, protocol: Protocol, thresholds: dict = THRESHOLDS, tau=0.5, k=2.0, per_edge=True, both_ends=False):
        super().__init__(protocol, thresholds)
        self.tau, self.k, self.per_edge, self.both_ends = tau, k, per_edge, both_ends
        self.b = {s.id: 0.0 for s in self.steps}
        self.first: dict[int, tuple[int, Event]] = {}       # step -> (idx, event) that first made it done (self.done)
        self.pred = {s.id: [p.id for p in self.steps if p.id != s.id and self._follows(s.id, p.id)] for s in self.steps}
        self.succ = {s.id: [d for d in self.pred if s.id in self.pred[d]] for s in self.steps}
        self.last = {s.id: -1e9 for s in self.steps}
        self.evs: dict[int, list[tuple[int, Event]]] = {s.id: [] for s in self.steps}

    def _ready(self, sid):
        p = self.pred[sid]
        return sum(self.b[x] >= self.tau for x in p) / len(p) if p else 1.0

    def _assign(self, ev: Event):
        cands = [s for s in self.steps if (ev.target is None or s.target is None or self._known(ev.target)) and self._is(s, ev)]
        if cands:
            ready = [c for c in cands if c.id not in self.done and self._ready(c.id) == 1.0]
            cont = [c for c in cands if c.id in self.done and ev.t_start - self.last[c.id] <= CONTINUE_GAP]
            best = (ready or cont or sorted(cands, key=lambda s: ((1.0 if s.id not in self.done else 0.2) * (0.3 + 0.7 * self._ready(s.id)), -self.pos[s.id]), reverse=True))[0]
            if best.technique and ev.technique and not self._same_t(best, ev):
                return best, "technique"
            return best, "exact"
        s, how = super()._find(ev)
        return (s, how) if how in ("item", "technique") else (None, "none")

    def _same_t(self, s, ev):
        from labbench.pipeline.align import _same
        return _same(s.technique, ev.technique)

    def feed(self, ev: Event):
        ev = self._normalise(ev)
        self.seen.append(ev)
        cur = (self._n, ev)
        self._n += 1
        out = self._issue(ev, cur)
        step, how = self._assign(ev)
        if step is not None and self._consistent(step, ev):
            out += self._step(ev, cur, step, how)
        return [v for v in out if v] + self._release(ev.t_end)

    def _step(self, ev, cur, step, how):
        t = ev.t_end
        if how == "item" and not (ev.conf >= STRONG and self._corroborated(ev)):
            return []
        out = []
        if how == "technique":
            out.append(self._emit(t, VerdictKind.WRONG_TECHNIQUE, step.id,
                (f"Step {step.id}: the expert used '{step.technique}' on the {step.obj}; this looks like '{ev.technique}'.",
                 f"Step {step.id}: did you {ev.technique} the {step.obj}? The expert used '{step.technique}'."), [cur]))
        elif how == "item":
            out.append(self._emit(t, VerdictKind.WRONG_ITEM, step.id,
                (f"Step {step.id} calls for {step.obj}, but this looks like {ev.obj}.",
                 f"Step {step.id} calls for {step.obj}; did you use {ev.obj}?"), [cur]))
        if how == "item":
            return [v for v in out if v]
        i = step.id
        self.last[i] = ev.t_end
        if step.id in self.done and how != "technique":    # a repeat of a done step: more evidence only
            self.b[i] = 1 - (1 - self.b[i]) * (1 - ev.conf)
            return [v for v in out if v]
        self.b[i] = 1 - (1 - self.b[i]) * (1 - ev.conf)
        self.evs[i].append(cur)
        if self.b[i] >= self.tau and i not in self.done:
            self.done[i] = cur
        done = [d for d in self.done if self.b[d] >= self.tau]
        # order: a done step d whose prerequisite is only now done, GAP later
        viol = sorted((d for d in done if d != i and i in self.pred[d] and ev.t_start - self.done[d][1].t_end >= ORDER_GAP and i in self.done),
                      key=self.pos.get)
        if not self.per_edge:
            viol = viol[:1]
        for d in viol:
            st = self.steps[self.pos[d]]
            out.append(self._emit(t, VerdictKind.WRONG_ORDER, d,
                (f"'{st.label}' (step {d}) came before step {i} ('{step.label}'); the expert did them the other way round.",
                 f"Did you do '{st.label}' (step {d}) before '{step.label}' (step {i})? The expert did them the other way round."),
                [self.done[d], self.done[i]], conf=min(self.b[d], self.b[i])))
            if self.both_ends:
                out.append(self._emit(t, VerdictKind.WRONG_ORDER, i,
                    (f"'{step.label}' (step {i}) should have come before step {d} ('{st.label}').",
                     f"Did you do '{st.label}' (step {d}) before '{step.label}' (step {i})?"),
                    [self.done[d], self.done[i]], conf=min(self.b[d], self.b[i])))
        # skip: undone m with enough done dependants
        for m in self.steps:
            if m.id in self.done or m.optional or not m.visible:
                continue
            deps = [d for d in self.succ[m.id] if d in self.done]
            if deps and sum(self.b[d] for d in deps) >= self.k:
                conf = 1.0
                for d in deps:
                    conf *= 1 - self.b[d]
                out.append(self._emit(t, VerdictKind.SKIPPED, m.id,
                    (f"Step {m.id} ('{m.label}') looks skipped.", f"Did you do '{m.label}' (step {m.id})?"),
                    [self.done[d] for d in deps], conf=1 - conf))
        return [v for v in out if v]

    def finish(self, t_end: float):
        seen = list(self.done.values())
        conf = min((e.conf for _, e in seen), default=0.5)
        for s in self.steps:
            if s.id not in self.done and not s.optional and s.visible:
                self._emit(t_end, VerdictKind.SKIPPED, s.id,
                    (f"Step {s.id} ('{s.label}') was never done.", f"Did you do '{s.label}' (step {s.id})?"), seen[-1:], conf, gated=False)
        return []
