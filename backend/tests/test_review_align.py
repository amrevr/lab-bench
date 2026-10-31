import unittest
from unittest import mock

from labbench.pipeline import align
from labbench.pipeline.align import Aligner
from labbench.schema import Event, Protocol, Step, VerdictKind


def _run(steps, events, finish=None, **thresholds):
    al = Aligner(Protocol(name="p", source_clip="c", steps=steps), **thresholds)
    for e in events:
        al.feed(Event(**e))
    if finish:
        al.finish(finish)
    return al


def _ev(t, action, obj, target=None, tech=None, conf=0.9):
    return dict(t_start=t, t_end=t + 1, action=action, obj=obj, target=target, technique=tech, conf=conf)


class PrerequisiteTests(unittest.TestCase):
    steps = [Step(id=119, label="cut", action="cut", obj="tomato", technique="slice"),
             Step(id=120, label="place", action="place", obj="tomato", target="plate", after=[119]),
             Step(id=121, label="salt", action="sprinkle", obj="salt", target="plate"),
             Step(id=122, label="pepper", action="sprinkle", obj="pepper", target="plate")]

    def test_step_done_without_its_prerequisite_is_one_skip_not_two(self):
        al = _run(self.steps, [_ev(0, "place", "tomato", "plate"), _ev(10, "sprinkle", "salt", "plate"),
                               _ev(20, "sprinkle", "pepper", "plate")], finish=30)
        self.assertIn(120, al.done)
        got = [(v.kind, v.step_id) for v in al.verdicts if v.mode != "ok"]
        self.assertEqual(got, [(VerdictKind.SKIPPED, 119)])

    def test_event_before_its_prerequisite_is_still_not_the_step(self):
        al = _run(self.steps, [_ev(0, "place", "tomato", "plate"), _ev(10, "cut", "tomato", tech="slice")])
        self.assertEqual(sorted(al.done), [119])


class CalibrationGateTests(unittest.TestCase):
    steps = [Step(id=1, label="salt", action="sprinkle", obj="salt", target="plate")]

    def test_weak_uncorroborated_item_only_surfaces_with_gates_open(self):
        weak = [_ev(0, "sprinkle", "paprika", "plate", conf=0.6)]
        zero = {VerdictKind.WRONG_ITEM: (0.0, 0.0)}
        self.assertEqual(_run(self.steps, weak, thresholds=zero).verdicts, [])
        with mock.patch.multiple(align, STRONG=0.0, MIN_CONF=0.0, ITEM_NEEDS_REPEAT=False):
            self.assertEqual([v.kind for v in _run(self.steps, weak, thresholds=zero).verdicts], [VerdictKind.WRONG_ITEM])

    def test_event_below_min_conf_is_ignored_unless_gate_open(self):
        low = [_ev(0, "sprinkle", "salt", "plate", conf=0.3)]
        self.assertEqual(_run(self.steps, low).done, {})
        with mock.patch.multiple(align, STRONG=0.0, MIN_CONF=0.0):
            self.assertEqual(list(_run(self.steps, low).done), [1])
