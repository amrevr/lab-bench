import json
import pathlib
import unittest

from labbench.answerkey import parse_error
from labbench.eval import by_category, score_clip
from labbench.pipeline.align import Aligner
from labbench.run import ROOT, load_protocol, run_clip
from labbench.schema import ASK_THRESHOLD, FLAG_THRESHOLD, Event, Step, Verdict, VerdictKind, mode_for


def non_ok(session):
    return [v for v in session["verdicts"] if v["mode"] != "ok"]


def _protocol():
    st = lambda i, a, o, t=None, g=None: Step(id=i, label=f"s{i}", action=a, obj=o, target=t, group=g)
    return [st(1, "cut", "tomato"), st(2, "place", "tomato", "plate"), st(3, "sprinkle", "salt", "plate", "s"),
            st(4, "sprinkle", "pepper", "plate", "s")]


def _feed(events):
    from labbench.schema import Protocol
    al = Aligner(Protocol(name="p", source_clip="c", steps=_protocol()))
    for e in events:
        al.feed(Event(**e))
    return [(v.kind.value, v.step_id) for v in al.verdicts if v.mode != "ok"]


class AlignerMatchingTests(unittest.TestCase):
    def test_target_is_soft_for_known_destinations_and_synonyms(self):
        for target in ("tray", "tomato", None):       # tray = plate; the recipe names tomato; unknown = absent
            self.assertEqual(_feed([dict(t_start=0, t_end=1, action="sprinkle", obj="salt", target=target, conf=0.9)]), [])
        self.assertEqual(_feed([dict(t_start=0, t_end=1, action="cut", obj="tomato", conf=0.9),
                                dict(t_start=1, t_end=2, action="place", obj="tomato", target="platter", conf=0.9)]), [])

    def test_unknown_destination_is_a_different_action(self):
        # putting the tomato in the fridge is not "place the slices on the plate"
        self.assertEqual(_feed([dict(t_start=0, t_end=1, action="place", obj="tomato", target="fridge", conf=0.9),
                                dict(t_start=5, t_end=6, action="cut", obj="tomato", conf=0.9)]), [])

    def test_events_within_one_chunk_are_not_ordered(self):
        place = dict(t_start=0, t_end=1, action="place", obj="tomato", target="plate", conf=0.9)
        cut = lambda t: dict(t_start=t, t_end=t + 1, action="cut", obj="tomato", conf=0.9)
        self.assertEqual(_feed([place, cut(1)]), [])
        self.assertEqual(_feed([place, cut(10)]), [("wrong_order", 2)])

    def test_group_steps_may_come_in_any_order(self):
        ev = lambda t, o: dict(t_start=t, t_end=t + 1, action="sprinkle", obj=o, target="plate", conf=0.9)
        self.assertNotIn(("wrong_order", 4), _feed([ev(0, "pepper"), ev(20, "salt")]))
        self.assertIn(("wrong_order", 4), _feed([ev(0, "pepper"), dict(t_start=20, t_end=21, action="place", obj="tomato", target="plate", conf=0.9)]))


class PipelineTests(unittest.TestCase):
    def test_expert_run_is_clean(self):
        s = run_clip(ROOT, "12_5")
        self.assertEqual(non_ok(s), [])
        self.assertEqual({v["kind"] for v in s["verdicts"]}, {"on_track"})    # on-track is shown as mode ok

    def test_all_fixture_errors_caught_without_false_alarms(self):
        for d in sorted(ROOT.iterdir()):
            key_path = d / "answer_key.json"
            if not key_path.exists():
                continue
            r = score_clip(run_clip(ROOT, d.name), json.loads(key_path.read_text()))
            self.assertTrue(all(e["caught"] for e in r["errors"]), r)
            self.assertEqual(r["false_alarms"], 0, r)

    def test_unsure_technique_asks_instead_of_flagging(self):
        v = non_ok(run_clip(ROOT, "12_26"))[0]
        self.assertEqual(v["mode"], "ask")

    def test_mode_thresholds(self):
        k = VerdictKind.SKIPPED
        self.assertEqual(mode_for(k, FLAG_THRESHOLD), "flag")
        self.assertEqual(mode_for(k, ASK_THRESHOLD), "ask")
        self.assertEqual(mode_for(k, ASK_THRESHOLD - 0.01), "ok")

    def test_strict_kinds_need_higher_confidence(self):
        self.assertEqual(mode_for(VerdictKind.WRONG_TECHNIQUE, 0.8), "ask")
        self.assertEqual(mode_for(VerdictKind.WRONG_ITEM, 0.85), "flag")
        self.assertEqual(mode_for(VerdictKind.WRONG_TECHNIQUE, 0.49), "ok")

    def test_order_and_skipped_use_minimum_event_confidence(self):
        # fixture 12_41: the order error spans two events; weaken one and the verdict follows it
        events = json.loads((ROOT / "12_41" / "events.json").read_text())
        events[2]["conf"] = 0.6
        al = Aligner(load_protocol(ROOT))
        for e in sorted(map(Event.model_validate, events), key=lambda e: e.t_end):
            al.feed(e)
        (o,) = [v for v in al.verdicts if v.kind == VerdictKind.WRONG_ORDER]
        self.assertEqual((o.conf, o.mode), (0.6, "ask"))
        self.assertEqual(len(o.evidence), 2)

    def test_skipped_at_end_uses_event_confidence(self):
        al = Aligner(load_protocol(ROOT))
        al.feed(Event(t_start=0, t_end=1, action="cut", obj="tomato", technique="slice", conf=0.6))
        self.assertEqual({(v.conf, v.mode) for v in al.finish(10)}, {(0.6, "ask")})

    def test_alignment_ignores_step_text(self):
        p = load_protocol(ROOT)
        events = json.loads((ROOT / "12_43" / "events.json").read_text())
        def run(proto):
            al = Aligner(proto)
            for e in map(Event.model_validate, events):
                al.feed(e)
            al.finish(99)
            return [(v.kind, v.step_id, v.mode) for v in al.verdicts]
        p2 = p.model_copy(update={"steps": [s.model_copy(update={"label": "x", "recipe_text": "y", "narration": "z"})
                                            for s in p.steps]})
        self.assertEqual(run(p), run(p2))

    def test_evidence_points_at_event_and_frames(self):
        (v,) = non_ok(run_clip(ROOT, "12_26"))
        (ev,) = v["evidence"]
        self.assertEqual((ev["clip"], ev["event_idx"], ev["frame_t"]), ("12_26", 1, [13.0, 16.0]))

    def test_old_json_still_loads(self):
        self.assertEqual(Step.model_validate({"id": 1, "text": "Slice", "action": "cut", "obj": "tomato"}).label, "Slice")
        v = Verdict.model_validate({"t": 1, "step_id": 1, "kind": "skipped_step", "conf": 0.9, "mode": "flag",
                                    "message": "m", "frame_refs": [2.0, 5.0]})
        self.assertEqual(v.evidence[0].frame_t, [2.0, 5.0])

    def test_real_key_scored_per_step_and_category(self):
        key = {"clip": "x", "errors": [parse_error(e) for e in
               ["125 Temperature", "121 Missing Step", "120 Order/Technique", "119 Measurement/Technique"]]}
        def vd(kind, step, conf=0.9):
            return {"t": 0, "step_id": step, "kind": kind, "conf": conf, "mode": "flag", "message": ""}
        sess = {"verdicts": [vd("wrong_item", 125), vd("wrong_order", 121), vd("wrong_technique", 119),
                             vd("skipped_step", 99), {**vd("skipped_step", 120), "mode": "ok"}]}
        r = score_clip(sess, key)
        self.assertEqual(by_category([r]), {"Temperature": (1, 1), "Missing Step": (0, 1), "Order": (0, 1),
                                            "Technique": (1, 2), "Measurement": (1, 1)})
        self.assertEqual(r["false_alarms"], 2)               # step 99, and wrong_order on 121 (the key says Missing Step)
        self.assertEqual([x["hit"] for x in r["verdicts"]], [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
