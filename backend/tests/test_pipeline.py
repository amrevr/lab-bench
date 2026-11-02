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
    return [Step(id=1, label="s1", action="cut", obj="tomato", technique="slice"), st(2, "place", "tomato", "plate"), st(3, "sprinkle", "salt", "plate", "s"),
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
        cut = lambda t: dict(t_start=t, t_end=t + 1, action="cut", obj="tomato", technique="slice", conf=0.9)
        self.assertEqual(_feed([place, cut(1)]), [])
        self.assertEqual(_feed([place, cut(10)]), [("wrong_order", 2)])

    def test_technique_is_judged_on_later_events_of_an_attempted_step(self):
        cut = lambda t, tech: dict(t_start=t, t_end=t + 1, action="cut", obj="tomato", technique=tech, conf=0.95)
        self.assertEqual(_feed([cut(0, "slice"), cut(5, "slice")]), [])
        self.assertEqual(_feed([cut(0, "slice"), cut(5, "dice"), cut(9, "dice")]), [("wrong_technique", 1)])   # once per step

    def test_group_steps_may_come_in_any_order(self):
        ev = lambda t, o: dict(t_start=t, t_end=t + 1, action="sprinkle", obj=o, target="plate", conf=0.9)
        self.assertNotIn(("wrong_order", 4), _feed([ev(0, "pepper"), ev(20, "salt")]))
        self.assertIn(("wrong_order", 4), _feed([ev(0, "pepper"), dict(t_start=20, t_end=21, action="place", obj="tomato", target="plate", conf=0.9)]))


class FalseAlarmControlTests(unittest.TestCase):
    weak_cut = dict(t_start=0, t_end=1, action="cut", obj="tomato", technique="slice", conf=0.9, chunk=[0, 3])

    def test_weak_lone_event_is_not_spoken_but_corroborated_one_is(self):
        place = lambda t, w: dict(t_start=t, t_end=t + 1, action="place", obj="tomato", target="plate", conf=0.55, chunk=[w, w + 3])
        cut = dict(t_start=10, t_end=11, action="cut", obj="tomato", technique="slice", conf=0.9, chunk=[10, 13])
        self.assertEqual(_feed([place(0, 0), cut]), [])                                   # one 0.6 window: parked, then dropped
        self.assertEqual(_feed([place(0, 0), cut, place(11.5, 11.5)]), [])                # too far from the first: still lone
        self.assertEqual(_feed([place(0, 0), place(1.5, 1.5), cut]), [("wrong_order", 2)])    # seen in two windows
        self.assertEqual(_feed([place(0, 0), cut, place(1.5, 1.5)]), [("wrong_order", 2)])    # corroborated later: spoken then

    def test_weak_technique_needs_the_same_technique_again(self):
        cut = lambda t, w, tech: dict(t_start=t, t_end=t + 1, action="cut", obj="tomato", technique=tech, conf=0.55, chunk=[w, w + 3])
        strong = {**self.weak_cut, "t_start": 0, "t_end": 1}
        self.assertEqual(_feed([strong, cut(5, 5, "tear"), cut(6.5, 6.5, "slice")]), [])
        self.assertEqual(_feed([strong, cut(5, 5, "tear"), cut(5.5, 5.5, "tear")]), [("wrong_technique", 1)])

    def test_parked_skip_is_withdrawn_when_the_step_happens(self):
        ev = lambda t, a, o, tg=None, c=0.55: dict(t_start=t, t_end=t + 1, action=a, obj=o, target=tg, conf=c, chunk=[t, t + 3])
        ok = [ev(0, "cut", "tomato", c=0.9)]
        later = [ev(10, "sprinkle", "salt", "plate"), ev(20, "sprinkle", "pepper", "plate")]    # weak: step 2 looks skipped, parked
        again = [ev(10.5, "sprinkle", "salt", "plate"), ev(20.5, "sprinkle", "pepper", "plate")]   # corroboration arrives later
        self.assertIn(("skipped_step", 2), _feed(ok + later + again))
        self.assertNotIn(("skipped_step", 2), _feed(ok + later + [ev(15, "place", "tomato", "plate", 0.9)] + again))


class PrincipledAlignerTests(unittest.TestCase):
    ev = staticmethod(lambda t, a, o, tg=None, c=0.9, tech=None, w=None: dict(
        t_start=t, t_end=t + 1, action=a, obj=o, target=tg, technique=tech, conf=c, chunk=[t if w is None else w] * 2))

    def run_al(self, steps, events):
        from labbench.schema import Protocol
        al = Aligner(Protocol(name="p", source_clip="c", steps=steps))
        for e in events:
            al.feed(Event(**e))
        return al

    def test_one_event_missing_the_target_or_technique_is_not_the_step(self):
        e = self.ev
        place = lambda t, w=None: e(t, "place", "tomato", None, w=w)
        cut = e(20, "cut", "tomato", tech="slice")
        self.assertEqual(_feed([place(0), cut]), [])                                  # a lone target-less place is not step 2
        self.assertEqual(_feed([place(0), place(1.5), cut]), [("wrong_order", 2)])     # seen in a second window it is
        self.assertEqual(_feed([e(0, "cut", "tomato"), e(10, "place", "tomato", "plate")]), [])   # lone technique-less cut is not step 1

    def test_prerequisite_step_must_have_been_attempted(self):
        steps = [Step(id=1, label="s1", action="cut", obj="tomato", technique="slice"),
                 Step(id=2, label="s2", action="place", obj="tomato", target="plate", after=[1])]
        e = self.ev
        al = self.run_al(steps, [e(0, "place", "tomato", "plate"), e(10, "cut", "tomato", tech="slice")])
        self.assertEqual(sorted(al.done), [1])
        al = self.run_al(steps, [e(0, "cut", "tomato", tech="slice"), e(10, "place", "tomato", "plate")])
        self.assertEqual(sorted(al.done), [1, 2])

    def test_target_named_as_the_ingredient_is_not_a_wrong_item(self):
        e = self.ev
        for noun, act in (("tomato", "sprinkle"), ("plate", "sprinkle"), ("cutting board", "pour"), ("tomato", "season")):
            self.assertEqual(_feed([e(0, "cut", "tomato", tech="slice"), e(10, act, noun, "plate", w=10), e(10.5, act, noun, "plate", w=10.5)]), [])
        al = self.run_al(_protocol(), [e(0, "season", "tomato", None)])
        self.assertEqual((al.seen[0].action, al.seen[0].obj, al.seen[0].target), ("sprinkle", "", "tomato"))

    def test_unknown_noun_is_wrong_item_only_when_strong_and_repeated(self):
        e = self.ev
        cut = e(0, "cut", "tomato", tech="slice")
        pap = lambda t, c=0.9: e(t, "sprinkle", "paprika", "plate", c=c, w=t)
        self.assertEqual(_feed([cut, pap(10)]), [])
        self.assertEqual(_feed([cut, pap(10, 0.55), pap(10.5, 0.55)]), [])
        self.assertEqual(_feed([cut, pap(10), pap(10.5)]), [("wrong_item", 3)])

    def test_unordered_peers_are_no_evidence_of_a_skip(self):
        g = lambda i, o: Step(id=i, label=f"s{i}", action="sprinkle", obj=o, target="plate", group="g")
        steps = [Step(id=1, label="s1", action="cut", obj="tomato", technique="slice"), g(2, "salt"), g(3, "pepper"), g(4, "cumin")]
        e = self.ev
        al = self.run_al(steps, [e(0, "cut", "tomato", tech="slice"), e(10, "sprinkle", "pepper", "plate"), e(20, "sprinkle", "cumin", "plate")])
        self.assertEqual([v.kind for v in al.verdicts if v.mode != "ok"], [])

    def test_ask_is_a_question_and_flag_a_statement(self):
        e = self.ev
        order = lambda c: self.run_al(_protocol(), [e(0, "place", "tomato", "plate", c=c, w=0), e(1.5, "place", "tomato", "plate", c=c, w=1.5),
                                                    e(10, "cut", "tomato", tech="slice")]).verdicts
        ask, flag = [v for v in order(0.6) if v.mode == "ask"], [v for v in order(0.9) if v.mode == "flag"]
        self.assertTrue(ask and flag)
        self.assertTrue(ask[0].message.startswith("Did you") and "?" in ask[0].message)
        self.assertNotIn("?", flag[0].message)


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
        events.append({**events[2], "t_start": events[2]["t_end"], "t_end": events[2]["t_end"] + 1,   # weak events need a second window
                       "chunk": [events[2]["chunk"][0] + 3, events[2]["chunk"][1] + 3]})
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


class KeyframesClipEndTests(unittest.TestCase):
    def test_missing_last_frame_is_skipped(self):
        import tempfile
        from unittest import mock
        from labbench.pipeline import chunker

        def fake_run(cmd):   # ffmpeg writes nothing for a seek past the end
            if float(cmd[cmd.index("-ss") + 1]) < 10.0:
                pathlib.Path(cmd[-1]).write_bytes(b"x")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(chunker, "_run", fake_run):
            kf = chunker.keyframes(pathlib.Path("c.mp4"), 9.0, 10.5, pathlib.Path(d))   # short final window
            self.assertTrue(all(p.exists() for _, p in kf))
        self.assertEqual([round(t, 2) for t, _ in kf], [9.19, 9.56, 9.94])


if __name__ == "__main__":
    unittest.main()


class IssueTests(unittest.TestCase):
    def test_visible_mishap_is_a_technique_flag_on_that_step_even_when_already_done(self):
        from labbench.pipeline.align import Aligner
        from labbench.schema import Protocol
        p = Protocol(name="t", source_clip="t", steps=[Step(id=1, label="fill", action="fill", obj="milk", target="mug")])
        al = Aligner(p)
        mk = lambda t, issue=None, c=0.9: Event(t_start=t, t_end=t + 1, action="fill", obj="milk", target="mug",
                                                 issue=issue, conf=c, chunk=[t, t + 3])
        self.assertEqual([v.kind.value for v in al.feed(mk(0))], ["on_track"])
        self.assertEqual([(v.kind.value, v.step_id) for v in al.feed(mk(5, "spill"))], [("wrong_technique", 1)])
        self.assertEqual(Aligner(p).feed(mk(0, "none")), Aligner(p).feed(mk(0)))      # "none" is no issue


class NounMatchTests(unittest.TestCase):
    def test_names_match_by_contained_words(self):
        from labbench.pipeline.align import _same
        self.assertTrue(_same("tomato slices", "tomato"))
        self.assertTrue(_same("black pepper", "pepper"))
        self.assertTrue(_same("Platter", "plate"))
        self.assertFalse(_same("salt", "pepper"))
        self.assertFalse(_same("brown sugar", "salt"))
        self.assertTrue(_same("olive oil", "olives"))   # known limit: a modifier word matches its noun


class VisibilityTests(unittest.TestCase):
    def test_a_step_not_visible_in_the_expert_run_is_never_asked_about(self):
        from labbench.schema import Protocol
        steps = [Step(id=1, label="a", action="cut", obj="tomato"), Step(id=2, label="b", action="wipe", obj="knife", visible=False),
                 Step(id=3, label="c", action="place", obj="tomato", target="plate"), Step(id=4, label="d", action="pour", obj="oil")]
        al = Aligner(Protocol(name="t", source_clip="t", steps=steps))
        for t, a, o, tg in ((0, "cut", "tomato", None), (10, "place", "tomato", "plate"), (20, "pour", "oil", None)):
            al.feed(Event(t_start=t, t_end=t + 1, action=a, obj=o, target=tg, conf=0.9, chunk=[t, t + 3]))
        al.finish(30)
        self.assertNotIn(2, [v.step_id for v in al.verdicts if v.mode != "ok"])


    def test_a_mishap_and_a_technique_word_on_one_step_are_both_spoken(self):
        from labbench.schema import Protocol
        p = Protocol(name="t", source_clip="t", steps=[Step(id=1, label="cut", action="cut", obj="tomato", technique="slice")])
        al = Aligner(p)
        mk = lambda t, tech, issue=None: Event(t_start=t, t_end=t + 1, action="cut", obj="tomato", technique=tech,
                                                issue=issue, conf=0.9, chunk=[t, t + 3])
        al.feed(mk(0, "slice", "spill"))
        al.feed(mk(5, "dice"))
        self.assertEqual(sorted(((v.kind.value, v.source or "") for v in al.verdicts if v.mode != "ok")),
                         [("wrong_technique", ""), ("wrong_technique", "issue")])
        self.assertEqual(Aligner(p).feed(mk(0, "slice", "n/a")), Aligner(p).feed(mk(0, "slice")))
