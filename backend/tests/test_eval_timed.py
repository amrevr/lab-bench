import unittest

from labbench.eval import score_timed, summarize_timed


def rec(rid, steps):
    return {"recording_id": rid, "step_annotations": steps}


def step(sid, a, b, *tags):
    return {"step_id": sid, "start_time": a, "end_time": b, **({"errors": [{"tag": t} for t in tags]} if tags else {})}


def sess(verdicts, last_t=100.0, end=120.0):
    return {"verdicts": [dict(mode="ask", **v) for v in verdicts], "events": [{"t_end": last_t}], "run": {"end": end}}


class TimedScoringTests(unittest.TestCase):
    def test_technique_needs_right_kind_and_time(self):
        r = rec("c", [step(1, 10, 20, "Technique Error")])
        self.assertTrue(score_timed(sess([{"step_id": 1, "kind": "wrong_technique", "t": 25}]), r)["rows"][0]["strict"])
        late = score_timed(sess([{"step_id": 1, "kind": "wrong_technique", "t": 60}]), r)
        self.assertEqual((late["rows"][0]["strict"], late["rows"][0]["lenient"], late["fa_mid"]), (False, True, 1))
        wrong = score_timed(sess([{"step_id": 1, "kind": "skipped_step", "t": 15}]), r)
        self.assertFalse(wrong["rows"][0]["strict"])

    def test_missing_step_counts_at_end_but_is_marked(self):
        r = rec("c", [step(1, -1, -1, "Missing Step"), step(2, 5, 9)])
        out = score_timed(sess([{"step_id": 1, "kind": "skipped_step", "t": 120}]), r)
        self.assertEqual((out["rows"][0]["strict"], out["rows"][0]["at_end"], out["fa_end"]), (True, True, 0))
        self.assertEqual(out["omission"], (1, 1, 1))

    def test_order_swap_is_timed_from_first_misplaced_step_and_accepts_a_skip_ask(self):
        r = rec("c", [step(3, 50, 60, "Order Error"), step(2, 80, 90, "Order Error")])
        out = score_timed(sess([{"step_id": 2, "kind": "skipped_step", "t": 55}]), r)
        row = next(x for x in out["rows"] if x["step_id"] == 2)
        self.assertEqual((row["strict"], row["delay_s"]), (True, 5.0))

    def test_not_visible_steps_are_neither_rows_nor_false_alarms(self):
        r = rec("c", [step(1, 10, 20, "Measurement Error")])
        out = score_timed(sess([{"step_id": 1, "kind": "wrong_item", "t": 15}]), r)
        self.assertEqual((out["rows"], out["not_visible"], out["fa_mid"]), ([], [1], 0))

    def test_recording_level_uses_mid_clip_alarms_on_normal_clips(self):
        normal = score_timed(sess([{"step_id": 4, "kind": "skipped_step", "t": 120}]), rec("n", [step(4, 1, 2)]))
        error = score_timed(sess([{"step_id": 1, "kind": "wrong_technique", "t": 15}]), rec("e", [step(1, 10, 20, "Technique Error")]))
        s = summarize_timed([normal, error])
        self.assertEqual((s["rec_tpr"], s["rec_fpr"], s["fa_end"]), ((1, 1), (0, 1), 1))


    def test_only_normal_clips_with_alarms_does_not_divide_by_zero(self):
        normal = score_timed(sess([{"step_id": 4, "kind": "skipped_step", "t": 50}]), rec("n", [step(4, 1, 2)]))
        self.assertEqual(summarize_timed([normal])["rec_f1_balanced"], 0.0)


if __name__ == "__main__":
    unittest.main()
