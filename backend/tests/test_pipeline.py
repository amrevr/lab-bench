import json
import pathlib
import unittest

from labbench.eval import score_clip
from labbench.run import ROOT, run_clip
from labbench.schema import ASK_THRESHOLD, FLAG_THRESHOLD, VerdictKind, mode_for


class PipelineTests(unittest.TestCase):
    def test_expert_run_is_clean(self):
        self.assertEqual(run_clip(ROOT, "12_5")["verdicts"], [])

    def test_all_fixture_errors_caught_without_false_alarms(self):
        for d in sorted(ROOT.iterdir()):
            key_path = d / "answer_key.json"
            if not key_path.exists():
                continue
            r = score_clip(run_clip(ROOT, d.name), json.loads(key_path.read_text()))
            self.assertTrue(all(e["caught"] for e in r["errors"]), r)
            self.assertEqual(r["false_alarms"], 0, r)

    def test_unsure_technique_asks_instead_of_flagging(self):
        v = run_clip(ROOT, "12_26")["verdicts"][0]
        self.assertEqual(v["mode"], "ask")

    def test_mode_thresholds(self):
        k = VerdictKind.SKIPPED
        self.assertEqual(mode_for(k, FLAG_THRESHOLD), "flag")
        self.assertEqual(mode_for(k, ASK_THRESHOLD), "ask")
        self.assertEqual(mode_for(k, ASK_THRESHOLD - 0.01), "ok")


if __name__ == "__main__":
    unittest.main()
