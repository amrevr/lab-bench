import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from labbench import replay
from labbench.schema import Event

PROTOCOL = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "recipe12" / "protocol.json"


def fake_events(self, kf, tracks, t0, t1, clip=None):
    return [Event(t_start=t0, t_end=t1, action="wash", obj="tomato", conf=0.9, clip=clip)] if t0 == 0 else []


class ReplayTests(unittest.TestCase):
    def run_replay(self, *extra):
        out = pathlib.Path(tempfile.mkdtemp()) / "s.json"
        argv = ["replay", "--clip", "/x/12_26.mp4", "--protocol", str(PROTOCOL), "--backend", "spark", "--start", "0",
                "--end", "8", "--pace", "fast", "--out", str(out), "--workdir", str(out.parent), *extra]
        with mock.patch("sys.argv", argv), mock.patch.dict(os.environ, {"META_API_KEY": ""}), \
                mock.patch.object(replay.chunker, "cut_chunk"), mock.patch.object(replay.chunker, "keyframes", return_value=[]), \
                mock.patch.object(replay.MetaSam, "track", lambda self, c, p, t0: setattr(self, "frames", self.frames + 150) or []), \
                mock.patch.object(replay.Vision, "events", fake_events), self.assertRaises(SystemExit) as ex:
            replay.main()
        return ex.exception.code, out

    def test_session_has_timing_and_cost(self):
        code, out = self.run_replay("--max-spend", "1", "--finish")
        self.assertEqual(code, 0)
        s = json.loads(out.read_text())
        self.assertEqual(s["clip"], "12_26")
        v = s["verdicts"][0]
        self.assertEqual((v["kind"], v["step_id"], v["closed_t"]), ("on_track", 125, 3.0))
        self.assertGreaterEqual(v["latency_s"], 0)
        self.assertGreater(v["emitted_at"], 1e9)
        self.assertEqual(s["run"]["billed_frames"], 150 * 4)
        self.assertEqual(s["verdicts"][-1]["kind"], "skipped_step")   # --finish: the rest was never done

    def test_spend_guard_refuses_when_estimate_exceeds_cap(self):
        code, out = self.run_replay("--max-spend", "0.07")            # 4 windows cost $0.12
        self.assertIn("over --max-spend", code)
        self.assertFalse(out.exists())

if __name__ == "__main__":
    unittest.main()
