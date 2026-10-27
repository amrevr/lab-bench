import unittest

from labbench.pipeline.glimmer import parse_events
from labbench.pipeline.sam import Track, link_tracks
from labbench.pipeline.wonder import coach_text
from labbench.schema import Verdict, VerdictKind


class ClientTests(unittest.TestCase):
    def test_parse_events_tolerates_prose_and_clamps(self):
        txt = 'Sure!\n[{"t_start":0,"t_end":9,"action":"cut","obj":"tomato","conf":0.7}]'
        (e,) = parse_events(txt, 2.0, 6.0)
        self.assertEqual((e.t_start, e.t_end), (2.0, 6.0))

    def test_parse_events_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_events("no json here")

    def test_link_tracks_keeps_identity_across_chunks(self):
        prev = [Track("a", "hand", [(3.9, 0.1, 0.1, 0.3, 0.3)]), Track("b", "hand", [(3.9, 0.6, 0.6, 0.9, 0.9)])]
        new = [Track("x", "hand", [(3.5, 0.62, 0.6, 0.9, 0.88)]), Track("y", "hand", [(3.5, 0.1, 0.12, 0.3, 0.3)])]
        self.assertEqual(link_tracks(prev, new), {"x": "b", "y": "a"})

    def test_wonder_offline_returns_finding(self):
        v = Verdict(t=1, step_id=1, kind=VerdictKind.SKIPPED, conf=0.9, mode="flag", message="m")
        self.assertEqual(coach_text(v), "m")


if __name__ == "__main__":
    unittest.main()
