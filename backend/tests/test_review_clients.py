import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from unittest import mock

from labbench.pipeline import chunker
from labbench.pipeline.sam import MetaSam
from labbench.pipeline.vision import Vision, VisionError, parse_events

GOOD = "<0f>1<|box;x1=10;y1=10;x2=20;y2=20;w=100;h=100;c=0.9|>"
EV = '[{"t_start":%s,"t_end":%s,"action":"cut","obj":"tomato"}]'


class ReviewClientTests(unittest.TestCase):
    def test_malformed_sam_frame_skips_only_that_concept(self):
        sam, errors = MetaSam(api_key="k"), []
        bad = {"hand": "<0f>1<|box;x1=1;y1=1;x2=2;y2=2|>", "cup": "<0f>1<|box;x1=1;y1=1;x2=2;y2=2;w=0;h=0|>"}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(MetaSam, "_request", side_effect=lambda u, c: bad.get(c, GOOD)):
            chunk = pathlib.Path(d, "c.mp4")
            chunk.write_bytes(b"x")
            tracks = sam.track(chunk, ["hand", "cup", "tomato"], 0.0, errors)
        self.assertEqual([t.label for t in tracks], ["tomato"])
        self.assertEqual(len(errors), 2)

    def test_reply_without_choices_is_vision_error(self):
        resp = io.BytesIO(json.dumps({"error": "overloaded"}).encode())
        with mock.patch("urllib.request.urlopen", return_value=mock.MagicMock(__enter__=lambda s: resp, __exit__=lambda *a: False)):
            with self.assertRaises(VisionError):
                Vision("glimmer").events([], [], 0, 3)

    def test_event_times_are_ordered_inside_window(self):
        (e,) = parse_events(EV % (0.5, 2.0), 120, 123)          # window-relative reply
        self.assertEqual((e.t_start, e.t_end), (120.5, 122.0))
        (e,) = parse_events(EV % (119, 121), 120, 123)          # straddles the start
        self.assertEqual((e.t_start, e.t_end), (120, 121))
        (e,) = parse_events(EV % (130, 140), 120, 123)          # after the window
        self.assertEqual((e.t_start, e.t_end), (123, 123))

    def test_no_sleep_after_final_attempt(self):
        err = urllib.error.URLError("down")
        with mock.patch.object(Vision, "_chat", side_effect=[err, err]), mock.patch("time.sleep") as sl:
            with self.assertRaises(VisionError):
                Vision("glimmer").events([], [], 0, 3)
        self.assertEqual(sl.call_count, 1)

    def test_stale_keyframe_is_not_returned(self):
        with tempfile.TemporaryDirectory() as d:
            stale = pathlib.Path(d, f"{1.5:08.2f}.jpg")
            stale.write_bytes(b"old")
            with mock.patch.object(chunker, "_run"):             # ffmpeg writes nothing
                self.assertEqual(chunker.keyframes(pathlib.Path("v.mp4"), 1.0, 2.0, pathlib.Path(d), n=1), [])


if __name__ == "__main__":
    unittest.main()
