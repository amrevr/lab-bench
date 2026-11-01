import http.server
import json
import threading
import time
import os
import pathlib
import tempfile
import unittest
import urllib.error
from unittest import mock

from labbench.pipeline.chunker import process_chunks, windows
from labbench.pipeline.sam import MetaSam, Track, link_tracks, parse_stream
from labbench.pipeline.vision import Vision, VisionError, parse_events
from labbench.pipeline.wonder import coach_text
from labbench.schema import Verdict, VerdictKind


class ClientTests(unittest.TestCase):
    def test_parse_events_tolerates_prose_and_clamps(self):
        txt = 'Sure!\n[{"t_start":0,"t_end":9,"action":"cut","obj":"tomato","conf":0.7}]'
        (e,) = parse_events(txt, 2.0, 6.0)
        self.assertEqual((e.t_start, e.t_end), (2.0, 6.0))

    def test_parse_events_normalizes_names(self):
        (e,) = parse_events('[{"t_start":0,"t_end":1,"action":"place","obj":"Paper_Towel","target":"cutting_board"}]')
        self.assertEqual((e.obj, e.target), ("paper towel", "cutting board"))

    def test_parse_events_ignores_trailing_junk(self):
        self.assertEqual(parse_events("[]]"), [])            # a doubled closing bracket is not a parse failure
        self.assertEqual(len(parse_events('[{"t_start":1,"t_end":2,"action":"cut","obj":"tomato"}]]')), 1)
        with self.assertRaises(ValueError):                  # an empty array then the real one is not idle
            parse_events('[] Actually: [{"t_start":1,"t_end":2,"action":"cut","obj":"tomato"}]')

    def test_parse_events_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_events("no json here")

    def test_link_tracks_keeps_identity_across_chunks(self):
        prev = [Track("a", "hand", [(3.5, 0.1, 0.1, 0.3, 0.3)]), Track("b", "hand", [(3.5, 0.6, 0.6, 0.9, 0.9)])]
        new = [Track("x", "hand", [(3.5, 0.62, 0.6, 0.9, 0.88)]), Track("y", "hand", [(3.5, 0.1, 0.12, 0.3, 0.3)])]
        self.assertEqual(link_tracks(prev, new), {"x": "b", "y": "a"})

    def test_link_tracks_compares_same_timestamp_not_last_vs_first(self):
        # A fast hand: at the previous chunk's last box (t=4.0) it is far from where the new chunk
        # first sees it (t=3.5); only the same-timestamp comparison matches.
        prev = [Track("a", "hand", [(3.5, 0.1, 0.1, 0.3, 0.3), (4.0, 0.7, 0.7, 0.9, 0.9)])]
        new = [Track("x", "hand", [(3.5, 0.11, 0.1, 0.3, 0.3), (4.0, 0.7, 0.7, 0.9, 0.9)])]
        self.assertEqual(link_tracks(prev, new), {"x": "a"})
        self.assertEqual(link_tracks(prev, [Track("z", "hand", [(5.0, 0.1, 0.1, 0.3, 0.3)])]), {"z": "z"})

    def test_parse_stream_normalizes_boxes_and_keeps_ids_per_response(self):
        text = ("<0f>1<|box;x1=640;y1=360;x2=1280;y2=720;w=1280;h=720;c=0.94|><|mask;x=0;y=0;c=0.9;data=AB,CD|>,"
                "2<|box;x1=0;y1=0;x2=128;y2=72;w=1280;h=720;c=0.8|><|mask;x=0;y=0;c=0.8;data=Z|>"
                "<1f>1<|box;x1=640;y1=360;x2=1280;y2=720;w=1280;h=720;c=0.9|><|mask;x=0|>")
        a, b = parse_stream(text, "hand", t0=2.0, fps=30)
        self.assertEqual((a.id, len(a.boxes), round(a.conf, 2)), ("hand:1", 2, 0.92))
        self.assertEqual(a.boxes[0], (2.0, 0.5, 0.5, 1.0, 1.0))
        self.assertAlmostEqual(a.boxes[1][0], 2 + 1 / 30)
        self.assertEqual(b.boxes, [(2.0, 0.0, 0.0, 0.1, 0.1)])
        self.assertEqual(parse_stream("", "bowl"), [])      # empty stream: no match, not an error

    def test_windows_default_3s_with_half_second_overlap(self):
        self.assertEqual(list(windows(8.0)), [(0.0, 3.0), (2.5, 5.5), (5.0, 8.0), (7.5, 8.0)])

    def test_process_chunks_runs_concurrently_and_keeps_order(self):
        def fn(a, b):
            time.sleep(0.3)
            return a
        t = time.time()
        out = process_chunks(8.0, fn, workers=4)
        self.assertEqual(out, [0.0, 2.5, 5.0, 7.5])
        self.assertLess(time.time() - t, 0.7)             # sequential would take 1.2 s

    def test_wonder_offline_returns_finding(self):
        v = Verdict(t=1, step_id=1, kind=VerdictKind.SKIPPED, conf=0.9, mode="flag", message="m")
        with mock.patch.dict(os.environ, {"META_API_KEY": ""}):
            self.assertEqual(coach_text(v), "m")

    def test_wonder_uses_responses_with_capped_effort(self):
        v = Verdict(t=1, step_id=1, kind=VerdictKind.SKIPPED, conf=0.9, mode="flag", message="m")
        reply = {"output": [{"type": "reasoning", "summary": []},
                            {"type": "message", "content": [{"type": "output_text", "text": " Try step 1. "}]}]}
        url, seen = self._serve([reply], wrap=False)
        with mock.patch.dict(os.environ, {"META_API_KEY": "k", "META_API_BASE": url}):
            self.assertEqual(coach_text(v), "Try step 1.")
        path, auth, body = seen[0]
        self.assertEqual((path, body["model"], body["reasoning"]), ("/v1/responses", "muse-wonder-1.0", {"effort": "low"}))
        self.assertGreaterEqual(body["max_output_tokens"], 600)

    def _serve(self, replies, wrap=True):
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(h):
                seen.append((h.path, h.headers.get("Authorization"), json.loads(h.rfile.read(int(h.headers["Content-Length"])))))
                body = json.dumps({"choices": [{"message": replies.pop(0)}]} if wrap else replies.pop(0)).encode()
                h.send_response(200)
                h.send_header("Content-Length", str(len(body)))
                h.end_headers()
                h.wfile.write(body)

            def log_message(h, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_port}/v1", seen

    def test_vision_backends_send_low_effort_and_retry_once(self):
        ok = '[{"t_start":1,"t_end":3,"action":"cut","obj":"tomato","conf":0.9}]'
        for backend, model in (("glimmer", "glimmer"), ("spark", "meta-models/muse-spark-open-1026")):
            url, seen = self._serve([{"content": None, "reasoning_content": "thinking"}, {"content": ok, "reasoning_content": "x"}])
            v = Vision(backend, base_url=url, api_key="k")
            tracks = [Track("hand:1", "hand", [(1.0, 0, 0, 1, 1)], conf=0.9), Track("tomato:2", "tomato", conf=0.7),
                      Track("plate:3", "plate", conf=0.5)]
            (e,) = v.events([], tracks, 0.0, 3.0, clip="c1")
            self.assertEqual(len(seen), 2)                  # first reply had no content -> retried once
            self.assertGreater(seen[1][2]["max_tokens"], seen[0][2]["max_tokens"])   # empty = ran out on reasoning
            path, auth, body = seen[0]
            self.assertEqual((path, auth, body["model"]), ("/v1/chat/completions", "Bearer k", model))
            if backend == "glimmer":      # vLLM ignores reasoning_effort for Glimmer
                self.assertEqual(body["chat_template_kwargs"], {"reasoning_strength": "low"})
                self.assertNotIn("reasoning_effort", body)
            else:
                self.assertEqual(body["reasoning_effort"], "low")
            self.assertGreaterEqual(body["max_tokens"], 600)
            self.assertEqual((e.clip, e.chunk, e.track_ids, e.sam_conf), ("c1", [0.0, 3.0], ["hand:1", "tomato:2"], 0.7))

    def test_vision_failure_is_raised_not_swallowed(self):
        url, seen = self._serve([{"content": "nope"}, {"content": "still nope"}])
        with self.assertRaises(VisionError):
            Vision("glimmer", base_url=url).events([], [], 0, 3)
        self.assertEqual(len(seen), 2)

    def test_vision_idle_is_empty_list_and_request_errors_retry_once(self):
        v, ok = Vision("glimmer"), '[{"t_start":1,"t_end":3,"action":"cut","obj":"tomato"}]'
        with mock.patch("labbench.pipeline.vision.BACKOFF", 0):
            with mock.patch.object(Vision, "_chat", side_effect=["[]"]):
                self.assertEqual(v.events([], [], 0, 3), [])           # a valid empty array is idle
            err = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
            with mock.patch.object(Vision, "_chat", side_effect=[err, ok]) as chat:
                self.assertEqual(len(v.events([], [], 0, 3)), 1)
                self.assertEqual(chat.call_count, 2)
            with mock.patch.object(Vision, "_chat", side_effect=[err, err]), self.assertRaises(VisionError):
                v.events([], [], 0, 3)

    def test_sam_retries_a_failed_concept_once_and_reports_it(self):
        sam, errors = MetaSam(api_key="k"), []
        sam.backoff = 0
        calls = []

        def req(data_url, concept):
            calls.append(concept)
            if concept == "plate" or (concept == "hand" and calls.count("hand") == 1):
                raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
            return ""
        chunk = pathlib.Path(tempfile.mkdtemp()) / "c.mp4"
        chunk.write_bytes(b"x")
        with mock.patch.object(sam, "_request", req):
            self.assertEqual(sam.track(chunk, ["hand", "plate"], 0.0, errors), [])
        self.assertEqual(sorted(calls), ["hand", "hand", "plate", "plate"])     # both retried
        self.assertEqual(len(errors), 1)                                         # only plate failed twice
        self.assertIn("plate", errors[0])


if __name__ == "__main__":
    unittest.main()
