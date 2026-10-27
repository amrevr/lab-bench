#!/usr/bin/env python3
"""Fake OpenAI-compatible streaming server for dry-running the benchmark without a GPU."""
import http.server
import json
import time

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        if self.path in ("/start_profile", "/stop_profile"):
            self.send_response(200); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        time.sleep(0.15)
        for tok in ['[{"t_start":170,', '"t_end":174,', '"action":"cut",', '"obj":"tomato",', '"conf":0.8}]']:
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": tok}}]}) + "\n\n").encode()); self.wfile.flush(); time.sleep(0.02)
        self.wfile.write(b"data: [DONE]\n\n")

if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
