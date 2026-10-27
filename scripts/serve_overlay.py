"""Serve the overlay with a session. python scripts/serve_overlay.py --clip 12_43 [--video data/clips/12_43.mp4]
Opens http://localhost:8765/overlay/?session=/out/<clip>.session.json"""
import argparse, functools, http.server, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from labbench.run import run_clip  # noqa: E402
from labbench.pipeline.wonder import coach_text  # noqa: E402
from labbench.schema import Verdict  # noqa: E402
import json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clip", default="12_43"); ap.add_argument("--root", default=str(ROOT / "fixtures" / "synthetic"))
ap.add_argument("--video"); ap.add_argument("--port", type=int, default=8765)
a = ap.parse_args()
s = run_clip(pathlib.Path(a.root), a.clip)
for v in s["verdicts"]:
    v["coach"] = coach_text(Verdict(**v))
out = ROOT / "out"; out.mkdir(exist_ok=True)
(out / f"{a.clip}.session.json").write_text(json.dumps(s, indent=1))
vid = f"&video=/{a.video}" if a.video else ""
print(f"http://localhost:{a.port}/overlay/?session=/out/{a.clip}.session.json{vid}")
os.chdir(ROOT)
http.server.ThreadingHTTPServer(("", a.port), http.server.SimpleHTTPRequestHandler).serve_forever()
