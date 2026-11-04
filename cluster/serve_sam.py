"""SAM 3.1 text-prompt video tracking service for `replay --sam-url`. POST /track?prompts=a,b&fps=10[&stride=2] with a raw mp4 body.

Needs the `sam3` package and the gated facebook/sam3.1 checkpoint. Run on the GPU host:
  SAM_CKPT=/path/sam3.1_multiplex.pt SAM_PORT=18081 python serve_sam.py
Optional: off by default in the pipeline, and no verdict reads its tracks (docs/measurements.md)."""
import sys, json, os, shutil, tempfile, time, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import cv2
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor

CKPT = os.environ.get("SAM_CKPT", "sam3.1_multiplex.pt")
TMP = os.environ.get("SAM_TMP", os.path.join(tempfile.gettempdir(), "sam-serve"))
USE_FA3 = os.environ.get("SAM_FA3", "0") == "1"
sem = threading.Semaphore(int(os.environ.get("SAM_CONC", "3")))  # bounds GPU memory
os.makedirs(TMP, exist_ok=True)
pred = build_sam3_multiplex_video_predictor(checkpoint_path=CKPT, use_fa3=USE_FA3)

# add_prompt toggles model.use_batched_grounding on the shared model; make it thread-local so
# concurrent requests never see each other's temporary False.
_m, _tl, _dflt = pred.model, threading.local(), pred.model.use_batched_grounding
vars(_m).pop("use_batched_grounding", None)
type(_m).use_batched_grounding = property(lambda s: getattr(_tl, "v", _dflt), lambda s, v: setattr(_tl, "v", v))

def sub_frames(path, stride, d):
    cap, i, n = cv2.VideoCapture(path), 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            cv2.imwrite(f"{d}/{n:05d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
        i += 1
    assert n, "no frames decoded"

def track(path, prompts, fps, stride=1):
    # SAM 3.1 add_prompt holds ONE text per session (it calls reset_state): one pass per prompt,
    # but the session (decoded frames) is shared across prompts.
    if stride > 1:
        d = tempfile.mkdtemp(dir=TMP)
        try:
            sub_frames(path, stride, d)
            return track(d, prompts, fps / stride)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    sid = pred.handle_request(dict(type="start_session", resource_path=path))["session_id"]
    tracks = []
    try:
        for label in prompts:
            pred.handle_request(dict(type="reset_session", session_id=sid))
            pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=label))
            per = {}  # obj_id -> {conf, boxes}
            try:
                for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                    o, t = r["outputs"], r["frame_index"] / fps
                    for i, oid in enumerate(o["out_obj_ids"].tolist()):
                        if not o["out_binary_masks"][i].any():
                            continue
                        x, y, w, h = [float(v) for v in o["out_boxes_xywh"][i]]
                        d = per.setdefault(oid, {"conf": 0.0, "boxes": []})
                        d["conf"] = max(d["conf"], float(o["out_probs"][i]))
                        d["boxes"].append([round(t, 3), x, y, x + w, y + h])
            except RuntimeError as e:
                if "No points are provided" not in str(e):  # prompt matched nothing: not an error
                    raise
            tracks += [{"label": label, "id": oid, **d} for oid, d in per.items()]
    finally:
        pred.handle_request(dict(type="close_session", session_id=sid))
    return tracks

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/track":
            return self.send_error(404)
        q = parse_qs(u.query)
        prompts = [p for p in q.get("prompts", ["hand,tomato,plate"])[0].split(",") if p]
        fps = float(q.get("fps", ["10"])[0])
        stride = max(1, int(q.get("stride", ["1"])[0]))
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        t0 = time.time()
        try:
            with sem, tempfile.NamedTemporaryFile(suffix=".mp4", dir=TMP) as f, \
                    torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                f.write(body); f.flush()
                tracks = track(f.name, prompts, fps, stride)
            out, code = {"tracks": tracks, "elapsed_s": round(time.time() - t0, 3)}, 200
        except Exception as e:
            traceback.print_exc(); sys.stderr.flush()
            out, code = {"error": repr(e)}, 500
        b = json.dumps(out).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

print("model loaded, serving", flush=True)
ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("SAM_PORT", "18081"))), H).serve_forever()
