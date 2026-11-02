"""Shared helpers for E3/E4: frame tiling (ffmpeg), Glimmer chat (1 call = 1 retry-aware request), AUC."""
import base64, json, os, pathlib, re, subprocess, threading, time, urllib.request
FFMPEG = os.environ.get("FFMPEG", "/home/agent/.local/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2")
CLIPS = pathlib.Path("/Users/r/SharedContainer/lab-bench/.claude/worktrees/review-findings/data/clips")
FX = pathlib.Path(__file__).resolve().parents[2] / "fixtures"
TMP = pathlib.Path("/home/agent/.claude/jobs/50da9733/tmp/exp/e34"); (TMP / "frames").mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8010/v1/chat/completions"
_lock = threading.Lock(); CALLS = [0]

def _run(cmd): subprocess.run(cmd, check=True, capture_output=True)

def tile(clip, t0, t1, n=6, cols=3, width=320):
    """n frames evenly over [t0,t1] tiled cols wide, row-major, one jpg. Missing tail frames are dropped."""
    key = TMP / "frames" / f"{clip}_{t0:.1f}_{t1:.1f}_{n}_{cols}_{width}.jpg"
    if key.exists(): return key
    d = TMP / "frames" / f"{key.stem}_parts"; d.mkdir(exist_ok=True); got = []
    for i in range(n):
        t = t0 + (t1 - t0) * (i + 0.5) / n; p = d / f"{i}.jpg"
        subprocess.run([FFMPEG, "-y", "-ss", f"{t:.2f}", "-i", str(CLIPS / f"{clip}.mp4"), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", "-q:v", "4", str(p)], capture_output=True)
        if p.exists() and p.stat().st_size: got.append(p)
    if not got: raise RuntimeError(f"no frames {clip} {t0}-{t1}")
    rows = -(-len(got) // cols)
    _run([FFMPEG, "-y", "-framerate", "1", "-start_number", "0", "-i", str(d / "%d.jpg"), "-vf", f"tile={cols}x{rows}", "-frames:v", "1", "-q:v", "4", str(key)])
    return key

def chat(system, parts, max_tokens=500):
    """parts: list of str | Path. Returns (parsed_json_or_None, raw, seconds). Counts every HTTP call."""
    content = []
    for p in parts:
        if isinstance(p, str): content.append({"type": "text", "text": p})
        else: content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    t0 = time.time(); raw = ""
    for attempt in range(2):
        body = json.dumps({"model": "glimmer", "messages": msgs, "temperature": 0.1, "max_tokens": max_tokens * (attempt + 1),
                           "chat_template_kwargs": {"reasoning_strength": "low"}}).encode()
        with _lock: CALLS[0] += 1
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=120) as r:
                raw = json.load(r)["choices"][0]["message"].get("content") or ""
        except Exception as x:
            raw = f"ERR {type(x).__name__}: {x}"; time.sleep(3); continue
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try: return json.loads(m.group(0)), raw, time.time() - t0
            except json.JSONDecodeError: pass
    return None, raw, time.time() - t0

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def prec_at_recall(pos, neg, rec):
    """Best precision over thresholds with recall >= rec."""
    best = 0.0
    for th in sorted(set(pos + neg)):
        tp = sum(p >= th for p in pos); fp = sum(n >= th for n in neg)
        if tp / len(pos) >= rec and tp + fp: best = max(best, tp / (tp + fp))
    return best
