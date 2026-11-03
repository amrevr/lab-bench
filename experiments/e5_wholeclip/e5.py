"""E5: whole clip at low fps (tiled timestamped grids) + written protocol -> Verdict-shaped dicts, scored without realign.
Usage: PYTHONPATH=<pillow dir> e5.py frames|single|twopass R   (single/twopass call Glimmer: wrap in glimmer_queue.sh)
Output: SCRATCH/<mode>_r<R>/<clip>.session.json (+ raw replies in <clip>.raw.json). Frames cached in SCRATCH/frames/<clip>/."""
import base64, json, pathlib, re, subprocess, sys, urllib.request, concurrent.futures as cf
from PIL import Image, ImageDraw, ImageFont
ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRATCH = pathlib.Path("/home/agent/.claude/jobs/50da9733/tmp/exp/e5")
CLIPS = pathlib.Path("/Users/r/SharedContainer/lab-bench/.claude/worktrees/review-findings/data/clips")
FF = "/home/agent/.local/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2"
URL = "http://localhost:8010/v1/chat/completions"
SESS = {"12": "/home/agent/.claude/jobs/a4f6fa6d/tmp/errclips", "8": "/home/agent/.claude/jobs/a4f6fa6d/tmp/r8_base"}
FONT = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 22)
CALLS = [0]

def duration(c):
    o = subprocess.run([FF, "-i", str(CLIPS / f"{c}.mp4")], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", o).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)

def grids(c, t0, t1, n_grids, side, tw=320, th=180):
    """n_grids images of side x side frames evenly spaced over [t0,t1], each frame stamped with its time."""
    n = n_grids * side * side
    ts = [t0 + (i + .5) * (t1 - t0) / n for i in range(n)]
    d = SCRATCH / "frames" / c; d.mkdir(parents=True, exist_ok=True)
    out = []
    for g in range(n_grids):
        p = d / f"{t0:.0f}_{t1:.0f}_{side}_{g}.jpg"
        if not p.exists():
            im = Image.new("RGB", (tw * side, th * side))
            for k, t in enumerate(ts[g * side * side:(g + 1) * side * side]):
                f = d / f"f{t:.1f}.jpg"
                if not f.exists():
                    subprocess.run([FF, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(CLIPS / f"{c}.mp4"), "-frames:v", "1",
                                    "-vf", f"scale={tw}:{th}", str(f)], check=True)
                fr = Image.open(f); dr = ImageDraw.Draw(fr)
                dr.rectangle([0, 0, 92, 28], fill=(0, 0, 0)); dr.text((4, 2), f"{t:5.0f}s", fill=(255, 255, 0), font=FONT)
                im.paste(fr, ((k % side) * tw, (k // side) * th))
            im.save(p, quality=85)
        out.append((p, ts[g * side * side], ts[min((g + 1) * side * side, n) - 1]))
    return out

def protocol_text(R):
    P = json.loads((ROOT / "fixtures" / f"recipe{R}" / "protocol.json").read_text())
    L = []
    for s in P["steps"]:
        x = f"- step_id {s['id']}: {s.get('recipe_text') or s['label']}"
        if s.get("technique"): x += f" (technique: {s['technique']})"
        if s.get("after"): x += f" [must come after step_id {', '.join(map(str, s['after']))}]"
        if s.get("group"): x += f" [same group '{s['group']}': order within the group does not matter]"
        L.append(x)
    return P, "\n".join(L)

KINDS = ("kinds: skipped_step = a protocol step was never done; wrong_order = a step done at the wrong point (report the step that is out of place);"
         " wrong_technique = a step done with the wrong method (cut shape, how it was added/mixed/handled, wrong tool use); wrong_item = wrong ingredient or utensil;"
         " also use wrong_technique for a visible preparation mishap (spill, drop, wrong amount of mess).")
OUT = ('Reply with ONLY JSON: {"timeline":[{"step_id":<int>,"t_start":<s>,"t_end":<s>}, ...] (every protocol step you saw performed, in the order performed),'
       ' "errors":[{"kind":"skipped_step|wrong_order|wrong_technique|wrong_item","step_id":<int>,"t":<seconds when it happened>,"message":"<short>"}, ...]}.'
       " Use [] for errors if the person followed the protocol. Report only errors you can see in the frames; the person may skip, repeat or reorder steps.")

def single_prompt(R, ptxt, dur, gr):
    head = (f"You watch a person following a written recipe in one cooking video of {dur:.0f} s. The whole video is given as {len(gr)} images; each image is a grid of "
            "timestamped thumbnails (row by row, left to right, time increasing; the yellow label is the time in seconds). "
            f"Protocol:\n{ptxt}\n\nFind procedural errors the person makes relative to the protocol. {KINDS}\n{OUT}")
    c = [{"type": "text", "text": head}]
    for p, a, b in gr:
        c += [{"type": "text", "text": f"Image covering {a:.0f}-{b:.0f}s:"}, img(p)]
    return c

def img(p):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}}

def chat(content, max_tokens, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = json.dumps({"model": "glimmer", "messages": msgs, "temperature": 0.1, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"reasoning_strength": "high"}}).encode()
    CALLS[0] += 1
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1500) as r:
        j = json.load(r)
    ch = j["choices"][0]
    return ch["message"].get("content") or "", j.get("usage"), ch.get("finish_reason")

def parse(text):
    i = text.find("{")
    return json.JSONDecoder().raw_decode(text[i:])[0]

def ask(content, max_tokens):
    """One call, one retry on empty/unparseable (counts as a call)."""
    for a in range(2):
        try:
            txt, use, fin = chat(content, max_tokens)
            return parse(txt), {"usage": use, "finish": fin, "text": txt}
        except Exception as x:
            err = repr(x)[:300]
    return None, {"error": err}

def to_session(c, R, P, dur, ans):
    ids = {s["id"] for s in P["steps"]}
    tl = [e for e in (ans or {}).get("timeline", []) if isinstance(e, dict) and e.get("step_id") in ids]
    ev = [{"t_start": float(min(max(e.get("t_start", 0), 0), dur - .5)), "t_end": float(min(max(e.get("t_end", 0), 0), dur - .5)),
           "action": "step", "obj": str(e["step_id"]), "conf": .8} for e in tl]
    vs = []
    for e in (ans or {}).get("errors", []):
        if not isinstance(e, dict) or e.get("kind") not in ("skipped_step", "wrong_order", "wrong_technique", "wrong_item") or e.get("step_id") not in ids: continue
        t = dur if e["kind"] == "skipped_step" else float(min(max(e.get("t", 0), 0), dur))
        vs.append({"t": t, "step_id": e["step_id"], "kind": e["kind"], "conf": .8, "mode": "flag", "message": str(e.get("message", ""))[:200], "evidence": []})
    return {"clip": c, "events": ev, "verdicts": vs, "timeline": tl, "run": {"end": dur}}

def clips(R):
    src = {"12": "12_5", "8": "8_15"}[R]       # the protocol's expert run is not a test
    return [p.name.split(".")[0] for p in sorted(pathlib.Path(SESS[R]).glob("*.session.json")) if p.name.split(".")[0] != src]

def run_single(R, c, P, ptxt):
    out = SCRATCH / f"single_r{R}"; out.mkdir(exist_ok=True)
    if (out / f"{c}.session.json").exists(): return
    dur = duration(c)
    ans, raw = ask(single_prompt(R, ptxt, dur, grids(c, 0, dur, 4, 4, 288, 162)), 9000)
    (out / f"{c}.raw.json").write_text(json.dumps(raw))
    if ans: (out / f"{c}.session.json").write_text(json.dumps(to_session(c, R, P, dur, ans)))   # a failed clip is retried next run
    print(c, "ok" if ans else "FAILED", raw.get("usage"), raw.get("finish"), flush=True)

P1 = ("You watch part of a cooking video ({a:.0f}-{b:.0f}s of {dur:.0f}s), given as {n} images, each a grid of timestamped thumbnails (row by row, left to right, time increasing; "
      "yellow label = seconds). The person follows this written recipe:\n{ptxt}\n\nWrite a precise timeline of what the person actually does in this part: each action with t_start, t_end, "
      "which protocol step_id it corresponds to (null if none), and how it was done (object, tool, technique/shape of cuts, amounts, anything unusual: wrong ingredient, spill, wrong order, skipped, repeated). "
      'Describe what you see, do not assume the recipe is followed. Reply with ONLY JSON: {{"events":[{{"t_start":<s>,"t_end":<s>,"step_id":<int|null>,"what":"<short>"}}]}}.')
P2 = ("A person followed this written recipe in a cooking video of {dur:.0f}s:\n{ptxt}\n\nBelow is an observer's timeline of what they actually did (from three consecutive parts of the video). "
      "Decide which procedural errors occurred. {kinds}\n{out}\n\nObserver timeline:\n{tl}")

def run_two(R, c, P, ptxt):
    out = SCRATCH / f"twopass_r{R}"; out.mkdir(exist_ok=True)
    if (out / f"{c}.session.json").exists(): return
    dur = duration(c); segs = [(dur * i / 3, dur * (i + 1) / 3) for i in range(3)]
    def seg(ab):
        a, b = ab
        gr = grids(c, a, b, 4, 3, 320, 180)
        cont = [{"type": "text", "text": P1.format(a=a, b=b, dur=dur, n=4, ptxt=ptxt)}]
        for p, x, y in gr: cont += [{"type": "text", "text": f"Image covering {x:.0f}-{y:.0f}s:"}, img(p)]
        return ask(cont, 6000)
    with cf.ThreadPoolExecutor(2) as ex: res = list(ex.map(seg, segs))
    tl = "\n".join(f"{e.get('t_start')}-{e.get('t_end')}s step {e.get('step_id')}: {e.get('what')}" for a, _ in res if a for e in a.get("events", []) if isinstance(e, dict))
    ans, raw = ask([{"type": "text", "text": P2.format(dur=dur, ptxt=ptxt, kinds=KINDS, out=OUT, tl=tl or "(nothing)")}], 8000)
    (out / f"{c}.raw.json").write_text(json.dumps({"pass1": [r for _, r in res], "tl": tl, "pass2": raw}))
    if ans: (out / f"{c}.session.json").write_text(json.dumps(to_session(c, R, P, dur, ans)))
    print(c, "ok" if ans else "FAILED", [bool(a) for a, _ in res], raw.get("usage"), flush=True)

if __name__ == "__main__":
    mode, R = sys.argv[1], sys.argv[2]; only = sys.argv[3:]
    P, ptxt = protocol_text(R)
    cs = only or clips(R)
    if mode == "frames":
        for c in cs: dur = duration(c); grids(c, 0, dur, 4, 4, 288, 162); [grids(c, dur * i / 3, dur * (i + 1) / 3, 4, 3) for i in range(3)]; print(c, dur)
    else:
        f = run_single if mode == "single" else run_two
        if mode == "single":
            with cf.ThreadPoolExecutor(2) as ex: list(ex.map(lambda c: f(R, c, P, ptxt), cs))
        else:
            for c in cs: f(R, c, P, ptxt)     # 2 workers inside each clip (pass 1 segments)
    print("calls this run:", CALLS[0])
