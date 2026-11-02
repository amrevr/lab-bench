"""E1: verification (candidates + decoys, yes/no/unclear) vs the current describe prompt, on sampled 3 s windows.
usage: e1.py prep R | e1.py run R COND[,COND..]   (COND = V1 V2 D; `run` calls Glimmer: wrap in glimmer_queue.sh)"""
import json, os, pathlib, random, sys, time, base64, site
sys.path.append(site.getusersitepackages())   # `python3 -I` drops user site-packages, where pydantic lives
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/r/SharedContainer/lab-bench/.claude/worktrees/design-leads")
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("FFMPEG", "/home/agent/.local/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2")
from labbench.pipeline import chunker                      # noqa: E402
from labbench.pipeline.align import Aligner, MIN_CONF      # noqa: E402
from labbench.pipeline.vision import Vision, parse_events  # noqa: E402
from labbench.schema import Protocol                       # noqa: E402

OUT = pathlib.Path("/home/agent/.claude/jobs/50da9733/tmp/exp/e1")
CLIPS = ROOT.parent / "review-findings/data/clips"
N, L = 100, 3.0

SYS1 = ("You watch short cooking-bench clips as keyframes. You get a list of candidate actions. For each candidate, decide "
        "whether it is being performed in these keyframes. Answer yes, no or unclear. Judge each candidate on its own; "
        "several, one or none may be true. Reply with ONLY a JSON array with one entry per candidate, in the order listed: "
        '[{"id":"A","answer":"yes|no|unclear","conf":0.0-1.0}]. conf in 0..1; lower it when unsure.')
SYS2 = ("You watch short cooking-bench clips as keyframes. You get a list of candidate actions. For each candidate, decide "
        "whether it is being performed in these keyframes. Answer yes only if you can see the hands and the named food or "
        "object doing exactly that in at least one keyframe; answer no if you see something else or nothing; unclear if the "
        "frames do not let you tell. Several, one or none may be true. Reply with ONLY a JSON array with one entry per "
        "candidate, in the order listed: "
        '[{"id":"A","evidence":"what you see for this candidate, max 10 words","answer":"yes|no|unclear","conf":0.0-1.0}]. conf in 0..1.')
SYS = {"V1": SYS1, "V2": SYS2}


def load(r):
    p = Protocol.model_validate_json((ROOT / f"fixtures/recipe{r}/protocol.json").read_text())
    ann = json.loads((ROOT / f"fixtures/recipe{r}/cc4d_error_annotations.json").read_text())
    return p, [a for a in ann if a["recording_id"] != p.model_dump().get("source_clip", "") and (CLIPS / f"{a['recording_id']}.mp4").exists()
               and a["recording_id"] != json.loads((ROOT / f"fixtures/recipe{r}/protocol.json").read_text())["source_clip"]]


def prep(r):
    p, recs = load(r)
    rng = random.Random(r)
    steps = [s for s in p.steps if s.visible]
    pos = {s.id: i for i, s in enumerate(steps)}
    segs = {a["recording_id"]: [(s["step_id"], s["start_time"], s["end_time"]) for s in a["step_annotations"] if s["start_time"] >= 0 and s["step_id"] in pos] for a in recs}
    def valid(rec, sid):   # segments of sid in rec, shrunk to where no other segment overlaps
        out = []
        for i, a, b in segs[rec]:
            if i != sid or b - a < L + 1:
                continue
            others = [(x, y) for j, x, y in segs[rec] if j != sid]
            lo, hi = a + .5, b - .5
            for x, y in others:
                if x < hi and y > lo:            # overlap: keep the larger remaining side
                    lo, hi = (lo, x) if x - lo >= hi - y else (y, hi)
            if hi - lo >= L:
                out.append((lo, hi))
        return out
    wins, k, used = [], 0, set()
    while len(wins) < N and k < 5000:
        k += 1
        s = steps[len(wins) % len(steps)]
        opts = [(rec, lo, hi) for rec in segs for lo, hi in valid(rec, s.id)]
        if not opts:
            if all(not any(valid(rec, t.id) for rec in segs) for t in steps if t.id == s.id):
                steps = [t for t in steps if t.id != s.id]; pos = {t.id: i for i, t in enumerate(steps)}
            continue
        rec, lo, hi = rng.choice(opts)
        t0 = round(rng.uniform(lo, hi - L), 1)
        if (rec, t0 // 6) in used:
            continue
        used.add((rec, t0 // 6))
        i = pos[s.id]
        near = sorted((t for t in steps if t.id != s.id), key=lambda t: (abs(pos[t.id] - i), rng.random()))
        is_pos = (len(wins) // len(steps)) % 2 == 0
        plaus = ([s] if is_pos else []) + near[:2 if is_pos else 3]
        decoys = rng.sample([t for t in near[3:]] or near[2:], min(2, len(near[3:]) or len(near[2:])))
        cands = [(t.id, "true" if t.id == s.id else "plausible") for t in plaus] + [(t.id, "decoy") for t in decoys]
        rng.shuffle(cands)
        wins.append({"i": len(wins), "rec": rec, "t0": t0, "t1": round(t0 + L, 1), "step": s.id, "pos": is_pos,
                     "cands": [{"id": "ABCDE"[j], "step": sid, "role": role} for j, (sid, role) in enumerate(cands)]})
    (OUT / f"windows{r}.json").write_text(json.dumps(wins, indent=1))
    print(r, len(wins), "windows", sum(w["pos"] for w in wins), "pos")


def frames(r, w):
    d = OUT / "frames" / f"{r}" / w["rec"]
    return chunker.keyframes(CLIPS / f"{w['rec']}.mp4", w["t0"], w["t1"], d, n=4)


def content(w, kf, tail=""):
    c = [{"type": "text", "text": f"Clip window {w['t0']:.1f}-{w['t1']:.1f}s. Keyframes follow with their timestamps."}]
    for t, p in kf:
        c += [{"type": "text", "text": f"t={t:.1f}s"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}}]
    return c + [{"type": "text", "text": tail}] if tail else c


def run_v(v, cond, p, w, kf):
    lab = {s.id: s.label for s in p.steps}
    tail = "Candidates:\n" + "\n".join(f"{c['id']}. {lab[c['step']]}" for c in w["cands"])
    msgs = [{"role": "system", "content": SYS[cond]}, {"role": "user", "content": content(w, kf, tail)}]
    t = time.time(); err = None
    for att in range(2):
        try:
            txt = v._chat(msgs, 900 * (att + 1))
            import re, json as j
            m = re.search(r"\[.*\]", txt, re.S)
            arr = j.JSONDecoder().raw_decode(m.group(0))[0]
            ans = {a["id"]: (str(a["answer"]).lower(), a.get("conf")) for a in arr}
            assert all(c["id"] in ans for c in w["cands"])
            return {"ans": ans, "sec": time.time() - t, "raw": txt}
        except Exception as x:
            err = f"{type(x).__name__}: {x}"[:200]
            msgs.append({"role": "user", "content": "That was not valid. Reply with only the JSON array."})
    return {"err": err, "sec": time.time() - t}


def run_d(v, p, w, kf):
    al = Aligner(p)
    t = time.time()
    try:
        evs = v.events(kf, [], w["t0"], w["t1"], w["rec"])
    except Exception as x:
        return {"err": str(x)[:200], "sec": time.time() - t}
    sec = time.time() - t
    det = set()
    for e in evs:
        e = al._normalise(e)
        if e.conf is not None and e.conf < MIN_CONF:
            continue
        s, how = al._find(e)
        if s is not None and how in ("exact", "technique"):   # 'item' = wrong object, not a detection of the step
            det.add(s.id)
    return {"det": sorted(det), "events": [e.model_dump(include={"action", "obj", "target", "technique", "conf"}) for e in evs], "sec": sec}


def run(r, conds):
    p, _ = load(r)
    wins = json.loads((OUT / f"windows{r}.json").read_text())
    names = sorted({n for st in p.steps for n in (st.obj, st.target) if n})
    vd = Vision("glimmer", base_url="http://localhost:8010/v1", names=names, steps=p.steps, issues=True)
    vv = Vision("glimmer", base_url="http://localhost:8010/v1")
    for cond in conds:
        res_f = OUT / f"res{r}_{cond}.json"
        done = json.loads(res_f.read_text()) if res_f.exists() else {}
        def one(w):
            if str(w["i"]) in done:
                return
            kf = frames(r, w)
            done[str(w["i"])] = run_d(vd, p, w, kf) if cond == "D" else run_v(vv, cond, p, w, kf)
            res_f.write_text(json.dumps(done))
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(one, wins))
        print(r, cond, "done", sum("err" in x for x in done.values()), "errors", flush=True)


def report(r, cond):
    wins = json.loads((OUT / f"windows{r}.json").read_text())
    res = json.loads((OUT / f"res{r}_{cond}.json").read_text())
    tp = n_t = fp = n_p = dy = n_d = unc = n_ans = err = 0
    by, lat = {}, []
    for w in wins:
        x = res.get(str(w["i"]))
        if not x or "err" in x:
            err += 1; continue
        lat.append(x["sec"])
        for c in w["cands"]:
            if cond == "D":
                yes, u = c["step"] in x["det"], False
            else:
                a = x["ans"][c["id"]][0]; yes, u = a == "yes", a == "unclear"
            n_ans += 1; unc += u
            if c["role"] == "true":
                n_t += 1; tp += yes; b = by.setdefault(c["step"], [0, 0]); b[0] += yes; b[1] += 1
            elif c["role"] == "plausible":
                n_p += 1; fp += yes
            else:
                n_d += 1; dy += yes
    lat.sort(); q = lambda f: round(lat[min(len(lat) - 1, int(f * len(lat)))], 1) if lat else None
    print(f"R{r} {cond}: windows={len(wins)} err={err} TPR={tp}/{n_t}={tp/max(n_t,1):.2f} FPR(plausible)={fp}/{n_p}={fp/max(n_p,1):.2f} "
          f"decoy-yes={dy}/{n_d}={dy/max(n_d,1):.2f} unclear={unc}/{n_ans}={unc/max(n_ans,1):.2f} p50={q(.5)}s p90={q(.9)}s")
    print("   by step TPR:", {k: f"{a}/{b}" for k, (a, b) in sorted(by.items())})


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "prep": prep(int(sys.argv[2]))
    elif c == "run": run(int(sys.argv[2]), sys.argv[3].split(","))
    else:
        for k in sys.argv[3].split(","): report(int(sys.argv[2]), k)
