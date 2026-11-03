"""E3: second-look confirm/refute on every non-ok verdict of saved sessions. Run inside glimmer_queue.sh, cwd backend:
 python experiments/e34_compare/e3.py run   |   python experiments/e34_compare/e3.py report"""
import json, pathlib, sys, glob, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, ".")
from common import *
from labbench.eval import realign, _strict, VISIBLE
from labbench.schema import Protocol
out = TMP / "e3.jsonl"; J = pathlib.Path("/home/agent/.claude/jobs/a4f6fa6d/tmp")
SYS = ("You are double-checking an alert from a cooking-video monitor. The first image is an expert doing the step; the remaining images are the novice's video "
       "over the relevant time span, 12 frames in time order (each image is 2x2, left-to-right then top-to-bottom, images in order).")
def jobs():
    L = []
    for R, d in ((8, "r8_base"), (12, "errclips")):
        fx = FX / f"recipe{R}"; proto = Protocol.model_validate_json((fx / "protocol.json").read_text())
        recs = {r["recording_id"]: r for r in json.loads((fx / "cc4d_error_annotations.json").read_text())}
        steps = {s.id: s for s in proto.steps}
        from labbench.eval import cc4d_rows
        exp = {s["step_id"]: s for s in recs[proto.source_clip]["step_annotations"] if s["start_time"] >= 0}
        for f in sorted(glob.glob(f"{J}/{d}/*.session.json")):
            c = pathlib.Path(f).name.split(".")[0]
            if c == proto.source_clip or c not in recs: continue
            s = realign(json.load(open(f)), proto); last_t = max(e["t_end"] for e in s["events"]); rows = cc4d_rows(recs[c])
            vis = [r for r in rows if set(r["tags"]) & VISIBLE]; hidden = {r["step_id"] for r in rows if not set(r["tags"]) & VISIBLE}
            oks = [v for v in s["verdicts"] if v["mode"] == "ok"]
            for i, v in enumerate(v for v in s["verdicts"] if v["mode"] != "ok"):
                st = steps.get(v["step_id"]); e = exp.get(v["step_id"])
                if not st or not e: continue
                tp = any(_strict(v, r, 30.0, last_t) for r in vis)
                if v["kind"] == "skipped_step":
                    pre = [o["closed_t"] for o in oks if o["closed_t"] is not None and exp.get(o["step_id"]) and exp[o["step_id"]]["start_time"] < e["start_time"]]
                    nxt = [o["evidence"][0]["t_start"] for o in oks if o["evidence"] and o["evidence"][0]["t_start"] is not None and exp.get(o["step_id"]) and exp[o["step_id"]]["start_time"] > e["start_time"]]
                    lo = max(pre, default=0.0); hi = min([x for x in nxt if x > lo], default=last_t); hi = min(max(hi, lo + 10), lo + 150, last_t)
                else:
                    ev = v["evidence"]; lo = max(0, min(x["t_start"] for x in ev) - 4); hi = max(x["t_end"] for x in ev) + 4; hi = max(hi, lo + 10)
                L.append({"R": R, "clip": c, "i": i, "kind": v["kind"], "step": v["step_id"], "t": v["t"], "msg": v["message"], "label": st.label,
                          "desc": getattr(st, "recipe_text", None) or st.label, "lo": lo, "hi": hi, "elo": e["start_time"], "ehi": e["end_time"],
                          "src": proto.source_clip, "tp": tp, "fa": (not tp) and v["step_id"] not in hidden})
    return L

def run(j):
    t0 = time.time()
    if j["kind"] == "skipped_step": claim = f"The monitor claims the novice SKIPPED the step '{j['desc']}' (never performed it, or not in this span)."
    elif j["kind"] == "wrong_order": claim = f"The monitor claims the novice did step '{j['desc']}' OUT OF ORDER: {j['msg']}"
    else: claim = f"The monitor claims the novice used a WRONG TECHNIQUE for step '{j['desc']}': {j['msg']}"
    ex = tile(j["src"], j["elo"], j["ehi"], 6, 3, 320)
    nv = [tile(j["clip"], j["lo"] + (j["hi"] - j["lo"]) * k / 3, j["lo"] + (j["hi"] - j["lo"]) * (k + 1) / 3, 4, 2, 400) for k in range(3)]
    te = time.time() - t0
    q = (f"{claim}\nNovice span {j['lo']:.0f}-{j['hi']:.0f}s. Look carefully at the novice frames. Does the claim hold?\n"
         'Reply with only JSON: {"verdict": "confirm|refute", "confidence": <0-100 that the claim is TRUE>, "reason": "<one sentence>"}')
    r, raw, dt = chat(SYS, [q, "Expert:", ex, "Novice:"] + nv, 400)
    try: c = float((r or {}).get("confidence"))
    except (TypeError, ValueError): c = None
    return {**{k: j[k] for k in ("R", "clip", "kind", "step", "t", "tp", "fa", "lo", "hi")}, "conf": c, "v": (r or {}).get("verdict"), "reason": (r or {}).get("reason"), "sec": round(dt, 1), "extract_sec": round(te, 1), "raw": None if r else raw[:150]}

def report():
    R = [json.loads(l) for l in out.read_text().splitlines()]
    for tag, sel in (("all", R), ("R8", [r for r in R if r["R"] == 8]), ("R12", [r for r in R if r["R"] == 12])):
        for kind in ("all", "skipped_step", "wrong_order", "wrong_technique"):
            S = [r for r in sel if (kind == "all" or r["kind"] == kind) and r["conf"] is not None]
            if not S: continue
            tp = [r for r in S if r["tp"]]; fa = [r for r in S if r["fa"]]
            line = f"{tag:3s} {kind:15s} n={len(S)} TP={len(tp)} FA={len(fa)} other={len(S)-len(tp)-len(fa)} | AUC(conf, TP vs FA)={auc([r['conf'] for r in tp],[r['conf'] for r in fa]):.2f}"
            for th in (30, 50, 70):    # drop verdict when confidence the claim is true < th
                line += f" | th{th}: TP kept {sum(r['conf']>=th for r in tp)}/{len(tp)}, FA dropped {sum(r['conf']<th for r in fa)}/{len(fa)}"
            print(line)
    print("mean call sec %.1f, mean frame-extract sec %.1f, parse fail %d/%d" % (sum(r["sec"] for r in R)/len(R), sum(r["extract_sec"] for r in R)/len(R), sum(r["conf"] is None for r in R), len(R)))

if __name__ == "__main__":
    if sys.argv[1] == "report": report(); sys.exit()
    L = jobs(); print(len(L), "verdicts", flush=True)
    k = lambda r: (r["R"], r["clip"], r["kind"], r["step"], r["t"])
    ok = [r for r in map(json.loads, out.read_text().splitlines()) if r["conf"] is not None] if out.exists() else []   # resume: keep answered rows
    have = {k(r) for r in ok}; todo = [j for j in L if k(j) not in have]; print(len(todo), "to (re)run", flush=True)
    with ThreadPoolExecutor(4) as ex, out.open("w") as f:
        for r in ok: f.write(json.dumps(r) + "\n")
        for r in ex.map(run, todo): f.write(json.dumps(r) + "\n"); f.flush()
    print("done calls", CALLS[0])
