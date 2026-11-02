"""E4: expert-vs-novice step comparison. Usage (inside glimmer_queue.sh, cwd backend):
 python experiments/e34_compare/e4.py RECIPE --variants v1,v2,base [--neg N]  -> TMP/e4_R<recipe>.jsonl (resumable)
 python experiments/e34_compare/e4.py RECIPE --report v1,base"""
import argparse, json, random, re, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from common import *
POS = {"Technique", "Preparation", "Measurement"}
CAT = {"Technique": "technique", "Preparation": "preparation", "Measurement": "amount"}
ap = argparse.ArgumentParser(); ap.add_argument("recipe"); ap.add_argument("--variants", default=""); ap.add_argument("--report", default="")
ap.add_argument("--neg", type=int, default=40); a = ap.parse_args()
fx = FX / f"recipe{a.recipe}"; proto = json.loads((fx / "protocol.json").read_text()); steps = {s["id"]: s for s in proto["steps"]}
recs = {r["recording_id"]: r for r in json.loads((fx / "cc4d_error_annotations.json").read_text())}
src = proto["source_clip"]; exp = {s["step_id"]: s for s in recs[src]["step_annotations"] if s["start_time"] >= 0}
out = TMP / f"e4_R{a.recipe}.jsonl"

def instances():
    inst = []
    for c, r in recs.items():
        if c == src: continue
        for s in r["step_annotations"]:
            if s["start_time"] < 0 or s["step_id"] not in exp or s["step_id"] not in steps: continue
            tg = sorted({e["tag"].replace(" Error", "") for e in s.get("errors", [])})
            lab = 1 if set(tg) & POS else 0 if not tg else None
            if lab is None: continue
            inst.append({"clip": c, "step": s["step_id"], "tags": tg, "label": lab, "t0": s["start_time"], "t1": s["end_time"],
                         "desc": "; ".join(e["description"] for e in s.get("errors", []) if e["tag"].replace(" Error", "") in POS)})
    pos = [i for i in inst if i["label"]]; neg = [i for i in inst if not i["label"]]
    random.Random(0).shuffle(neg)   # sample first, then drop clips with no local video, so the already-run set stays the same
    return sorted((i for i in pos + neg[:a.neg] if (CLIPS / f'{i["clip"]}.mp4').exists()), key=lambda i: (i["clip"], i["step"]))

SYS_CMP = "You compare how a novice performed one cooking step against an expert reference video of the same step. Frames of each are tiled left-to-right, top-to-bottom in time order."
Q = {
 "v1": ("Image 1 = expert doing the step. Image 2 = novice doing the same step.\nStep: {label} ({text}).\n"
        "Did the novice perform it the same way as the expert? Look at TECHNIQUE (how the action is done, e.g. slice vs chop, sprinkle vs pour), "
        "PREPARATION (which ingredient variant or tool, how prepared, e.g. wet vs dry, clean vs dirty) and AMOUNT. "
        "Ignore camera angle, background, hands, lighting and speed.\n"
        'Reply with only JSON: {{"differs": <0-100 confidence that the novice did it differently>, "category": "technique|preparation|amount|none", "difference": "<one short sentence, empty if same>"}}', 500),
 "v2": ("Image 1 = expert doing the step. Image 2 = novice doing the same step.\nStep: {label} ({text}).\n"
        "First state what ingredient, tool and motion the expert used and in what state (one sentence), then the same for the novice, then compare. "
        "Assume the novice did it the SAME way unless you can see a clear difference in the ingredient/tool used, its preparation or state, the way the motion is done, or the amount. "
        "Ignore camera angle, background, lighting, speed and which hand.\n"
        'Reply with only JSON: {{"expert": "...", "novice": "...", "differs": <0-100 confidence the novice did it differently>, "category": "technique|preparation|amount|none", "difference": "<empty if same>"}}', 700),
 "base": ("Image = a novice doing a cooking step.\nStep: {label}. Written description: \"{text}\".\n"
          "Was the step done correctly according to this description? Check the right ingredient (and variant), the right tool, the right technique (how the action is done), its preparation, and the amount. "
          "Ignore camera angle, background, lighting and speed.\n"
          'Reply with only JSON: {{"differs": <0-100 confidence the step was NOT done correctly>, "category": "technique|preparation|amount|none", "difference": "<one short sentence, empty if correct>"}}', 500)}

def run(v, i):
    st = steps[i["step"]]; q, mt = Q[v]; text = st.get("recipe_text") or st["label"]
    q = q.format(label=st["label"], text=text)
    nov = tile(i["clip"], i["t0"], i["t1"])
    parts = ([q, "Expert:", tile(src, exp[i["step"]]["start_time"], exp[i["step"]]["end_time"]), "Novice:", nov] if v != "base" else [q, nov])
    j, raw, dt = chat(SYS_CMP if v != "base" else "You check whether a cooking step in a video was done correctly. Frames are tiled left-to-right, top-to-bottom in time order.", parts, mt)
    try: sc = float((j or {}).get("differs"))
    except (TypeError, ValueError): sc = None
    return {"v": v, **{k: i[k] for k in ("clip", "step", "tags", "label", "desc")}, "score": sc, "cat": (j or {}).get("category"),
            "diff": (j or {}).get("difference"), "sec": round(dt, 1), "raw": None if j else raw[:200]}

def report(vs):
    rows = list({(r["v"], r["clip"], r["step"]): r for r in map(json.loads, out.read_text().splitlines())}.values())
    for v in vs:
        R = [r for r in rows if r["v"] == v]; ok = [r for r in R if r["score"] is not None]
        P = [r["score"] for r in ok if r["label"]]; N = [r["score"] for r in ok if not r["label"]]
        line = f"{v:5s} n={len(R)} (parse fail {len(R)-len(ok)}) pos={len(P)} neg={len(N)} AUC={auc(P,N):.2f}"
        for t in ("Technique", "Preparation", "Measurement"):
            Pt = [r["score"] for r in ok if t in r["tags"]]; line += f" | {t[:4]} n={len(Pt)} AUC={auc(Pt,N):.2f}"
        line += " | P@R30=%.2f P@R50=%.2f P@R70=%.2f" % tuple(prec_at_recall(P, N, x) for x in (.3, .5, .7))
        th = 50; tp = sum(p >= th for p in P); fp = sum(n >= th for n in N)
        line += f" | @score>=50: recall {tp}/{len(P)} FP {fp}/{len(N)} | mean sec {sum(r['sec'] for r in R)/max(len(R),1):.1f}"
        print(line)
        hit = [r for r in ok if r["label"] and r["score"] >= 50]
        catm = sum(r["cat"] in {CAT[t] for t in r["tags"] if t in CAT} for r in hit)
        stop = set("the a an of to and in on with it is was not instead used use did".split())
        tok = lambda s: {w[:5] for w in re.findall(r"[a-z]{4,}", (s or "").lower())} - {w[:5] for w in stop}
        lex = sum(bool(tok(r["diff"]) & tok(r["desc"])) for r in hit)
        print(f"      flagged positives {len(hit)}: category matches tag {catm}, lexical overlap with annotation {lex}")
        for r in hit: print(f"      [{r['clip']} s{r['step']} {','.join(r['tags'])}] ann: {r['desc'][:70]} || model: {r['cat']}: {(r['diff'] or '')[:90]}")

if a.report: report(a.report.split(",")); sys.exit()
done = {(r["v"], r["clip"], r["step"]) for r in map(json.loads, out.read_text().splitlines()) if r["score"] is not None} if out.exists() else set()  # failed rows get retried; report() keeps the last row per key
inst = instances(); print(len(inst), "instances", sum(i["label"] for i in inst), "pos", flush=True)
for v in a.variants.split(","):
    todo = [i for i in inst if (v, i["clip"], i["step"]) not in done]
    with ThreadPoolExecutor(4) as ex, out.open("a") as f:
        for r in ex.map(lambda i: run(v, i), todo):
            f.write(json.dumps(r) + "\n"); f.flush()
    print(v, "done; calls so far", CALLS[0], flush=True)
