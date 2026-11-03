"""E3b: non-leading LOCALIZATION second look for skipped_step alerts. No alert is mentioned: the model is asked where in
12 novice frames the step happens, or "none". A found step refutes the skip. A decoy step from the far end of the recipe
(not expected in the span) is asked the same way to measure false finds. cwd backend, inside glimmer_queue.sh:
 python ../experiments/e34_compare/e3b.py run | report"""
import json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, ".")
from common import *
import e3
from labbench.schema import Protocol
out = TMP / "e3b.jsonl"
SYS = ("You locate cooking steps in an egocentric video. The first image shows an expert doing the step. The next 3 images are the novice's video, "
       "12 frames in time order: each image is a 2x2 grid read left-to-right, top-to-bottom; frames are numbered 1-12 across the images.")

def ask(j, step_text, elo, ehi):
    ex = tile(j["src"], elo, ehi, 6, 3, 320)
    nv = [tile(j["clip"], j["lo"] + (j["hi"] - j["lo"]) * k / 3, j["lo"] + (j["hi"] - j["lo"]) * (k + 1) / 3, 4, 2, 400) for k in range(3)]
    q = (f"Step: '{step_text}'. In which novice frames (1-12), if any, is the novice clearly doing this step? It may not appear at all.\n"
         'Reply with only JSON: {"frames": [<frame numbers, empty if none>], "confidence": <0-100 that the step appears>}')
    r, raw, dt = chat(SYS, [q, "Expert:", ex, "Novice:"] + nv, 300)
    try: c = float((r or {}).get("confidence")); fr = list((r or {}).get("frames") or [])
    except (TypeError, ValueError): c, fr = None, []
    return c, fr, dt, (None if r else raw[:150])

def run(j):
    proto = Protocol.model_validate_json((FX / f"recipe{j['R']}/protocol.json").read_text())
    recs = {r["recording_id"]: r for r in json.loads((FX / f"recipe{j['R']}/cc4d_error_annotations.json").read_text())}
    exp = [s for s in recs[j["src"]]["step_annotations"] if s["start_time"] >= 0]
    far = max(exp, key=lambda s: abs(s["start_time"] - j["elo"]))          # decoy: the expert step furthest in time
    dlabel = next((s.label for s in proto.steps if s.id == far["step_id"]), far["description"])
    c, fr, dt, raw = ask(j, j["desc"], j["elo"], j["ehi"])
    dc, dfr, ddt, draw = ask(j, dlabel, far["start_time"], far["end_time"])
    return {**{k: j[k] for k in ("R", "clip", "kind", "step", "t", "tp", "fa", "lo", "hi")}, "conf": c, "frames": fr,
            "decoy_step": far["step_id"], "decoy_conf": dc, "decoy_frames": dfr, "sec": round(dt + ddt, 1), "raw": raw or draw}

def report():
    R = [r for r in map(json.loads, out.read_text().splitlines()) if r["conf"] is not None]
    tp = [r for r in R if r["tp"]]; fa = [r for r in R if r["fa"]]
    # skip-confidence = 100 - P(step appears): high means "really skipped"
    print(f"n={len(R)} TP={len(tp)} FA={len(fa)} | AUC(skip-conf, TP vs FA)={auc([100-r['conf'] for r in tp],[100-r['conf'] for r in fa]):.2f}")
    for th in (30, 50, 70):
        print(f"  found-if-conf>={th}: TP kept {sum(r['conf']<th for r in tp)}/{len(tp)}, FA dropped {sum(r['conf']>=th for r in fa)}/{len(fa)}, "
              f"decoy false-find {sum((r['decoy_conf'] or 0)>=th for r in R)}/{len(R)}")
    print("mean sec per alert (2 calls) %.1f" % (sum(r["sec"] for r in R) / max(1, len(R))))

if __name__ == "__main__":
    if sys.argv[1] == "report": report(); sys.exit()
    L = [j for j in e3.jobs() if j["kind"] == "skipped_step"]; print(len(L), "skip alerts", flush=True)
    k = lambda r: (r["R"], r["clip"], r["step"], r["t"])
    ok = [r for r in map(json.loads, out.read_text().splitlines()) if r["conf"] is not None] if out.exists() else []
    have = {k(r) for r in ok}; todo = [j for j in L if k(j) not in have]
    with ThreadPoolExecutor(3) as ex, out.open("w") as f:
        for r in ok: f.write(json.dumps(r) + "\n")
        for r in ex.map(run, todo): f.write(json.dumps(r) + "\n"); f.flush()
    print("done calls", CALLS[0])
