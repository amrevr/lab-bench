"""E4-lite (no GPU): expert-vs-novice V-JEPA distance per step instance; AUC for Technique/Preparation/Measurement-error steps.
Usage: PYTHONPATH=<numpy> python3 experiments/e4_vjepa.py FEATS_DIR RECIPE..."""
import json, pathlib, sys
import numpy as np
FX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
HOW = {"Technique", "Preparation", "Measurement"}

def seg_mean(z, a, b):
    m = (z["seg_t"] >= a - 0.5) & (z["seg_t"] < b)
    if not m.any(): return None
    v = z["seg_emb"][m].astype(np.float32).mean(0); return v / np.linalg.norm(v)

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))

feats, pooled = pathlib.Path(sys.argv[1]), {"expert": ([], []), "consensus": ([], []), "expert_z": ([], [])}
for R in sys.argv[2:]:
    proto = json.loads((FX / f"recipe{R}/protocol.json").read_text()); exp_id = proto["source_clip"]
    recs = {r["recording_id"]: r for r in json.loads((FX / f"recipe{R}/cc4d_error_annotations.json").read_text())}
    Z = {c: np.load(feats / f"{c}.npz") for c in recs if (feats / f"{c}.npz").exists()}
    inst = []   # (clip, step, vec, is_how_error)
    for c, z in Z.items():
        for st in recs[c]["step_annotations"]:
            if st["start_time"] < 0: continue
            v = seg_mean(z, st["start_time"], st["end_time"])
            tags = {e["tag"].replace(" Error", "") for e in st.get("errors", [])}
            if v is not None: inst.append((c, st["step_id"], v, bool(tags & HOW)))
    exp = {s: v for c, s, v, _ in inst if c == exp_id}
    res = {k: ([], []) for k in pooled}
    by_step = {}
    for c, s, v, y in inst:
        if c == exp_id or s not in exp: continue
        others = [w for c2, s2, w, _ in inst if s2 == s and c2 != c]
        cons = np.mean(others, 0); cons /= np.linalg.norm(cons)
        d_exp, d_con = 1 - float(v @ exp[s]), 1 - float(v @ cons)
        res["expert"][y].append(d_exp) if False else res["expert"][0 if y else 1].append(d_exp)
        res["consensus"][0 if y else 1].append(d_con)
        by_step.setdefault(s, []).append((d_exp, y))
    for s, L in by_step.items():   # per-step z-score: some steps just vary more
        d = np.array([x for x, _ in L]); mu, sd = d.mean(), d.std() + 1e-6
        for x, y in L: res["expert_z"][0 if y else 1].append((x - mu) / sd)
    n_pos, n_neg = len(res["expert"][0]), len(res["expert"][1])
    print(f"recipe {R} (expert {exp_id}): {n_pos} how-error steps vs {n_neg} clean | " +
          " | ".join(f"{k} AUC {auc(*res[k]):.2f}" for k in res))
    for k in pooled: pooled[k][0].extend(res[k][0]); pooled[k][1].extend(res[k][1])
print("pooled (raw values across recipes; z is per-step):", " | ".join(f"{k} AUC {auc(*pooled[k]):.2f}" for k in pooled))
