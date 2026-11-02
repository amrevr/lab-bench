"""Score E5 sessions (verdict-shaped, NOT realigned) and the windowed pipeline's sessions (realigned).
Usage: score.py RECIPE DIR [--realign]   (DIR of *.session.json)"""
import json, pathlib, sys, collections
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from labbench.eval import score_timed, summarize_timed, realign, _strict, VISIBLE
from labbench.schema import Protocol

def tau(a, b):
    c = [x for x in a if x in b]; n = len(c)
    if n < 2: return None
    pos = {x: i for i, x in enumerate(b)}
    s = sum((1 if pos[c[i]] < pos[c[j]] else -1) for i in range(n) for j in range(i + 1, n))
    return s / (n * (n - 1) / 2)

def main(R, D, re_):
    fx = ROOT / "fixtures" / f"recipe{R}"
    proto = Protocol.model_validate_json((fx / "protocol.json").read_text())
    recs = {r["recording_id"]: r for r in json.loads((fx / "cc4d_error_annotations.json").read_text())}
    res, tag, hit, len_, fak = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    rec_n = rec_t = 0; taus = []
    for p in sorted(pathlib.Path(D).glob("*.session.json")):
        c = p.name.split(".")[0]
        if c == proto.source_clip or c not in recs: continue
        s = json.loads(p.read_text())
        if re_: s = realign(s, proto)
        r = score_timed(s, recs[c]); res.append(r)
        for row in r["rows"]:
            for t in row["tags"]: tag[t] += 1; hit[t] += row["strict"]; len_[t] += row["lenient"]
        for v in s["verdicts"]:
            if v["mode"] != "ok" and not any(_strict(v, row, 30.0, 1e9) for row in r["rows"]): fak[v["kind"]] += 1
        gt = sorted((st["start_time"], st["step_id"]) for st in recs[c]["step_annotations"]
                    if st["start_time"] >= 0 and not any(e["tag"] == "Missing Step" for e in st.get("errors", [])))
        gt_ids = [i for _, i in gt]
        if "timeline" in s:
            got = [e["step_id"] for e in sorted(s["timeline"], key=lambda e: e["t_start"])]
            got = list(dict.fromkeys(got))
        else:       # windowed pipeline: order of its "step done" verdicts
            got = list(dict.fromkeys(v["step_id"] for v in sorted(s["verdicts"], key=lambda v: v["t"]) if v["mode"] == "ok"))
        rec_n += len(gt_ids); rec_t += sum(i in got for i in gt_ids)
        t_ = tau(got, gt_ids)
        if t_ is not None: taus.append(t_)
    S = summarize_timed(res)
    print(f"clips {len(res)}  strict {S['steps_strict']} lenient {S['steps_lenient']}  FA mid {S['fa_mid']} end {S['fa_end']}  rec TPR {S['rec_tpr']} FPR {S['rec_fpr']} bF1 {S['rec_f1_balanced']}")
    print("by tag strict/lenient/n:", {t: f"{hit[t]}/{len_[t]}/{n}" for t, n in tag.most_common()})
    print("unmatched verdict kinds:", dict(fak))
    print(f"step recognition: listed {rec_t}/{rec_n} GT-performed steps; mean Kendall tau {sum(taus)/max(1,len(taus)):.2f} (n={len(taus)})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--realign" in sys.argv)
