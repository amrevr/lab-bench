"""E2 eval: current Aligner vs belief Tracker on identical saved events (realign-style, offline).
Usage: python experiments/e2_eval.py [--sets oracle,real,noise] [--tau .5 --k 2 --no-edge --both]"""
import argparse, collections, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "experiments"))
from labbench.eval import score_timed, summarize_timed
from labbench.pipeline.align import Aligner
from labbench.schema import Event, Protocol
from e2_tracker import Tracker

HELD12 = {"12_48", "12_51", "12_9", "12_13", "12_16"}
J = pathlib.Path("/home/agent/.claude/jobs")
ORA = J / "50da9733/tmp/exp/oracle"; REAL = J / "a4f6fa6d/tmp"


def run(cls, proto, sess, **kw):
    al = cls(proto, **kw)
    for e in sess["events"]:
        al.feed(Event.model_validate(e))
    al.finish(float(sess["run"]["end"]))
    return {**sess, "verdicts": [v.model_dump(mode="json") for v in al.verdicts]}


def evaluate(recipe, d, cls, clips=None, w=30.0, **kw):
    fx = ROOT / "fixtures" / f"recipe{recipe}"
    proto = Protocol.model_validate_json((fx / "protocol.json").read_text())
    recs = {r["recording_id"]: r for r in json.loads((fx / "cc4d_error_annotations.json").read_text())}
    res = []
    for p in sorted(pathlib.Path(d).glob("*.session.json")):
        c = p.name.split(".")[0]
        if c == proto.source_clip or c not in recs or (clips is not None and c not in clips):
            continue
        res.append(score_timed(run(cls, proto, json.loads(p.read_text()), **kw), recs[c], w))
    return res


def line(res):
    s = summarize_timed(res)
    tag = collections.defaultdict(lambda: [0, 0, 0])
    for x in res:
        for r in x["rows"]:
            for t in r["tags"]:
                tag[t][0] += r["strict"]; tag[t][1] += r["lenient"]; tag[t][2] += 1
    f = lambda t: f"{t[0]}/{t[1]}"
    keys = ["Order", "Missing Step", "Technique", "Preparation"]
    return (f"strict {f(s['steps_strict'])} len {f(s['steps_lenient'])} | " +
            " ".join(f"{k[:5]} {tag[k][0]}/{tag[k][2]}" for k in keys if k in tag) +
            f" | FA mid {s['fa_mid']} end {s['fa_end']} | rec TPR {f(s['rec_tpr'])} FPR {f(s['rec_fpr'])} F1 {s['rec_f1_balanced']}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", default="oracle,real,noise")
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--no-edge", action="store_true"); ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    kw = dict(tau=a.tau, k=a.k, per_edge=not a.no_edge, both_ends=a.both)
    jobs = []
    if "oracle" in a.sets:
        jobs += [(f"oracle r{n}", n, ORA / f"r{n}", None) for n in (12, 8, 10, 1)]
    if "real" in a.sets:
        jobs += [("real r12 tune", 12, REAL / "errclips", "tune"), ("real r12 held", 12, REAL / "errclips", "held"),
                 ("real r8 run1", 8, REAL / "r8_base", None), ("real r8 run2", 8, REAL / "r8_base2", None),
                 ("real r10", 10, REAL / "r10_base", None)]
    if "noise" in a.sets:
        for n in (12, 8, 10, 1):
            for sd in (0, 1, 2):
                jobs.append((f"noise r{n} s{sd}", n, ORA.parent / "e2" / f"noise_r{n}_s{sd}", None))
    for name, n, d, grp in jobs:
        clips = None
        if grp:
            allc = {p.name.split(".")[0] for p in pathlib.Path(d).glob("*.session.json")}
            clips = (allc & HELD12) if grp == "held" else (allc - HELD12)
        for lab, cls, k in (("current", Aligner, {}), ("tracker", Tracker, kw)):
            print(f"{name:14s} {lab:8s}", line(evaluate(n, d, cls, clips, **k)))

if __name__ == "__main__":
    main()
