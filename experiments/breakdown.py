"""Per-tag strict catch breakdown + false-alarm kinds for a sessions dir (realigned). Usage: breakdown.py RECIPE SESSIONS"""
import json, pathlib, sys, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from labbench.eval import score_timed, realign, _strict
from labbench.schema import Protocol
R, S = sys.argv[1], pathlib.Path(sys.argv[2])
fx = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / f"recipe{R}"
proto = Protocol.model_validate_json((fx / "protocol.json").read_text())
recs = {r["recording_id"]: r for r in json.loads((fx / "cc4d_error_annotations.json").read_text())}
tag = collections.Counter(); hit = collections.Counter(); fak = collections.Counter()
for p in sorted(S.glob("*.session.json")):
    c = p.name.split(".")[0]
    if c == proto.source_clip or c not in recs: continue
    s = realign(json.loads(p.read_text()), proto); res = score_timed(s, recs[c])
    for r in res["rows"]:
        for t in r["tags"]: tag[t] += 1; hit[t] += r["strict"]
    vis = res["rows"]
    for v in s["verdicts"]:
        if v["mode"] != "ok" and not any(_strict(v, r, 30.0, 1e9) for r in vis): fak[v["kind"]] += 1
print("strict by tag:", {t: f"{hit[t]}/{n}" for t, n in tag.most_common()})
print("unmatched verdict kinds:", dict(fak))
