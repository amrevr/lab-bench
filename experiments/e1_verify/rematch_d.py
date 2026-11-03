"""Re-match B's stored D-condition events with the CURRENT aligner (_normalise/_find), no model calls. Writes res<R>_<OUT>.json.
Usage: LB_E1=<dir> python rematch_d.py R OUTCOND"""
import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))
from labbench.pipeline.align import Aligner, MIN_CONF
from labbench.schema import Event, Protocol
R, OUTC = sys.argv[1], sys.argv[2]; D = pathlib.Path(os.environ["LB_E1"])
p = Protocol.model_validate_json((pathlib.Path(__file__).resolve().parents[2] / f"fixtures/recipe{R}/protocol.json").read_text())
res = json.loads((D / f"res{R}_D.json").read_text()); out = {}
for k, x in res.items():
    if "err" in x: out[k] = x; continue
    al = Aligner(p); det = set()
    for e in x["events"]:
        e = al._normalise(Event.model_validate({"t_start": 0, "t_end": 1, **e}))
        if e.conf is not None and e.conf < MIN_CONF: continue
        s, how = al._find(e)
        if s is not None and how in ("exact", "technique"): det.add(s.id)
    out[k] = {**x, "det": sorted(det)}
(D / f"res{R}_{OUTC}.json").write_text(json.dumps(out)); print("changed windows:", sum(set(out[k].get("det", [])) != set(res[k].get("det", [])) for k in res))
