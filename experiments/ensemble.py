"""Self-consistency: keep a run's events only if >= K other runs saw the same (action, obj) within GAP s.
Usage: ensemble.py OUT K RUN_DIR RUN_DIR... (writes OUT/<clip>.session.json filtered from the FIRST run dir)"""
import json, pathlib, sys
out, K, runs = pathlib.Path(sys.argv[1]), int(sys.argv[2]), [pathlib.Path(p) for p in sys.argv[3:]]
GAP = 3.0; out.mkdir(parents=True, exist_ok=True)
key = lambda e: (e["action"].strip().lower(), e["obj"].strip().lower())
for p in sorted(runs[0].glob("*.session.json")):
    others = [json.loads((r / p.name).read_text())["events"] for r in runs[1:] if (r / p.name).exists()]
    if len(others) < K: continue
    s = json.loads(p.read_text())
    def support(e):
        return sum(any(key(o) == key(e) and o["t_start"] - GAP <= e["t_end"] and e["t_start"] - GAP <= o["t_end"] for o in evs) for evs in others)
    s["events"] = [e for e in s["events"] if support(e) >= K]
    (out / p.name).write_text(json.dumps(s))
