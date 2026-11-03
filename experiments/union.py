"""Union of runs: first run's events plus other runs' events with no same (action, obj) within GAP s in the merged list.
Usage: union.py OUT RUN_DIR RUN_DIR..."""
import json, pathlib, sys
out, runs = pathlib.Path(sys.argv[1]), [pathlib.Path(p) for p in sys.argv[2:]]; GAP = 3.0; out.mkdir(parents=True, exist_ok=True)
key = lambda e: (e["action"].strip().lower(), e["obj"].strip().lower())
for p in sorted(runs[0].glob("*.session.json")):
    s = json.loads(p.read_text()); evs = list(s["events"])
    for r in runs[1:]:
        if not (r / p.name).exists(): continue
        for e in json.loads((r / p.name).read_text())["events"]:
            if not any(key(o) == key(e) and o["t_start"] - GAP <= e["t_end"] and e["t_start"] - GAP <= o["t_end"] for o in evs):
                evs.append(e)
    s["events"] = sorted(evs, key=lambda e: (e["chunk"][0] if e.get("chunk") else e["t_start"], e["t_start"]))
    s["run"]["end"] = max(float(s["run"]["end"]), *(e["t_end"] for e in evs)) if evs else s["run"]["end"]
    (out / p.name).write_text(json.dumps(s))
