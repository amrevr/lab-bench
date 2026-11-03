"""E0 oracle: ground-truth CaptainCook4D step segments -> perfect events -> current aligner -> timed scorer.
Optional perception noise: drop each true step with p=--drop, add --fp spurious protocol-step events per clip.
Usage: python experiments/oracle.py RECIPE OUTDIR [--drop P] [--fp N] [--seed S]"""
import argparse, json, pathlib, random
ap = argparse.ArgumentParser(); ap.add_argument("recipe"); ap.add_argument("out", type=pathlib.Path)
ap.add_argument("--drop", type=float, default=0.0); ap.add_argument("--fp", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed)
fx = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / f"recipe{a.recipe}"
proto = json.loads((fx / "protocol.json").read_text()); steps = {s["id"]: s for s in proto["steps"]}
a.out.mkdir(parents=True, exist_ok=True)
for r in json.loads((fx / "cc4d_error_annotations.json").read_text()):
    segs = [s for s in r["step_annotations"] if s.get("start_time", -1) >= 0 and s["step_id"] in steps]
    end = max([s["end_time"] for s in segs], default=60.0) + 10
    def ev(st, t0, t1):
        return {"t_start": round(t0, 2), "t_end": round(t1, 2), "action": st["action"], "obj": st["obj"], "target": st.get("target"),
                "technique": st.get("technique", ""), "conf": 1.0, "clip": r["recording_id"], "chunk": [t0, t1]}
    evs = [ev(steps[s["step_id"]], s["start_time"], min(s["start_time"] + 3, s["end_time"])) for s in segs if rng.random() >= a.drop]
    for _ in range(a.fp):
        t = rng.uniform(0, end - 10); evs.append(ev(steps[rng.choice(list(steps))], t, t + 3))
    evs.sort(key=lambda e: e["t_start"])
    sess = {"clip": r["recording_id"], "protocol": proto.get("name"), "events": evs, "verdicts": [], "run": {"end": end, "finish": True}}
    (a.out / f'{r["recording_id"]}.session.json').write_text(json.dumps(sess))
print(f"wrote {len(list(a.out.glob('*.session.json')))} sessions to {a.out}")
