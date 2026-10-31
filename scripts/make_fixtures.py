"""Generate SYNTHETIC perception-event fixtures for recipe 12.

These stand in for Glimmer output until real clips can be fetched. They are NOT CaptainCook4D data.
Each clip has events.json (pipeline input) and answer_key.json (eval only, same role as the dataset's
error annotations). Usage: python scripts/make_fixtures.py
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "synthetic"

# (action, obj, target, technique)
CANON = [
    ("cut", "tomato", None, "slice"),
    ("cut", "mozzarella", None, "slice"),
    ("place", "tomato", "plate", None),
    ("place", "basil", "plate", None),
    ("pour", "olive oil", "plate", None),
    ("sprinkle", "salt", "plate", None),
]


def build(cid, steps, conf=None):
    events, t = [], 4.0
    for i, (a, o, tg, te) in enumerate(steps):
        c = (conf or {}).get(i, 0.9)
        events.append({"t_start": t, "t_end": t + 6, "action": a, "obj": o, "target": tg,
                       "technique": te, "conf": c, "sam_conf": 0.9, "track_ids": ["hand:1", f"{o}:1"],
                       "frame_refs": [t + 1, t + 4], "clip": cid, "chunk": [t, t + 3.0]})
        t += 8
    return events


clips = {
    "12_5": (CANON, {}, []),
    "12_43": ([s for i, s in enumerate(CANON) if i != 3], {},
              [{"type": "omission", "step_id": 4, "t": 36.0}]),
    "12_41": ([CANON[0], CANON[1], CANON[4], CANON[2], CANON[3], CANON[5]], {},
              [{"type": "order", "step_id": 5, "t": 28.0}]),
    "12_26": ([CANON[0], ("cut", "mozzarella", None, "tear")] + CANON[2:], {1: 0.8},
              [{"type": "technique", "step_id": 2, "t": 18.0}]),
    "12_38": (CANON[:4] + [("pour", "balsamic vinegar", "plate", None)] + CANON[5:], {},
              [{"type": "wrong_ingredient", "step_id": 5, "t": 42.0}]),
}
for cid, (steps, conf, key) in clips.items():
    d = OUT / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.json").write_text(json.dumps(build(cid, steps, conf), indent=1))
    (d / "answer_key.json").write_text(json.dumps({"clip": cid, "synthetic": True, "errors": key}, indent=1))
print("wrote", len(clips), "clips to", OUT)
