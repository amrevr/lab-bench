"""Suggest flag/ask thresholds from scored verdicts. python scripts/calibrate.py [--root DIR] [--manifest M]

Runs every clip under --root with thresholds off (so silent verdicts show up too), scores it with
labbench.eval, prints each verdict's kind, conf and hit / false alarm, then per threshold group:
flag just above the top false alarm, ask just below the lowest true hit. If a false alarm is at or
above the lowest hit the groups do not separate: use one ask band. Tuned on few clips: no recall claim.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from labbench.pipeline import align  # noqa: E402
from labbench.eval import score_root  # noqa: E402
from labbench.schema import ASK_THRESHOLD, FLAG_THRESHOLD, THRESHOLDS, VerdictKind  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=pathlib.Path, default=ROOT / "fixtures" / "synthetic")
ap.add_argument("--manifest", type=pathlib.Path)
a = ap.parse_args()
# thresholds alone leave the aligner's own gates on; open them too so weak verdicts get scored
align.STRONG, align.MIN_CONF, align.ITEM_NEEDS_REPEAT = 0.0, 0.0, False
results = score_root(a.root, a.manifest, thresholds={k: (0.0, 0.0) for k in VerdictKind})
rows = [(r["clip"], v) for r in results for v in r["verdicts"]]
for clip, v in rows:
    print(f'{clip:6s} {v["kind"]:16s} step {v["step_id"]:<4d} conf {v["conf"]:.2f}  {"hit" if v["hit"] else "FALSE ALARM"}')
strict = {k.value for k in THRESHOLDS}
for name, pick in (("technique / wrong_item", lambda k: k in strict), ("other kinds", lambda k: k not in strict)):
    vs = [v for _, v in rows if pick(v["kind"])]
    hits, fas = [v["conf"] for v in vs if v["hit"]], [v["conf"] for v in vs if not v["hit"]]
    cur = THRESHOLDS.get(VerdictKind.WRONG_TECHNIQUE, (FLAG_THRESHOLD, ASK_THRESHOLD)) if name[0] == "t" else (FLAG_THRESHOLD, ASK_THRESHOLD)
    line = f"{name}: {len(hits)} hits, {len(fas)} false alarms; current flag/ask {cur[0]}/{cur[1]}"
    if not hits:
        print(line + "; no hits, no suggestion")
    elif fas and max(fas) >= min(hits):
        print(line + f"; do not separate (top false alarm {max(fas):.2f} >= lowest hit {min(hits):.2f}): use a single ask band")
    else:
        flag = round(max(fas) + 0.01, 2) if fas else cur[0]
        print(line + f"; suggest flag {flag}, ask {round(min(hits) - 0.01, 2)}")
print(f"tuned on {len(results)} clips; do not quote as a general recall claim")
