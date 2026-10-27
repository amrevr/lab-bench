"""Score coaching verdicts against the annotation answer key. python -m labbench.eval [--root DIR]

Per clip: for each annotated error, caught = a flag or ask of the matching kind on the annotated
step within WINDOW seconds of the annotation time; report latency and whether it was an ask.
Non-ok verdicts that match no annotation count as false alarms. The key is never pipeline input.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from .run import ROOT, run_clip

WINDOW = 15.0
KIND = {"omission": "skipped_step", "order": "wrong_order", "technique": "wrong_technique",
        "wrong_ingredient": "wrong_item"}


def score_clip(session: dict, key: dict) -> dict:
    used, rows = set(), []
    for err in key["errors"]:
        hit = None
        for i, v in enumerate(session["verdicts"]):
            if (i not in used and v["step_id"] == err["step_id"] and v["kind"] == KIND[err["type"]]
                    and abs(v["t"] - err["t"]) <= WINDOW):
                hit = i
                break
        if hit is None:
            rows.append({**err, "caught": False})
        else:
            used.add(hit)
            v = session["verdicts"][hit]
            rows.append({**err, "caught": True, "mode": v["mode"], "latency_s": round(v["t"] - err["t"], 1)})
    false_alarms = [v for i, v in enumerate(session["verdicts"]) if i not in used]
    return {"clip": key["clip"], "errors": rows, "false_alarms": len(false_alarms),
            "synthetic": key.get("synthetic", False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    results = []
    for d in sorted(p for p in a.root.iterdir() if (p / "answer_key.json").exists()):
        key = json.loads((d / "answer_key.json").read_text())
        results.append(score_clip(run_clip(a.root, d.name), key))
    if a.json:
        print(json.dumps(results, indent=1))
        return
    caught = total = fa = 0
    for r in results:
        tag = " (synthetic)" if r["synthetic"] else ""
        print(f'{r["clip"]}{tag}: false alarms {r["false_alarms"]}')
        fa += r["false_alarms"]
        for e in r["errors"]:
            total += 1
            caught += e["caught"]
            res = f'CAUGHT {e["mode"]} @ {e["latency_s"]:+.1f}s' if e["caught"] else "MISSED"
            print(f'  {e["type"]:17s} step {e["step_id"]}: {res}')
    print(f"caught {caught}/{total}, false alarms {fa}")


if __name__ == "__main__":
    main()
