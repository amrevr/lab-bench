"""Score coaching verdicts against the annotation answer key.
python -m labbench.eval [--root DIR] [--manifest manifest.json]

Synthetic keys (type/step_id/t): caught = a flag or ask of the matching kind on the annotated step
within WINDOW seconds of the annotation time; report latency and whether it was an ask.
Real keys (answerkey.load_keys, per step, no timestamps): Missing Step -> skipped_step, Order ->
wrong_order, Technique -> wrong_technique must match kind; Measurement/Preparation/Temperature
count as caught if any non-ok verdict lands on the step. Reported per category.
Non-ok verdicts that match no annotation count as false alarms. The key is never pipeline input.
Quantities and heat are not visible to the pipeline, so Measurement/Temperature rates will be low.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from .answerkey import load_keys
from .run import ROOT, run_clip

WINDOW = 15.0
KIND = {"omission": "skipped_step", "order": "wrong_order", "technique": "wrong_technique",
        "wrong_ingredient": "wrong_item"}
STEP_KIND = {"Missing Step": "skipped_step", "Order": "wrong_order", "Technique": "wrong_technique"}


def score_clip(session: dict, key: dict) -> dict:
    verdicts = [v for v in session["verdicts"] if v["mode"] != "ok"]
    used, rows = set(), []
    if any("categories" in e for e in key["errors"]):
        for err in key["errors"]:
            for sid in err["step_ids"]:
                for cat in err["categories"]:
                    hits = [i for i, v in enumerate(verdicts) if v["step_id"] == sid
                            and (cat not in STEP_KIND or v["kind"] == STEP_KIND[cat])]
                    used.update(hits)
                    rows.append({"step_id": sid, "category": cat, "caught": bool(hits)})
    else:
        for err in key["errors"]:
            hit = None
            for i, v in enumerate(verdicts):
                if (i not in used and v["step_id"] == err["step_id"] and v["kind"] == KIND[err["type"]]
                        and abs(v["t"] - err["t"]) <= WINDOW):
                    hit = i
                    break
            if hit is None:
                rows.append({**err, "caught": False})
            else:
                used.add(hit)
                v = verdicts[hit]
                rows.append({**err, "caught": True, "mode": v["mode"], "latency_s": round(v["t"] - err["t"], 1)})
    return {"clip": key["clip"], "errors": rows, "false_alarms": len(verdicts) - len(used),
            "synthetic": key.get("synthetic", False),
            "verdicts": [{"kind": v["kind"], "conf": v["conf"], "mode": v["mode"], "step_id": v["step_id"],
                          "hit": i in used} for i, v in enumerate(verdicts)]}


def by_category(results: list[dict]) -> dict[str, tuple[int, int]]:
    """Real keys only: category -> (caught, total) over all clips."""
    out: dict[str, list[int]] = {}
    for r in results:
        for e in r["errors"]:
            if "category" in e:
                c = out.setdefault(e["category"], [0, 0])
                c[0] += e["caught"]
                c[1] += 1
    return {k: tuple(v) for k, v in out.items()}


def score_root(root: pathlib.Path, manifest: pathlib.Path | None = None, thresholds: dict | None = None) -> list[dict]:
    """Score every clip under root: synthetic answer_key.json files, or novice clips of a manifest."""
    kw = {"thresholds": thresholds} if thresholds else {}
    if manifest:
        keys = [k for k in load_keys(manifest).values() if k["role"] == "novice"]
        return [score_clip(run_clip(root, k["clip"], **kw), k) for k in keys if (root / k["clip"] / "events.json").exists()]
    return [score_clip(run_clip(root, d.name, **kw), json.loads((d / "answer_key.json").read_text()))
            for d in sorted(p for p in root.iterdir() if (p / "answer_key.json").exists())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--manifest", type=pathlib.Path, help="real CaptainCook4D manifest; --root holds protocol.json and <clip>/events.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    results = score_root(a.root, a.manifest)
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
            res = ("CAUGHT" + (f' {e["mode"]} @ {e["latency_s"]:+.1f}s' if "mode" in e else "")) if e["caught"] else "MISSED"
            print(f'  {e.get("type") or e["category"]:17s} step {e["step_id"]}: {res}')
    print(f"caught {caught}/{total}, false alarms {fa}")
    cats = by_category(results)
    for cat, (c, n) in cats.items():
        print(f"  {cat:13s} {c}/{n} ({c / n:.0%})")
    if cats:
        print("Quantities and heat are not visible to the pipeline; Measurement/Temperature catches are incidental.")


if __name__ == "__main__":
    main()
