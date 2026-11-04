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
from .pipeline.align import Aligner
from .schema import Event, Protocol

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
    ap.add_argument("--cc4d", type=pathlib.Path, help="CaptainCook4D error_annotations.json: timed scoring of --sessions")
    ap.add_argument("--sessions", type=pathlib.Path, help="directory of <clip>.session.json (with --cc4d)")
    ap.add_argument("--held-out", default="", help="comma-separated clip ids reported separately (with --cc4d)")
    ap.add_argument("--realign", type=pathlib.Path, help="protocol.json: re-run the aligner (current settings) on each session's events")
    a = ap.parse_args()
    if a.cc4d:
        if not a.sessions:
            ap.error("--cc4d needs --sessions")
        timed_main(a.cc4d, a.sessions, set(filter(None, a.held_out.split(","))), a.json, a.realign)
        return
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


def realign(session: dict, protocol: Protocol) -> dict:
    """The session's verdicts as the current aligner would give them for the same events (no video, no model calls)."""
    al = Aligner(protocol)
    for e in session["events"]:
        al.feed(Event.model_validate(e))
    end = float(session["run"]["end"])
    if session["run"].get("finish", any(v["kind"] == "skipped_step" and v["t"] == end for v in session["verdicts"])):
        al.finish(end)    # only a session that was finished had end-of-clip skips (older ones: inferred from its verdicts)
    return {**session, "verdicts": [v.model_dump(mode="json") for v in al.verdicts]}


def timed_main(cc4d: pathlib.Path, sessions: pathlib.Path, held_out: set[str], as_json: bool,
               protocol: pathlib.Path | None = None) -> None:
    recs = {r["recording_id"]: r for r in json.loads(cc4d.read_text())}
    sess = {p.name.split(".")[0]: json.loads(p.read_text()) for p in sorted(sessions.glob("*.session.json"))}
    if protocol:
        proto = Protocol.model_validate_json(protocol.read_text())
        if sess.pop(proto.source_clip, None) is not None:     # the expert run the protocol learned from is not a test
            print(f"not scored: {proto.source_clip} (the protocol's expert run)")
        sess = {c: realign(s, proto) for c, s in sess.items()}
    missing = sorted(set(recs) - set(sess) - ({proto.source_clip} if protocol else set()))
    groups = {"tune": sorted(set(sess) - held_out), "held-out": sorted(set(sess) & held_out)}
    report = {g: {w: summarize_timed([score_timed(sess[c], recs[c], w) for c in clips if c in recs])
                  for w in (15.0, 30.0, 60.0)} for g, clips in groups.items() if clips}
    if as_json:
        print(json.dumps({"report": report, "no_session": missing}, indent=1))
        return
    f = lambda t: f"{t[0]}/{t[1]}"
    for g, by_w in report.items():
        print(f"== {g}: {', '.join(groups[g])}")
        for w, s in by_w.items():
            print(f"  W={w:>4.0f}s  steps strict {f(s['steps_strict'])} (in time {f(s['steps_strict_in_time'])}), "
                  f"lenient {f(s['steps_lenient'])} | false alarms mid-clip {s['fa_mid']}, at end {s['fa_end']} | "
                  f"recordings: error clips caught {f(s['rec_tpr'])}, normal clips with a mid-clip alarm {f(s['rec_fpr'])}, "
                  f"balanced F1 {s['rec_f1_balanced']}")
        s = by_w[30.0]
        print(f"  issue (mishap) verdicts: {s['issue_tp']} caught an annotated error, {s['issue_fa']} false alarms")
        print(f"  EgoPER EDA {f(s['eda'])}, omission acc {f(s['omission_acc'])}, omission IoU {f(s['omission_iou'])}; "
              f"delay to strict catch (s) {s['delay_s']}, observation ratio {s['obs_ratio']}")
    if missing:
        print(f"not scored (no session): {', '.join(missing)}")
    print("Measurement/Temperature-only steps are not visible to the pipeline and are left out of every line above (EDA included).")



# ---- Timed scoring against CaptainCook4D's own error annotations (per-step times + error tags) ----
# Spec (reviewed against CaptainCook4D, EgoPER, PREGO/TI-PREGO, MistExit): rows are annotated error steps; a strict
# catch is the right step, a compatible kind and on time; lenient is any non-ok verdict on the step at any time.
# No paper defines a tolerance in seconds, so W is reported as a sensitivity, not a tuned constant.
VISIBLE = {"Technique", "Order", "Missing Step", "Preparation"}       # Measurement / Temperature: not visible
KINDS = {"Technique": {"wrong_technique"}, "Order": {"wrong_order", "skipped_step"},   # a mid-clip skip ask is the right question during a swap
         "Missing Step": {"skipped_step"}, "Preparation": None}        # None: any non-ok kind


def cc4d_rows(rec: dict) -> list[dict]:
    """One row per annotated error step: step_id, start, end (-1 when the step never happened), tags."""
    rows = [{"step_id": st["step_id"], "start": st["start_time"], "end": st["end_time"],
             "tags": sorted({e["tag"].replace(" Error", "") for e in st["errors"]})}
            for st in rec["step_annotations"] if st.get("errors")]
    order = [r for r in rows if "Order" in r["tags"] and r["start"] >= 0]
    for r in order:   # a swapped step is timed from the first later step done ahead of it (that is when it can be asked about)
        ahead = [x["start"] for x in order if x["step_id"] > r["step_id"] and x["start"] < r["start"]]
        if ahead:
            r["order_start"] = min(ahead)
    return rows


def _strict(v: dict, row: dict, w: float, last_t: float) -> bool:
    if v["step_id"] != row["step_id"]:
        return False
    for tag in set(row["tags"]) & VISIBLE:
        kinds = KINDS[tag]
        if kinds is not None and v["kind"] not in kinds:
            continue
        if tag == "Missing Step":       # an omission has no moment; a closing ask counts (reported as at-end)
            if row["start"] < 0 or v["t"] >= row["start"]:
                return True
            continue
        a, b = (row.get("order_start", row["start"]) if tag == "Order" else row["start"]), row["end"]
        if a >= 0 and a <= v["t"] <= b + w and v["t"] <= last_t:    # a step that never happened has no window
            return True
    return False


def score_timed(session: dict, rec: dict, w: float = 30.0) -> dict:
    """Score one session against one CaptainCook4D recording annotation."""
    verdicts = [v for v in session["verdicts"] if v["mode"] != "ok"]
    last_t = max((e["t_end"] for e in session["events"]), default=0.0)
    rows = cc4d_rows(rec)
    vis = [r for r in rows if set(r["tags"]) & VISIBLE]
    hidden = {r["step_id"] for r in rows if not set(r["tags"]) & VISIBLE}
    out_rows = []
    for r in vis:
        hit = next((v for v in verdicts if _strict(v, r, w, last_t)), None)
        out_rows.append({**r, "strict": hit is not None, "lenient": any(v["step_id"] == r["step_id"] for v in verdicts),
                         "at_end": bool(hit) and hit["t"] > last_t,
                         "delay_s": round(hit["t"] - r.get("order_start", r["start"]), 1) if hit and r["start"] >= 0 else None})
    tp_i = {i for i, v in enumerate(verdicts) if any(_strict(v, r, w, last_t) for r in vis)}
    tp = [v for i, v in enumerate(verdicts) if i in tp_i]
    fa = [v for i, v in enumerate(verdicts) if i not in tp_i and v["step_id"] not in hidden]   # neither, on a not-visible error step
    steps = {st["step_id"]: st for st in rec["step_annotations"]}
    flagged = {v["step_id"] for v in verdicts}
    erroneous = {r["step_id"] for r in vis}
    missing = {r["step_id"] for r in vis if "Missing Step" in r["tags"]}
    asked = {v["step_id"] for v in verdicts if v["kind"] == "skipped_step"}
    return {"clip": rec["recording_id"], "is_error": bool(vis), "rows": out_rows,
            "not_visible": sorted(hidden), "tp": len(tp),
            "fa_mid": sum(v["t"] <= last_t for v in fa), "fa_end": sum(v["t"] > last_t for v in fa),
            "mid_clip_nonok": sum(v["t"] <= last_t for v in verdicts),
            # EgoPER EDA: every annotated step is a segment, predicted erroneous if any non-ok verdict lands on it
            "eda": (sum((s in flagged) == (s in erroneous) for s in steps if s not in hidden),
                    sum(s not in hidden for s in steps)),
            "issue_tp": sum(i in tp_i for i, v in enumerate(verdicts) if v.get("source") == "issue"),
            "issue_fa": sum(v.get("source") == "issue" for v in fa),
            "omission": (len(missing & asked), len(missing), len(missing | asked)),
            "obs_ratio": [round(v["t"] / float(session["run"]["end"]), 2) for v in tp if v["t"] <= last_t]}


def summarize_timed(results: list[dict]) -> dict:
    """Step-level strict/lenient, false alarms, EgoPER EDA and omission metrics, recording-level TPR/FPR/balanced F1."""
    rows = [r for x in results for r in x["rows"]]
    err = [x for x in results if x["is_error"]]
    norm = [x for x in results if not x["is_error"]]
    tp_rec = sum(x["tp"] > 0 for x in err)
    fp_rec = sum(x["mid_clip_nonok"] > 0 for x in norm)
    # TI-PREGO balanced precision: false positives weighted by positives/negatives, since most clips are normal
    denom = tp_rec + len(err) / max(1, len(norm)) * fp_rec
    pb = tp_rec / denom if denom else 0.0      # no error clips and only false alarms: precision 0
    rc = tp_rec / max(1, len(err))
    om = [x["omission"] for x in results]
    return {"steps_strict": (sum(r["strict"] for r in rows), len(rows)),
            "steps_strict_in_time": (sum(r["strict"] and not r["at_end"] for r in rows), len(rows)),
            "steps_lenient": (sum(r["lenient"] for r in rows), len(rows)),
            "tp_verdicts": sum(x["tp"] for x in results),
            "fa_mid": sum(x["fa_mid"] for x in results), "fa_end": sum(x["fa_end"] for x in results),
            "eda": (sum(x["eda"][0] for x in results), sum(x["eda"][1] for x in results)),
            "issue_tp": sum(x["issue_tp"] for x in results), "issue_fa": sum(x["issue_fa"] for x in results),
            "omission_acc": (sum(o[0] for o in om), sum(o[1] for o in om)),
            "omission_iou": (sum(o[0] for o in om), sum(o[2] for o in om)),
            "rec_tpr": (tp_rec, len(err)), "rec_fpr": (fp_rec, len(norm)),
            "rec_f1_balanced": round(2 * pb * rc / (pb + rc), 2) if pb + rc else 0.0,
            "delay_s": sorted(r["delay_s"] for r in rows if r["delay_s"] is not None and not r["at_end"]),
            "obs_ratio": sorted(o for x in results for o in x["obs_ratio"])}


if __name__ == "__main__":
    main()
