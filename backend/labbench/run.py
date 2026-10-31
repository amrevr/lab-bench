"""Run a clip's events through the aligner. python -m labbench.run --clip 12_43 [--root fixtures/synthetic]"""
from __future__ import annotations

import argparse
import json
import pathlib

from .pipeline.align import Aligner
from .schema import THRESHOLDS, Event, Protocol

ROOT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "synthetic"


def load_protocol(root: pathlib.Path) -> Protocol:
    return Protocol.model_validate_json((root / "protocol.json").read_text())


def run_clip(root: pathlib.Path, clip: str, thresholds: dict = THRESHOLDS) -> dict:
    protocol = load_protocol(root)
    events = [Event.model_validate(e) for e in json.loads((root / clip / "events.json").read_text())]
    events.sort(key=lambda e: e.t_end)     # feed order = event_idx in verdict evidence
    al = Aligner(protocol, thresholds)
    for ev in events:
        al.feed(ev)
    al.finish(max((e.t_end for e in events), default=0.0) + 2)
    return {"clip": clip, "protocol": protocol.model_dump(),
            "events": [e.model_dump() for e in events],
            "verdicts": [v.model_dump(mode="json") for v in al.verdicts]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--root", type=pathlib.Path, default=ROOT)
    p.add_argument("--out", type=pathlib.Path)
    a = p.parse_args()
    s = run_clip(a.root, a.clip)
    text = json.dumps(s, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    for v in s["verdicts"]:
        print(f'{v["t"]:6.1f}s {v["mode"]:5s} {v["kind"]:16s} {v["message"]}')


if __name__ == "__main__":
    main()
