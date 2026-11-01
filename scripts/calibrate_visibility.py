"""Mark protocol steps the vision model never recognizes in the expert run as not visible.

python3 scripts/calibrate_visibility.py --protocol fixtures/recipe10/protocol.json --expert-session out/10_16.session.json

The aligner then never asks "did you skip X?" about them (a step nobody can see being missed is noise), but still
credits and judges them when they are seen. One expert run is all a new procedure needs; score without that run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from labbench.pipeline.align import Aligner  # noqa: E402
from labbench.schema import Event, Protocol  # noqa: E402


def calibrate(protocol: dict, session: dict) -> dict:
    al = Aligner(Protocol.model_validate({**protocol, "steps": [{**s, "visible": True} for s in protocol["steps"]]}))
    for e in session["events"]:
        al.feed(Event.model_validate(e))
    for s in protocol["steps"]:
        s["visible"] = s["id"] in al.done
    return protocol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=pathlib.Path, required=True)
    ap.add_argument("--expert-session", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, help="default: overwrite --protocol")
    a = ap.parse_args()
    p = calibrate(json.loads(a.protocol.read_text()), json.loads(a.expert_session.read_text()))
    (a.out or a.protocol).write_text(re.sub(r"\n\s+(?=[^\s{\[])", " ", json.dumps(p, indent=1)) + "\n")
    print("not visible:", [s["id"] for s in p["steps"] if not s["visible"]])


if __name__ == "__main__":
    main()
