"""Parse CaptainCook4D error strings from the manifest into step-level answer keys.

Eval only. Strings look like "119 Measurement/Technique", "127,123,124,121 Missing Step" or
"126,124,123,127 before 120 (Order/Preparation/Technique)". The manifest has no timestamps,
so real-clip scoring is per step: did we raise any non-ok verdict on that step?
"""
from __future__ import annotations

import json
import pathlib
import re

CATS = ("Missing Step", "Order", "Technique", "Measurement", "Preparation", "Temperature")


def parse_error(s: str) -> dict:
    ids = [int(x) for x in re.findall(r"\d+", s.split("(")[0].split("before")[0])]
    cats = [c for c in CATS if c.lower().split()[0] in s.lower()]
    return {"step_ids": ids, "categories": cats, "raw": s}


def load_keys(manifest: pathlib.Path) -> dict[str, dict]:
    m = json.loads(manifest.read_text())
    return {c["id"]: {"clip": c["id"], "role": c["role"], "duration_sec": c["duration_sec"],
                      "errors": [parse_error(e) for e in c["errors"]]}
            for c in m["clips"]}
