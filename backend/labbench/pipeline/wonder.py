"""Muse Wonder: turn the aligner's findings into short coaching text.

Wonder never sees video; it gets the protocol step and the event text. With no META_API_KEY it
returns the aligner's own message, so the demo runs offline. Meta `/v1/responses`, model
`muse-wonder-1.0`; the default reasoning effort is max and slow, `low` answers in ~2 s but needs
max_output_tokens >= ~600 (reasoning tokens count) or it returns no text.
"""
from __future__ import annotations

import json
import os
import urllib.request

from ..schema import Verdict

PROMPT = (
    "You are a calm kitchen-bench coach. Rewrite this finding as one or two short sentences for the "
    "person doing the task. If mode is 'ask', ask them to confirm instead of accusing. No alarm words.\n"
    "mode: {mode}\nfinding: {message}"
)


def coach_text(v: Verdict) -> str:
    key = os.environ.get("META_API_KEY")
    if not key:
        return v.message
    base = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
    body = json.dumps({"model": os.environ.get("WONDER_MODEL", "muse-wonder-1.0"),
                       "reasoning": {"effort": "low"}, "max_output_tokens": 800,
                       "input": PROMPT.format(mode=v.mode, message=v.message)}).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/responses", body,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            parts = [c["text"] for o in json.load(r)["output"] if o["type"] == "message" for c in o["content"]]
        return "".join(parts).strip() or v.message
    except Exception:
        return v.message
