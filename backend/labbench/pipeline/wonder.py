"""Muse Wonder: turn the aligner's findings into short coaching text.

Wonder never sees video; it gets the protocol step and the event text. With no META_API_KEY it
returns the aligner's own message, so the demo runs offline. The endpoint shape is an assumption
(OpenAI-compatible) to be checked against the Meta Model API docs once the key lands.
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
    key, base = os.environ.get("META_API_KEY"), os.environ.get("META_API_BASE")
    if not key or not base:
        return v.message
    body = json.dumps({"model": os.environ.get("WONDER_MODEL", "muse-wonder"),
                       "messages": [{"role": "user", "content": PROMPT.format(mode=v.mode, message=v.message)}],
                       "max_tokens": 120}).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip() or v.message
    except Exception:
        return v.message
