"""Muse Glimmer: tracks + keyframes -> timestamped events.

Talks to any OpenAI-compatible chat endpoint (local vLLM or Baseten). There is no JSON mode, so
the reply is parsed leniently, validated against Event, and retried once on failure.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import urllib.request

from pydantic import ValidationError

from ..schema import Event

SYSTEM = (
    "You watch short cooking-bench clips as keyframes plus object-track summaries. Reply with ONLY a "
    "JSON array of events: {t_start,t_end,action,obj,target,technique,conf}. action is one of "
    "wash,dry,cut,place,pour,sprinkle,mix,pick. Use lowercase nouns. conf in 0..1; lower it when unsure. "
    "Return [] if nothing is happening."
)


def parse_events(text: str, t0: float = 0.0, t1: float = 1e9) -> list[Event]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in reply")
    out = []
    for raw in json.loads(m.group(0)):
        ev = Event.model_validate(raw)
        ev.t_start, ev.t_end = max(ev.t_start, t0), min(ev.t_end, t1)
        out.append(ev)
    return out


class Glimmer:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("GLIMMER_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("GLIMMER_API_KEY", "")
        self.model = model or os.environ.get("GLIMMER_MODEL", "muse-glimmer-30b")

    def _chat(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.1,
                           "max_tokens": 400}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body,
                                     {"Content-Type": "application/json",
                                      "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def events(self, keyframes: list[tuple[float, pathlib.Path]], track_summary: str,
               t0: float, t1: float) -> list[Event]:
        content = [{"type": "text", "text": f"Clip window {t0:.1f}-{t1:.1f}s. Tracks: {track_summary or 'none'}. "
                                            "Keyframes follow with their timestamps."}]
        for t, p in keyframes:
            b64 = base64.b64encode(p.read_bytes()).decode()
            content += [{"type": "text", "text": f"t={t:.1f}s"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        for attempt in range(2):
            try:
                evs = parse_events(self._chat(msgs), t0, t1)
                for e in evs:
                    e.frame_refs = [t for t, _ in keyframes]
                return evs
            except (ValueError, ValidationError, json.JSONDecodeError):
                msgs.append({"role": "user", "content": "That was not valid. Reply with only the JSON array."})
        return []
