"""Vision model: tracks + keyframes -> timestamped events.

One client, two OpenAI-compatible backends chosen by env VISION_BACKEND: `glimmer` (default; self-hosted
on the lab's H100s, forwarded to localhost:8000) or `spark` (Muse Spark Open on Baseten). There is no JSON
mode, so the reply is parsed leniently, validated against Event, and retried once on failure.
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
from .sam import Track

# backend -> (base url, served model, env var holding the key)
BACKENDS = {
    "glimmer": (os.environ.get("GLIMMER_BASE_URL", "http://localhost:8000/v1"), "glimmer", "GLIMMER_API_KEY"),
    "spark": ("https://inference.baseten.co/v1", "meta-models/muse-spark-open-1026", "BASETEN_API_KEY"),
}


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


def summarize(tracks: list[Track]) -> str:
    return "; ".join(f"{t.label} {t.id} ({len(t.boxes)} boxes, sam conf {t.conf:.2f})" for t in tracks)


class Vision:
    def __init__(self, backend: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        backend = backend or os.environ.get("VISION_BACKEND", "glimmer")
        url, name, key_env = BACKENDS[backend]
        self.base_url = (base_url or url).rstrip("/")
        self.model = model or name
        self.api_key = api_key or os.environ.get(key_env, "")

    def _chat(self, messages: list[dict]) -> str:
        # Spark Open burns the whole budget on reasoning without reasoning_effort, and returns no content
        # under ~600 tokens; any `reasoning_content` in the reply is ignored.
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.1,
                           "max_tokens": 800, "reasoning_effort": "low"}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body, headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"].get("content") or ""

    def events(self, keyframes: list[tuple[float, pathlib.Path]], tracks: list[Track],
               t0: float, t1: float, clip: str | None = None) -> list[Event]:
        content = [{"type": "text", "text": f"Clip window {t0:.1f}-{t1:.1f}s. Tracks: {summarize(tracks) or 'none'}. "
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
                    e.frame_refs, e.clip, e.chunk = [t for t, _ in keyframes], clip, [t0, t1]
                    mine = [tr for tr in tracks if tr.label in (e.obj, e.target, "hand")]
                    e.track_ids = [tr.id for tr in mine]
                    e.sam_conf = round(min((tr.conf for tr in mine), default=1.0), 2) if mine else None
                return evs
            except (ValueError, ValidationError, json.JSONDecodeError):
                msgs.append({"role": "user", "content": "That was not valid. Reply with only the JSON array."})
        return []
