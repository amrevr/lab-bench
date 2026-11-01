"""Build a pipeline protocol for any CaptainCook4D recipe from its task graph, with Glimmer naming each step.

python3 scripts/make_protocol.py --task-graph spicedhotchocolate.json --step-desc step_idx_description.json \
    --annotations complete_step_annotations.json --recipe 8 --out fixtures/recipe8/protocol.json

The graph's edges become `after` (prerequisites); step order follows a normal (error-free) recording; Glimmer
(text only, the self-hosted model at GLIMMER_BASE_URL) turns each step description into action/obj/target/technique.
Nothing is hand-labelled, so the same command works for every recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import urllib.request

PROMPT = (
    "Each line is one step of a cooking recipe, as 'id: description'. For every step return a JSON object "
    "{id, label, action, aliases, seen_as, obj, target, technique}: action is one short lowercase verb a camera could see (add, pour, cut, "
    "mix, microwave, heat, spread, place, wash, ...); obj is what is moved or worked on, as a short lowercase noun "
    "(milk in 'fill a mug with milk'; for a step on everything in a container, such as mix, heat or microwave, the "
    "container itself, e.g. 'mug'); target is the container or surface it goes into or onto (or null); technique is the "
    "method only when the step names one that a camera could check (slice, dice, chop, ...), else null; label is the "
    "step in 3-6 plain words; aliases lists up to 3 other verbs a viewer might use for "
    "the same visible action (fill: pour, add; microwave: heat, warm); seen_as lists up to 2 other ways the step "
    "looks on camera as {action, target} with everyday verbs (microwave the mug: {\"action\": \"place\", \"target\": \"microwave\"}), or []. Use the same noun for the same thing in every step. Reply with ONLY a JSON array.")


def ask_glimmer(lines: list[str]) -> list[dict]:
    url = os.environ.get("GLIMMER_BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
    body = {"model": "glimmer", "temperature": 0.0, "max_tokens": 3000,
            "chat_template_kwargs": {"reasoning_strength": "low"},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": "\n".join(lines)}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    return json.JSONDecoder().raw_decode(text[text.index("["):])[0]


def share_forms(steps: list[dict]) -> list[dict]:
    """Steps whose verbs are aliases of each other (heat ~ microwave) look the same on camera: share their seen_as."""
    verbs = lambda st: {st["action"], *st.get("aliases", [])}
    for a in steps:
        for b in steps:
            if a is not b and (a["action"] in verbs(b) or b["action"] in verbs(a)):
                for v in b.get("seen_as", []):
                    if v not in a.setdefault("seen_as", []):
                        a["seen_as"].append(v)
    for st in steps:
        if not st.get("seen_as"):
            st.pop("seen_as", None)
    return steps


def build(graph: dict, desc: dict, annotations: dict, recipe: int) -> dict:
    by_text = {v.strip(): int(k) for k, v in desc.items()}
    local = {k: v for k, v in graph["steps"].items() if v not in ("START", "END")}
    gid = {k: by_text[v.strip()] for k, v in local.items()}          # task-graph id -> dataset step id
    after = {gid[str(b)]: sorted(gid[str(a)] for a, b2 in graph["edges"] if b2 == b and str(a) in gid)
             for b in {e[1] for e in graph["edges"]} if str(b) in gid}
    recs = sorted((r for r in annotations.values() if r["activity_id"] == recipe), key=lambda r: r["recording_id"])
    normal = next(r for r in recs if not any(s.get("has_errors") for s in r["steps"]))
    order = list(dict.fromkeys(s["step_id"] for s in normal["steps"] if s["step_id"] in gid.values()))   # repeats once
    order += [g for g in gid.values() if g not in order]             # a step the normal run skipped (optional)
    named = {int(x["id"]): x for x in ask_glimmer([f"{g}: {local[k]}" for k, g in gid.items()])}
    steps = []
    for g in order:
        n = named[g]
        steps.append({"id": g, "label": n.get("label") or desc[str(g)], "recipe_text": desc[str(g)],
                      "action": n["action"].lower().strip(),
                      **({"aliases": sorted({a.lower().strip() for a in n["aliases"]} - {n["action"].lower().strip()})}
                         if n.get("aliases") else {}),
                      **({"seen_as": [{"action": v["action"].lower().strip(),
                                       **({"target": v["target"].lower().strip()} if v.get("target") else {})}
                                      for v in n["seen_as"] if v.get("action")]} if n.get("seen_as") else {}),
                      "obj": n["obj"].lower().strip(),
                      **({"target": n["target"].lower().strip()} if n.get("target") else {}),
                      **({"technique": n["technique"].lower().strip()} if n.get("technique") else {}),
                      **({"after": after[g]} if after.get(g) else {})})
    return {"name": f"{normal['activity_name']} (CaptainCook4D recipe {recipe}, generated)",
            "source_clip": normal["recording_id"], "partial_order": True, "steps": share_forms(steps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-graph", type=pathlib.Path, required=True)
    ap.add_argument("--step-desc", type=pathlib.Path, required=True)
    ap.add_argument("--annotations", type=pathlib.Path, required=True, help="complete_step_annotations.json")
    ap.add_argument("--recipe", type=int, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    p = build(json.loads(a.task_graph.read_text()), json.loads(a.step_desc.read_text()),
              json.loads(a.annotations.read_text()), a.recipe)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(re.sub(r"\n\s+(?=[^\s{\[])", " ", json.dumps(p, indent=1)) + "\n")
    print(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
