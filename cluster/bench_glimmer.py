#!/usr/bin/env python3
"""Latency benchmark for an OpenAI-compatible vision endpoint (vLLM). Stdlib only, Python 3.6+.

Matches the Glimmer call in the app: N keyframes + a short prompt, streaming, short JSON output.
Reports TTFT and total latency p50/p90 plus output tokens/s. Optional vLLM torch-profiler capture.

  python3 bench_glimmer.py --url http://localhost:8000/v1 --model MODEL --frames frames/*.jpg \
      --runs 20 --warmup 3 --out results/glimmer.json [--profile]
"""
import argparse
import base64
import glob
import json
import statistics
import sys
import time
import urllib.request

SYSTEM = ("You watch short cooking-bench clips as keyframes. Reply with ONLY a JSON array of events: "
          "{t_start,t_end,action,obj,target,technique,conf}. action is one of wash,dry,cut,place,pour,"
          "sprinkle,mix,pick. Return [] if nothing is happening.")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def build(frames, n):
    content = [{"type": "text", "text": "Clip window 170.0-174.0s. Tracks: none. Keyframes follow."}]
    for i, p in enumerate(frames[:n]):
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        content += [{"type": "text", "text": "t=%.1fs" % (170 + i)},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def post(url, path, body=None):
    req = urllib.request.Request(url.rstrip("/").rsplit("/v1", 1)[0] + path if path.startswith("/start") or path.startswith("/stop")
                                 else url.rstrip("/") + path,
                                 json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def one(url, model, msgs, max_tokens):
    body = {"model": model, "messages": msgs, "temperature": 0.1, "max_tokens": max_tokens, "stream": True}
    t0 = time.time()
    ttft, chunks, text = None, 0, ""
    with post(url, "/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
            except (KeyError, IndexError, ValueError):
                continue
            if delta:
                if ttft is None:
                    ttft = time.time() - t0
                chunks += 1
                text += delta
    total = time.time() - t0
    return {"ttft": ttft if ttft is not None else total, "total": total, "chunks": chunks, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--nframes", type=int, default=4)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default="results/glimmer.json")
    ap.add_argument("--profile", action="store_true", help="capture one vLLM torch profile (needs VLLM_TORCH_PROFILER_DIR on the server)")
    a = ap.parse_args()
    frames = sorted(f for g in a.frames for f in glob.glob(g))
    if not frames:
        sys.exit("no frames found")
    msgs = build(frames, a.nframes)
    for _ in range(a.warmup):
        one(a.url, a.model, msgs, a.max_tokens)
    runs = []
    for i in range(a.runs):
        r = one(a.url, a.model, msgs, a.max_tokens)
        runs.append(r)
        print("run %2d  ttft %.2fs  total %.2fs  chunks %d" % (i + 1, r["ttft"], r["total"], r["chunks"]), flush=True)
    if a.profile:
        try:
            post(a.url, "/start_profile").read()
            one(a.url, a.model, msgs, a.max_tokens)
            post(a.url, "/stop_profile").read()
            print("profile captured in the server's VLLM_TORCH_PROFILER_DIR")
        except Exception as e:  # noqa
            print("profile request failed (server needs VLLM_TORCH_PROFILER_DIR set):", e)
    tt, tot = [r["ttft"] for r in runs], [r["total"] for r in runs]
    summary = {"model": a.model, "nframes": min(a.nframes, len(frames)), "runs": a.runs,
               "ttft_p50": statistics.median(tt), "ttft_p90": pct(tt, .9),
               "total_p50": statistics.median(tot), "total_p90": pct(tot, .9),
               "chunks_per_s_p50": statistics.median([r["chunks"] / max(r["total"] - r["ttft"], 1e-6) for r in runs]),
               "sample_output": runs[-1]["text"][:600]}
    json.dump({"summary": summary, "runs": runs}, open(a.out, "w"), indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k != "sample_output"}, indent=1))
    print("sample output:", summary["sample_output"])


if __name__ == "__main__":
    main()
