"""Replay a recorded clip through the live pipeline at real-time pace.

python -m labbench.replay --clip 12_43.mp4 --protocol ../fixtures/recipe12/protocol.json \
    --backend spark --start 0 --end 200 --pace realtime --max-spend 5 --out session.json

Each 3 s window (0.5 s overlap) is cut with ffmpeg, then SAM (hand/tomato/plate, one request per concept)
and the vision model run concurrently on it; windows run pipelined (a window takes longer than real time).
Results are fed to the aligner in window order. Verdicts print as soon as the aligner emits them; Muse
Wonder's coaching text is appended when it arrives and never gates the verdict. `latency_s` is the real
mistake-to-flag delay as far as the pipeline is concerned: wall time of emission minus wall time the
window closed (`--pace realtime`) or was submitted (`--pace fast`).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg

os.environ.setdefault("FFMPEG", imageio_ffmpeg.get_ffmpeg_exe())   # static build, no system ffmpeg needed

from .pipeline import chunker  # noqa: E402
from .pipeline.align import Aligner  # noqa: E402
from .pipeline.sam import CONCEPTS, LocalSam, MetaSam, NullSam, link_tracks  # noqa: E402
from .pipeline.vision import Vision, VisionError, attach_tracks  # noqa: E402
from .pipeline.wonder import coach_text  # noqa: E402
from .schema import Protocol  # noqa: E402

FRAME_COST = 0.0002          # USD per billed SAM frame
WORKERS = 5                  # windows in flight
VISION_DOWN = 8              # that many vision request failures in a row abort the replay
SAM_WORKERS = 3              # SAM chunks in flight; tracks attach late and never hold a verdict


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def persist_tracks(prev, tracks, counts: dict[str, int]) -> dict[str, str]:
    """Rename tracks in place to replay-wide ids; return raw id -> persistent id (for event track_ids)."""
    link = link_tracks(prev, tracks)
    for t in tracks:
        if link[t.id] == t.id:      # unmatched: fresh id ('#' never occurs in SAM ids, so no collisions)
            counts[t.label] = counts.get(t.label, 0) + 1
            link[t.id] = f"{t.label}#{counts[t.label]}"
    for t in tracks:
        t.id = link[t.id]
    return link


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=pathlib.Path, required=True)
    ap.add_argument("--protocol", type=pathlib.Path, required=True)
    ap.add_argument("--backend", choices=["spark", "glimmer"], default="spark")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, required=True, help="clip time (s) to stop at")
    ap.add_argument("--pace", choices=["realtime", "fast"], default="realtime")
    ap.add_argument("--max-spend", type=float, required=True, help="USD of SAM frames before aborting")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--protocol-prompt", default="", choices=["", "full", "steps", "vocab", "techdefs", "steps+techdefs"],
                    help="experimental prompt variant (see vision.prompt_extra)")
    ap.add_argument("--nframes", type=int, default=4, help="keyframes per window sent to the vision model")
    ap.add_argument("--vision-timeout", type=float, default=60, help="seconds before a vision call is a failure "
                    "(raise for offline accuracy runs on a shared GPU so no window is dropped)")
    ap.add_argument("--pair", action="store_true", help="send keyframes two per image (8 frames in 4 images)")
    ap.add_argument("--issues", action=argparse.BooleanOptionalAction, default=True,
                    help="ask the vision model for visible mishaps (spills, ...); on by default: recall up on recipes 8/10/12")
    ap.add_argument("--sam-workers", type=int, default=SAM_WORKERS, help="SAM chunks in flight (match the service's workers)")
    ap.add_argument("--sam-url", help="self-hosted SAM service (e.g. http://localhost:8001) instead of the hosted API")
    ap.add_argument("--no-sam", action="store_true", help="skip SAM (no tracks, no cost): verdicts do not use tracks")
    ap.add_argument("--finish", action="store_true", help="at --end, call every step still undone skipped")
    ap.add_argument("--workdir", type=pathlib.Path, help="chunks and keyframes (default: a temp dir)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    if a.pair and a.nframes % 2:
        sys.exit("--pair needs an even --nframes")

    clip, protocol = a.clip.stem, Protocol.model_validate_json(a.protocol.read_text())
    work = a.workdir or pathlib.Path(tempfile.mkdtemp(prefix="replay-"))
    names = sorted({n for st in protocol.steps for n in (st.obj, st.target) if n})   # the recipe's own nouns
    sam, vision, al = (NullSam() if a.no_sam else LocalSam(a.sam_url, fps=a.fps) if a.sam_url else MetaSam(fps=a.fps)), Vision(a.backend, names=names, steps=protocol.steps, style=a.protocol_prompt, issues=a.issues, paired=a.pair, timeout=a.vision_timeout), Aligner(protocol)
    wins = [(a.start + s, a.start + e) for s, e in chunker.windows(a.end - a.start)]
    est = 0 if a.no_sam or a.sam_url else len(CONCEPTS) * max(50, round(3.0 * a.fps))   # billed frames per window (50 min per request)
    print(f"{len(wins)} windows, estimated SAM cost ${len(wins) * est * FRAME_COST:.2f} (cap ${a.max_spend:.2f})")
    if len(wins) * est * FRAME_COST > a.max_spend:
        sys.exit("estimate is over --max-spend; shorten the range or lower --fps/--height")

    def see(kf, s, e):
        """Vision events, or (None, why) when the reply failed: idle ([]) and failed are different."""
        try:
            return vision.events(kf, [], s, e, clip), None
        except VisionError as x:
            return [], str(x)

    def track_window(i: int, s: float, e: float):
        """-> (tracks, failed concepts, seconds). Runs beside the verdict path: tracks never delay a verdict."""
        failed, t1 = [], time.time()
        try:
            chunk = chunker.cut_chunk(a.clip, s, e, work / clip / f"chunk{i:04d}.mp4", a.height, a.fps)
            return sam.track(chunk, list(CONCEPTS), s, failed), failed, round(time.time() - t1, 2)
        except Exception as x:
            return [], failed + [f"{type(x).__name__}: {x}"[:200]], round(time.time() - t1, 2)
        finally:
            with sam_lock:
                sam_done[0] += 1

    def work_on(i: int, s: float, e: float):
        """-> (events, health, SAM future or None); health = {"vision": error or None, "t": stage seconds}."""
        h = {"vision": None, "t": {}}
        t = time.time
        fut = None if a.no_sam else sam_pool.submit(track_window, i, s, e)
        try:
            t0 = t()
            kf = chunker.keyframes(a.clip, s, e, work / "frames" / clip, n=a.nframes)
            if a.pair:
                kf = chunker.pair_frames(kf)
            h["t"]["keyframes"] = round(t() - t0, 2)
            evs, h["vision"] = see(kf, s, e)
            h["t"]["vision"] = round(t() - t0 - h["t"]["keyframes"], 2)
            h["t"]["window"] = round(t() - t0, 2)
            return evs, h, fut
        except Exception as x:   # one bad window must not end the replay
            h["vision"] = f"{type(x).__name__}: {x}"
            return [], h, fut
        finally:
            done[0] += 1

    done, events, prev, errors, aborted, timings = [0], [], [], [], False, []
    sam_lock = threading.Lock()
    down = 0                   # windows in a row whose vision request failed
    sam_done = [0] if not a.no_sam else done          # windows whose SAM call has finished (NullSam: none are made)
    sam_pool = ThreadPoolExecutor(a.sam_workers)
    pending: list = []          # (s, e, events of the window as (Event, session dict), SAM future), window order

    def drain(wait: bool = False):
        """Attach SAM tracks to already-judged events, in window order (track ids persist across windows)."""
        nonlocal prev
        while pending and (wait or pending[0][3].done()):
            s, e, pairs, fut = pending.pop(0)
            tracks, failed, secs = fut.result()
            stats["sam_failed"] += len(failed)
            sam_s.append(secs)
            for err in failed:
                errors.append(f"{s:.1f}-{e:.1f}: sam {err}")
            evs = [ev for ev, _ in pairs]
            attach_tracks(evs, tracks)
            link = persist_tracks(prev, tracks, counts)
            prev = tracks or prev
            for ev, d in pairs:
                d["track_ids"], d["sam_conf"] = [link.get(t, t) for t in ev.track_ids], ev.sam_conf

    counts: dict[str, int] = {}
    sam_s: list[float] = []
    stats = {"windows": 0, "vision_failed": 0, "vision_idle": 0, "sam_failed": 0}
    session = {"clip": clip, "protocol": protocol.model_dump(), "events": events, "verdicts": al.verdicts}
    wonder_pool = ThreadPoolExecutor(2)
    wall0 = time.time()

    def coach(v):
        v.coach = coach_text(v)
        print(f"        coach (+{time.time() - v.emitted_at:.1f}s) step {v.step_id}: {v.coach}", flush=True)

    def emit(vs, closed_t, fed_at):
        for v in vs:
            v.closed_t, v.emitted_at = closed_t, time.time()
            v.latency_s = round(v.emitted_at - fed_at, 1)
            print(f"{v.closed_t:7.1f}s {v.mode:5s} {v.kind.value:15s} step {v.step_id} conf {v.conf:.2f} "
                  f"latency {v.latency_s:.1f}s  {v.message}", flush=True)
            if v.mode != "ok":
                wonder_pool.submit(coach, v)

    q: queue.Queue = queue.Queue()

    def submit_windows(pool):
        nonlocal aborted
        for i, (s, e) in enumerate(wins):
            if a.pace == "realtime":
                time.sleep(max(0.0, wall0 + (e - a.start) - time.time()))
            if (sam.frames + (i - sam_done[0]) * est) * FRAME_COST > a.max_spend:   # SAM still in flight counts
                print(f"ABORT: max spend ${a.max_spend:.2f} would be exceeded", file=sys.stderr)
                aborted = True
                break
            q.put((s, e, time.time(), pool.submit(work_on, i, s, e)))
        q.put(None)

    with ThreadPoolExecutor(WORKERS) as pool:
        threading.Thread(target=submit_windows, args=(pool,), daemon=True).start()
        while item := q.get():                                    # window order, so the aligner sees time order
            s, e, fed_at, f = item
            evs, h, sam_fut = f.result()
            stats["windows"] += 1
            timings.append({"s": s, "queued": round(time.time() - fed_at, 2), **h.get("t", {})})
            stats["vision_idle"] += not evs and not h["vision"]
            stats["vision_failed"] += bool(h["vision"])
            down = down + 1 if h["vision"] and "request" in h["vision"] else 0
            if down >= VISION_DOWN:      # the model is unreachable (tunnel/server down): a session now would be empty
                print(f"ABORT: {down} windows in a row could not reach the vision model; no session written",
                      file=sys.stderr, flush=True)
                os._exit(3)              # without waiting for the windows still queued
            if h["vision"]:
                errors.append(f"{s:.1f}-{e:.1f}: vision: {h['vision']}")
                print(f"window {s:.1f}-{e:.1f} failed: vision: {h['vision']}", file=sys.stderr)
            pairs = []
            for ev in sorted(evs, key=lambda x: x.t_end):
                events.append(ev.model_dump())
                pairs.append((ev, events[-1]))
                emit(al.feed(ev), e, fed_at)
            if sam_fut:
                pending.append((s, e, pairs, sam_fut))
            drain()
            print(f"[{e:6.1f}s] {len(evs)} events, billed {sam.frames} frames ~${sam.frames * FRAME_COST:.2f}", flush=True)
    drain(wait=True)
    sam_pool.shutdown(wait=True)
    if a.finish and not aborted:
        emit(al.finish(a.end), a.end, time.time())
    wonder_pool.shutdown(wait=True)

    lat = [v.latency_s for v in al.verdicts if v.mode != "ok"]
    session["verdicts"] = [v.model_dump(mode="json") for v in al.verdicts]
    session["run"] = {"backend": a.backend, "pace": a.pace, "start": a.start, "end": a.end, "fps": a.fps,
                      "height": a.height, "billed_frames": sam.frames, "cost_usd": round(sam.frames * FRAME_COST, 3),
                      "errors": errors, "aborted": aborted, "finish": a.finish and not aborted, "no_sam": a.no_sam, "protocol_prompt": a.protocol_prompt, "issues": a.issues, "nframes": a.nframes, "pair": a.pair, "timings": timings,
                      "sam_s": {"p50": pct(sam_s, .5), "p90": pct(sam_s, .9)} if sam_s else None, **stats}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(session, indent=1))
    print(f"windows {stats['windows']}: vision failed {stats['vision_failed']}, idle {stats['vision_idle']}, "
          f"SAM requests failed {stats['sam_failed']}")
    print(f"wrote {a.out}: {len(events)} events, {len(al.verdicts)} verdicts, cost ${sam.frames * FRAME_COST:.2f}"
          + (f", flag latency p50 {pct(lat, .5):.1f}s p90 {pct(lat, .9):.1f}s" if lat else ""))
    sys.exit(2 if aborted else 0)


if __name__ == "__main__":
    main()
