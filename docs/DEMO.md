# Demo video: storyboard and shot list (3 min, screen recording)

Length and format of the video are not in the official rules text (issue #4): **[open]**. This plan assumes
about 3 minutes. The recording is made by the user; nothing here has been recorded.

Ground rules: every clip is a recording from CaptainCook4D processed live through the pipeline at real-time
pace, and the video says so. Show video time and flag time on screen. Say what the system cannot see. Do not
say "real time", and do not quote a recall number.

## On-screen layout

Left: the overlay (`overlay/index.html`) playing the clip, with the video clock. Right or below: the terminal
running `labbench.replay`, whose lines carry the video time the chunk closed, the verdict, its confidence and
`latency_s`, and (indented) the Muse Wonder text. Keep both in frame. Burn in a caption "Recording, processed live
at real-time pace" for the whole video.

Note: the overlay places a verdict at the verdict's video time `t`; the terminal shows when it was emitted
(chunk close plus processing). Use the terminal for flag time. The overlay has no separate flag clock
(adding one is optional work).

## Storyboard

| Time | Shot | Narration (draft, adapt) |
|---|---|---|
| 0:00-0:20 | Title card, then expert clip 12_5 for a few seconds beside the 9-step protocol | "Lab Bench Copilot learns a procedure from one expert run and its written steps, then coaches a novice. This is cooking, standing in for bench work." |
| 0:20-0:40 | Architecture slide (diagram from the README) | "Muse Glimmer (or Muse Spark Open) reads the actions from keyframes, a deterministic aligner compares them to the protocol, and Muse Wonder words the coaching. Glimmer runs on our own H100s; Spark and Wonder are hosted." |
| 0:40-1:20 | Clean run: 12_10 (fallback 12_2 or 12_119), terminal and overlay side by side, checklist ticking, no flags | "Everything you see is a recording processed live, not a live camera. This cook makes no mistake. The coach stays quiet. On three clean clips the final code raised no false alarms; that is a small sample." Show the clock and `latency_s` on the on-track lines. |
| 1:20-2:10 | End-of-clip ask: 12_119 (not yet checked live as a demo), replay with `--finish`: the end-of-clip "Did you do ... (step 124)?" question | "Skips are only called once later steps happen or the clip ends, and when unsure the coach asks. This question is an example of that behaviour: the vision step really missed the mozzarella, so the coach asks instead of asserting." |
| 2:10-2:30 | A verdict's coaching text and source frames in the overlay | "Wonder rewords the finding as a question; the evidence is the keyframes the model saw." |
| 2:30-3:00 | Limits card, then close | "It cannot see technique, quantities or spills. Latency is about 10 to 15 seconds from the mistake, not real time. Recall is unmeasured on fresh clips. Next: more clips, more procedures." |

The 12_43 "skipped step caught" example is **not** shown: it is only an end-of-clip ask from a stored session made
with older code (offline re-alignment gave asks for steps 124, 126 and 121, and step 126 was not actually skipped),
and it needs a live check on the final code before it is shown. Instead show the 12_119 end-of-clip question for
step 124 (stored session https://files.catbox.moe/x062ss.json, clip https://files.catbox.moe/xe5dwu.mp4). Label it
on screen as an honest example of the "ask" behaviour: the mozzarella really was missed by the vision step. It is
one ask, not a demonstration that skipped steps are caught. Runs may differ.

The full 12_10 run is 394 s, longer than the video. Either record the whole run and cut with a visible
"cut" caption and unbroken video-time clock, or record a window (`--start/--end`). Never speed up playback.

## Commands

Run from `backend/` with clips in `data/clips/<id>.mp4` (see README; the URLs for 12_10, 12_2, 12_119 are in the
README, with manifest_v3 https://files.catbox.moe/7we43k.json). Cost is about $0.03 per 2.5 s of video.

```
# clean run, ~$4.7 for the whole clip (394 s); cap 6
python -m labbench.replay --clip ../data/clips/12_10.mp4 --protocol ../fixtures/recipe12/protocol.json \
    --backend spark --start 0 --end 394 --pace realtime --max-spend 6 --out ../out/12_10.session.json
# same for 12_2 (--end 429) and 12_119 (--end 442)

# skipped steps, ~$2.4 (200 s); cap 3
python -m labbench.replay --clip ../data/clips/12_43.mp4 --protocol ../fixtures/recipe12/protocol.json \
    --backend spark --start 0 --end 200 --pace realtime --max-spend 3 --finish --out ../out/12_43.session.json
```

Overlay on a stored session (from the repo root; `serve_overlay.py` only replays synthetic fixtures, so use a
plain static server instead):

```
python3 -m http.server 8765
# http://localhost:8765/overlay/?session=/out/12_43.session.json&video=/data/clips/12_43.mp4&frames=/out/frames
```

Key frames for the evidence thumbnails are written under the replay `--workdir` (`frames/<clip>/`); run with
`--workdir ../out` and point `frames=` at `/out/frames` to show them.

## Before recording

- `META_API_KEY` (SAM, Wonder) and `BASETEN_API_KEY` (Spark) available to the process; in the sandbox they are
  injected by the proxy. Never show them on screen.
- Do a 20 s trial: `--start 120 --end 140 --max-spend 0.5`. Check the output has no "failed" lines.
- Backend `glimmer` only: forward `ssh -N -L 8000:127.0.0.1:18080 <gpu-host>`, check the server is up
  (docs/RUNNING.md). Glimmer ran slower than Spark in the pipeline (p50 15.8 s on 12_17), so use `spark` for the
  main recording unless the Glimmer story is wanted; say plainly which backend ran.
- Clean terminal font size, overlay wide enough to read the checklist, caption on.

## Fallback if a live API call fails

Play the stored session through the overlay (above) with the same video; say "stored session from an earlier run
of the same pipeline" in the narration. Stored sessions for earlier runs are linked on issue #7 (12_43, 12_17,
12_10, 12_2, 12_119). Note the 12_10/12_2/12_119 sessions were produced by the final code; 12_43's
post-fix session predates PR #21/#22, so its verdicts differ from a fresh run.
