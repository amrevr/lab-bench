# Lab Bench Copilot: submission write-up (draft)

Status: DRAFT for the Meta Global AI Developer Hackathon, Indie track. Every number below comes from the
repo or from issues #7, #10, #14 (measured by us, one run each, small samples). Items marked **[open]** are not
decided or not confirmed. Nothing here claims more than was measured; see "Limitations and honesty".

## Problem and idea

Procedures done by hand (a lab protocol, an assembly, a recipe) go wrong in quiet ways: a skipped step, steps in
the wrong order, the wrong ingredient. Idea: learn the procedure from one expert run plus its written steps, then
watch a novice run in short chunks, tick off steps as they happen, and say something when a step looks
missed, out of order or wrong. When the system is not sure it asks ("Quick check: did you do X?") instead of
alarming.

Cooking stands in for bench work (CaptainCook4D recipe 12, tomato mozzarella salad, 9 steps). We do not claim
anything about real lab procedures.

## Architecture and where each Meta tool is used

```
novice clip -> chunker (3 s windows, 0.5 s overlap, 10 fps / 480p H.264, 4 keyframes)
  -> (optional, off by default) SAM 3.1 tracks for hand, tomato, plate: stored with events, used by no verdict
  -> vision model: keyframes -> timestamped events + confidence
  -> aligner (deterministic): events vs protocol -> verdicts (ok / ask / flag)
  -> Muse Wonder: rewords a verdict as a short coaching sentence
  -> overlay: video, checklist, flags, source frames
```

| Component | Model | Where it runs |
|---|---|---|
| Object tracks (optional, off by default; no verdict uses them) | SAM 3.1 | Hosted Meta Model API (`--sam`); a self-hosted service was used in experiments but is not in this repo |
| Event extraction, backend `spark` | Muse Spark Open (`muse-spark-open-1026`) | Hosted, Baseten |
| Event extraction, backend `glimmer` | Muse Glimmer 30B, FP8, vLLM, tensor parallel 2 | Self-hosted on 2x H100 (shared machine), reached over an SSH forward |
| Coaching text | Muse Wonder (`muse-wonder-1.0`, effort low) | Hosted, Meta Model API |

Both vision backends are implemented behind one client (`VISION_BACKEND` / `--backend`) and both ran in the
pipeline. Almost all measured runs used `spark`; Glimmer was run in the pipeline on the first 75-80 s of
12_26 and on 12_17 (see Results). Wonder never sees video and does not gate a verdict: the aligner's verdict
is shown immediately and Wonder's text is appended when it arrives (3-10 s later in replay). Without a Meta
key the aligner's own message is used. Muse Image is not used.

## Design decisions and why

- **Ask instead of alarm.** Confidence maps to silent / ask / flag. Defaults flag >= 0.75, ask 0.40-0.75;
  technique and wrong-item are stricter (flag 0.85, ask 0.50). Ask messages are worded as questions. Thresholds
  were set on a few clips and calibration found no clean separation for non-technique kinds.
- **Confidence gate.** A verdict that rests on an event below confidence 0.8 is parked until the same
  action/object is seen again in another window within 1 s, else dropped. Added after a clean expert run
  produced 5 false asks.
- **Principled aligner fixes (PR #22).** A lone event counts as a step only if it shows what the step defines
  (target, technique) or is seen again; seasonings are an unordered group; wrong-item only for a noun outside
  the protocol, strong and repeated; obvious vision misplacements (target in the object slot) are normalised.
- **Chunking 3 s with 0.5 s overlap, pipelined** (issue #10): hosted SAM is slower than real time for one stream,
  so windows overlap in flight; overlap lets tracks be linked by box IoU at the same timestamp.
- **10 fps / 480p** (issue #14): 150 billed frames per window instead of 270, SAM latency 2.9-5.2 s instead of
  6.2-8.1 s, detections and confidence about the same.
- **Replay as the demo.** A recorded clip is processed live through the pipeline at real-time pace; the demo is
  labelled as a recording.

## Data

CaptainCook4D (Apache-2.0), recipe 12, 10 clips used so far (expert 12_5 plus novices); clips archived on
catbox. The dataset's error annotations are used only as an evaluation key, never as pipeline input. The
protocol (`fixtures/recipe12/protocol.json`) is the recipe's step list with hand-set ids, order hints,
a seasoning group and one prerequisite (`after`).

## Results, exactly as measured

All runs: one run each, Spark Open unless stated, 10 fps / 480p, real-time pace. Credit is by step id, so some
credited hits are incidental or of the wrong kind. These are not a benchmark.

| Set | Clips | Code | Result | Caveat |
|---|---|---|---|---|
| Dev | 12_43, 12_26, 12_38 (errors) | PR #19 | 12/25 steps credited, 4 false alarms; Technique 0/8 | First runs; thresholds not yet tuned |
| Dev (re-scored offline) | same | PR #20 | 12/25, Technique 2/8; two named false alarms removed | Changes were motivated by these clips |
| Dev, 5 clips (offline) | + 12_41, 12_5 | PR #21 | 17/42 -> 13/42 credited, scored false alarms 4 -> 1; expert 12_5 false asks 5 -> 0 | Gate tuned on these 5 clips |
| Held-out 1 (run once, code frozen) | 12_41 (order), 12_5 (clean) | PR #20 | 12_41: 6/17 credited, 0 false alarms, but the real order mistake was not flagged at the right time; 12_5: 5 asks, 0 flags, all false | Both now consumed as dev data |
| Held-out 2 (fresh) | clean 12_17, 12_16, 12_15; errors 12_51, 12_48 | PR #21 | Clean: 5 asks + 7 flags, all false alarms (12_17 alone gave 9 of 12). Errors: **1 genuine detection of 9** (PR #20 had 2/9) | Did not generalise; the earlier "design held" note was true only on 12_5 |
| Held-out 3 (fresh) | clean 12_10, 12_2, 12_119 | PR #22 (final) | **0 false asks/flags** on 3 clean clips (PR #21 aligner on the same events: 5). One end-of-clip skip ask on 12_119 is a real vision miss (mozzarella read as "place") | Small sample; aligner developed on earlier clips; no fresh error clips, so says nothing about recall |

Dev-set categories (PR #19, 3 clips): Missing Step 6/6, Preparation 3/4, Measurement 3/6 (incidental),
Temperature 0/1, Technique 0/8. These were measured on the clips used to tune and **did not carry over** to
held-out 2.

**Latency, chunk close to verdict** (live, sequential): p50 6.0-11.0 s, p90 9.9-17.7 s across held-out 2;
on_track verdicts on held-out 3: p50 5.9-9.9 s, p90 10.4-13.1 s. Concurrent early runs: p50 9.1 s, p90 12.5 s,
max 13.2 s. Add about 1.5 s average for the chunk to close: mistake to flag is roughly 10-15 s, not
real time. The mistake-to-flag delay itself was not measured (the manifest has no timestamps); the one
well-timed hit (12_26 slice vs dice) was flagged about 3 s of video time after the first dice event plus about 9 s
to emit.

**Backends.** Glimmer 30B in isolation (4 keyframes, reasoning strength low, 10 runs): 7.3-7.7 s p50 total,
JSON valid 10/10 (default reasoning: p50 18 s, 5/8 valid). Spark Open about 6 s for 4 keyframes. In the
pipeline on 12_17, Glimmer gave 51 events (Spark 73), 4 asks (all false alarms) vs 9, with chunk-close to
verdict p50 15.8 s / p90 21.3 s, slower than Spark. On the first 75-80 s of 12_26: Spark 5.7-8.4 s per chunk vs
Glimmer 14.7-18.6 s (shared machine, rough).

**Cost.** SAM bills $0.0002 per frame, 150 frames per 2.5 s step of video (3 concepts, 50-frame minimum per
request), i.e. about $0.7 per minute of video, dominated by SAM. Held-out 3 cost $15.21 for 21.1 min of video.
About $67.8 of the $150 Meta credits spent in total (the Baseten share is small). A 3-minute demo run is about
$2.2 (replay `--max-spend` enforces a cap).

**What failed.** Technique detection (2/8 on dev at best, not carried over); quantity and heat errors
are not visible; failed/empty vision replies (3 empty replies over 5 clips in held-out 2, 1 each on 12_10 and
12_2 from a clip-end keyframe bug, not fixed); 30-93 idle windows per clip.

## Limitations and honesty

- **Not detected:** technique, quantity/measurement and spill errors. Measurement and Temperature catches in the
  tables are incidental, not detection. PR #22 did not add any of these.
- **Recall is unknown.** On the second held-out set, 1 genuine detection of 9 annotated errors (before the last
  fixes). There are no fresh error clips for recipe 12, so the final code has no recall measurement. Fewer
  verdicts can mean lost catches (the confidence gate lost some).
- **False-alarm control and the aligner fixes were developed on earlier clips.** The third fresh set of
  3 clean clips gave 0 false alarms with the final code: a small sample, the upper bound is still around one per
  clip. All 10 CaptainCook4D clips are now consumed as development data; more tuning on them would not give an honest number.
- **Vision misreads** pepper, salt and Italian seasoning and sometimes the verb; steps can be missed because
  of it.
- **Clean runs differ in step order** from the protocol's single order; the protocol only partly models this
  (seasonings are an unordered group), so some order/skip verdicts reflect the protocol, not just vision errors.
- **Skipped steps are only called** after later steps happen or at the end of the clip.
- **Latency about 10-15 s from the mistake, not real time.** The earlier "about 10 s" claim (issue #10) was
  corrected on #7 and #10.
- **Demo clips are replays of recordings** processed live at real-time pace, not a live camera.
- Scoring credits by step id and the dataset key has no timestamps.

We do not claim technique, order or quantity detection, real-time operation, or a general recall number.

## Rules and submission facts (issue #4)

Verified from the Official Rules text: judging weights Impact 30 / Use of Meta AI tooling 25 / Demo 25 /
Technical depth and execution 20. Submit through the Cerebral Valley portal by **Nov 4, 9:00 PM ET**. Indie teams
are up to 4 and all members must be named. Original work; open-source software and public datasets allowed
under their licences (CaptainCook4D is Apache-2.0). Organizers get a perpetual, irrevocable, non-exclusive
licence to use submissions.

Open items, **not** in the official rules text: demo video length and format, whether the repo must be public,
whether a write-up or deck is required, and the "only work done during the event" rule (secondary sources
only).

## Team **[open]**

Names of all members to be filled in by the user.
