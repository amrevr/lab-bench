# Lab Bench Copilot: submission write-up

Meta Global AI Developer Hackathon, Indie track. Every number below comes from `docs/measurements.md` and was
measured by us, one run per clip. Nothing here claims more than was measured; see "Limitations".

## Problem and idea

Procedures done by hand (a lab protocol, an assembly, a recipe) go wrong in quiet ways: a skipped step, steps in
the wrong order, the wrong ingredient, a spill. Lab Bench Copilot learns a procedure from its written steps and
one expert recording. It then watches a novice's video in 3-second windows, ticks steps off as they happen, and
speaks up when a step looks missed, out of order or wrong. When it is unsure it asks ("Quick check: did you add the
salt?") instead of alarming.

Cooking stands in for bench work. Everything is evaluated on six CaptainCook4D recipes against the dataset's own
timed error annotations. We make no claim about real lab procedures.

## Architecture and where each Meta model is used

```
novice video -> chunker: 3 s windows, 0.5 s overlap, 4 keyframes each, several windows in flight
  -> vision model: keyframes -> JSON events {action, obj, target, technique, issue (mishap), conf}
  -> aligner (deterministic): events vs the procedure's prerequisite graph -> verdicts
       on_track | ask ("Quick check: did you ...?") | flag (skipped, wrong order, wrong item, mishap)
  -> Muse Wonder: rewrites each verdict as a short coaching line, asynchronously (never delays a flag)
  -> overlay: video, step checklist, flags, the keyframes behind each verdict
```

| Model | Role | Where it runs |
|---|---|---|
| Muse Glimmer 30B + `Muse-Glimmer-30B-assistant` DFlash drafter | vision events per window; writes protocols from step text | self-hosted: vLLM, FP8, tensor parallel 2, 2x H100 |
| Muse Spark Open | alternate vision backend (`--backend spark`) | hosted on Baseten |
| Muse Wonder | coaching text; never sees video, never gates a verdict | Meta Model API |
| SAM 3.1 | object tracks; optional (`--sam`), off by default, no verdict uses them | Meta Model API |
| V-JEPA 2 ViT-L | evaluated as features for error detection and step recognition; not adopted | self-hosted |

Both vision backends sit behind one client and run the same pipeline, aligner and scorer. Glimmer keeps video on
your own machines and gives fewer false alarms; Spark is the live option and catches more.

## Design decisions and why

- **Learning a new procedure without labels.** `scripts/make_protocol.py` turns a CaptainCook4D task graph into a
  protocol: Glimmer names each step from its text (action, object, target, verb synonyms, how it looks on camera)
  and the graph's edges become prerequisites. Recipes 1, 3, 8, 10 and 27 use generated protocols, not hand-tuned.
- **Visibility calibration from one expert run.** `scripts/calibrate_visibility.py` replays the expert recording and
  marks steps the vision model never sees as not visible, so the system doesn't ask about steps it can't perceive.
  On recipe 10 this cut false alarms from 83 to 18.
- **Ask before alarming.** A verdict needs a confident event (>= 0.6) or the same evidence seen again; otherwise it
  becomes a question or is dropped. Mishap flags need confidence >= 0.8 or a repeat, which took mishap flags from
  ~15% to 36% precise across five recipes.
- **Partial order.** Steps the task graph leaves unordered (seasonings, independent prep) are not called out of order.
- **Real-time replay as the demo.** A recorded clip is processed live at real-time pace and labelled as a
  recording; the dataset's error annotations are used only to score, never as pipeline input.

## Results

Strict = right step and compatible verdict, during the step or within 30 s. Lenient = any warning on that step.
The expert recording is never scored. Glimmer with the drafter unless stated.

| Recipe | Role | Errors caught strict / lenient | Error recordings caught | Normal recordings with a mid-clip alarm |
|---|---|---|---|---|
| 12 tomato salad | tuning | 7/24 / 12/24 | 4/4 | 2/7 |
| 12 | held-out clips | 0/7 / 2/7 | 0/2 | 1/3 |
| 8 hot chocolate | development | 15/41 / 28/41 | 9/9 | 2/6 |
| 10 pinwheels | development | 19/91 / 32/91 | 7/8 | 3/3 |
| 1 egg sandwich | first clean test, later informed a fix | 18/77 / 40/77 | 11/13 | 2/4 |
| 3 mug pizza | clean test | 9/43 / 21/43 | 5/7 | 5/5 |
| 27 cheese pimiento | clean test of the current code (protocol frozen before any run) | 12/45 / 16/45 | 8/9 | 5/5 |

**Hosted vs self-hosted on the clean recipe 27:**

| | Hosted: Muse Spark Open | Self-hosted: Muse Glimmer 30B |
|---|---|---|
| Errors caught on time / lenient | **15/45** / **24/45** | 12/45 / 16/45 |
| Error recordings with a catch | 8/9 | 8/9 |
| Mid-clip false alarms | 39 | **20** |
| Video leaves your machines | yes | no |

**The aligner alone,** on identical recipe-27 Glimmer events: false alarms fell from 90 to 26 (71%) against the
aligner before this work, with more strict and on-time catches (10 -> 12).

**Latency.** Hosted Spark on 12_41 at real-time pace kept up end to end: alerts about 9 s after the window closed
(median), never more than 17 s, no backlog over the 470 s clip. Glimmer on dedicated H100s takes 1.8 s per call
(4.1 s p90) at real-time pace, below the 2.5 s window stride; the DFlash drafter halved this from 3.6 s. On a
shared, saturated machine Glimmer calls took ~15 s and alerts fell 1.5-4 minutes behind, so real time needs
dedicated GPUs or the hosted backend. The idle-GPU end-to-end run was not re-measured.

**Tried, measured, dropped:** SAM tracks in the verdict path (no verdict changed; 12_41 fell 16 min behind),
V-JEPA 2 features (error AUC ~0.55), a Glimmer "second look" investigator (confirms nearly every alert: AUC 0.48),
expected-step verification (no end-to-end gain on recipe 27 at twice the calls), skipping Glimmer's reasoning (misses most steps), 8 frames per window, protocol text in the prompt,
ensembling runs.

## Limitations

- **Precision on unseen recipes is the main weakness:** about 2-5 false alarms per recording (recipe 27: 1.9;
  recipe 3: 4.6), and on recipes 3 and 27 every normal recording got at least one mid-clip alarm.
- **Not reliably detected:** technique, quantity and temperature errors; most aren't visible as steps.
- **Accuracy depends on how visible a recipe's steps are** to the vision model (recipe 1's microwave steps are
  mostly invisible); a single headline number across recipes would mislead.
- **One run per clip; repeat runs vary** by about 4 strict catches or 7 false alarms.
- **Every recipe except 27 informed some decision.** Recipe 27 is the one clean test of the current code.
- **Skipped steps** are called only after later steps happen, or at the end of the clip.
- **Demo clips are replays** of recordings processed at real-time pace, not a live camera.

## Try it

Organizer guide: [SETUP.md](SETUP.md), from a 5-second offline check to a self-hosted run. Or ask a coding
agent to "run the lab-bench tests"; it follows `.claude/skills/lab-bench/SKILL.md`.

## Team

*Your Unified Recording Interpreter*
