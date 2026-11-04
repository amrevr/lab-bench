# Latency measurements (issue #14)

| Stage | Hardware | p50 | p90 | Notes |
|---|---|---|---|---|
| ffmpeg cut 4 s chunk, 480p H.264 | build container CPU | 1.6 s | n/a | single run, 2026-10-27 |
| ffmpeg 4 keyframes, 640 px | build container CPU | 1.1 s | n/a | single run |
| SAM 3.1 upload / first frame / total | n/a | not measured | | needs key or H100 |
| Glimmer call, 4 keyframes 640 px, reasoning low, 1 in flight | 2x H100 TP=2 FP8, idle | 2.2 s | 4.8 s | 1416 prompt tokens, 50-400 output (mostly reasoning); 45 windows of 12_119, 2026-10-31 |
| Same, 5 in flight (pipeline load) | same | 3.4 s | 6.9 s | all 177 windows of 12_119 |

Glimmer notes (2026-10-31, 12_119 keyframes, no SAM, events fed to the aligner):
- Decode dominates: ~75 tok/s per request, and the reasoning channel is most of the output. Shrinking frames to
  336 px cuts prompt tokens 3x (1270 -> 410) but not latency.
- There is no reasoning-off value (the template default is `high`; "off"/"none" reason more). Skipping reasoning by
  prefilling the answer channel (`assistant` + `[`, `continue_final_message`) gave p50 2.2 s / p90 3.7 s at 5 in
  flight, but 17 event windows instead of 63, 1 of 9 steps on track instead of 4, and 5 false asks instead of 3.
  Not used.
- DFlash drafter (`meta-models/Muse-Glimmer-30B-assistant`, `DFLASH=1 cluster/40_serve_glimmer_h100.sh`), all
  177 windows, reasoning low, events fed to the aligner (2026-10-31/11-01, GPUs otherwise idle):

  | Server | 1 in flight p50 / p90 | real-time pace (1 call per 2.5 s) p50 / p90 | 5 in flight p50 / p90 |
  |---|---|---|---|
  | no drafter | 2.2 / 4.8 s | 3.6 / 7.7 s | 3.4 / 6.9 s |
  | drafter, 15 draft tokens (recipe) | 1.8 / 3.4 s | not run | 6.0 / 10.8 s |
  | drafter, 4 draft tokens (kept) | 1.6 / 2.7 s | 1.8 / 4.1 s | 5.9 / 10.3 s |

  Acceptance length 3.5 of 15 (k=15) and 2.7 of 4 (k=4). The drafter wins at real-time pace and loses on bursts,
  where verifying draft tokens for many requests saturates the GPUs: keep vision calls near one in flight. Quality
  on clean 12_119 is unchanged (55-61 event windows, 3 steps on track, 3-4 false asks in every run). The drafter
  needs `--gpu-memory-utilization 0.44` (about 32 GB per GPU instead of 26).

## Accuracy and generalization (2026-11-01/03)

Scored with `python -m labbench.eval --cc4d ... --realign <protocol>` against CaptainCook4D's own error annotations
and step times (strict = right step, compatible verdict, during the step or within 30 s, not an end-of-clip ask
except for omissions; lenient = any non-ok verdict on the step). The protocol's own expert clip is never scored.
Glimmer 30B with the DFlash drafter, SAM off the verdict path, `--issues` on, visibility calibrated from the expert
run. Recipe 12 uses its hand-written protocol; recipes 8 and 10 use protocols generated from the CaptainCook4D
task graphs by Glimmer (`scripts/make_protocol.py`), not hand-tuned.

| Recipe (clips) | Split | Error steps caught strict (in time) | Lenient | False alarms mid / end | Error clips caught | Normal clips with a mid-clip alarm | EgoPER EDA |
|---|---|---|---|---|---|---|---|
| 12 tomato salad (16) | tune | 8/24 (4) | 13/24 | 4 / 8 | 4/4 | 3/7 | 77/95 |
| 12 | held-out | 1/7 (1) | 3/7 | 7 / 2 | 1/2 | 1/3 | 33/44 |
| 8 hot chocolate (15) | dev (generalization fixes tuned here) | 16/41 (10) | 33/41 | 14 / 26 | 9/9 | 2/6 | 65/89 |
| 10 pinwheels (11) | first untouched, then informed visibility calibration | 17/91 (17) | 37/91 | 27 / 7 | 7/8 | 3/3 | 108/180 |
| 1 microwave egg sandwich (7 of 17 so far) | clean test: protocol and settings frozen before any result | 2/21 (2) | 9/21 | 13 / 9 | 1/4 | 1/3 | 53/77 |
| 3 mug pizza (12) | clean test: protocol and settings frozen before any result, run on dsp at afde3bd | 13/43 (11) | 28/43 | 56 / 17 | 6/7 | 5/5 | 102/157 |

Numbers are with the aligner and scorer fixes from the PR #27 review (e7e83b3, 912a19b); all sessions were
re-scored offline from their saved Glimmer events. EDA leaves out Measurement/Temperature-only steps, like every
other line. Mishap ("issue") verdicts are counted separately: 12 caught an annotated error and 19 were false alarms
across recipes 12, 8 and 10 (~39% precise).

Recipe 3 is the clean out-of-sample result (recipe 1 informed the aligner fix 030118f): it finds mistakes about
as well as the development recipes (strict 30%, lenient 65%, 6 of 7 error clips) but raises about 6 false alarms per
clip, every normal clip gets a mid-clip alarm, and mishap flags are only 13% precise there (4 right, 26 wrong).

Recipe 1 was the first clean out-of-sample result: its protocol was generated and committed before any run, and
nothing was changed after seeing its numbers. It is much weaker than recipes 8 and 12: Glimmer recognizes its steps
(spraying a ramekin, three microwave steps, assembly) poorly, and calibration hides only 1 of its 12 steps, so the
aligner asks about steps Glimmer simply missed. Accuracy depends strongly on how visible a recipe's steps are to
the vision model; a single headline number across recipes would mislead. The row is interim (7 of 17 scorable
clips) and will be completed.

Recipe 10's first, fully untouched result (before visibility calibration): strict 25/91, lenient 48/91, 83 false
alarms, EDA 49%.

Run-to-run variance (same pipeline, recipe 8, two runs): strict 11 vs 15, in time 3 vs 9, mid-clip false alarms
10 vs 3. Treat single-run differences under about 4 catches or 7 false alarms as noise. Glimmer call failures
(timeouts under load) are about 1%.

What moved accuracy (measured):
- Visible mishaps (`--issues`): above both baseline runs on recipe 8 (strict 11 and 13 -> 16, lenient 28 -> 33,
  error clips 9/9) and on recipe 10 (strict 6 -> 17, lenient 16 -> 37); issue verdicts are ~39% precise, so more
  "was there a spill?" asks (recipe 8 mid-clip false alarms 14 vs 5-10 without it).
- Generated protocols + lenient noun matching + verb aliases / container steps / on-camera forms: recipe 12's
  generated protocol matches the hand-written one; recipe 8 false alarms 36 -> 32 at equal catches.
- Visibility calibration from the expert run: recipe 10 false alarms 83 -> 18, EDA 49% -> 59%, at a recall cost.
- Gate 0.6 instead of 0.8: same catches, warnings up to 5 min earlier.

Ruled out (measured): skipping Glimmer's reasoning (misses most steps); protocol in the prompt with techniques
(Glimmer echoes "slice"), steps-only or technique-definition prompts (more false alarms on 17 clips); gating
end-of-clip asks on later steps (loses 5/6 omissions); merging two Glimmer runs (no better than the better run);
frozen V-JEPA 2 ViT-L probes (error detection AUC ~0.55, below chance across recipes; step recognition 0.22-0.28
from one expert run); SAM 3.1 in the verdict path (12_41 fell 16 min behind).

8 frames per window (`--nframes 8 --pair`, recipe 12, 14 of 17 clips, same clips compared): no gain. Tune strict
8/24 both, lenient 12 vs 13, mid-clip false alarms 5 vs 4; held-out 1/3 vs 0/3 (noise). Each Glimmer call costs
about 2x. Not adopted; the recipe-8 part of the comparison was skipped.

Not re-measured: latency under today's shared-GPU load (another user's jobs pushed a 4-frame Glimmer call from ~2 s
to ~12 s). Runs from the sandbox depend on an SSH tunnel that dropped several times a day; `replay.py` now aborts a
clip without writing a session when the vision model is unreachable, and offline accuracy runs use
`--vision-timeout 300` (at 60 s, a loaded GPU dropped 10-30% of windows).

Evaluation caveats (from code review of PR #27): the 0.6 gate was chosen on the same recipe-12 clips reported as
"tune"; recipe 8 was development data; `--issues` default and visibility calibration were decided after seeing
recipes 8, 10 and 12. The clean generalization numbers are the first recipe-10 run above and recipe 1.
