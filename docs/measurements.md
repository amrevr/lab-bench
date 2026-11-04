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

## End-to-end real-time run with the current defaults (2026-11-04)

`replay --pace realtime` (live 60 s vision timeout, 4 keyframes, `--issues`, no SAM, Glimmer with the drafter),
run on dsp with no other job of ours on the Glimmer server, on 12_41 (470 s) and 8_31:

| Clip | Glimmer call p50 / p90 | Window close -> events reach the aligner, p50 / p90 / max | Failed windows |
|---|---|---|---|
| 12_41 | 15.7 / 29.8 s | 120 / 205 / 230 s | 0 of 189 |
| 8_31 | 14.8 / 31.6 s | 89 / 132 / 150 s | 0 of 107 |

Not real time on this run: the H100s were at 100% utilization from another user's jobs (about 32 GB and 9 GB,
started during the run), and each Glimmer call took ~8x its idle-GPU time (1.8 s p50 at real-time pace, table
above), so windows queued and verdicts fell 1.5-4 minutes behind. With GPUs to ourselves the measured per-call
latency (1.8 s p50, 4.1 s p90) is below the 2.5 s window stride, which is what real time needs; that end-to-end
claim was not re-measured on an idle machine. Real time therefore needs dedicated GPUs (or the hosted backend),
not a shared machine.

## Accuracy and generalization (2026-11-01/03)

Scored with `python -m labbench.eval --cc4d ... --realign <protocol>` against CaptainCook4D's own error annotations
and step times (strict = right step, compatible verdict, during the step or within 30 s, not an end-of-clip ask
except for omissions; lenient = any non-ok verdict on the step). The protocol's own expert clip is never scored.
Glimmer 30B with the DFlash drafter, SAM off the verdict path, `--issues` on, visibility calibrated from the expert
run. Recipe 12 uses its hand-written protocol; recipes 8 and 10 use protocols generated from the CaptainCook4D
task graphs by Glimmer (`scripts/make_protocol.py`), not hand-tuned.

| Recipe (clips) | Split | Error steps caught strict (in time) | Lenient | False alarms mid / end | Error clips caught | Normal clips with a mid-clip alarm | EgoPER EDA |
|---|---|---|---|---|---|---|---|
| 12 tomato salad (16) | tune | 7/24 (3) | 12/24 | 3 / 8 | 4/4 | 2/7 | 77/95 |
| 12 | held-out | 0/7 (0) | 2/7 | 6 / 2 | 0/2 | 1/3 | 33/44 |
| 8 hot chocolate (15) | dev (generalization fixes tuned here) | 15/41 (9) | 28/41 | 8 / 25 | 9/9 | 2/6 | 60/89 |
| 10 pinwheels (11) | dev (informed visibility calibration and the aligner fixes) | 19/91 (19) | 32/91 | 21 / 7 | 7/8 | 3/3 | 105/180 |
| 1 microwave egg sandwich (17) | dev (informed aligner fix 030118f) | 18/77 (17) | 40/77 | 34 / 14 | 11/13 | 2/4 | 121/181 |
| 3 mug pizza (12) | clean test at afde3bd (see below); rescored here with the later rules | 9/43 (7) | 21/43 | 38 / 17 | 5/7 | 5/5 | 100/157 |
| 27 cheese pimiento (14) | **clean test of the current code** (protocol frozen before any run) | 12/45 (9) | 16/45 | 20 / 6 | 8/9 | 5/5 | 88/139 |

Numbers are with the aligner and scorer fixes from the PR #27 review (e7e83b3, 912a19b); all sessions were
re-scored offline from their saved Glimmer events. EDA leaves out Measurement/Temperature-only steps, like every
other line. Mishap ("issue") verdicts are counted separately: 12 caught an annotated error and 19 were false alarms
across recipes 12, 8 and 10 (~39% precise).

Rows above use the current code: aligner fix 030118f and mishap flags backed only at confidence >= 0.8 or when the
same mishap is seen again (ISSUE_STRONG). That rule was chosen on all five recipes together: it cuts 33 of 192 false
alarms for 8 of 63 strict catches, and mishap flags go from ~15% to 36% precise (8 right, 14 wrong). Every recipe
here has now informed a decision; the next clean test is a fresh recipe.

Recipe 27 is the clean test of the current code (030118f aligner fix and ISSUE_STRONG): protocol generated and
committed before any run, visibility calibrated from its expert run 27_13 (3 of 12 steps hidden), nothing changed
after seeing the result. It catches at least one error in 8 of 9 error recordings (strict 27%, lenient 36%), with
about 1.9 false alarms per recording (recipe 3: about 4.6), but still a mid-clip alarm on every normal recording.
Mishap flags were 4 right, 1 wrong (recipe 3 before ISSUE_STRONG: 4 right, 26 wrong). Different recipes, one run
each: this is consistent with the precision rules generalizing, not proof.

Before/after on the same clean set: recipe 27's saved Glimmer events (15 sessions, expert excluded, same calibrated
protocol, same scorer), replayed through three versions of the aligner:

| Aligner | Strict (in time) | Lenient | False alarms mid / end | Error recordings caught | Mishap flags right / wrong |
|---|---|---|---|---|---|
| `main` before this work (75b4208) | 10/45 (5) | 33/45 | 32 / 58 | 7/9 | 0 / 0 |
| PR #27 before the last two rules (e7e83b3) | 12/45 (9) | 17/45 | 28 / 6 | 8/9 | 4 / 8 |
| current | 12/45 (9) | 16/45 | 20 / 6 | 8/9 | 4 / 1 |

On identical perception, false alarms fall from 90 to 26 (71%) with more strict and on-time catches; the old
aligner's high lenient count comes from warning on most steps of every recording. This isolates the aligner: the
events already include this work's prompt changes (`--issues`, protocol-derived actions), so it understates the
total change, and the old aligner never saw generated protocols in practice.

Recipe 3's clean result, as run before these two changes (afde3bd): strict 13/43 (11), lenient 28/43, false alarms
56 / 17, error clips 6/7, every normal clip with a mid-clip alarm, EDA 102/157. Recipe 3 is the clean out-of-sample result (recipe 1 informed the aligner fix 030118f): it finds mistakes about
as well as the development recipes (strict 30%, lenient 65%, 6 of 7 error clips) but raises about 6 false alarms per
clip, every normal clip gets a mid-clip alarm, and mishap flags are only 13% precise there (4 right, 26 wrong).

Recipe 1 was the first clean out-of-sample result: its protocol was generated and committed before any run, and
nothing was changed after seeing its numbers. It is much weaker than recipes 8 and 12: Glimmer recognizes its steps
(spraying a ramekin, three microwave steps, assembly) poorly, and calibration hides only 1 of its 12 steps, so the
aligner asks about steps Glimmer simply missed. Accuracy depends strongly on how visible a recipe's steps are to
the vision model; a single headline number across recipes would mislead.

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
from one expert run); SAM 3.1 in the verdict path (12_41 fell 16 min behind). Expected-step verification (`--verify` on branch exp/verify-step:
a second Glimmer call per window asking yes/no about the next 3 expected steps plus a decoy; per-window recall
0.04 -> 0.14 on recipe 8 in a sampled test) did not carry over end to end: on recipe 27, the same run with and
without its events gives strict 9 vs 9, lenient 14 vs 13, mid-clip false alarms 21 vs 19, and the decoy got a
"yes" in 505 of 4619 windows (11%, vs ~1% in the sampled test), at twice the Glimmer calls. Not merged.

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
