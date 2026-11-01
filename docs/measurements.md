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
