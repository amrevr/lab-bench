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
- Next lever: the DFlash drafter (`DFLASH=1 cluster/40_serve_glimmer_h100.sh`), not yet measured.
