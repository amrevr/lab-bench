# Latency measurements (issue #14)

| Stage | Hardware | p50 | p90 | Notes |
|---|---|---|---|---|
| ffmpeg cut 4 s chunk, 480p H.264 | build container CPU | 1.6 s | n/a | single run, 2026-10-27 |
| ffmpeg 4 keyframes, 640 px | build container CPU | 1.1 s | n/a | single run |
| SAM 3.1 upload / first frame / total | n/a | not measured | | needs key or H100 |
| Glimmer TTFT / total | n/a | not measured | | needs H100 or Baseten |
