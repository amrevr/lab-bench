---
date: 2026-11-04
author: claude-design-leads
type: finding
repo: lab-bench
links: [https://github.com/amrevr/lab-bench/tree/exp/design-leads]
---
# Whole-clip Glimmer pass recognizes steps far better on r8, but its verdicts are no better

E5 (whole clip as 4 grids of 16 timestamped frames + protocol, reasoning high), scored on PR #27 code:
- r8 single pass: steps recognized 72/88 (82%) vs windowed pipeline 38/95 (40%); strict 12/41 vs 11/41; mid false alarms 17 vs 10.
- r12 single pass: steps 107/129 (83%) vs 104/138 (75%); strict 8/31 vs 9/31.
- Two-pass (timeline per quarter, then text-only reasoning): strict 12/31 (r12) and 12/41 (r8) but mid false alarms 40-41; worse overall.
Lead: use a slower long-context pass for the step timeline only and let the graph tracker (which helps when perception is good) produce verdicts. Not real time as is (~15-20 min per clip on the shared GPUs).
Two clips failed (one in single r12, one in single r8) and were not retried.
