---
date: 2026-11-03
author: claude-design-leads
type: finding
repo: lab-bench
links: [https://github.com/amrevr/lab-bench/tree/exp/design-leads]
---
# Design-lead experiments: perception is the bottleneck, not windowing or rules

Scored on PR #27 code (e5de81c); scripts on branch exp/design-leads.
- Oracle: perfect GT step events through the aligner catch < 1/3 of errors (strict r12 6/32, r8 14/41, r10 8/91, r1 22/77); Technique/Preparation ~0 by construction.
- Belief tracker over the prerequisite graph: helps only with perfect events (Order r8 4->8, r1 7->13); ties or is worse on real Glimmer events.
- One positive lead: expected-step verification prompt (2-3 candidate steps + decoys, yes/no/unclear). r12 TPR 0.65 vs 0.52 (describe prompt), FPR 0.04 vs 0.01; held-out r8 0.14 vs 0.04, FPR 0.01; decoy echo 0/200.
- Small label-free gain: leave-one-out step detectability gating. r10 F1 0.48->0.64, r8 run2 0.86->0.94, r12 unchanged.
- Ruled out: investigator second look (confirm AUC 0.48; localize AUC 0.40, decoy "found" 60%); expert comparison (V-JEPA AUC < 0.5 on r8/r10; Glimmer step text 0.69 r12 -> 0.53 r8; expert frames worse); ensembling repeat runs; whole-clip high-reasoning ceiling (r12 strict 8/31 vs windowed 9/31).
- Untested: SAM tracks in the Glimmer prompt (needs hosted SAM or a service restart).
- Long Glimmer runs go on dsp (/data/vermar/lab-bench-exp), not through the flapping host tunnel.
