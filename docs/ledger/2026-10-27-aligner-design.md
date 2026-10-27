---
date: 2026-10-27
author: claude-thread
type: decision
repo: lab-bench
links: [https://github.com/amrevr/lab-bench/pull/16]
---
# Aligner: one verdict per mistake, flag at 0.75, ask at 0.40

First version cascaded (one omission produced 3 verdicts). Now: an attempted step counts as done even if
wrong; a gap is called skipped only after two later steps happen without it; a missing step that appears after
a later one becomes wrong_order on the later step; leftovers at clip end are skipped. Thresholds in schema.py.
Self-test on SYNTHETIC events: 4/4 injected errors caught, 0 false alarms, expert clean. This says nothing yet
about real perception quality.
