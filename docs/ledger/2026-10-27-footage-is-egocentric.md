---
date: 2026-10-27
author: claude-thread
type: finding
repo: lab-bench
links: []
---
# Footage is head-mounted and mostly walking around the kitchen

Contact sheet of 12_43 (1 frame per 20 s): fridge, drawer, stove, cutting board, hands visible only during
prep. Implications: (1) many 4 s chunks hold no task activity, so Glimmer must return [] cheaply and Wonder
must stay silent on them; (2) fast head motion will break SAM track continuity, so link_tracks may rarely fire
and keyframe-based Glimmer is the more reliable path; (3) a pure motion rules fallback would be weak on this
footage. No GPU in the build container (nvidia-smi absent), so none of this is measured yet.
