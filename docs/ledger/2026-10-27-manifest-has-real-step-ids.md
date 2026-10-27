---
date: 2026-10-27
author: claude-thread
type: finding
repo: lab-bench
links: []
---
# Manifest steps are ids 119-127 and not in execution order

Expert order: 125 rinse, 122 dry, 119 slice, 120 place, 124 sprinkle mozzarella, 127 italian seasoning,
126 pepper, 123 salt, 121 olive oil. Aligner uses list position, never id order.
Annotation strings carry no timestamps (e.g. "127,123,124,121 Missing Step"), so real-clip scoring is
per step, not per time window. Wrong-ingredient is not a category: 12_38 is Preparation/Measurement/Missing Step,
so 12_38 is a weak test of "wrong ingredient" and the pitch should not claim that case from it.
