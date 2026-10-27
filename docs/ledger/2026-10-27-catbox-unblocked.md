---
date: 2026-10-27
author: claude-thread
type: finding
repo: lab-bench
links: []
---
# catbox allowlisted; all 5 recipe 12 clips downloaded

Egress proxy first 403'd files.catbox.moe; after amrevr added it, the manifest and all five clips
(12_5, 12_43, 12_41, 12_26, 12_38; 60-170 MB each, 1280x720 H.264) downloaded into gitignored data/clips.
Gotcha: two large downloads got a connection reset mid-transfer on the first try and succeeded on retry.
