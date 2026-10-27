---
date: 2026-10-27
author: claude-thread
type: plan
repo: lab-bench
links: [https://github.com/amrevr/lab-bench/pull/16]
---
# PR 16 merged: skeleton on main; real perception still unwired

On main: schema, aligner, chunker (ffmpeg, 4 s chunk 1.6 s, 4 keyframes 1.1 s on CPU), SAM track linking with a
stubbed Meta client, Glimmer OpenAI-compatible client with validated JSON, Wonder with offline fallback,
eval + answerkey parser, overlay page, README. Blocked: Meta API key (~Oct 29) and any GPU access from this
container (none seen). Next: self-hosted/Baseten Glimmer test on keyframes from real clips if a GPU or key
appears; otherwise a motion/colour rules fallback on real frames; expert protocol check against 12_5.
