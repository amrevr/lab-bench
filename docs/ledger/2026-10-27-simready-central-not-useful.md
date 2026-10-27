---
date: 2026-10-27
author: claude-thread
type: decision
repo: lab-bench
links: [https://huggingface.co/spaces/nvidia/simready-central, https://docs.omniverse.nvidia.com/simready/latest/simready-central.html]
---
# Skip NVIDIA SimReady Central; it does not help this pipeline

What it is (from search snippets; the Space page itself had no readable README, and docs.omniverse.nvidia.com and
explainx.ai are not on the network allowlist, so the asset list was not inspected): a free index of 2,000+ OpenUSD
3D assets with physics properties for robot simulation and factory digital twins. The Space runs on an RTX PRO 6000.
Assets sit in contributors' own HF datasets; the index links them.

Fit against Lab Bench:
- Test clips: none. It supplies 3D assets, not video; rendering real-looking egocentric kitchen footage needs
  Omniverse/Isaac plus a GPU (none in the container) and days of scene work. Synthetic clips would also be a
  domain gap for Glimmer/SAM and say nothing about CaptainCook4D recipe 12.
- Eval data: no. The answer key is the CaptainCook4D error annotations; synthetic renders add no ground truth we trust.
- Demo assets: marginal at best. The demo plays the real novice clip with the overlay.
- Rules: assets from a third party are allowed as data, but building a scene pipeline eats time before Nov 4.

Recommendation: do not use. Revisit only if footage is lost and a pure-synthetic fallback is needed.
Caveat: if amrevr wants the asset catalog checked for lab/kitchen items, allow the two hosts above first.
