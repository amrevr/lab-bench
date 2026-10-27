---
date: 2026-10-27
author: claude-thread
type: decision
repo: lab-bench
links: [https://huggingface.co/spaces/nvidia/simready-central, https://docs.omniverse.nvidia.com/simready/latest/simready-central.html]
---
# Skip NVIDIA SimReady Central; it does not help this pipeline

What it is (rechecked 2026-10-27 after amrevr allowlisted docs.omniverse.nvidia.com and explainx.ai; see Recheck below): a free index of 2,000+ OpenUSD
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
## Recheck (hosts now reachable)
- docs.omniverse.nvidia.com/simready/latest/simready-central.html: OpenUSD packages (assets, metadata, license) in
  contributors' HF datasets, joined by a shared index. Aimed at robotics, simulation, digital twins. Names no
  categories and no kitchen, lab, food or cooking items. Licenses are creator-chosen (CC-BY-4.0 recommended; non-commercial
  allowed). Mentions Isaac Sim, Kit Apps and DSX Sim as consumers; no video or rendering tool of its own.
- Press coverage (unite.ai, dealroom, daily.dev): catalog is robot arms, grippers, warehouse equipment, racks and cooling
  from OEMs (FANUC, Universal Robots, Franka, Vertiv, LG and others); 2,000+ assets, launched around Oct 21 2026.
  Nothing kitchen or lab related.
- explainx.ai: reachable, but its homepage and a search turned up no SimReady Central page. Nothing to add.
- The HF Space page still renders only a loading status, so the per-asset list was not browsed. Not worth more time.

Recommendation confirmed: do not use. The recheck found no kitchen or lab assets and no video tooling.
