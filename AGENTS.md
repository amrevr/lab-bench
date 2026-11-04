# Agent instructions

To run the tests, verify the setup, re-score results or demo the overlay, follow
[`.claude/skills/lab-bench/SKILL.md`](.claude/skills/lab-bench/SKILL.md) (the same skill is linked at
`.agents/skills/lab-bench/`). The human guide is [`docs/SETUP.md`](docs/SETUP.md).

- Quick check: `python3 -m pip install -r backend/requirements.txt && scripts/check.sh` (offline, free).
- Ask the user before anything that calls a paid API (`labbench.replay --backend spark`) or uses GPUs.
- Keys come from environment variables only (`.env.example`). Never commit or print them.
