---
name: lab-bench
description: Set up, test and demo the Lab Bench Copilot repo (procedure-error detection on CaptainCook4D video). Use when asked to run the tests, verify the repo works, reproduce or re-score results, open the overlay, or run a replay on a clip.
---

# Lab Bench: run and verify

The full human guide is `docs/SETUP.md`. It has four levels, and this skill covers them in order. **Do levels 1
and 2 without asking.** They are free and offline apart from two downloads. Before level 3 or 4, ask the user,
because those cost money or GPU time and need their keys or hardware.

## Level 1: offline checks (always do this)

From the repo root:

```
python3 --version                                   # needs 3.10+
python3 -m pip install -r backend/requirements.txt  # if `scripts/check.sh` reports missing dependencies
scripts/check.sh
```

Pass means the last line is `ALL OFFLINE CHECKS PASSED` and the exit code is 0. The script runs the unit tests
(`cd backend && python3 -m unittest discover -s tests`), the synthetic eval (`python3 -m labbench.eval`, which
must print `caught 4/4, false alarms 0`) and a parse of every fixture JSON.

If pip refuses with "externally managed environment", create a venv first:
`python3 -m venv .venv && . .venv/bin/activate`. On a failure, report the failing test name and its traceback
verbatim. Do not edit code to make a test pass unless the user asks.

## Level 2: a stored real session (free, needs internet)

Follow "2. A stored real session" in `docs/SETUP.md` exactly. It downloads one recorded session and its clip,
re-scores the session with the current aligner, and serves the overlay. Report:

- the eval output's `false alarms mid-clip` / `at end` numbers for W=30s
- the overlay URL to open. Start `python3 -m http.server 8765` in the background, and tell the user the port must be
  reachable from their browser.

12_119 has no annotated errors, so strict/lenient 0/0 is expected and is not a failure.

## Level 3 / 4: live replay (ask first)

- Level 3 needs `BASETEN_API_KEY` in the environment and a clip in `data/clips/`. Confirm the spend with the user,
  then run the `labbench.replay` command from `docs/SETUP.md` with a short `--end` (60-120 s) first.
- Level 4 needs a vLLM server for Muse Glimmer (`cluster/40_serve_glimmer_h100.sh`, 2x H100). Never start, stop
  or kill GPU processes on a shared machine without the user's explicit go-ahead.
- Never print, log or commit API keys. They come from the environment or `.env`, which is gitignored.

## Report back

Give a short table: level, command, pass/fail, and the key numbers (tests run, synthetic caught/false alarms,
stored-session false alarms). Then add one line on anything you skipped and why.
