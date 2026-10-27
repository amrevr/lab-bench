# lab-bench

## Subagents

- Prefer delegating to subagents whenever the work can be handed off (research, downloads, probing, multi-file reading), keeping the main thread for decisions and talking with the user.
- Default subagent model: Sonnet. If a task looks like it would benefit from Opus (hard reasoning, tricky debugging, high-stakes design), ask the user before using Opus.
