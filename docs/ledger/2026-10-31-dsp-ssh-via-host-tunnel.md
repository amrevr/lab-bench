---
date: 2026-10-31
author: claude-thread
type: gotcha
repo: lab-bench
links: []
---
# dsp reachable from sandbox via host port forward: use `ssh dsp-tunnel`

dsp is vermar@dspserver.cs.rpi.edu:22, VPN-gated; `dsp` is only a host-side ssh alias. Host must keep running
`ssh -N -L 0.0.0.0:2222:localhost:22 dsp`; sandbox allows localhost:2222; sandbox ~/.ssh/config `Host dsp-tunnel`
-> host.docker.internal:2222 with a dedicated ed25519 key `dsp_claude` (pubkey in dsp authorized_keys; rotate by
removing the `claude-sandbox` line there) and HostKeyAlias dspserver.cs.rpi.edu.
Gotcha: an older `Host dsp` block (direct to the real host) already existed; ssh takes the first value per option,
so a second `Host dsp` block inherits its HostName. Hence the separate name. Plain `ssh dsp` in the sandbox fails
without VPN.
Failure: `Connection closed ... port 2222` / kex_exchange_identification = host tunnel down. `sbx secret` is for
proxy-injected service tokens, not ssh keys.
