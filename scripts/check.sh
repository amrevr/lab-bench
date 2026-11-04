#!/usr/bin/env bash
# Offline checks: no API keys, no GPU, no network, no cost. Run from anywhere: scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/../backend"
PY=${PYTHON:-python3}

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "need Python 3.10+, have " + sys.version.split()[0]'
"$PY" -c 'import pydantic, imageio_ffmpeg' 2>/dev/null || {
  echo "missing dependencies: $PY -m pip install -r backend/requirements.txt" >&2; exit 1; }

echo "== unit tests"
"$PY" -m unittest discover -s tests

echo "== synthetic eval (4 injected errors, 1 clean clip)"
out=$("$PY" -m labbench.eval)
echo "$out"
grep -q '^caught 4/4, false alarms 0$' <<<"$out" || { echo "FAIL: synthetic eval expected 'caught 4/4, false alarms 0'" >&2; exit 1; }

echo "== fixtures parse"
"$PY" - <<'EOF'
import json, pathlib
n = 0
for p in sorted(pathlib.Path("../fixtures").glob("recipe*/*.json")):
    json.loads(p.read_text()); n += 1
assert n, "no fixtures found"
print(f"{n} fixture files OK")
EOF

echo "ALL OFFLINE CHECKS PASSED"
