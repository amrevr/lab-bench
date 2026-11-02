#!/bin/bash
# realign every saved set with the current worktree aligner; print strict/FA line (W=30) per set
J=/home/agent/.claude/jobs; cd "$(dirname "$0")/../backend"
for s in "12 $J/50da9733/tmp/exp/oracle/r12" "8 $J/50da9733/tmp/exp/oracle/r8" "10 $J/50da9733/tmp/exp/oracle/r10" "1 $J/50da9733/tmp/exp/oracle/r1" \
         "12 $J/a4f6fa6d/tmp/errclips" "8 $J/a4f6fa6d/tmp/r8_base" "8 $J/a4f6fa6d/tmp/r8_base2" "10 $J/a4f6fa6d/tmp/r10_base"; do
  set -- $s; echo "$(basename $2) r$1: $(python3 -m labbench.eval --cc4d ../fixtures/recipe$1/cc4d_error_annotations.json --sessions $2 --realign ../fixtures/recipe$1/protocol.json --held-out 12_48,12_51,12_9,12_13,12_16 2>&1 | grep 'W=  30' | sed 's/.*steps//' | tr '\n' ' ')"
done
