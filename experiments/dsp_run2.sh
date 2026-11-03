#!/bin/bash
# runs ON dsp: E1 recipe 8 (V2 chosen on r12, plus D baseline), then the E5 whole-clip stages
B=/data/vermar/lab-bench-exp; PY=/data/vermar/envs/glimmer-vllm/bin/python; mkdir -p $B/logs
export LB_URL=http://127.0.0.1:18080/v1 LB_CLIPS=/data/vermar/lab-bench-vjepa/clips LB_E1=$B/scratch/e1 LB_E34=$B/scratch/e34 LB_E5=$B/scratch/e5
export LB_SESS12=$B/sessnames/r12 LB_SESS8=$B/sessnames/r8
export FFMPEG=$(ls $B/pylib/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-*) PYTHONPATH=$B/pylib
log() { echo "[$(date +%F' '%T)] $*" >> $B/logs/runner.log; }
run() { local dir=$1 lg=$2; shift 2; log "start: $*"; (cd $B/repo/$dir && nice -n 10 $PY "$@") >> $B/logs/$lg.log 2>&1; log "exit $?: $*"; }
run experiments/e1_verify e1 e1.py run 8 V2,D
for st in "single 12" "single 8" "twopass 12" "twopass 8"; do run experiments/e5_wholeclip e5 e5.py $st; done
log "ALL DONE 2"
