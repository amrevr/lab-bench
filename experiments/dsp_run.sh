#!/bin/bash
# runs ON dsp: lab-bench design-lead experiments, one at a time, Glimmer at 127.0.0.1:18080 (no sandbox tunnel in the loop)
B=/data/vermar/lab-bench-exp; PY=/data/vermar/envs/glimmer-vllm/bin/python; mkdir -p $B/logs
export LB_URL=http://127.0.0.1:18080/v1 LB_CLIPS=/data/vermar/lab-bench-vjepa/clips LB_E1=$B/scratch/e1 LB_E34=$B/scratch/e34 LB_E5=$B/scratch/e5
export FFMPEG=$(ls $B/pylib/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-*) PYTHONPATH=$B/pylib
log() { echo "[$(date +%F' '%T)] $*" >> $B/logs/runner.log; }
run() { local dir=$1; shift; log "start: $*"; (cd $B/repo/$dir && nice -n 10 $PY "$@") >> $B/logs/$LOGNAME_.log 2>&1; log "exit $?: $*"; }
LOGNAME_=e1;  run experiments/e1_verify e1.py run 12 V1,D,V2
LOGNAME_=e4;  run backend ../experiments/e34_compare/e4.py 8 --variants base,v2 --neg 48
LOGNAME_=e5;  for st in "single 12" "single 8" "twopass 12" "twopass 8"; do run experiments/e5_wholeclip e5.py $st; done
log "ALL DONE"
