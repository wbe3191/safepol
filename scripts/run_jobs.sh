#!/usr/bin/env bash
# Run a list of (cmd method seed [steps]) jobs with bounded parallelism inside WSL.
# usage: scripts/run_jobs.sh JOBFILE PARALLEL
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/safepol-venv/bin/python
JOBFILE=$1; PAR=${2:-6}
mkdir -p results/logs
run_one() {
  local cmd=$1 method=$2 seed=$3 steps=${4:-150000} tag=${5:-eval}
  local log=results/logs/${cmd}_${method}_s${seed}_${tag}.log
  if [ "$cmd" = "train" ] && [ -f results/models/${method}_s${seed}.pt ]; then echo "skip $method s$seed (model exists)"; return; fi
  if [ "$cmd" = "eval" ] && [ -f results/${tag}/${method}_s${seed}.csv ]; then echo "skip eval $method s$seed"; return; fi
  echo "start $cmd $method s$seed ($(date +%H:%M:%S))"
  $PY -m safepol.experiments.run $cmd --method $method --seed $seed --steps $steps --tag $tag > $log 2>&1
  echo "done  $cmd $method s$seed ($(date +%H:%M:%S)) rc=$?"; tail -1 $log
}
export -f run_one; export PY
grep -v '^#' "$JOBFILE" | grep -v '^\s*$' | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
