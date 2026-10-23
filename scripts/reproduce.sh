#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper from scratch (WSL/Linux, ~3.5 h on 8 cores).
#   bash scripts/reproduce.sh            # full pipeline
#   bash scripts/reproduce.sh tables     # only statistics/tables/figures from existing results/evaltest
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-~/safepol-venv/bin/python}
if [ "${1:-all}" = "all" ]; then
  $PY tests/smoke.py
  bash scripts/run_jobs.sh scripts/jobs_main_a.txt 6          # SAFEPOL/B5/B4 seed 0-2 + static baselines seed 0
  bash scripts/run_jobs.sh scripts/jobs_main_a_eval.txt 5
  bash scripts/make_jobs_b.sh
  bash scripts/run_jobs.sh scripts/jobs_main_b_train.txt 6    # remaining seeds + ablations
  bash scripts/run_jobs.sh scripts/jobs_main_b_eval.txt 5
  bash scripts/run_jobs.sh scripts/jobs_b8.txt 5                # B8 shielded runbook (static)
  bash scripts/run_jobs.sh scripts/jobs_h.txt 5                 # SAFEPOL-H hybrid: train
  bash scripts/run_jobs.sh scripts/jobs_h_eval.txt 5
  $PY scripts/scalability.py                                  # results/scalability.csv (~40 min)
fi
$PY scripts/stats.py results/evaltest                         # results/tables/*.{csv,tex} + paper/tables/*.tex
$PY scripts/figures.py results/evaltest                       # paper/figures/*.pdf
$PY scripts/extra_stats.py                                    # per-scenario / per-group numbers quoted in the text
$PY scripts/k8s/summarise.py                                  # paper/tables/k8s_e2e.tex from results/k8s (cluster runs: see README)
echo "done; compile with: cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main"
