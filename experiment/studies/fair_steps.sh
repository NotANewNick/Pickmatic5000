#!/usr/bin/env bash
# Fair test at equal steps: MaskGIT's own schedule at 1-4 steps, 1,000 test digits, each with a
# FID range.
# This is a trimmed rewrite, not the exact script that ran. The run also made diffusion at 1-4
# steps (before the step-spacing fix; superseded by fair_steps_diffusion.sh) and copied in the
# old fixed-recipe picker's recipes (removed), and it wrote summary.md and fid_range.txt for all
# of them. results/fair_steps/ keeps only the MaskGIT rows, as summary_maskgit.md and
# fid_range_maskgit.txt (the names used below); their numbers are unchanged.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
py=.venv/bin/python D=work/p2/data R=work/p2/runs O=work/p2/eval_steps
log() { echo "[$(date +%T)] $*"; }
mkdir -p $O
for s in 1 2 3; do
    log "maskgit $s steps"
    $py experiment/baselines.py generate maskgit --model $R/M_maskgit --data $D --out $O/maskgit_s$s --steps $s > $O/maskgit_s$s.log 2>&1 || log "FAILED maskgit $s"
done
cp -r work/p2/eval/maskgit_s4 $O/
RUNS=$(ls -d $O/maskgit_s?)
log "evaluate"
$py experiment/evaluate.py $RUNS --data $D --out $O/summary_maskgit.md > $O/evaluate.log 2>&1 || log "FAILED evaluate"
log "fid range"
$py experiment/fid_range.py $RUNS --data $D --resamples 20 > $O/fid_range_maskgit.txt 2>&1 || log "FAILED fid_range"
cat $O/fid_range_maskgit.txt
log "done"
