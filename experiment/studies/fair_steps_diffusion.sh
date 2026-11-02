#!/usr/bin/env bash
# Diffusion rerun after the step-spacing fix (the old spacing wasted one step), 1,000 test digits.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
py=.venv/bin/python D=work/p2/data R=work/p2/runs O=work/p2/eval_steps
log() { echo "[$(date +%T)] $*"; }
for s in 1 2 3 4 8 10 20; do
    log "diffusion $s steps"; rm -rf $O/diffusion_s$s
    $py experiment/baselines.py generate diffusion --model $R/D_diffusion --data $D --out $O/diffusion_s$s --steps $s --cfg 1 > $O/diffusion_s$s.log 2>&1 || log "FAILED diffusion $s"
done
RUNS=$(ls -d $O/diffusion_s*/)
$py experiment/evaluate.py $RUNS --data $D --out $O/summary_diffusion.md > $O/evaluate_diffusion.log 2>&1 || log "FAILED evaluate"
$py experiment/fid_range.py $RUNS --data $D --resamples 20 > $O/fid_range_diffusion.txt 2>&1 || log "FAILED fid_range"
grep -v WARNING $O/fid_range_diffusion.txt
log "done"
