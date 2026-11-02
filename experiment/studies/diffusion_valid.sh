#!/usr/bin/env bash
# Diffusion (DDIM, no guidance) at 1-20 steps on the 1,000 valid digits: the diffusion rows of
# reference/digits-28x28/results/tool_picker/summary_diffusion.md. (Run by hand on 2026-11-01;
# reconstructed from the run logs afterwards.) The FID ranges (fid_range_diffusion.txt) were
# computed later, on 2026-11-02 on the CPU, from the same stored runs; maskgit_s8 is a check row
# and matches fid_range.txt.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
py=.venv/bin/python D=work/p2/data R=work/p2/runs V=work/p2/eval_tp
log() { echo "[$(date +%T)] $*"; }
for s in 1 2 3 4 8 10 20; do
    log "diffusion $s steps on valid"
    $py experiment/baselines.py generate diffusion --model $R/D_diffusion --data $D --out $V/diffusion_s$s --steps $s --cfg 1 --split valid > $V/diffusion_s$s.log 2>&1 || log "FAILED diffusion $s"
done
RUNS=$(ls -d $V/diffusion_s*/)
$py experiment/evaluate.py $RUNS --data $D --out $V/summary_diffusion.md > $V/evaluate_diffusion.log 2>&1 || log "FAILED evaluate"
log "fid range"
CUDA_VISIBLE_DEVICES="" $py experiment/fid_range.py $V/maskgit_s8 $V/diffusion_s{2,4,8,20} --data reference/digits-28x28/data \
    --resamples 20 > $V/fid_range_diffusion.txt 2>&1 || log "FAILED fid_range"
log "done"
