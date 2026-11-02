#!/usr/bin/env bash
# Training of the two baseline models in reference/digits-28x28/baselines/ (M_maskgit and
# D_diffusion), 60 epochs each (the defaults). Taken from the removed phase2.sh (2026-10-29),
# which also trained the old row-by-row models; only these two lines are kept.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
py=.venv/bin/python M=work/p2/data R=work/p2/runs
mkdir -p $R
$py experiment/baselines.py train diffusion --data $M --out $R/D_diffusion > $R/D_diffusion.log 2>&1
$py experiment/baselines.py train maskgit --data $M --out $R/M_maskgit > $R/M_maskgit.log 2>&1
