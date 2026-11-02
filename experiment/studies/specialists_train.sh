#!/usr/bin/env bash
# The first three pattern tools (ring, lines, holes), each with the MaskGIT baseline's
# network shape and training budget (60 epochs of the same 20,000 digits), on 28x28 digits.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
py=.venv/bin/python D=work/p2/data R=work/p2/runs
log() { echo "[$(date +%T)] $*"; }
for t in ring lines holes; do
    log "train specialist $t"
    $py experiment/specialists.py train --tool $t --data $D --out $R/S_$t > $R/S_$t.log 2>&1 || log "FAILED $t"
done
log "check: fixed plans on valid"
for p in "holes" "ring:0,holes" "lines:0:2:0,holes" "ring:0,lines:45:3:0,holes"; do
    n=$(echo "$p" | tr ':,' '_-')
    $py experiment/specialists.py draw --tools $R/S --data $D --plan "$p" --split valid --out work/p2/eval_spec/$n > /dev/null 2>&1 || log "FAILED draw $p"
done
log "done"
