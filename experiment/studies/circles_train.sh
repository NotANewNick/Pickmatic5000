#!/usr/bin/env bash
# The circles tool (concentric circles in one pass) as its own specialist, trained
# like the other three: round 1 on real contexts (MaskGIT's network shape and budget, 60
# epochs), round 2 on the round-2 specialists' drawings (as S2). S2_circles then joins the
# S2 toolbox. Then fixed plans on valid next to the matching line plans.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
# Retry a run that dies of "Illegal instruction" (exit 132; MKL on this old Xeon, despite
# the settings above; reruns succeed).
py() { local i s; for i in 1 2 3; do .venv/bin/python "$@"; s=$?; [ $s -ne 132 ] && return $s; echo "[$(date +%T)] illegal instruction, retry $i: ${*:1:3}" >&2; done; return 132; }
D=work/p2/data R=work/p2/runs E=work/p2/eval_spec
log() { echo "[$(date +%T)] $*"; }
if [ ! -f $R/S_circles/specialist.safetensors ]; then
    log "round 1: circles"
    py experiment/specialists.py train --tool circles --data $D --out $R/S_circles > $R/S_circles.log 2>&1 || { log "FAILED round 1"; exit 1; }
fi
# C1 = the round-2 specialists plus the round-1 circles; round 2 for circles starts there
for t in ring lines holes; do ln -sfn S2_$t $R/C1_$t; done
ln -sfn S_circles $R/C1_circles
if [ ! -f $R/S2_circles/specialist.safetensors ]; then
    log "round 2: circles"
    py experiment/specialists.py train --tool circles --data $D --out $R/S2_circles --context drawn --init $R/C1 \
        --epochs 20 --lr 1e-4 --minutes 30 > $R/S2_circles.log 2>&1 || { log "FAILED round 2"; exit 1; }
fi
log "check: fixed plans on valid"
RUNS=""
for p in "holes" "lines:0:2:0,holes" "circles:mid:2:0,holes" "circles:top:2:0,holes" "circles:bottom:2:0,holes" \
         "ring:0,lines:45:3:0,holes" "ring:0,circles:mid:3:0,holes"; do
    n=$(echo "$p" | tr ':,' '_-')
    py experiment/specialists.py draw --tools $R/S2 --data $D --plan "$p" --split valid --out $E/S2c_$n > /dev/null 2>&1 \
        || log "FAILED draw $p"
    RUNS="$RUNS $E/S2c_$n"
done
py experiment/evaluate.py $RUNS --data $D --out $E/summary_circles.md > $E/evaluate_circles.log 2>&1 || log "FAILED evaluate"
log "done"
