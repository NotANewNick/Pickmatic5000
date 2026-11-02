#!/usr/bin/env bash
# The dots tool (a grid of single tiles, every k-th tile of every k-th row, in one pass)
# as its own specialist, trained like the others: round 1 on real
# contexts (60 epochs), round 2 on the round-2 specialists' drawings (circles included).
# S2_dots then joins the S2 toolbox. Then fixed plans on valid.
# Waits for the whole circles chain (PID $1) first: two trainings (5.4 GB each) do not fit
# on the 11 GB card together.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
# Retry a run that dies of "Illegal instruction" (exit 132; MKL on this old Xeon, despite
# the settings above; reruns succeed).
py() { local i s; for i in 1 2 3; do .venv/bin/python "$@"; s=$?; [ $s -ne 132 ] && return $s; echo "[$(date +%T)] illegal instruction, retry $i: ${*:1:3}" >&2; done; return 132; }
D=work/p2/data R=work/p2/runs E=work/p2/eval_spec
log() { echo "[$(date +%T)] $*"; }
CIRCLES=${1:-}
[ -n "$CIRCLES" ] && while kill -0 "$CIRCLES" 2>/dev/null; do sleep 30; done
if [ ! -f $R/S_dots/specialist.safetensors ]; then
    log "round 1: dots"
    py experiment/specialists.py train --tool dots --data $D --out $R/S_dots > $R/S_dots.log 2>&1 || { log "FAILED round 1"; exit 1; }
fi
[ -f $R/S2_circles/specialist.safetensors ] || { log "no S2_circles"; exit 1; }
for t in ring lines holes circles; do ln -sfn S2_$t $R/D1_$t; done
ln -sfn S_dots $R/D1_dots
log "round 2: dots"
py experiment/specialists.py train --tool dots --data $D --out $R/S2_dots --context drawn --init $R/D1 \
    --epochs 20 --lr 1e-4 --minutes 30 > $R/S2_dots.log 2>&1 || { log "FAILED round 2"; exit 1; }
log "check: fixed plans on valid"
RUNS=""
for p in "dots:2:0:0,holes" "dots:2:0:0,dots:2:1:1,holes" "dots:4:0:0,dots:2:0:0,holes" "dots:3:1:1,holes" \
         "lines:0:2:0,holes" "ring:0,lines:45:3:0,holes"; do
    n=$(echo "$p" | tr ':,' '_-')
    py experiment/specialists.py draw --tools $R/S2 --data $D --plan "$p" --split valid --out $E/S2d_$n > /dev/null 2>&1 \
        || log "FAILED draw $p"
    RUNS="$RUNS $E/S2d_$n"
done
py experiment/evaluate.py $RUNS --data $D --out $E/summary_dots.md > $E/evaluate_dots.log 2>&1 || log "FAILED evaluate"
log "done"
