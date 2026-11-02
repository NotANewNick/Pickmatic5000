#!/usr/bin/env bash
# The fractal tool (Sierpinski triangle from 4 corners at 2 tile sizes, the
# Sierpinski carpet, or only the carpet's holes; one pass) as its own specialist, trained
# like the others: round 1 on real contexts (60 epochs), round 2 on the round-2
# specialists' drawings (circles and dots included). S2_fractal then joins the S2
# toolbox. Then fixed plans on valid. Waits for the dots chain (PID $1) first.
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
# Retry a run that dies of "Illegal instruction" (exit 132; MKL on this old Xeon, despite
# the settings above; reruns succeed).
py() { local i s; for i in 1 2 3; do .venv/bin/python "$@"; s=$?; [ $s -ne 132 ] && return $s; echo "[$(date +%T)] illegal instruction, retry $i: ${*:1:3}" >&2; done; return 132; }
D=work/p2/data R=work/p2/runs E=work/p2/eval_spec
log() { echo "[$(date +%T)] $*"; }
BEFORE=${1:-}
[ -n "$BEFORE" ] && while kill -0 "$BEFORE" 2>/dev/null; do sleep 30; done
if [ ! -f $R/S_fractal/specialist.safetensors ]; then
    log "round 1: fractal"
    py experiment/specialists.py train --tool fractal --data $D --out $R/S_fractal > $R/S_fractal.log 2>&1 || { log "FAILED round 1"; exit 1; }
fi
for t in ring lines holes circles dots; do [ -f $R/S2_$t/specialist.safetensors ] && ln -sfn S2_$t $R/F1_$t; done
ln -sfn S_fractal $R/F1_fractal
log "round 2: fractal"
py experiment/specialists.py train --tool fractal --data $D --out $R/S2_fractal --context drawn --init $R/F1 \
    --epochs 20 --lr 1e-4 --minutes 30 > $R/S2_fractal.log 2>&1 || { log "FAILED round 2"; exit 1; }
log "check: fixed plans on valid"
RUNS=""
for p in "fractal:carpetholes,holes" "fractal:tri:tl:1,holes" "fractal:tri:br:2,holes" "fractal:carpet,holes" \
         "lines:0:2:0,holes" "ring:0,lines:45:3:0,holes"; do
    n=$(echo "$p" | tr ':,' '_-')
    py experiment/specialists.py draw --tools $R/S2 --data $D --plan "$p" --split valid --out $E/S2f_$n > /dev/null 2>&1 \
        || log "FAILED draw $p"
    RUNS="$RUNS $E/S2f_$n"
done
py experiment/evaluate.py $RUNS --data $D --out $E/summary_fractal.md > $E/evaluate_fractal.log 2>&1 || log "FAILED evaluate"
log "done"
