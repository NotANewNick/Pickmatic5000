#!/usr/bin/env bash
# The whole chain after the round-1 specialists (specialists_train.sh, or the last tool's
# training script: PID $1), on 28x28 digits, with every tool in the S2 toolbox (ring, lines,
# holes, and circles and dots when trained):
#   MaskGIT on the valid digits (the bar) -> round 2: specialists fine-tuned on each other's
#   drawings -> both pickers (urgency is an input) -> round 3: specialists fine-tuned on the
#   pickers' own states -> pickers continued on them -> valid draws at five urgencies, scored
#   -> MaskGIT trained as long as all specialists together (fairness), scored the same way.
# Test digits are not drawn here (only once, at the end, after the choice on valid).
set -uo pipefail
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX PYTHONPATH="$PWD/experiment"
# Retry a run that dies of "Illegal instruction" (exit 132; MKL on this old Xeon, despite
# the settings above; reruns succeed).
py() { local i s; for i in 1 2 3; do .venv/bin/python "$@"; s=$?; [ $s -ne 132 ] && return $s; echo "[$(date +%T)] illegal instruction, retry $i: ${*:1:3}" >&2; done; return 132; }
D=work/p2/data R=work/p2/runs O=${O:-work/p2/tp} V=work/p2/eval_tp C=work/p2/picker_noplan/critics
log() { echo "[$(date +%T)] $*"; }
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
mkdir -p $O $V
for s in 1 2 3 4 6 8; do
    [ -f $V/maskgit_s$s/generated.npy ] && continue
    log "maskgit $s steps on valid"
    py experiment/baselines.py generate maskgit --model $R/M_maskgit --data $D --out $V/maskgit_s$s --steps $s --split valid > $V/maskgit_s$s.log 2>&1 || log "FAILED maskgit $s"
done
for s in 1 2 3 4 6 8; do  # the second competitor: MaskGIT with the Halton order (Besnier et al. 2025)
    [ -f $V/maskgit_halton_s$s/generated.npy ] && continue
    log "maskgit halton $s steps on valid"
    py experiment/baselines.py generate maskgit --model $R/M_maskgit --data $D --out $V/maskgit_halton_s$s --steps $s --split valid --order halton > $V/maskgit_halton_s$s.log 2>&1 || log "FAILED maskgit halton $s"
done
# Safeguard after each fine-tuning round: the same fixed plan (outer ring, 45-degree lines,
# holes) drawn on valid with the old and the new specialists; the chain goes on with the
# better of the two (judge FID on valid), so a broken round cannot feed the pickers.
better() {  # better OLD NEW -> prints the prefix to use
    for x in "$1" "$2"; do
        py experiment/specialists.py draw --tools $R/$x --data $D --plan "ring:0,lines:45:3:0,holes" --split valid --out $V/check_$x > /dev/null 2>&1
    done
    py experiment/evaluate.py $V/check_$1 $V/check_$2 --data $D --out $V/check.md > /dev/null 2>&1
    py -c "
import json
a, b = (json.load(open('$V/check_' + x + '/summary.json'))['fid'] for x in ('$1', '$2'))
print('$2' if b < a else '$1')
import sys; print(f'check: $1 FID {a:.1f}, $2 FID {b:.1f}', file=sys.stderr)"
}
TOOLS="ring lines holes $(for t in circles dots; do [ -f $R/S2_$t/specialist.safetensors ] && echo -n "$t "; done)"
log "tools: $TOOLS"
for t in ring lines holes; do
    [ -f $R/S2_$t/specialist.safetensors ] && continue  # done earlier (circles and dots train their own rounds 1-2)
    log "round 2: $t on the specialists' own drawings"
    py experiment/specialists.py train --tool $t --data $D --out $R/S2_$t --context drawn --init $R/S \
        --epochs 20 --lr 1e-4 --minutes 30 > $R/S2_$t.log 2>&1 || log "FAILED round 2 $t"
done
T2=$(better S S2 2>> $V/checks.log); tail -1 $V/checks.log; log "pickers use $T2"
for kind in step plan; do
    log "picker $kind on $T2 (60 min)"
    py experiment/tool_picker.py train --kind $kind --tools $R/$T2 --data $D --critics $C \
        --out $O/$kind --updates 100000 --minutes 60 --check 100 > $O/$kind.log 2>&1 || log "FAILED $kind"
done
for t in $TOOLS; do
    log "round 3: $t on the pickers' states"
    py experiment/specialists.py train --tool $t --data $D --out $R/S3_$t --context picker --init $R/$T2 \
        --pickers $O/step $O/plan --epochs 20 --lr 1e-4 --minutes 30 > $R/S3_$t.log 2>&1 || log "FAILED round 3 $t"
done
T3=$(better $T2 S3 2>> $V/checks.log); tail -1 $V/checks.log; log "continued pickers use $T3"
for kind in step plan; do
    log "picker $kind continued on $T3 (30 min)"
    py experiment/tool_picker.py train --kind $kind --tools $R/$T3 --data $D --critics $C --init $O/$kind \
        --out $O/${kind}3 --updates 100000 --minutes 30 --check 100 > $O/${kind}3.log 2>&1 || log "FAILED ${kind}3"
done
log "valid draws"
for kind in step plan step3 plan3; do
    for u in 0 0.05 0.1 0.15 0.2; do
        py experiment/tool_picker.py draw --picker $O/$kind --urgency $u --split valid --out $V/${kind}_u$u > /dev/null 2>&1 || log "FAILED draw $kind $u"
    done
done
RUNS=$(ls -d $V/*/)
py experiment/evaluate.py $RUNS --data $D --out $V/summary.md > $V/evaluate.log 2>&1 || log "FAILED evaluate"
py experiment/fid_range.py $RUNS --data $D > $V/fid_range.txt 2>&1 || log "FAILED fid_range"
grep -v -i "warn\|MKL" $V/fid_range.txt
# Fairness: MaskGIT gets as many training minutes as all specialist rounds together
E=$(py -c "
import json, glob
m = sum(json.load(open(f))['train_minutes'] for f in glob.glob('$R/S*_*/specialist.json'))
print(max(60, round(60 * m / 56)))")
log "maskgit, $E epochs (as long as all specialists together)"
py experiment/baselines.py train maskgit --data $D --out $R/M_maskgit_long --epochs $E > $R/M_maskgit_long.log 2>&1 || log "FAILED maskgit long"
for s in 1 2 3 4 6 8; do
    py experiment/baselines.py generate maskgit --model $R/M_maskgit_long --data $D --out $V/maskgit_long_s$s --steps $s --split valid > /dev/null 2>&1 || log "FAILED long $s"
    py experiment/baselines.py generate maskgit --model $R/M_maskgit_long --data $D --out $V/maskgit_long_halton_s$s --steps $s --split valid --order halton > /dev/null 2>&1 || log "FAILED long halton $s"
done
RUNS=$(ls -d $V/*/)
py experiment/evaluate.py $RUNS --data $D --out $V/summary.md > $V/evaluate.log 2>&1 || log "FAILED evaluate"
py experiment/fid_range.py $RUNS --data $D > $V/fid_range.txt 2>&1 || log "FAILED fid_range"
grep -v -i "warn\|MKL" $V/fid_range.txt
log "done"
