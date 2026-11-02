#!/usr/bin/env bash
# One command: sets up the environment on first use (NVIDIA, AMD or CPU), draws digits with the
# included pattern tools and pickers, and scores them (FID, correct digit, copy check; the
# hardware is recorded with every result). Nothing is trained; everything comes from reference/.
# The model weights (*.safetensors, 392 MB) are on Hugging Face (dontknowmyname/Pickmatic5000);
# they are downloaded into reference/ the first time, and skipped if already there.
#
#   ./run.sh                 both pickers (step, plan) at urgency 0, 0.05 and 0.1, plus the
#                            baselines (MaskGIT, Halton-order MaskGIT, diffusion), then scores
#   ./run.sh pickers         only the pickers
#   ./run.sh baselines       only the baselines (MaskGIT and diffusion at 2, 4 and 8 steps,
#                            Halton-order MaskGIT at 8)
#   ./run.sh plan "ring:0,lines:45:3:0,holes"
#                            one fixed tool sequence, drawn by the specialists (no picker)
#   ./run.sh models          only download the model weights (skipped if already there;
#                            while the HF repo is private, first run .venv/bin/hf auth login)
#   ./run.sh update          get the latest code (git pull) and any missing model weights
#
# Options:  --count N      digits per run (default 1000; the results in README use 1,000)
#           --split NAME   valid (default, as in README) or test
#           --out DIR      where runs and scores go (default work/run-DATE-TIME)
#           --cpu          run on the CPU even if a GPU is present (slow: use a small --count;
#                          the setup still installs the GPU build if a GPU is found)
#           --help         show this text
#
# The scores are in OUT/summary.md (all measures) and OUT/fid_range.txt (FID with its range).
# On old Xeons a step can crash now and then with "Illegal instruction" (MKL); it is retried once.
set -euo pipefail
cd "$(dirname "$0")"
help() { sed -n '2,/^set /{/^#/s/^# \{0,1\}//p}' run.sh; }

mode=all
if [[ $# -gt 0 && "$1" != -* ]]; then mode="$1"; shift; fi
plan=""
if [[ "$mode" == plan ]]; then
    [[ $# -gt 0 && "$1" != -* ]] || { echo "Usage: ./run.sh plan \"ring:0,lines:45:3:0,holes\"" >&2; exit 1; }
    plan="$1"; shift
fi
count=1000 split=valid out="" cpu=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --count) count="$2"; shift 2 ;;
        --split) split="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --cpu) cpu=1; shift ;;
        -h|--help) help; exit 0 ;;
        *) echo "Unknown option $1 (see ./run.sh --help)" >&2; exit 1 ;;
    esac
done
case "$split" in valid|test) ;; *) echo "--split must be valid or test" >&2; exit 1 ;; esac

R=reference/digits-28x28
DATA=$R/data TOOLS=$R/specialists/S2 PICKERS=$R/results/tool_picker/pickers
export PYTHONPATH="$PWD/experiment"
# Old Xeons: keep MKL off AVX-512 ("Illegal instruction" crashes otherwise)
export MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX
log() { echo "[$(date +%T)] $*"; }
py() {  # .venv's python; one retry after an "Illegal instruction" crash (exit 132)
    local rc=0
    .venv/bin/python "$@" || rc=$?
    if [[ $rc == 132 ]]; then
        log "Illegal instruction (old CPU and MKL), retrying once"
        rc=0; .venv/bin/python "$@" || rc=$?
    fi
    return $rc
}

HF_REPO=dontknowmyname/Pickmatic5000
WEIGHTS=(
    $R/data/classifier_mnist_28.safetensors
    $R/critics/classifier2.safetensors
    $R/baselines/M_maskgit/maskgit.safetensors
    $R/baselines/D_diffusion/diffusion.safetensors
    $R/results/tool_picker/pickers/step/picker.safetensors
    $R/results/tool_picker/pickers/plan/picker.safetensors
)
for t in circles dots fractal holes lines ring; do WEIGHTS+=("${TOOLS}_$t/specialist.safetensors"); done

models() {  # download the weights from Hugging Face into the same reference/ paths, if missing
    local f missing=0
    for f in "${WEIGHTS[@]}"; do [[ -f "$f" ]] || missing=$((missing + 1)); done
    if [[ $missing == 0 ]]; then return 0; fi
    log "downloading $missing model files from https://huggingface.co/$HF_REPO"
    if [[ ! -x .venv/bin/hf ]]; then
        .venv/bin/python -m pip install -q "huggingface_hub==0.36.2"
    fi
    # (hf can exit 0 after a failed download, so check the files instead)
    .venv/bin/hf download "$HF_REPO" --include "*.safetensors" --local-dir . || true
    for f in "${WEIGHTS[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo "Download failed ($f is missing). While the Hugging Face repo is private, log in" >&2
            echo "first with your own token (https://huggingface.co/settings/tokens):" >&2
            echo "  .venv/bin/hf auth login     then run ./run.sh models again" >&2
            exit 1
        fi
    done
    log "model weights ready"
}

pickers() {
    for kind in step plan; do
        for u in 0 0.05 0.1; do
            log "picker $kind, urgency $u"
            py experiment/tool_picker.py draw --picker $PICKERS/$kind --tools $TOOLS --data $DATA \
                --split "$split" --count "$count" --urgency $u --out "$out/${kind}_u$u"
        done
    done
}

baselines() {
    for s in 2 4 8; do
        log "MaskGIT, $s steps"
        py experiment/baselines.py generate maskgit --model $R/baselines/M_maskgit --data $DATA \
            --split "$split" --count "$count" --steps $s --out "$out/maskgit_s$s"
        log "diffusion, $s steps"
        py experiment/baselines.py generate diffusion --model $R/baselines/D_diffusion --data $DATA \
            --split "$split" --count "$count" --steps $s --cfg 1 --out "$out/diffusion_s$s"
    done
    log "MaskGIT with the Halton order, 8 steps"
    py experiment/baselines.py generate maskgit --model $R/baselines/M_maskgit --data $DATA \
        --split "$split" --count "$count" --steps 8 --order halton --out "$out/maskgit_halton_s8"
}

score() {
    log "scoring"
    runs=("$out"/*/)
    py experiment/evaluate.py "${runs[@]}" --data $DATA --out "$out/summary.md" > "$out/evaluate.log"
    py experiment/fid_range.py "${runs[@]}" --data $DATA > "$out/fid_range.txt" 2> "$out/fid_range.log"
    echo
    cat "$out/fid_range.txt"
    echo
    echo "All measures: $out/summary.md   Pictures: $out/*/samples.png"
}

case "$mode" in
    all|pickers|baselines|plan)
        experiment/setup_env.sh
        models
        # After the setup, so it still checks the GPU build it installed
        if [[ $cpu == 1 ]]; then export CUDA_VISIBLE_DEVICES=""; fi
        out="${out:-work/run-$(date +%Y%m%d-%H%M%S)}"
        mkdir -p "$out"
        case "$mode" in
            all) pickers; baselines ;;
            pickers) pickers ;;
            baselines) baselines ;;
            plan)
                log "fixed plan $plan"
                py experiment/specialists.py draw --tools $TOOLS --data $DATA --split "$split" \
                    --count "$count" --plan "$plan" --out "$out/plan" ;;
        esac
        score ;;
    models)
        experiment/setup_env.sh
        models
        log "all ${#WEIGHTS[@]} model files are in place" ;;
    update)
        git pull --ff-only
        experiment/setup_env.sh
        models ;;
    *)
        help; exit 1 ;;
esac
