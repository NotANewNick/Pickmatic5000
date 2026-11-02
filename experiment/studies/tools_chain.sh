#!/usr/bin/env bash
# The overnight chain: the added tools (circles, dots, fractal; each script skips
# rounds already trained), then the picker queue on all six tools (O = picker folder).
set -uo pipefail
cd "$(dirname "$0")/../.."
for t in circles dots fractal; do
    experiment/studies/${t}_train.sh >> work/p2/runs/${t}_train.log 2>&1 || echo "[$(date +%T)] $t chain failed"
done
O=${O:-work/p2/tp5} experiment/studies/tool_picker_queue.sh
