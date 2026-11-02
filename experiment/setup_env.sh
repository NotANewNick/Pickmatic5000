#!/usr/bin/env bash
# Create .venv in the project root with the right PyTorch build for this machine.
#   ./experiment/setup_env.sh            # auto-detect NVIDIA (CUDA), AMD (ROCm) or CPU
#   ./experiment/setup_env.sh cuda|rocm|cpu
# Safe to rerun: does nothing if .venv already matches the backend.
set -euo pipefail
cd "$(dirname "$0")/.."

backend="${1:-auto}"
if [[ "$backend" == auto ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        backend=cuda
    elif command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q gfx; then
        backend=rocm
    elif [[ -e /dev/kfd ]]; then
        backend=rocm  # AMD kernel driver present, ROCm user tools not installed
    else
        echo "No NVIDIA or AMD GPU found: installing the CPU build (drawing and scoring only, slow)."
        backend=cpu
    fi
fi
# All results were made with torch 2.8.0 (keep it, so they reproduce). These indexes carry 2.8.0 builds.
case "$backend" in
    cuda) index=https://download.pytorch.org/whl/cu128 ;;
    rocm) index=https://download.pytorch.org/whl/rocm6.4 ;;
    cpu) index=https://download.pytorch.org/whl/cpu ;;
    *) echo "Unknown backend '$backend' (use cuda, rocm or cpu)" >&2; exit 1 ;;
esac

stamp=.venv/.ready
# Environments made before the rename keep working (their extra packages are unused).
[[ -f .venv/.alodlm-ready && ! -f "$stamp" ]] && mv .venv/.alodlm-ready "$stamp"
if [[ -f "$stamp" && "$(cat "$stamp")" == "$backend" ]]; then
    exit 0
fi
echo "Setting up for $backend using $index (one-time, downloads ~3-6 GB)"

# uv fetches Python 3.12 and installs packages quickly. If it is missing, put a
# private copy in .tools/ (nothing outside the project is changed).
if ! command -v uv >/dev/null; then
    if [[ ! -x .tools/uv ]]; then
        echo "Installing uv into .tools/"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh
    fi
    export PATH="$PWD/.tools:$PATH"
fi
rm -rf .venv
uv venv --seed --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.8.0" --index-url "$index"
uv pip install --python .venv/bin/python pillow numpy "safetensors==0.8.0" "huggingface_hub==0.36.2"
.venv/bin/python - "$backend" <<'EOF'
import sys
import torch
wanted = sys.argv[1]
print(f"torch {torch.__version__}; GPU available: {torch.cuda.is_available()}; "
      f"devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  {i}: {torch.cuda.get_device_name(i)}")
if wanted != "cpu" and not torch.cuda.is_available():
    hint = {"cuda": "NVIDIA: needs driver 570 or newer (check nvidia-smi).",
            "rocm": "AMD: needs the amdgpu driver and your user in the 'render' and "
                    "'video' groups (sudo usermod -aG render,video $USER, then log in again)."}
    raise SystemExit(f"PyTorch cannot see the GPU. {hint[wanted]}")
EOF
echo "$backend" > "$stamp"
echo "Environment ready ($backend)."
