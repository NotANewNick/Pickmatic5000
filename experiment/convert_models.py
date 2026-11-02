"""Convert old PyTorch model files (.pt, pickled) to .safetensors + .json.

    experiment/convert_models.py FILE.pt ...        (no arguments: every .pt under reference/)

FILE.safetensors gets the weights (the state dict) and FILE.json the other entries (sizes,
tool name, training notes), if there are any. Optimizer state is dropped (not needed to
draw). Every tensor is read back and checked bit for bit against the .pt file. The .pt files
are left in place; delete them yourself after the check. This is how the included models in
reference/ were converted (2026-11-02); the training code now writes this format directly.
"""

import sys
from pathlib import Path

import torch

from common import PROJECT, load_model, save_model

OPTIMIZER = {"optimizer", "optim", "opt", "scheduler"}


def convert(pt):
    state = torch.load(pt, map_location="cpu", weights_only=True)
    if all(isinstance(v, torch.Tensor) for v in state.values()):
        weights, settings = state, None  # a plain state dict
    else:
        tables = [k for k, v in state.items() if isinstance(v, dict) and v
                  and all(isinstance(x, torch.Tensor) for x in v.values()) and k not in OPTIMIZER]
        if len(tables) != 1:
            raise SystemExit(f"{pt}: expected one state dict, found {tables}")
        weights = state[tables[0]]
        settings = {k: v for k, v in state.items() if k != tables[0] and k not in OPTIMIZER}
        if any(isinstance(v, torch.Tensor) for v in settings.values()):
            raise SystemExit(f"{pt}: loose tensors outside the state dict")
    out = pt.with_suffix(".safetensors")
    save_model(out, weights, settings)
    back, _ = load_model(out)
    assert back.keys() == weights.keys()
    for k, v in weights.items():
        assert back[k].dtype == v.dtype and torch.equal(back[k], v), f"{pt}: {k} differs"
    files = [out] + ([out.with_suffix(".json")] if settings is not None else [])
    print(f"{pt.relative_to(PROJECT)}: {pt.stat().st_size:,} -> "
          + " + ".join(f"{f.name} {f.stat().st_size:,}" for f in files) + " bytes, identical", flush=True)


def main():
    paths = [Path(p).resolve() for p in sys.argv[1:]] or sorted((PROJECT / "reference").rglob("*.pt"))
    for pt in paths:
        convert(pt)


if __name__ == "__main__":
    main()
