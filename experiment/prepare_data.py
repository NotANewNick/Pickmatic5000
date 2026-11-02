"""Make a data folder from data/raw: info.json (the split settings) and the 2x2 tile codebook.

The requests of each split (index and label) need no files: common.test_cases draws them from
info.json and data/raw. The judge (classifier_<dataset>_28.safetensors) is trained by evaluate.py the
first time it is needed.

The digits' folder is included: reference/digits-28x28/data (its codebook, rebuilt this way
on a CPU, is byte-identical; its judge is the one all results used, so keep it).
Use this for a new dataset, e.g. Fashion-MNIST:

  python experiment/prepare_data.py --dataset fashion --out work/fashion/data
"""

import argparse
from pathlib import Path

import numpy as np

from common import HELD_OUT, SIZE, build_codebook, download, training_split, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="mnist", choices=("mnist", "fashion"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--hint-rows", type=int, default=8, help="real rows given in the request (even)")
    parser.add_argument("--train-images", type=int, default=20000)
    parser.add_argument("--test-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.hint_rows % 2:
        parser.error("--hint-rows must be even (the tools work on 2x2 tiles)")
    out = Path(args.out)
    download(args.dataset)
    info = {"dataset": args.dataset, "size": SIZE, "hint_rows": args.hint_rows,
            "train_images": args.train_images, "test_prompts": args.test_prompts,
            **{f"{split}_prompts": n for split, n in HELD_OUT.items()}, "seed": args.seed}
    write_json(out / "info.json", info)
    codebook = out / "codebook_tile.npy"
    if not codebook.is_file():
        print("building the tile codebook (k-means; about 6 minutes on a CPU)", flush=True)
        np.save(codebook, build_codebook(training_split(out)[0]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
