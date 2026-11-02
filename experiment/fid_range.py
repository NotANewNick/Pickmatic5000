"""FID with a resampling range. Run after evaluate.py (reads each run's summary.json).

The point value is the FID on all generated images (as in summary.json). The range comes
from resampling the image pairs (generated, true) with replacement. A resample repeats
some images, and repeats raise FID, so the resampled FIDs sit above the point value. The
range is therefore moved down by that shift: from point - (median - p10) to
point + (p90 - median), with p10, median and p90 taken over the resampled FIDs. The point
always lies inside. (Splitting into halves avoids repeats but gives FIDs on half as many
images, which have their own, larger shift, so we use the shifted resampling range.)

--real-floor also prints the FID of real against real: the true test images of the prompts
against other test images with the same class mix, over the same number of draws. Only
compare it with the "vs other real" column it adds to each run (the run's images against
those same draws). Do not compare it with the main FID: each generated image keeps the top
rows of its own true image, which the other real images do not, so the main FID can sit
below the floor even for a model worse than real data.

Runs on a held-out split (meta.json "split": practice or valid) take their true images from
the training split, and their "other real" images from the other held-out digits (never
from the training draw).

  python experiment/fid_range.py --data reference/digits-28x28/data RUN [RUN ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate
from common import cpu_name, device_info, held_out_index, read_json, split_images


def other_real(images, labels, index, draws, pool=None):
    """Draws of other test images (or other images of `pool`) with the same labels as the
    prompts (same draws every call)."""
    rng = np.random.default_rng(1)
    rest = np.setdiff1d(np.arange(len(images)) if pool is None else pool, index)
    pools = {c: rest[labels[rest] == c] for c in np.unique(labels[index])}
    counts = {c: int((labels[index] == c).sum()) for c in pools}
    return [np.concatenate([rng.choice(pools[c], counts[c], replace=False) for c in pools])
            for _ in range(draws)]


def spread(values):
    low, middle, high = np.percentile(values, [10, 50, 90])
    return f"{middle:6.1f} ({low:6.1f}-{high:6.1f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data", required=True)
    parser.add_argument("--resamples", type=int, default=20)
    parser.add_argument("--real-floor", action="store_true",
                        help="also print the FID of real test images against other real test images")
    args = parser.parse_args()
    device = "cuda" if evaluate.torch.cuda.is_available() else "cpu"
    hardware = device_info()["device"] if device == "cuda" else cpu_name()
    print(f"range: 10-90% over {args.resamples} resamples, moved by their shift (see --help); "
          f"features on {hardware}", flush=True)
    floors = {}
    for run in map(Path, args.runs):
        meta, summary = read_json(run / "meta.json"), read_json(run / "summary.json")
        size = meta.get("size", 28)
        model = evaluate.classifier(meta["dataset"], size, args.data, device)
        samples = [json.loads(line) for line in (run / "samples.jsonl").open()]
        generated = np.load(run / "generated.npy")
        split = meta.get("split", "test")
        images, labels = split_images(meta, split)
        index = np.array([s["index"] for s in samples])
        truth = images[index]
        key = (meta["dataset"], size, split, index.tobytes())
        if args.real_floor and key not in floors:
            pool = None
            if split != "test":  # all held-out digits, never the training draw
                info = read_json(Path(args.data) / "info.json")
                pool = np.concatenate(list(held_out_index(len(images), info["seed"],
                                                          info["train_images"]).values()))
            floors[key] = other_real(images, labels, index, args.resamples, pool)
            values = [evaluate.fid(model, truth, images[other], device) for other in floors[key]]
            name = meta["dataset"] + ("" if split == "test" else ", " + split)
            print(f"{'real vs real (' + name + ')':34s} FID {spread(values)}  "
                  f"{len(index)} true vs {len(index)} other real images, {args.resamples} draws; "
                  f"compare with 'vs other real' below", flush=True)
        rng = np.random.default_rng(0)
        values = [evaluate.fid(model, generated[i], truth[i], device)
                  for i in (rng.integers(0, len(generated), len(generated)) for _ in range(args.resamples))]
        low, middle, high = np.percentile(values, [10, 50, 90])
        point = summary["fid"]
        low, high = point - (middle - low), point + (high - middle)
        other = ""
        if args.real_floor:
            values = [evaluate.fid(model, generated, images[o], device) for o in floors[key]]
            other = f"  vs other real {spread(values)}"
        print(f"{run.name:34s} correct {100 * summary['class_accuracy']:5.1f}%  FID {point:6.1f} "
              f"({low:6.1f}-{high:6.1f})  copy {summary['copy_ratio']:.2f}  "
              f"steps {summary['sequential_steps_per_image']:6.1f}{other}", flush=True)


if __name__ == "__main__":
    main()
