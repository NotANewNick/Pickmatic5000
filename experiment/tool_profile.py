"""Where each tool (or baseline) is good or bad: quality per digit class and per kind of digit.

All runs must have drawn the same prompts (same split, same order). For every run and every
group of prompts it reports "correct digit" (the judge of evaluate.py) and the FID of that
group against the true images of the same prompts, with a 10-90% resampling range moved by
its shift as in fid_range.py.

Groups:
  class      the requested digit (about 100 prompts each on the 1,000 test digits)
  typical    terciles of how unusual the TRUE digit is: the distance of its judge features
             from the mean of its class (over the prompts), split within each class so every
             tercile has the same class mix. "unusual" digits are the hard ones to match.

FID on 100-333 images sits well above FID on 1,000 (fewer images, same features), so only
compare runs within the same group, never a group's FID with the overall one.

  python experiment/tool_profile.py --data reference/digits-28x28/data --out work/profile RUN [RUN ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate
from common import device_info, read_json, split_images, write_json


def features(model, images, device):
    with torch.no_grad():
        return torch.cat([model.features(evaluate.to_tensor(images[i:i + 500], device))
                          for i in range(0, len(images), 500)]).double().cpu()


def frechet(a, b):
    """evaluate.fid on precomputed features."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = torch.cov(a.T), torch.cov(b.T)
    values, vectors = torch.linalg.eigh(cov_a)
    root = vectors @ torch.diag(values.clamp(min=0).sqrt()) @ vectors.T
    middle = torch.linalg.eigvalsh(root @ cov_b @ root).clamp(min=0).sqrt().sum()
    return float(((mu_a - mu_b) ** 2).sum() + cov_a.trace() + cov_b.trace() - 2 * middle)


def fid_with_range(a, b, resamples, rng):
    point = frechet(a, b)
    values = []
    for _ in range(resamples):
        pick = torch.as_tensor(rng.integers(0, len(a), len(a)))
        values.append(frechet(a[pick], b[pick]))
    low, middle, high = np.percentile(values, [10, 50, 90])
    return point, point - (middle - low), point + (high - middle)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="folder for profile.md and profile.json")
    parser.add_argument("--resamples", type=int, default=20)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [Path(r) for r in args.runs]
    meta = read_json(runs[0] / "meta.json")
    model = evaluate.classifier(meta["dataset"], meta.get("size", 28), args.data, device)
    split = meta.get("split", "test")
    images, _ = split_images(meta, split)
    index = None
    drawn = {}
    for run in runs:
        samples = [json.loads(line) for line in (run / "samples.jsonl").open()]
        this = np.array([s["index"] for s in samples])
        if index is None:
            index, labels = this, np.array([s["label"] for s in samples])
        elif not np.array_equal(index, this) or read_json(run / "meta.json").get("split", "test") != split:
            raise SystemExit(f"{run} drew other prompts than {runs[0]}")
        generated = np.load(run / "generated.npy")
        with torch.no_grad():
            predicted = torch.cat([model(evaluate.to_tensor(generated[i:i + 500], device)).argmax(1).cpu()
                                   for i in range(0, len(generated), 500)]).numpy()
        steps = float(np.mean([s["sequential_steps"] for s in samples]))
        drawn[run.name] = (features(model, generated, device), predicted == labels, steps)
    truth = features(model, images[index], device)

    groups = {f"class {c}": labels == c for c in np.unique(labels)}
    distance = np.zeros(len(labels))
    tercile = np.zeros(len(labels), int)
    for c in np.unique(labels):
        member = np.where(labels == c)[0]
        distance[member] = (truth[member] - truth[member].mean(0)).norm(dim=1).numpy()
        rank = np.argsort(np.argsort(distance[member]))
        tercile[member] = rank * 3 // len(member)
    for t, name in enumerate(("typical", "middle", "unusual")):
        groups[name] = tercile == t

    rng = np.random.default_rng(0)
    result = {"hardware": device_info()["device"], "split": split, "prompts": len(labels),
              "groups": {g: int(m.sum()) for g, m in groups.items()}, "runs": {}}
    for name, (feats, correct, steps) in drawn.items():
        entry = {"steps": steps, "all": {"correct": float(correct.mean()),
                                         "fid": fid_with_range(feats, truth, args.resamples, rng)}}
        for g, member in groups.items():
            m = torch.as_tensor(member)
            entry[g] = {"correct": float(correct[member].mean()),
                        "fid": fid_with_range(feats[m], truth[m], args.resamples, rng)}
        result["runs"][name] = entry
        print(f"{name}: done", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "profile.json", result)
    names = list(result["runs"])
    lines = [f"# Tool profile ({split}, {len(labels)} prompts, {result['hardware']})", "",
             "Each cell: correct digit % / FID (10-90% range). Lower FID = closer to real. "
             "Compare within a column only.", ""]
    columns = ["all"] + list(groups)
    lines.append("| tool | steps | " + " | ".join(columns) + " |")
    lines.append("|---|---|" + "---|" * len(columns))
    for name in names:
        entry = result["runs"][name]
        cells = [f"{100 * entry[c]['correct']:.1f} / {entry[c]['fid'][0]:.1f} "
                 f"({entry[c]['fid'][1]:.1f}-{entry[c]['fid'][2]:.1f})" for c in columns]
        lines.append(f"| {name} | {entry['steps']:.1f} | " + " | ".join(cells) + " |")
    lines += ["", "Best FID per column (ties: ranges overlap with the best):", ""]
    for c in columns:
        best = min(names, key=lambda n: result["runs"][n][c]["fid"][0])
        ceiling = result["runs"][best][c]["fid"][2]
        tied = [n for n in names if n != best and result["runs"][n][c]["fid"][1] <= ceiling]
        lines.append(f"- {c}: {best}" + (f" (tied: {', '.join(tied)})" if tied else ""))
    (out / "profile.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
