"""Score generated images and compare compute across runs.

Quality:  a small CNN classifier (trained here once, cached) checks that the image
          shows the requested class; MSE compares the generated rows to the real test image.
Realism:  "correct class" alone rewards blurry average or copied digits, so two more:
          FID on the classifier's 128 features (generated vs the real test images of the
          same prompts; lower is better, and real vs real gives the noise floor), and a
          copy check: each image's distance to its nearest training image, divided by the
          same for the real test images (about 1 is normal, well below 1 means copying).
Compute:  token-layer evaluations and sequential steps (passes) per image
          (hardware-independent), plus wall time (only comparable on the same machine).
Split:    each run's meta.json says which prompts it drew ("split": test, or the held-out
          practice/valid digits from the training split; see common.test_cases); the true
          images come from there. The copy check always compares with the training draw.
Judge:    <data>/classifier_<dataset>_<size>.safetensors. Always use the included judge
          (reference/digits-28x28/data): a judge trained anew differs, and so do the FIDs.

Run: python experiment/evaluate.py --data reference/digits-28x28/data --out work/eval/summary.md RUN [RUN ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CLASSES, device_info, load_model, load_split, read_json, save_model, split_images, training_split,
                    write_json)


class Classifier(nn.Module):
    def __init__(self, size, classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * (size // 4) ** 2, 128)
        self.fc2 = nn.Linear(128, classes)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.fc1(x.flatten(1)))

    def forward(self, x):
        return self.fc2(self.features(x))


def fid(model, generated, real, device):
    """Frechet distance between the classifier features of two image sets."""
    with torch.no_grad():
        a = torch.cat([model.features(to_tensor(generated[i:i + 500], device))
                       for i in range(0, len(generated), 500)]).double().cpu()
        b = torch.cat([model.features(to_tensor(real[i:i + 500], device))
                       for i in range(0, len(real), 500)]).double().cpu()
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = torch.cov(a.T), torch.cov(b.T)
    # trace(sqrt(cov_a cov_b)) = trace(sqrt(R cov_b R)) with R = sqrt(cov_a), both symmetric
    values, vectors = torch.linalg.eigh(cov_a)
    root = vectors @ torch.diag(values.clamp(min=0).sqrt()) @ vectors.T
    middle = torch.linalg.eigvalsh(root @ cov_b @ root).clamp(min=0).sqrt().sum()
    return float(((mu_a - mu_b) ** 2).sum() + cov_a.trace() + cov_b.trace() - 2 * middle)


def nearest_distance(images, train, hint_rows):
    """Per image: root mean square distance of the body rows to the nearest training image."""
    x = torch.tensor(images[:, hint_rows:].reshape(len(images), -1), dtype=torch.float32)
    t = torch.tensor(train[:, hint_rows:].reshape(len(train), -1), dtype=torch.float32)
    best = torch.cat([torch.cdist(x[i:i + 250], t).min(1).values for i in range(0, len(x), 250)])
    return (best / x.shape[1] ** 0.5).numpy()


TRAIN_CACHE = {}


def training_images(data, dataset, size):
    """The training images the models saw (common.training_split)."""
    key = (str(data), dataset, size)
    if key not in TRAIN_CACHE:
        TRAIN_CACHE[key] = training_split(data)[0]
    return TRAIN_CACHE[key]


def to_tensor(images, device):
    return torch.tensor(images, dtype=torch.float32, device=device).unsqueeze(1) / 255.0


def classifier(dataset, size, data, device):
    path = Path(data) / f"classifier_{dataset}_{size}.safetensors"
    model = Classifier(size, CLASSES[dataset]).to(device)
    if path.is_file():
        model.load_state_dict(load_model(path, device)[0])
        return model.eval()
    if path.with_suffix(".pt").is_file():
        raise SystemExit(f"old judge format: {path.with_suffix('.pt')}; "
                         f"run experiment/convert_models.py on it first")
    torch.manual_seed(0)
    images, labels = load_split(dataset, "train", size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3 if CLASSES[dataset] == 10 else 6):
        order = np.random.default_rng(epoch).permutation(len(images))
        for start in range(0, len(images), 256):
            index = order[start:start + 256]
            loss = F.cross_entropy(model(to_tensor(images[index], device)),
                                   torch.tensor(labels[index], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test_images, test_labels = load_split(dataset, "test", size)
    model.eval()
    with torch.no_grad():
        predicted = torch.cat([model(to_tensor(test_images[i:i + 1000], device)).argmax(1).cpu()
                               for i in range(0, len(test_images), 1000)])
    accuracy = float((predicted.numpy() == test_labels).mean())
    print(f"classifier test accuracy on real images: {accuracy:.4f}")
    save_model(path, model.state_dict())
    return model


def summarize(run, data, device, model):
    meta = read_json(run / "meta.json")
    samples = [json.loads(line) for line in (run / "samples.jsonl").open()]
    generated = np.load(run / "generated.npy")
    size = meta.get("size", 28)
    split = meta.get("split", "test")  # runs from before the held-out splits used the test digits
    truth_images, _ = split_images(meta, split)
    hint = meta["hint_rows"]
    truth = np.stack([truth_images[s["index"]] for s in samples])
    labels = np.array([s["label"] for s in samples])
    train = training_images(data, meta["dataset"], size)
    with torch.no_grad():
        predicted = model(to_tensor(generated, device)).argmax(1).cpu().numpy()
    body_error = ((generated[:, hint:].astype(np.float64) - truth[:, hint:]) / 255.0) ** 2

    def mean(key):
        values = [s[key] for s in samples if s[key] is not None]
        return float(np.mean(values)) if values else None

    tokens = mean("generated_tokens")
    summary = {
        "run": run.name, "split": split, "kind": meta["kind"], "ordering": meta["ordering"],
        "decode": meta["decode"], "images": len(samples),
        "hardware": meta["hardware"],
        "class_accuracy": float((predicted == labels).mean()),
        "body_mse": float(body_error.mean()),
        "fid": fid(model, generated, truth, device),
        "copy_ratio": float(np.median(nearest_distance(generated, train, hint))
                            / np.median(nearest_distance(truth, train, hint))),
        "complete_images": float(np.mean([s["generated_tokens"] == meta["body_tokens"]
                                          for s in samples])),
        "generated_tokens": tokens,
        "sequential_steps_per_image": mean("sequential_steps"),
        "sequential_layers_per_image": mean("sequential_layers") if "sequential_layers" in samples[0] else None,
        "tokens_per_sequential_step": (tokens / mean("sequential_steps")
                                       if mean("sequential_steps") else None),
        "decode_token_layers_per_image": mean("decode_token_layers"),
        "prefill_token_layers_per_image": mean("prefill_token_layers"),
        "wall_seconds_per_image": mean("wall_seconds"),
        "body_tokens": meta["body_tokens"],
    }

    write_json(run / "summary.json", summary)
    return summary


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/summary.md")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {(read_json(Path(r) / "meta.json")["dataset"],
                 read_json(Path(r) / "meta.json").get("size", 28)) for r in args.runs}
    if len(datasets) != 1:
        raise SystemExit(f"Runs use different datasets or sizes: {datasets}")
    dataset, size = datasets.pop()
    model = classifier(dataset, size, args.data, device)
    summaries = [summarize(Path(r), args.data, device, model) for r in args.runs]

    columns = [
        ("run", "run"), ("split", "split"), ("GPU", None), ("class acc", "class_accuracy"),
        ("body MSE", "body_mse"), ("FID", "fid"), ("copy", "copy_ratio"),
        ("complete", "complete_images"),
        ("steps/img", "sequential_steps_per_image"),
        ("tok/step", "tokens_per_sequential_step"),
        ("seq. layers/img", "sequential_layers_per_image"),
        ("token-layers/img", "decode_token_layers_per_image"),
        ("s/img", "wall_seconds_per_image"),
    ]
    lines = ["| " + " | ".join(name for name, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    for s in summaries:
        cells = []
        for name, key in columns:
            if key is None:
                hw = s["hardware"]
                cells.append(f"{hw.get('device', '?')} ({hw.get('backend', '?')})")
            else:
                cells.append(fmt(s.get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    note = ("\nHardware-independent: class acc, body MSE, steps/img, tok/step, "
            "token-layers/img.\nWall time (s/img) is only comparable "
            "between runs on the same GPU.\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table + "\n" + note)
    write_json(Path(args.out).with_suffix(".json"),
               {"evaluated_on": device_info(), "runs": summaries})
    print(table + "\n" + note)


if __name__ == "__main__":
    main()
