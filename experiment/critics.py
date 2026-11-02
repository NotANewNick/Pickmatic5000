"""The picker's critics (tool_picker.py): the second digit classifier p2 of the reward, and
the real-vs-drawn checker used with --realism checker.

p2 is deliberately unlike evaluate.py's judge (other layers, other seed), so the picker is
never trained against the classifier that scores it. It is trained on the training draw
only (never on the practice/valid/test digits).

The included p2 (reference/digits-28x28/critics/classifier2.safetensors, 98.86% on the real test
digits) was trained this way on an RTX 2080 Ti. GPU training is not bit-exact, so a new p2
differs a little; keep the included one to reproduce the included pickers.

  python experiment/critics.py --data reference/digits-28x28/data --out work/critics
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device_info, load_model, load_split, read_json, save_model, training_split, write_json
from evaluate import to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SecondClassifier(nn.Module):
    """The reward's digit classifier p2. Deliberately unlike evaluate.py's judge (2 convs +
    2 dense layers, seed 0): 3 convs with batch norm, dropout, 1 dense layer, seed 1."""

    def __init__(self, size=28, classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 5, padding=2), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(96 * (size // 8) ** 2, classes))

    def forward(self, x):
        return self.net(x)

    def features(self, x):
        """The flattened last conv layer (the input of the dense layer)."""
        return self.net[:-2](x)


def load_second_classifier(folder, device):
    """The second classifier in FOLDER (classifier2.safetensors + critics.json), in eval mode."""
    model = SecondClassifier(read_json(Path(folder) / "critics.json")["size"]).to(device)
    model.load_state_dict(load_model(Path(folder) / "classifier2.safetensors", device)[0])
    return model.eval()


class RealismChecker(nn.Module):
    """Real (logit > 0) or generated; log D(x) = logsigmoid(logit)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit(model, images, targets, steps, seed, loss_fn, batch=128, lr=1e-3):
    """Adam for `steps` updates, in epochs of shuffled batches."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    done = 0
    for epoch in range(10 ** 9):
        model.train()
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(images))
        for start in range(0, len(images), batch):
            index = order[start:start + batch]
            loss = loss_fn(model(to_tensor(images[index], DEVICE)),
                           torch.tensor(targets[index], device=DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            done += 1
            if done == steps:
                return model.eval()


@torch.no_grad()
def outputs(model, images):
    return torch.cat([model(to_tensor(images[i:i + 1000], DEVICE)).float().cpu()
                      for i in range(0, len(images), 1000)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    info = read_json(Path(args.data) / "info.json")
    size = info.get("size", 28)
    images, labels = training_split(args.data)
    p2 = fit(SecondClassifier(size).to(DEVICE), images, labels.astype(np.int64), args.steps, 1, F.cross_entropy)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_model(out / "classifier2.safetensors", p2.state_dict())
    test_images, test_labels = load_split(info["dataset"], "test", size)
    accuracy = float((outputs(p2, test_images).argmax(1).numpy() == test_labels).mean())
    write_json(out / "critics.json", {"size": size, "classifier2_test_accuracy": accuracy,
                                      "classifier_steps": args.steps, "hardware": device_info()})
    print(f"second classifier: {100 * accuracy:.2f}% on the real test digits; wrote {out}")


if __name__ == "__main__":
    main()
