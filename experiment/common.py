"""Shared helpers: dataset download and splits, the 2x2 tile codebook, prompts, image grids,
hardware info, model files (.safetensors + .json)."""

import gzip
import json
import platform
import struct
import urllib.request
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"  # shared dataset downloads
SIZE = 28
CODEBOOK_SIZE = 512
DATASETS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashion": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
CLASSES = {"mnist": 10, "fashion": 10}


def download(dataset):
    """Fetch the four IDX files once into data/raw/<dataset>."""
    target = RAW / dataset
    target.mkdir(parents=True, exist_ok=True)
    for images, labels in FILES.values():
        for name in (images, labels):
            path = target / name
            if not path.is_file():
                print(f"downloading {DATASETS[dataset]}{name}", flush=True)
                urllib.request.urlretrieve(DATASETS[dataset] + name, path)
    return target


def load_split(dataset, split, size=SIZE):
    """Return uint8 images [N, 28, 28] and int labels [N]."""
    if size != SIZE:
        raise ValueError("size must be 28")
    directory = download(dataset)
    images_file, labels_file = FILES[split]
    with gzip.open(directory / images_file, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051 or (rows, cols) != (SIZE, SIZE):
            raise ValueError(f"Unexpected image file {images_file}")
        images = np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, rows, cols)
    with gzip.open(directory / labels_file, "rb") as handle:
        magic, count_labels = struct.unpack(">II", handle.read(8))
        if magic != 2049 or count_labels != count:
            raise ValueError(f"Unexpected label file {labels_file}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8).astype(np.int64)
    return images, labels


def tiles_of(images):
    """[N, H, W] -> [N, H/2, W/2, 4] 2x2 tiles (top-left, top-right, bottom-left, bottom-right)."""
    n, h, w = images.shape
    return images.reshape(n, h // 2, 2, w // 2, 2).transpose(0, 1, 3, 2, 4).reshape(n, h // 2, w // 2, 4)


def build_codebook(images, size=CODEBOOK_SIZE, iterations=40, seed=0):
    """k-means over the 2x2 tiles of training images; returns [size, 4] uint8."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(tiles_of(images).reshape(-1, 4), dtype=torch.float32, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    centres = x[torch.randperm(len(x), generator=g, device=device)[:size]].clone()
    centres[0] = 0  # an exact blank tile
    for _ in range(iterations):
        nearest = torch.cat([torch.cdist(x[i:i + 65536], centres).argmin(1)
                             for i in range(0, len(x), 65536)])
        sums = torch.zeros_like(centres).index_add_(0, nearest, x)
        counts = torch.bincount(nearest, minlength=size).float()
        centres = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centres)
        centres[0] = 0
    return centres.round().clamp(0, 255).to(torch.uint8).cpu().numpy()


# Held-out digits: drawn from the dataset's training split, right after the training draw
# in the same permutation, so no model or baseline ever trains on them. "practice" is for
# learning a picker, "valid" for choosing checkpoints and settings; the 1,000 test digits
# stay untouched until the end.
SPLITS = ("test", "practice", "valid")
HELD_OUT = {"practice": 5000, "valid": 1000}


def held_out_index(count, seed, train_images):
    """Training-split indices of each held-out split (shorter or empty if the training
    split runs out)."""
    order = np.random.default_rng(seed).permutation(count)
    index, start = {}, train_images
    for split, size in HELD_OUT.items():
        index[split] = order[start:start + size]
        start += size
    return index


def split_images(info, split="test"):
    """Images and labels that the prompts' "index" points into: the test split for "test",
    the training split for the held-out splits."""
    return load_split(info["dataset"], "test" if split == "test" else "train", info.get("size", SIZE))


def training_split(data):
    """The training images and labels a data folder's models saw (the first train_images of
    the seed's permutation). Never includes the held-out splits."""
    info = read_json(Path(data) / "info.json")
    images, labels = load_split(info["dataset"], "train", info.get("size", SIZE))
    index = np.random.default_rng(info["seed"]).permutation(len(images))[:info["train_images"]]
    images, labels = images[index], labels[index]
    return images, labels


def test_cases(data, split="test"):
    """Index and label of each request of a split, in order. Drawn from info.json and
    data/raw, so a data folder needs no prompt files: one permutation (seed) of the training
    split gives the training draw (its first train_images) and then the held-out splits; the
    same rng's next permutation of the test split gives the test digits (the first
    test_prompts). Older workspaces' {split}_prompts*.jsonl files hold the same index and
    label (checked) and are read if present."""
    data = Path(data)
    for name in (f"{split}_prompts.jsonl", f"{split}_prompts_pixel.jsonl", f"{split}_prompts_tile.jsonl"):
        if (data / name).is_file():
            return [{k: p[k] for k in ("index", "label")} for p in map(json.loads, (data / name).open())]
    info = read_json(data / "info.json")
    train_labels = load_split(info["dataset"], "train", info.get("size", SIZE))[1]
    if split == "test":
        test_labels = load_split(info["dataset"], "test", info.get("size", SIZE))[1]
        rng = np.random.default_rng(info["seed"])
        rng.permutation(len(train_labels))  # the training draw comes first
        index = rng.permutation(len(test_labels))[:info["test_prompts"]]
        labels = test_labels
    else:
        index = held_out_index(len(train_labels), info["seed"], info["train_images"])[split]
        labels = train_labels
    return [{"index": int(i), "label": int(labels[i])} for i in index]


def save_grid(path, pairs, scale=3):
    """Rows of (ground truth | generated) image pairs."""
    from PIL import Image
    if not pairs:
        return
    size = pairs[0][0].shape[0]
    scale = scale * 28 // size
    columns = 5
    rows = (len(pairs) + columns - 1) // columns
    gap = 4
    width = columns * (2 * size + gap) * scale
    canvas = Image.new("L", (width, rows * (size + gap) * scale), 64)
    for i, (truth, generated) in enumerate(pairs):
        r, c = divmod(i, columns)
        pair = np.concatenate((truth, generated), 1)
        tile = Image.fromarray(pair).resize((2 * size * scale, size * scale), Image.NEAREST)
        canvas.paste(tile, (c * (2 * size + gap) * scale, r * (size + gap) * scale))
    canvas.save(path)


def device_info():
    """Describe the accelerator so NVIDIA and AMD results stay distinguishable."""
    import torch
    info = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": getattr(torch.version, "hip", None),
        "python": platform.python_version(),
    }
    if torch.cuda.is_available():
        info["backend"] = "rocm" if info["hip_runtime"] else "cuda"
        info["device"] = torch.cuda.get_device_name()
        props = torch.cuda.get_device_properties(0)
        info["device_memory_gb"] = round(props.total_memory / 2**30, 1)
        info["arch"] = (getattr(props, "gcnArchName", "gfx?") if info["hip_runtime"]
                        else f"sm_{props.major}{props.minor}")
    else:
        info["backend"] = "cpu"
        info["device"] = cpu_name()
    return info


def cpu_name():
    """CPU model from /proc/cpuinfo (platform.processor() often gives only x86_64)."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def save_model(path, weights, settings=None):
    """A model file: the weights (a state dict) in PATH (.safetensors; plain tensors, no pickled
    code) and, if given, the settings needed to rebuild the model in PATH with .json."""
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous() for k, v in weights.items()}, str(path))
    if settings is not None:
        write_json(path.with_suffix(".json"), settings)


def load_model(path, device="cpu"):
    """(weights, settings) of a file written by save_model; settings is {} if there are none."""
    from safetensors.torch import load_file
    path = Path(path)
    settings = path.with_suffix(".json")
    return load_file(str(path), device=str(device)), read_json(settings) if settings.is_file() else {}
