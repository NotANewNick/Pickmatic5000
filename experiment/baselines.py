"""Best known methods at this project's scale, for a fair "best method vs ours" comparison.

Same data (the workspace's training digits), same prompts (class + hint rows), same test
digits and measures (evaluate.py: correct digit, FID, copy check, steps).

Both are reimplemented from their papers (see REFERENCES.md), in portable PyTorch (NVIDIA,
AMD, CPU).

  diffusion  DDPM training (Ho et al. 2020) of a small U-Net on the pixels, conditioned on
             the class and the hint rows; DDIM sampling (Song et al. 2021) with a chosen
             number of steps; classifier-free guidance (Ho and Salimans 2022). One step =
             one U-Net pass (guidance runs its two passes side by side, not one after another).
  maskgit    MaskGIT (Chang et al. 2022) on the 2x2-tile tokens: a bidirectional
             transformer predicts all masked tiles at once; each round keeps the most
             confident ones (cosine schedule). One step = one transformer pass.
             --order halton: the tiles kept each step follow the fixed Halton order
             (Besnier et al. 2025) instead.

  python experiment/baselines.py train diffusion --data reference/digits-28x28/data --out work/D_diffusion
  python experiment/baselines.py generate diffusion --model reference/digits-28x28/baselines/D_diffusion \
      --data reference/digits-28x28/data --split valid --out work/eval/diffusion_s8 --steps 8 --cfg 1
  python experiment/baselines.py train maskgit --data reference/digits-28x28/data --out work/M_maskgit
  python experiment/baselines.py generate maskgit --model reference/digits-28x28/baselines/M_maskgit \
      --data reference/digits-28x28/data --split valid --out work/eval/maskgit_s8 --steps 8 [--order halton]
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (SPLITS, device_info, load_model, read_json, save_grid, save_model, split_images, test_cases, tiles_of,
                    training_split, write_json)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- diffusion

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None]
    return torch.cat((angles.sin(), angles.cos()), 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x))) + self.e(emb)[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class UNet(nn.Module):
    """14x14 -> 7x7 -> 14x14 (levels=1), or 28x28 -> 14x14 -> 7x7 and back (levels=2), so
    the middle is 7x7 at both sizes. Inputs: noisy image, hint image (hint rows, zeros
    below), and the hint mask; the class (or "none" for guidance) and the time step are
    embedded."""

    def __init__(self, classes=10, width=192, emb=256, levels=1):
        super().__init__()
        self.levels = levels
        self.classes, self.emb_dim = classes, emb
        self.time = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.label = nn.Embedding(classes + 1, emb)  # last = no class (guidance)
        self.inp = nn.Conv2d(3, width, 3, padding=1)
        self.d1, self.d2 = ResBlock(width, width, emb), ResBlock(width, width, emb)
        self.down = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.m1, self.m2 = ResBlock(2 * width, 2 * width, emb), ResBlock(2 * width, 2 * width, emb)
        self.up = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.u1, self.u2 = ResBlock(2 * width, width, emb), ResBlock(width, width, emb)
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))
        if levels == 2:  # an outer 28x28 level around the 14x14 one
            self.outer_in = nn.Conv2d(3, width, 3, padding=1)
            self.o1, self.o2 = ResBlock(width, width, emb), ResBlock(width, width, emb)
            self.outer_down = nn.Conv2d(width, width, 3, stride=2, padding=1)
            self.outer_up = nn.ConvTranspose2d(width, width, 2, stride=2)
            self.o3, self.o4 = ResBlock(2 * width, width, emb), ResBlock(width, width, emb)

    def forward(self, x, hint, mask, t, label):
        emb = self.time(timestep_embedding(t, self.emb_dim)) + self.label(label)
        if self.levels == 2:
            g1 = self.o2(self.o1(self.outer_in(torch.cat((x, hint, mask), 1)), emb), emb)
            h0 = self.outer_down(g1)
        else:
            h0 = self.inp(torch.cat((x, hint, mask), 1))
        h1 = self.d2(self.d1(h0, emb), emb)
        h = self.m2(self.m1(self.down(h1), emb), emb)
        h = self.u2(self.u1(torch.cat((self.up(h), h1), 1), emb), emb)
        if self.levels == 2:
            h = self.o4(self.o3(torch.cat((self.outer_up(h), g1), 1), emb), emb)
        return self.out(h)


def cosine_alphas(steps=1000):
    """Cumulative alphas of the cosine schedule (Nichol and Dhariwal 2021)."""
    s = torch.arange(steps + 1, dtype=torch.float64) / steps
    f = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    return (f[1:] / f[0]).clamp(1e-5, 1).float()


def to_unit(images):
    return torch.tensor(images, dtype=torch.float32)[:, None] / 127.5 - 1


def hint_inputs(images, hint_rows):
    x = to_unit(images)
    mask = torch.zeros_like(x)
    mask[:, :, :hint_rows] = 1
    return x * mask, mask


def train_diffusion(args):
    images, labels = training_split(args.data)
    hint_rows = read_json(Path(args.data) / "info.json")["hint_rows"]
    torch.manual_seed(0)
    levels = 2 if images.shape[1] >= 28 else 1
    model = UNet(levels=levels).to(DEVICE)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    alphas = cosine_alphas().to(DEVICE)
    x_all = to_unit(images).to(DEVICE)
    hint_all, mask_all = (t.to(DEVICE) for t in hint_inputs(images, hint_rows))
    y_all = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    body = (1 - mask_all[:1])  # loss on the generated rows only
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "hardware.json", device_info())
    step = 0
    for epoch in range(args.epochs):
        for index in torch.randperm(len(x_all), device=DEVICE).split(128):
            x0, y = x_all[index], y_all[index].clone()
            y[torch.rand(len(y), device=DEVICE) < 0.1] = model.classes  # drop the class 10%
            t = torch.randint(0, len(alphas), (len(x0),), device=DEVICE)
            noise = torch.randn_like(x0)
            a = alphas[t][:, None, None, None]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
            predicted = model(xt, hint_all[index], mask_all[index], t, y)
            loss = (((predicted - noise) ** 2) * body).sum() / (body.sum() * len(x0))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for e, p in zip(ema.parameters(), model.parameters()):
                    e.lerp_(p, 1 - 0.999)
            step += 1
        print(json.dumps({"epoch": epoch, "step": step, "loss": loss.item()}), flush=True)
    save_model(out / "diffusion.safetensors", ema.state_dict(),
               {"parameters": sum(p.numel() for p in model.parameters()), "plan": "none", "plan_dim": 0, "levels": levels})


@torch.inference_mode()
def sample_diffusion(model, hint, mask, labels, steps, cfg, alphas):
    """DDIM (eta = 0) from noise in `steps` steps; guidance mixes class and no-class."""
    x = torch.randn(len(labels), 1, hint.shape[2], hint.shape[3], device=DEVICE)
    # "trailing" spacing (Lin et al. 2024): the last step goes from t = T/steps to the image.
    # (linspace(T-1, 0, steps) wasted a step: its last step, at t = 0, changes nothing.)
    times = torch.linspace(len(alphas) - 1, 0, steps + 1, device=DEVICE)[:-1].round().long()
    none = torch.full_like(labels, model.classes)
    for i, t in enumerate(times):
        tt = t.repeat(len(labels))
        if cfg == 1:
            eps = model(x, hint, mask, tt, labels)
        else:  # both passes side by side: one sequential step
            both = model(torch.cat((x, x)), torch.cat((hint, hint)), torch.cat((mask, mask)),
                         torch.cat((tt, tt)), torch.cat((labels, none)))
            conditional, unconditional = both.chunk(2)
            eps = unconditional + cfg * (conditional - unconditional)
        a = alphas[t]
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()  # consistent with the clamped x0 (DDIM)
        a_next = alphas[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0, device=DEVICE)
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
    return x


# ------------------------------------------------------------------------------- maskgit

class MaskGIT(nn.Module):
    """Bidirectional transformer over [class, hint tiles, body tiles] (8 layers, width 384)."""

    def __init__(self, codes=512, classes=10, length=64, width=384, layers=8, heads=6):
        super().__init__()
        self.codes, self.mask_id = codes, codes
        self.token = nn.Embedding(codes + 1 + classes, width)  # codes, mask, classes
        self.position = nn.Embedding(length, width)
        layer = nn.TransformerEncoderLayer(width, heads, 3 * width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, codes)

    def forward(self, ids):
        h = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.norm(self.body(h)))


def tile_codes(images, codebook):
    tiles = tiles_of(images).astype(np.int64)  # [N, R, C, 4]
    flat = torch.tensor(tiles.reshape(-1, 4), dtype=torch.float32)
    centres = torch.tensor(codebook, dtype=torch.float32)
    # in chunks: all tiles of 200,000 augmented images at once would need 80 GB
    nearest = torch.cat([torch.cdist(part, centres).argmin(1) for part in flat.split(1 << 20)])
    return nearest.reshape(tiles.shape[:3]).numpy()


def maskgit_inputs(images, labels, hint_rows, codebook, codes=512):
    """Token rows: class token, hint tiles, body tiles (raster)."""
    grid = tile_codes(images, codebook)
    hint_tile_rows = hint_rows // 2
    hint = grid[:, :hint_tile_rows].reshape(len(images), -1)
    body = grid[:, hint_tile_rows:].reshape(len(images), -1)
    label = codes + 1 + np.asarray(labels)[:, None]
    return torch.tensor(np.concatenate((label, hint), 1)), torch.tensor(body)


def train_maskgit(args):
    images, labels = training_split(args.data)
    info = read_json(Path(args.data) / "info.json")
    codebook = np.load(Path(args.data) / "codebook_tile.npy")
    prefix, body = maskgit_inputs(images, labels, info["hint_rows"], codebook)
    prefix, body = prefix.to(DEVICE), body.to(DEVICE)
    torch.manual_seed(0)
    model = MaskGIT(length=prefix.shape[1] + body.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total = args.epochs * math.ceil(len(body) / 128)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "hardware.json", device_info())
    step = 0
    for epoch in range(args.epochs):
        for index in torch.randperm(len(body), device=DEVICE).split(128):
            target = body[index]
            ratio = torch.cos(torch.rand(len(target), 1, device=DEVICE) * math.pi / 2)
            masked = torch.rand(target.shape, device=DEVICE) < ratio
            masked[torch.arange(len(target)), torch.randint(0, target.shape[1], (len(target),))] = True
            ids = torch.cat((prefix[index], target.masked_fill(masked, model.mask_id)), 1)
            logits = model(ids)[:, prefix.shape[1]:]
            loss = F.cross_entropy(logits[masked], target[masked])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
        print(json.dumps({"epoch": epoch, "step": step, "loss": loss.item()}), flush=True)
    save_model(out / "maskgit.safetensors", model.state_dict(),
               {"length": prefix.shape[1] + body.shape[1], "parameters": sum(p.numel() for p in model.parameters()),
                "plan": "none", "values": 0})


def halton_order(rows, cols):
    """Tile order of the Halton scheduler (Besnier et al. 2025, arXiv 2503.17076): points of
    the 2D Halton sequence (bases 2 and 3) snapped to the grid, duplicates dropped, until
    every tile is covered. Fixed for every image, as in the paper."""
    def radical(i, base):
        value, f = 0.0, 1.0
        while i:
            f /= base
            value += f * (i % base)
            i //= base
        return value
    order, seen, i = [], set(), 0
    while len(order) < rows * cols:
        tile = int(radical(i, 3) * rows) * cols + int(radical(i, 2) * cols)
        if tile not in seen:
            seen.add(tile)
            order.append(tile)
        i += 1
    return order


@torch.inference_mode()
def sample_maskgit(model, prefix, n_body, steps, temperature=4.5, order="confidence", grid=None):
    """MaskGIT decoding: each round predicts all masked tiles, keeps the most confident
    (confidence plus annealed Gumbel noise), re-masks the rest (cosine schedule).
    order="halton": the tiles kept are the next ones of the fixed Halton order instead (same
    cosine schedule of how many; their values sampled as predicted); grid = (rows, cols)."""
    body = torch.full((len(prefix), n_body), model.mask_id, device=DEVICE)
    if order == "halton":
        rank = torch.empty(n_body, device=DEVICE)
        rank[torch.as_tensor(halton_order(*grid), device=DEVICE)] = torch.arange(n_body, device=DEVICE, dtype=torch.float)
    for i in range(steps):
        logits = model(torch.cat((prefix, body), 1))[:, prefix.shape[1]:].float()
        probs = logits.softmax(-1)
        sampled = torch.distributions.Categorical(probs=probs).sample()
        confidence = probs.gather(-1, sampled[..., None])[..., 0]
        unknown = body == model.mask_id
        sampled = torch.where(unknown, sampled, body)
        confidence = torch.where(unknown, confidence, torch.full_like(confidence, float("inf")))
        keep_masked = math.floor(n_body * math.cos(math.pi / 2 * (i + 1) / steps))
        if keep_masked > 0 and order == "halton":  # the last keep_masked tiles of the order stay masked
            sampled = torch.where(rank[None] >= n_body - keep_masked, torch.full_like(sampled, model.mask_id), sampled)
        elif keep_masked > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(confidence).clamp(1e-9, 1)))
            score = confidence.log() + temperature * (1 - (i + 1) / steps) * gumbel
            threshold = score.sort(-1).values[:, keep_masked - 1:keep_masked]
            sampled = torch.where(score <= threshold, torch.full_like(sampled, model.mask_id), sampled)
        body = sampled
    return body


# ------------------------------------------------------------------------- generation

def generate(args):
    info = read_json(Path(args.data) / "info.json")
    hint_rows, size = info["hint_rows"], info.get("size", 28)
    images, _ = split_images(info, args.split)
    prompts = test_cases(Path(args.data), args.split)[:args.count]
    index = np.array([p["index"] for p in prompts])
    labels = np.array([p["label"] for p in prompts])
    truth = images[index]
    name = "diffusion.safetensors" if args.method == "diffusion" else "maskgit.safetensors"
    weights, state = load_model(Path(args.model) / name, DEVICE)
    if state.get("plan", "none") != "none" or state.get("values", 0) or state.get("plan_dim", 0):
        raise SystemExit(f"{args.model} was trained with a plan input (removed on 2026-11-02; not included in this release)")
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    if args.method == "diffusion":
        model = UNet(levels=state.get("levels", 1)).to(DEVICE).eval()
        model.load_state_dict(weights)
        hint, mask = (t.to(DEVICE) for t in hint_inputs(truth, hint_rows))
        alphas = cosine_alphas().to(DEVICE)
        y = torch.tensor(labels, device=DEVICE)
        out = torch.cat([sample_diffusion(model, hint[i:i + 250], mask[i:i + 250], y[i:i + 250],
                                          args.steps, args.cfg, alphas)
                         for i in range(0, len(y), 250)])
        generated = ((out[:, 0] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        body_tokens, per_step = (size - hint_rows) * size, None
        decode = {"steps": args.steps, "cfg": args.cfg, "sampler": "ddim"}
        passes = 2 if args.cfg != 1 else 1
    else:
        model = MaskGIT(length=state["length"]).to(DEVICE).eval()
        model.load_state_dict(weights)
        codebook = np.load(Path(args.data) / "codebook_tile.npy")
        prefix, body = maskgit_inputs(truth, labels, hint_rows, codebook)
        prefix = prefix.to(DEVICE)
        rows, cols = (size - hint_rows) // 2, size // 2
        codes = torch.cat([sample_maskgit(model, prefix[i:i + 250], body.shape[1], args.steps,
                                          order=args.order, grid=(rows, cols))
                           for i in range(0, len(prefix), 250)]).cpu().numpy()
        tiles = codebook[codes.reshape(-1, rows, cols)]
        generated = tiles.reshape(-1, rows, cols, 2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 2 * rows, 2 * cols)
        body_tokens, per_step = body.shape[1], state["length"]
        decode = {"steps": args.steps, "temperature": 4.5, "sampler": "maskgit", "order": args.order}
        passes = 1
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / len(prompts)  # batched: not comparable to batch-1 runs
    full = truth.copy()
    full[:, hint_rows:] = generated[:, hint_rows:] if args.method == "diffusion" else generated
    layers = 8 if args.method == "maskgit" else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "meta.json", {
        "kind": args.method, "model": str(args.model), "ordering": args.method,
        "hint_rows": hint_rows, "body_tokens": body_tokens, "dataset": info["dataset"], "size": size,
        "split": args.split, "decode": decode, "guess": [], "fill_zero": False, "compile": False, "count": len(prompts),
        "plan": "none", "plan_source": None,
        "parameters": state["parameters"], "passes_per_step": passes, "hardware": device_info(),
        "batched_seconds_note": "wall_seconds are per image in batches of 250, not batch 1"})
    with (out_dir / "samples.jsonl").open("w") as handle:
        for p in prompts:
            handle.write(json.dumps({
                "index": p["index"], "label": p["label"], "generated_hex": "",
                "generated_tokens": body_tokens, "sequential_steps": args.steps,
                "rule_steps": 0,
                "sequential_layers": None if layers is None else args.steps * layers,
                "decode_token_layers": None if per_step is None else args.steps * per_step * layers,
                "prefill_token_layers": 0, "wall_seconds": seconds, "mean_exit_depth": None,
                "stop_reason": "length"}) + "\n")
    save_grid(out_dir / "samples.png", list(zip(truth[:50], full[:50])))
    np.save(out_dir / "generated.npy", full)
    print(f"wrote {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("train", "generate"))
    parser.add_argument("method", choices=("diffusion", "maskgit"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", help="generate: the trained model folder")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=25, help="generate: sequential steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="diffusion: guidance scale (1 = none)")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--split", choices=SPLITS, default="test",
                        help="generate: test, or the held-out practice/valid digits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order", choices=("confidence", "halton"), default="confidence",
                        help="maskgit generate: which tiles each step keeps")
    args = parser.parse_args()
    if args.action == "train":
        (train_diffusion if args.method == "diffusion" else train_maskgit)(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
