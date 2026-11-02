"""NotANewNick's picker over the pattern tools (specialists.py), learned from reward only.

Per image the picker chooses which tool to use (with its settings: which ring; which angle,
spacing and shift of lines; or the holes), in which order, how often (a tool may be used
again, e.g. a second ring further in) and when to stop. It is never told any of this: no
recipes, no first tool, no number of tools. The only limits are the actions themselves and
a cap of --max-passes passes per image. Tiles still blank when it stops stay empty (black);
the reward judges the image as it is.

Two pickers (NotANewNick's design):
  plan   sees only the request (the class and the hint rows) and writes its whole sequence
         of tools up front, one at a time, from the request and its own earlier choices
         (it never sees a drawn image)
  step   sees the image after every pass (drawn tiles, which tiles are still blank, the
         class, the pass number) and picks the next tool or stop

Reward (one number per finished image, nothing per step):
  reward = p2(class | image) * D(image) * (1 - urgency) ** passes
  p2     a second digit classifier (critics.py; not evaluate.py's judge), trained on the
         training draw only
  D      how real the image looks. Default (--realism score): the realism score of
         Kynkaanniemi et al. (NeurIPS 2019) in p2's features, against real training digits
         of the requested class, capped at 1 (inside the real digits' region = 1). It is
         fixed, so the reward does not move. On the step picker's valid drawings it gave
         0.69-0.74 to broken ones (FID 130-330), 0.87 to good ones (FID 6-7), 0.92 to
         real digits; the trained checker below sat at 0.49 for all of them.
         Or (--realism checker): a real-vs-drawn checker trained alongside the picker (as
         in AdaNAT, Ni et al. 2024), one update per picker update (--checker-every 1) or in
         fixed stretches (--checker-every 0 --checker-refresh N). Both swung the picker's
         valid FID between about 7 and 330 on 28x28 digits.
  urgency  what a pass costs (0 = quality only, 0.2 = each pass takes 20% off). It is a
         setting: an input of the picker, random per request in training (0..--max-urgency),
         chosen by whoever draws (--urgency). One picker covers every speed.
         A product, not "quality - cost * passes":
         with the difference, pickers in "Learning Unmasking Policies for Diffusion
         Language Models" (ICML 2026) collapsed to the cheapest action.

Choice in two levels, so no tool starts out favoured: first ring, lines, holes, circles, dots,
fractal (when the toolbox has them) or stop (equal odds at the start), then that tool's settings
(which ring; the angle, spacing and shift of the lines; the centre, spacing and shift of
the circles; the spacing and row and column shifts of the dots; which fractal; each its own
choice). The entropy bonus covers both levels.

Learning: REINFORCE with K rollouts per request and the mean reward of the other K-1 as the
baseline (RLOO: Kool et al. 2019), plus an entropy bonus. Requests: the practice digits
(held out of every drawer's training). Every --check updates the picker is saved and draws
the valid digits (choices sampled with a fixed seed, as in training; see valid_draw); the judge's FID and mean passes are logged. At the end
every saved picker is scored on the valid digits by the training objective
(p2 * D * (1 - urgency) ** passes, averaged over urgencies 0, half and the maximum, with the
same D for all of them: the fixed score, or the final checker) and the best is kept, so speed is not undone by the choice. Test digits: only in the final "draw".

  python experiment/tool_picker.py train --kind step --tools reference/digits-28x28/specialists/S2 \\
      --data reference/digits-28x28/data --critics reference/digits-28x28/critics --out work/tp/step
  python experiment/tool_picker.py draw --picker reference/digits-28x28/results/tool_picker/pickers/step \\
      --urgency 0.1 --split valid --out work/tp/step_valid
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate
from baselines import DEVICE
from common import device_info, load_model, read_json, save_model, split_images, test_cases, training_split, write_json
from critics import RealismChecker, load_second_classifier
from specialists import Toolbox, write_run
from tool_profile import frechet


class Encoder(nn.Module):
    """Image [N, C, 28, 28] + class -> vector."""

    def __init__(self, channels, width=256, classes=10, extra=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.label = nn.Embedding(classes, 64)
        self.extra = nn.Embedding(extra, 64) if extra else None
        self.mix = nn.Sequential(nn.Linear(64 * 49 + 64 + (64 if extra else 0), width), nn.ReLU())

    def forward(self, x, label, extra=None):
        parts = [self.conv(x), self.label(label)]
        if self.extra is not None:
            parts.append(self.extra(extra))
        return self.mix(torch.cat(parts, 1))


# Per tool, its settings in choice order: (head name, what sets its size). A "limited" head
# picks a shift below the spacing chosen just before it. Head names are kept from the
# earlier pickers so their saved weights still load.
SETTINGS = {
    "ring": [("ring", "rings")],
    "lines": [("angle", "angles"), ("spacing", "spacings"), ("shift", "limited")],
    "holes": [],
    "circles": [("centre", "centres"), ("circle_spacing", "spacings"), ("circle_shift", "limited")],
    "dots": [("dot_spacing", "spacings"), ("dot_row", "limited"), ("dot_col", "limited")],
    "fractal": [("fractal", "fractals")],
}
ACTION_TABLE = {"ring": "ring_action", "lines": "line_action", "circles": "circle_action", "dots": "dot_action",
                "fractal": "fractal_action"}


class Heads(nn.Module):
    """Two-level choice from a state vector: tool or stop, then the tool's settings."""

    def __init__(self, box, width):
        super().__init__()
        self.tools = list(box.tools)  # ring, lines, holes, then circles/dots if present
        self.choices = len(self.tools) + 1  # ... and stop
        self.tool = nn.Linear(width, self.choices)
        sizes = {"rings": box.rings, "angles": len(box.angles), "spacings": len(box.spacings),
                 "centres": len(box.centres), "fractals": box.fractals, "limited": max(box.spacings)}
        for tool in self.tools:
            for name, size in SETTINGS[tool]:
                self.add_module(name, nn.Linear(width, sizes[size]))
            if tool in ACTION_TABLE:
                self.register_buffer(ACTION_TABLE[tool], torch.as_tensor(getattr(box, ACTION_TABLE[tool])))
        self.register_buffer("spacings", torch.as_tensor(box.spacings))
        self.holes, self.stop = box.holes_action, len(box.actions)

    @staticmethod
    def choose(dist, greedy, generator):
        return dist.logits.argmax(1) if greedy else torch.multinomial(dist.probs, 1, generator=generator).squeeze(1)

    def forward(self, h, force_stop=False, greedy=False, generator=None):
        """Returns the action (index into box.actions, or stop), its log-prob and entropy."""
        tool_logits = self.tool(h)
        if force_stop:
            tool_logits = tool_logits.masked_fill(torch.arange(self.choices, device=h.device) < self.choices - 1,
                                                  float("-inf"))
        dist = torch.distributions.Categorical(logits=tool_logits)
        tool = self.choose(dist, greedy, generator)
        log_prob, entropy, p = dist.log_prob(tool), dist.entropy(), dist.probs
        action = torch.full_like(tool, self.stop)
        for t, name in enumerate(self.tools):
            picks, spacing = [], None
            used = (tool == t).float()
            for head, size in SETTINGS[name]:
                logits = getattr(self, head)(h)
                if size == "limited":
                    logits = logits.masked_fill(torch.arange(logits.shape[1], device=h.device)[None]
                                                >= self.spacings[spacing][:, None], float("-inf"))
                d = torch.distributions.Categorical(logits=logits)
                pick = self.choose(d, greedy, generator)
                if size == "spacings":
                    spacing = pick
                log_prob = log_prob + used * d.log_prob(pick)
                entropy = entropy + p[:, t] * d.entropy()
                picks.append(pick)
            chosen = (getattr(self, ACTION_TABLE[name])[tuple(picks)] if name in ACTION_TABLE
                      else torch.full_like(tool, self.holes))
            action = torch.where(tool == t, chosen, action)
        return action, log_prob, entropy


class StepPicker(nn.Module):
    """Sees the current image (pixels, blank-tile mask), the class, the pass number and the urgency."""

    def __init__(self, box, max_passes, width=256):
        super().__init__()
        self.encode = Encoder(2, width, extra=max_passes + 1)
        self.urgency = nn.Linear(1, width)
        self.body = nn.Sequential(nn.Linear(width, width), nn.ReLU())
        self.heads = Heads(box, width)

    def forward(self, pixels, blank, label, passes, urgency):
        return self.body(self.encode(torch.stack((pixels, blank), 1), label, passes) + self.urgency(urgency[:, None]))


class PlanPicker(nn.Module):
    """Sees only the request (hint rows as an image, the class), the urgency and its own earlier choices."""

    def __init__(self, box, width=256):
        super().__init__()
        n_actions = len(box.actions)
        self.encode = Encoder(1, width)
        self.previous = nn.Embedding(n_actions + 2, width)  # actions, stop, "start"
        self.cell = nn.GRUCell(width, width)
        self.urgency = nn.Linear(1, width)
        self.heads = Heads(box, width)
        self.start = n_actions + 1

    def first(self, request, label, urgency):
        return self.encode(request[:, None], label) + self.urgency(urgency[:, None])

    def forward(self, state, previous):
        return self.cell(self.previous(previous), state)


class Drawing:
    """The requests of one split on DEVICE: prompts for the tools, hint pixels, classes."""

    def __init__(self, box, data, split, count=None):
        info = read_json(Path(data) / "info.json")
        images, _ = split_images(info, split)
        self.prompts = test_cases(Path(data), split)[:count]
        self.truth = images[np.array([p["index"] for p in self.prompts])]
        self.labels = torch.tensor([p["label"] for p in self.prompts], device=DEVICE)
        self.prefix = box.prompts(self.truth, self.labels.cpu().numpy())
        self.hint = torch.tensor(self.truth, device=DEVICE, dtype=torch.float32) / 255
        self.hint[:, box.hint_rows:] = 0
        self.split, self.info = split, info


def to_pixels(box, codebook, hint, body):
    """Body codes -> [N, 28, 28] in 0..1 (blank tiles empty) and the blank-tile mask."""
    blank = body == box.mask_id
    tiles = codebook[body.masked_fill(blank, box.blank)]  # [N, tiles, 4]
    n, r, c = len(body), box.rows, box.cols
    drawn = tiles.view(n, r, c, 2, 2).permute(0, 1, 3, 2, 4).reshape(n, 2 * r, 2 * c)
    pixels = hint.clone()
    pixels[:, box.hint_rows:] = drawn
    mask = torch.zeros_like(pixels)
    mask[:, box.hint_rows:] = blank.view(n, r, c).float().repeat_interleave(2, 1).repeat_interleave(2, 2)
    return pixels, mask


def rollout(picker, kind, box, codebook, prefix, hint, labels, urgency, max_passes, greedy=False, generator=None,
            drawn_tiles=False, trace=None):
    """Draws a batch with the picker. Returns pixels, passes, sum of log-probs, entropy
    sum and the chosen action names per row. trace (a list) gets (body before, action) of
    every pass (action -1: none for that row)."""
    n, stop = len(prefix), len(box.actions)
    body = box.empty(n)
    alive = torch.ones(n, dtype=torch.bool, device=DEVICE)
    passes = torch.zeros(n, dtype=torch.long, device=DEVICE)
    log_prob = torch.zeros(n, device=DEVICE)
    entropy = torch.zeros(n, device=DEVICE)
    chosen = [[] for _ in range(n)]
    if kind == "plan":
        state = picker.first(hint, labels, urgency)
        previous = torch.full((n,), picker.start, device=DEVICE)
    for step in range(max_passes + 1):
        if kind == "step":
            pixels, blank = to_pixels(box, codebook, hint, body)
            h = picker(pixels, blank, labels, passes, urgency)
        else:
            h = state = picker(state, previous)
        # at the cap only stop is left
        action, lp, ent = picker.heads(h, force_stop=step == max_passes, greedy=greedy, generator=generator)
        log_prob = log_prob + torch.where(alive, lp, torch.zeros_like(log_prob))
        entropy = entropy + torch.where(alive, ent, torch.zeros_like(entropy))
        act = torch.where(alive & (action < stop), action, torch.full_like(action, -1))
        for i in torch.nonzero(act >= 0).squeeze(1).tolist():
            chosen[i].append(box.names[int(act[i])])
        if trace is not None:
            trace.append((body.clone(), act.clone()))
        if (act >= 0).any():
            body = box.apply(prefix, body, act, generator)
        passes = passes + (act >= 0).long()
        alive = alive & (action < stop)
        if kind == "plan":
            previous = action
        if not alive.any():
            break
    pixels, _ = to_pixels(box, codebook, hint, body)
    if drawn_tiles:
        return pixels, passes, log_prob, entropy, chosen, (body != box.mask_id).sum(1)
    return pixels, passes, log_prob, entropy, chosen


@torch.no_grad()
def valid_draw(picker, kind, box, codebook, split, max_passes, urgency, seed=0, greedy=False):
    """A whole split at one urgency, choices sampled as in training (the same seed for every
    call). Greedy choices (always the likeliest) looped on one redrawn line set up to the
    cap for a quarter of the requests at urgency 0 whenever "again" and "holes" were close:
    valid FID 12 -> 110 -> 12 -> 71 between checks, against 10-13 sampled."""
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    drawn, passes = [], []
    for i in range(0, len(split.prompts), 500):
        s = slice(i, i + 500)
        u = torch.full((len(split.prefix[s]),), float(urgency), device=DEVICE)
        px, ps, *_ = rollout(picker, kind, box, codebook, split.prefix[s], split.hint[s], split.labels[s], u,
                             max_passes, greedy=greedy, generator=generator)
        drawn.append(px)
        passes.append(ps)
    return torch.cat(drawn), torch.cat(passes).float()


@torch.no_grad()
def judge_fid(judge, pixels, truth):
    a = judge.features(pixels[:, None].float()).double().cpu()
    b = judge.features(torch.tensor(truth, device=DEVICE, dtype=torch.float32)[:, None] / 255).double().cpu()
    return frechet(a, b)


class RealismScore:
    """Realism score (Kynkaanniemi et al. 2019): max over real samples r of
    radius_k(r) / |f - r| in p2's features, where radius_k is r's distance to its k-th
    nearest real neighbour; real samples with radii above the median are dropped, as in the
    paper. Per requested class, capped at 1."""

    def __init__(self, p2, real, labels, k=3, per_class=20000):
        self.p2 = p2
        self.sets = {}
        rng = np.random.default_rng(0)
        for c in np.unique(labels):
            member = np.where(labels == c)[0]
            member = member[rng.permutation(len(member))[:per_class]]
            f = self.features(real[torch.as_tensor(member, device=real.device)][:, None])
            radius = torch.cat([torch.cdist(f[i:i + 2000], f).kthvalue(k + 1, dim=1).values
                                for i in range(0, len(f), 2000)])  # k + 1: each is its own nearest
            keep = radius <= radius.median()
            self.sets[int(c)] = (f[keep], radius[keep])

    @torch.no_grad()
    def features(self, x):
        return torch.cat([self.p2.features(x[i:i + 1000]) for i in range(0, len(x), 1000)])

    @torch.no_grad()
    def __call__(self, pixels, labels):
        f = self.features(pixels[:, None])
        score = torch.zeros(len(f), device=f.device)
        for c, (real, radius) in self.sets.items():
            m = labels == c
            if m.any():
                score[m] = (radius[None] / torch.cdist(f[m], real).clamp(min=1e-8)).max(1).values
        return score.clamp(max=1)


def load_picker(kind, box, max_passes):
    return (StepPicker(box, max_passes) if kind == "step" else PlanPicker(box)).to(DEVICE)


def train(args):
    torch.manual_seed(args.seed)
    box = Toolbox(args.tools, args.data, args.temperature)
    codebook = torch.tensor(box.codebook, device=DEVICE, dtype=torch.float32) / 255
    practice = Drawing(box, args.data, "practice")
    valid = Drawing(box, args.data, "valid")
    p2 = load_second_classifier(args.critics, DEVICE)
    judge = evaluate.classifier(practice.info["dataset"], practice.info.get("size", 28), args.data, DEVICE)
    real, real_labels = training_split(args.data)
    real = torch.tensor(real, device=DEVICE, dtype=torch.float32) / 255
    if args.realism == "score":
        realism = RealismScore(p2, real, real_labels)
        args.checker_warmup, args.checker_every, args.checker_refresh = 0, 0, 0
    else:
        realism = lambda pixels, labels: torch.sigmoid(checker(pixels[:, None]))
    checker = RealismChecker().to(DEVICE)
    checker_opt = torch.optim.Adam(checker.parameters(), lr=args.checker_lr, betas=(0.5, 0.999))
    picker = load_picker(args.kind, box, args.max_passes)
    if args.init:  # continue an earlier round's picker (e.g. on fine-tuned specialists)
        picker.load_state_dict(load_model(Path(args.init) / "picker.safetensors", DEVICE)[0])
    optimizer = torch.optim.Adam(picker.parameters(), lr=args.lr)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items()}
    settings["hardware"] = device_info()
    settings["actions"] = box.names
    write_json(out / "settings.json", settings)
    best, history, started = None, [], time.time()
    urgencies = (0.0, args.max_urgency / 2, args.max_urgency)  # where the valid digits are drawn
    log = (out / "train.log").open("a")
    rng = np.random.default_rng(args.seed)

    def checker_step(fake):
        idx = torch.as_tensor(rng.integers(0, len(real), len(fake)), device=DEVICE)
        x = torch.cat((real[idx], fake.detach()))[:, None]
        y = torch.cat((torch.ones(len(fake)), torch.zeros(len(fake)))).to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(checker(x), y)
        checker_opt.zero_grad()
        loss.backward()
        checker_opt.step()
        return float(loss.detach())

    # Warm up the checker on the untrained picker's images so the first rewards mean something.
    for _ in range(args.checker_warmup):
        rows = torch.as_tensor(rng.integers(0, len(practice.prompts), args.batch), device=DEVICE)
        with torch.no_grad():
            u = torch.as_tensor(rng.uniform(0, args.max_urgency, len(rows)), device=DEVICE, dtype=torch.float32)
            pixels, *_ = rollout(picker, args.kind, box, codebook, practice.prefix[rows], practice.hint[rows],
                                 practice.labels[rows], u, args.max_passes, generator=generator)
        checker_step(pixels)

    for update in range(1, args.updates + 1):
        picker.train()
        rows = torch.as_tensor(rng.integers(0, len(practice.prompts), args.batch), device=DEVICE)
        rows = rows.repeat_interleave(args.k)
        urgency = torch.as_tensor(rng.uniform(0, args.max_urgency, args.batch), device=DEVICE,
                                  dtype=torch.float32).repeat_interleave(args.k)
        pixels, passes, log_prob, entropy, _ = rollout(
            picker, args.kind, box, codebook, practice.prefix[rows], practice.hint[rows],
            practice.labels[rows], urgency, args.max_passes, generator=generator)
        with torch.no_grad():
            label = practice.labels[rows]
            quality = F.softmax(p2(pixels[:, None]), 1).gather(1, label[:, None]).squeeze(1)
            realness = realism(pixels, label)
            reward = quality * realness * (1 - urgency) ** passes.float()
            grouped = reward.view(args.batch, args.k)
            baseline = (grouped.sum(1, keepdim=True) - grouped) / (args.k - 1)
            advantage = (grouped - baseline).view(-1)
        loss = -(advantage * log_prob).mean() - args.entropy * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(picker.parameters(), 1.0)
        optimizer.step()
        checker_loss = checker_step(pixels) if args.checker_every and update % args.checker_every == 0 else None
        if args.checker_refresh and update % args.checker_refresh == 0:
            # fixed stretches: the checker is frozen for checker_refresh updates, then trained
            # on fresh drawings of the current picker (a steadier reward than every update)
            for _ in range(args.checker_refresh_steps):
                rows_c = torch.as_tensor(rng.integers(0, len(practice.prompts), args.batch), device=DEVICE)
                u = torch.as_tensor(rng.uniform(0, args.max_urgency, len(rows_c)), device=DEVICE, dtype=torch.float32)
                with torch.no_grad():
                    fresh, *_ = rollout(picker, args.kind, box, codebook, practice.prefix[rows_c], practice.hint[rows_c],
                                        practice.labels[rows_c], u, args.max_passes, generator=generator)
                checker_loss = checker_step(fresh)
        out_of_time = bool(args.minutes) and time.time() - started > 60 * args.minutes
        if update % args.check == 0 or update == args.updates or out_of_time:
            picker.eval()
            entry = {"update": update, "minutes": round((time.time() - started) / 60, 1),
                     "reward": float(reward.mean()), "quality": float(quality.mean()),
                     "realness": float(realness.mean()), "passes": float(passes.float().mean()),
                     "checker_loss": checker_loss, "valid": {}}
            for u in urgencies:
                drawn, valid_passes = valid_draw(picker, args.kind, box, codebook, valid, args.max_passes, u)
                with torch.no_grad():
                    entry["valid"][str(u)] = {
                        "fid": judge_fid(judge, drawn, valid.truth),
                        "correct": float((judge(drawn[:, None]).argmax(1) == valid.labels).float().mean()),
                        "passes": float(valid_passes.mean())}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            save_model(out / "checkpoints" / f"u{update:06d}.safetensors", picker.state_dict(),
                       {"kind": args.kind, "max_passes": args.max_passes, "entry": entry})
        if out_of_time:
            print(f"time limit of {args.minutes} minutes reached", flush=True)
            break
    # Keep the saved picker with the best training objective on the valid digits, all scored
    # with the final realism measure (the judge's FID is only logged, never used to choose).
    checker.eval()
    for entry in history:
        weights, state = load_model(out / "checkpoints" / f"u{entry['update']:06d}.safetensors", DEVICE)
        picker.load_state_dict(weights)
        picker.eval()
        objective = []
        for u in urgencies:
            drawn, valid_passes = valid_draw(picker, args.kind, box, codebook, valid, args.max_passes, u)
            with torch.no_grad():
                q = F.softmax(p2(drawn[:, None]), 1).gather(1, valid.labels[:, None]).squeeze(1)
                objective.append(float((q * realism(drawn, valid.labels) * (1 - u) ** valid_passes).mean()))
        entry["valid_objective"] = float(np.mean(objective))
        if best is None or entry["valid_objective"] > best["valid_objective"]:
            best = entry
            save_model(out / "picker.safetensors", weights, state)
    write_json(out / "history.json", {"best": best, "history": history})
    print(f"kept update {best['update']}: valid objective {best['valid_objective']:.4f}; per urgency "
          + ", ".join(f"{u}: FID {v['fid']:.2f}, {v['passes']:.2f} passes, {100 * v['correct']:.1f}% correct"
                      for u, v in best["valid"].items()))


def draw(args):
    """The kept picker draws a split once (sampled choices, --greedy for the likeliest) and writes a run folder for
    evaluate.py and fid_range.py."""
    settings = read_json(Path(args.picker) / "settings.json")
    settings["tools"], settings["data"] = args.tools or settings["tools"], args.data or settings["data"]
    box = Toolbox(settings["tools"], settings["data"], settings["temperature"])
    codebook = torch.tensor(box.codebook, device=DEVICE, dtype=torch.float32) / 255
    weights, state = load_model(Path(args.picker) / "picker.safetensors", DEVICE)
    picker = load_picker(state["kind"], box, state["max_passes"])
    picker.load_state_dict(weights)
    picker.eval()
    split = Drawing(box, settings["data"], args.split, args.count)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(args.seed)
    pixels, passes, plans, tiles = [], [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(split.prompts), 500):
            s = slice(i, i + 500)
            u = torch.full((len(split.prefix[s]),), args.urgency, device=DEVICE)
            px, ps, _, _, ch, nd = rollout(picker, state["kind"], box, codebook, split.prefix[s], split.hint[s],
                                           split.labels[s], u, state["max_passes"], greedy=args.greedy, generator=generator,
                                           drawn_tiles=True)
            tiles.extend(nd.tolist())
            pixels.append(px)
            passes.append(ps)
            plans.extend(ch)
    seconds = (time.perf_counter() - started) / len(split.prompts)
    full = (torch.cat(pixels) * 255).round().clamp(0, 255).byte().cpu().numpy()
    full[:, :box.hint_rows] = split.truth[:, :box.hint_rows]
    passes = torch.cat(passes).cpu().numpy()
    from collections import Counter
    write_run(args.out, split.info, args.split, split.prompts, full, split.truth, passes, plans, tiles,
              box.pass_layers, {"decode": {"picker": state["kind"], "urgency": args.urgency, "greedy": args.greedy,
                                           "max_passes": state["max_passes"], "temperature": settings["temperature"]},
        "picker": str(args.picker), "kind": state["kind"], "urgency": args.urgency,
        "tools_folder": settings["tools"], "temperature": settings["temperature"],
        "parameters": sum(box.parameters), "batched_seconds_per_image": seconds,
        "passes_histogram": dict(sorted(Counter(passes.tolist()).items())),
        "first_tool": dict(Counter(p[0] if p else "stop" for p in plans).most_common(10)),
        "plans": dict(Counter(" > ".join(p) for p in plans).most_common(20))})
    print(f"wrote {args.out}: mean passes {passes.mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("train")
    p.add_argument("--kind", required=True, choices=("plan", "step"))
    p.add_argument("--tools", required=True, help="specialists folder prefix (PREFIX_ring, ...)")
    p.add_argument("--data", required=True)
    p.add_argument("--critics", required=True, help="folder with classifier2.safetensors and critics.json (critics.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--max-urgency", type=float, default=0.2,
                   help="training urgencies are drawn from 0..this (the share a pass costs)")
    p.add_argument("--max-passes", type=int, default=10)
    p.add_argument("--init", help="an earlier picker folder to continue from")
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    p.add_argument("--batch", type=int, default=64, help="requests per update")
    p.add_argument("--k", type=int, default=4, help="rollouts per request")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--realism", choices=("score", "checker"), default="score",
                   help="D: the fixed realism score (default) or a checker trained alongside")
    p.add_argument("--checker-lr", type=float, default=1e-4)
    p.add_argument("--checker-every", type=int, default=1, help="one checker update every N picker updates (0 = off)")
    p.add_argument("--checker-refresh", type=int, default=0,
                   help="freeze the checker for N picker updates, then train it --checker-refresh-steps steps (with --checker-every 0 to use only this)")
    p.add_argument("--checker-refresh-steps", type=int, default=200)
    p.add_argument("--checker-warmup", type=int, default=200)
    p.add_argument("--check", type=int, default=250, help="valid check every this many updates")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("draw")
    p.add_argument("--picker", required=True)
    p.add_argument("--tools", help="specialists folder prefix (default: the picker's settings.json)")
    p.add_argument("--data", help="data folder (default: the picker's settings.json)")
    p.add_argument("--split", required=True, choices=("practice", "valid", "test"))
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--urgency", type=float, default=0.0, help="what a pass costs: 0 = quality only")
    p.add_argument("--greedy", action="store_true", help="always the likeliest choice (default: sampled)")
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"train": train, "draw": draw}[args.action](args)


if __name__ == "__main__":
    main()
