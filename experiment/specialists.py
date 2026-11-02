"""The pattern tools (NotANewNick's design): each tool is its own network, trained only for
its job, and draws its whole shape in ONE pass.

Tools on the body's tile grid (28x28 digits: 10 x 14 tiles of 2x2 pixels under the 4 hint
tile rows). A picker (tool_picker.py) chooses which tools to use, with which settings, in
which order and how often; nothing here fixes that.

  ring      one ring of the grid: ring 0 is the outer border, ring 1 one tile further in, ...
  lines     a set of parallel lines, one tile thick: angle 0-165 degrees in 15-degree steps
            (0 = rows, 90 = columns), every 2nd, 3rd or 4th line, shifted by 0..spacing-1
  circles   a set of concentric circles, one tile thick (for round strokes):
            centre the middle of the image, of its top half or of its bottom half
            (pixel row 14, 7 or 21, column 14), every 2nd, 3rd or 4th circle, shifted by
            0..spacing-1. Circle k = the tiles whose centre is k to k+1 tiles from the centre.
  dots      a grid of single tiles: every k-th tile of every k-th row,
            k = 2, 3 or 4, shifted by 0..k-1 rows and 0..k-1 columns. Coarse to
            fine (e.g. dots k=4, then k=2, then holes) is left for the picker to find; such
            multiscale filling is prior work (Reed et al., ICML 2017).
  fractal   a self-similar pattern (two families): the Sierpinski
            triangle (tiles whose row and column numbers share no binary 1, counted from
            one of the 4 corners, with tiles of size 1 or 2), or the Sierpinski carpet
            (3x3 blocks with the middle left out, at every scale) or only its holes
  holes     every tile not drawn yet

A toolbox uses the tools whose folders exist (ring, lines and holes always; circles, dots
and fractal if trained), so sets without them keep their actions unchanged.

Using a tool redraws its whole shape: its tiles are blanked and drawn again, seeing
everything else that is drawn. A specialist sees the prompt (class token and hint tiles)
and the drawn tiles, and predicts its shape's tiles in one pass; they are then sampled
independently (temperature --temperature).

Training (one network per tool, nothing shared): real training digits; a random context
of 0-10 shapes (rings and line sets, any settings) is shown as drawn, the rest is blank; the
loss is on the tool's own tiles (ring/lines: a random shape of that tool; holes: every
tile outside the context). For a quarter of the ring/lines examples the context is every
other tile, as when a shape is redrawn after the holes are filled (a picker may do that). Same network shape, data, optimiser, batch and number of updates
as the MaskGIT baseline (baselines.py, 60 epochs) unless --width/--layers make it smaller.
The network shape is a plain bidirectional transformer (the MaskGIT class in baselines.py);
the weights and training are the specialist's own.

Fine-tuning rounds (NotANewNick's design: every specialist also learns to work with, and fix,
the drawings of the others; a many-to-many grid), --init = the earlier round's specialists:
  --context drawn   round 2: for half of each batch the context is drawn by the earlier
                    round's specialists themselves (1-4 random tools with random settings,
                    any order), not real tiles; each context shape is drawn while its tool
                    sees the real digit around it, so the drawn tiles are near-misses of
                    the real digit. The target stays the real digit's tiles of the own
                    shape, so it learns to fit around, or replace, the others' mistakes.
                    (Contexts drawn freely from the request alone were tried first: they
                    often show another digit than the target, and the holes specialist
                    learned to ignore its context: FID 72 vs 15 for ring, 45° lines, holes.)
                    The other half keeps round 1's real contexts.
  --context picker  round 3: the same, but the tools in the context are the sequences the
                    trained pickers (--pickers) choose, at random urgencies, redrawn the same
                    way; the state just before they use this specialist's tool.
Like SUNDAE (Savinov et al., ICLR 2022: a denoiser trained on its own drafts) and scheduled
sampling (Bengio et al., NeurIPS 2015); across separate specialists, every pairing, here.

Prior work: masked-token transformers that fill chosen tiles in one pass (MaskGIT, Chang et
al. 2022); training on structured rather than random masks is NotANewNick's design.

  python experiment/specialists.py train --tool ring --data reference/digits-28x28/data --out work/S_ring
  python experiment/specialists.py draw --tools reference/digits-28x28/specialists/S2 \\
      --data reference/digits-28x28/data --split valid --plan "ring:0,lines:0:2:0,holes" --out work/plan1
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines import DEVICE, MaskGIT, maskgit_inputs
from common import SPLITS, device_info, load_model, read_json, save_grid, save_model, split_images, test_cases, training_split, write_json

TOOLS = ("ring", "lines", "holes", "circles", "dots", "fractal")
OPTIONAL = ("circles", "dots", "fractal")
CORNERS = ("tl", "tr", "bl", "br")
ANGLES = tuple(range(0, 180, 15))
SPACINGS = (2, 3, 4)
CENTRES = {"mid": 14, "top": 7, "bottom": 21}  # circle centres: pixel row (column 14)


def line_index(rows, cols, angle):
    """Per tile, the number of the 1-tile-thick line through it at this angle."""
    r, c = np.indices((rows, cols))
    a = math.radians(angle)
    if abs(math.cos(a)) >= abs(math.sin(a)) - 1e-9:  # mostly horizontal: one tile per column
        return np.round(r - c * math.tan(a)).astype(int)
    return np.round(c - r / math.tan(a)).astype(int)  # mostly vertical: one tile per row


def circle_index(rows, cols, centre):
    """Per tile, the number of the 1-tile-thick circle around this centre through it (the
    body grid sits under the hint rows of a square image of 2 * cols pixels)."""
    r, c = np.indices((rows, cols))
    y = 2 * cols - 2 * rows + 2 * r + 1  # tile centre in image pixels
    x = 2 * c + 1
    return np.floor(np.hypot(y - CENTRES[centre], x - cols) / 2).astype(int)


def carpet(rows, cols):
    """Sierpinski carpet: False where, at some scale, the tile is in the middle of a 3x3 block."""
    r, c = np.indices((rows, cols))
    keep = np.ones((rows, cols), bool)
    while r.any() or c.any():
        keep &= ~((r % 3 == 1) & (c % 3 == 1))
        r, c = r // 3, c // 3
    return keep


def fractals(rows, cols):
    """The fractal tool's settings: (name, mask [rows, cols])."""
    r, c = np.indices((rows, cols))
    out = []
    for corner in CORNERS:
        rr = rows - 1 - r if corner[0] == "b" else r
        cc = cols - 1 - c if corner[1] == "r" else c
        for size in (1, 2):
            out.append((f"tri:{corner}:{size}", ((rr // size) & (cc // size)) == 0))
    return out + [("carpet", carpet(rows, cols)), ("carpetholes", ~carpet(rows, cols))]


def actions(rows, cols, tools=TOOLS):
    """Every (tool, setting) the picker can choose, with its tiles; 'holes' is drawn from the
    blanks at the time. Returns a list of (name, tool, mask [rows*cols] bool or None)."""
    r, c = np.indices((rows, cols))
    ring_of = np.minimum(np.minimum(r, rows - 1 - r), np.minimum(c, cols - 1 - c))
    out = [(f"ring:{k}", "ring", (ring_of == k).ravel()) for k in range(ring_of.max() + 1)]
    for angle in ANGLES:
        index = line_index(rows, cols, angle)
        for spacing in SPACINGS:
            for shift in range(spacing):
                mask = (index % spacing == shift).ravel()
                if mask.any():
                    out.append((f"lines:{angle}:{spacing}:{shift}", "lines", mask))
    if "circles" in tools:
        for centre in CENTRES:
            index = circle_index(rows, cols, centre)
            for spacing in SPACINGS:
                for shift in range(spacing):
                    out.append((f"circles:{centre}:{spacing}:{shift}", "circles", (index % spacing == shift).ravel()))
    if "dots" in tools:
        for spacing in SPACINGS:
            for row in range(spacing):
                for col in range(spacing):
                    out.append((f"dots:{spacing}:{row}:{col}", "dots", ((r % spacing == row) & (c % spacing == col)).ravel()))
    if "fractal" in tools:
        out += [(f"fractal:{name}", "fractal", mask.ravel()) for name, mask in fractals(rows, cols)]
    out.append(("holes", "holes", None))
    return out


def grid_of(info):
    size = info.get("size", 28)
    return (size - info["hint_rows"]) // 2, size // 2


def shape_masks(rows, cols):
    """[shapes, rows*cols] bool on DEVICE for the ring, line and circle actions, and their tools."""
    acts = [a for a in actions(rows, cols) if a[2] is not None]
    masks = torch.tensor(np.stack([a[2] for a in acts]), device=DEVICE)
    tools = [a[1] for a in acts]
    return masks, tools


def random_context(masks, n, generator, most=10):
    """Union of 0..most random shapes per row: [n, tiles] bool."""
    count = torch.randint(0, most + 1, (n,), device=DEVICE, generator=generator)
    pick = torch.randint(0, len(masks), (n, most), device=DEVICE, generator=generator)
    use = torch.arange(most, device=DEVICE)[None] < count[:, None]
    return (masks[pick] & use[..., None]).any(1)


def generated_context(box, prefix, tool, generator, real, most=4):
    """Bodies drawn by the earlier specialists: 1..most random actions per row (no holes when
    training the holes tool, so blanks are left), each drawn while seeing the real digit
    around its cells, so the drawn tiles are near-misses of the real digit and the real
    target stays consistent with them. Returns body codes (mask_id = blank)."""
    n = len(prefix)
    body = box.empty(n)
    allowed = torch.tensor([i for i, name in enumerate(box.names) if not (tool == "holes" and name == "holes")],
                           device=DEVICE)
    count = torch.randint(1, most + 1, (n,), device=DEVICE, generator=generator)
    for p in range(most):
        action = allowed[torch.randint(0, len(allowed), (n,), device=DEVICE, generator=generator)]
        action = torch.where(p < count, action, torch.full_like(action, -1))
        body = box.apply(prefix, body, action, generator, reference=real)
    return body


def picker_context(box, pickers, prefix, images, labels, tool, generator, real):
    """The pickers' own sequences of tools (from a free drawing of the request), redrawn step
    by step while each tool sees the real digit around its cells (as generated_context);
    the state just before a use of this tool is the context. Returns body codes, the action
    chosen there, and which rows got one (the others fall back to generated_context)."""
    from tool_picker import rollout
    n = len(prefix)
    hint = torch.tensor(images, device=DEVICE, dtype=torch.float32) / 255
    hint[:, box.hint_rows:] = 0
    codebook = torch.tensor(box.codebook, device=DEVICE, dtype=torch.float32) / 255
    kind, picker, max_urgency, max_passes = pickers[int(torch.randint(0, len(pickers), (1,), generator=None))]
    urgency = torch.rand(n, device=DEVICE, generator=generator) * max_urgency
    trace = []
    with torch.no_grad():
        rollout(picker, kind, box, codebook, prefix, hint, labels, urgency, max_passes, generator=generator, trace=trace)
    t = box.tools.index(tool)
    found = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    body = box.empty(n)
    chosen = torch.full((n,), -1, dtype=torch.long, device=DEVICE)
    seen = torch.zeros(n, device=DEVICE)
    state = box.empty(n)
    for _, action in trace:  # keep one random use per row (reservoir sampling)
        use = torch.as_tensor([a >= 0 and box.tool_of[a] == t for a in action.tolist()], device=DEVICE)
        seen = seen + use.float()
        take = use & (torch.rand(n, device=DEVICE, generator=generator) * seen.clamp(min=1) < 1)
        body = torch.where(take[:, None], state, body)
        chosen = torch.where(take, action, chosen)
        found |= take
        if (action >= 0).any():
            state = box.apply(prefix, state, action, generator, reference=real)
    return body, chosen, found


def load_pickers(folders, box):
    from tool_picker import load_picker
    out = []
    for folder in folders:
        settings = read_json(Path(folder) / "settings.json")
        weights, state = load_model(Path(folder) / "picker.safetensors", DEVICE)
        picker = load_picker(state["kind"], box, state["max_passes"])
        picker.load_state_dict(weights)
        out.append((state["kind"], picker.eval(), settings["max_urgency"], state["max_passes"]))
    return out


def train(args):
    info = read_json(Path(args.data) / "info.json")
    images, labels = training_split(args.data)
    codebook = np.load(Path(args.data) / "codebook_tile.npy")
    prefix, body = maskgit_inputs(images, labels, info["hint_rows"], codebook)
    prefix, body = prefix.to(DEVICE), body.to(DEVICE)
    labels_t = torch.as_tensor(labels, device=DEVICE)
    rows, cols = grid_of(info)
    masks, tools = shape_masks(rows, cols)
    own = masks[[i for i, t in enumerate(tools) if t == args.tool]] if args.tool != "holes" else None
    torch.manual_seed(0)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(1 + TOOLS.index(args.tool) + (0 if args.context == "real" else 10 * len(args.context)))
    length = prefix.shape[1] + body.shape[1]
    box = pickers = None
    if args.init:  # a fine-tuning round: start from the earlier round's weights for this tool
        weights, state = load_model(Path(f"{args.init}_{args.tool}") / "specialist.safetensors", DEVICE)
        args.width, args.layers, args.heads = state["width"], state["layers"], state["heads"]
        box = Toolbox(args.init, args.data)  # frozen: draws the contexts
        if args.context == "picker":
            pickers = load_pickers(args.pickers, box)
    model = MaskGIT(length=length, width=args.width, layers=args.layers, heads=args.heads).to(DEVICE)
    if args.init:
        model.load_state_dict(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * math.ceil(len(body) / 128)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "hardware.json", device_info())
    started, step, stopped = time.time(), 0, False
    for epoch in range(args.epochs):
        for index in torch.randperm(len(body), device=DEVICE).split(128):
            target = body[index]
            context = random_context(masks, len(index), generator)
            if own is None:
                draw = ~context
            else:
                draw = own[torch.randint(0, len(own), (len(index),), device=DEVICE, generator=generator)]
                everything = torch.rand(len(index), 1, device=DEVICE, generator=generator) < 0.25
                context = torch.where(everything, ~draw, context & ~draw)
            shown = target.masked_fill(~context, model.mask_id)  # real tiles as context
            if args.context != "real":
                drawn = torch.rand(len(index), device=DEVICE, generator=generator) >= args.real_share
                rows_d = torch.nonzero(drawn).squeeze(1)
                if len(rows_d):
                    with torch.no_grad():
                        made = generated_context(box, prefix[index[rows_d]], args.tool, generator, target[rows_d])
                    d_draw = (made == model.mask_id) if own is None else draw[rows_d]
                    if args.context == "picker":
                        with torch.no_grad():
                            from_picker, chosen, found = picker_context(
                                box, pickers, prefix[index[rows_d]], images[index[rows_d].cpu().numpy()],
                                labels_t[index[rows_d]], args.tool, generator, target[rows_d])
                        made = torch.where(found[:, None], from_picker, made)
                        picked = (made == model.mask_id) if own is None else box.masks[chosen.clamp(min=0)]
                        d_draw = torch.where(found[:, None], picked, d_draw)
                    shown[rows_d] = made.masked_fill(d_draw, model.mask_id)
                    draw[rows_d] = d_draw
            draw[:, 0] |= ~draw.any(1)  # never an empty target
            shown = shown.masked_fill(draw, model.mask_id)
            ids = torch.cat((prefix[index], shown), 1)
            logits = model(ids)[:, prefix.shape[1]:]
            loss = F.cross_entropy(logits[draw], target[draw])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if args.minutes and time.time() - started > 60 * args.minutes:
                stopped = True
                break
        print(json.dumps({"epoch": epoch, "step": step, "loss": loss.item(),
                          "minutes": round((time.time() - started) / 60, 1)}), flush=True)
        if stopped:
            print(f"time limit of {args.minutes} minutes reached", flush=True)
            break
    save_model(out / "specialist.safetensors", model.state_dict(),
               {"length": length, "tool": args.tool, "width": args.width, "layers": args.layers, "heads": args.heads,
                "parameters": sum(p.numel() for p in model.parameters()),
                "epochs": args.epochs, "steps": step, "context": args.context, "init": args.init,
                "train_minutes": (time.time() - started) / 60})


class Toolbox:
    """The specialists (FOLDER_ring, FOLDER_lines, FOLDER_holes, and FOLDER_circles,
    FOLDER_dots and FOLDER_fractal if they exist) and every action, ready to draw batches on DEVICE."""

    def __init__(self, folder, data, temperature=1.0):
        info = read_json(Path(data) / "info.json")
        self.rows, self.cols = grid_of(info)
        self.hint_rows = info["hint_rows"]
        self.codebook = np.load(Path(data) / "codebook_tile.npy")
        self.blank = int(np.abs(self.codebook.astype(int)).sum(1).argmin())  # the empty tile
        self.tools = [t for t in TOOLS if t not in OPTIONAL or (Path(f"{folder}_{t}") / "specialist.safetensors").exists()]
        self.actions = actions(self.rows, self.cols, self.tools)
        self.names = [a[0] for a in self.actions]
        self.tool_of = [self.tools.index(a[1]) for a in self.actions]
        n = self.rows * self.cols
        self.masks = torch.stack([torch.as_tensor(a[2]) if a[2] is not None else torch.zeros(n, dtype=torch.bool)
                                  for a in self.actions]).to(DEVICE)
        self.models, self.parameters = [], []
        for tool in self.tools:
            weights, state = load_model(Path(f"{folder}_{tool}") / "specialist.safetensors", DEVICE)
            model = MaskGIT(length=state["length"], width=state["width"], layers=state["layers"],
                            heads=state["heads"]).to(DEVICE).eval()
            model.load_state_dict(weights)
            self.models.append(model)
            self.parameters.append(state["parameters"])
        # the picker's two-level choice: tool, then settings -> action index
        self.rings = sum(a[1] == "ring" for a in self.actions)
        self.angles, self.spacings = ANGLES, SPACINGS
        self.ring_action = [self.names.index(f"ring:{k}") for k in range(self.rings)]
        self.line_action = [[[self.names.index(f"lines:{a}:{s}:{o}") if o < s else 0 for o in range(max(SPACINGS))]
                             for s in SPACINGS] for a in ANGLES]
        self.holes_action = self.names.index("holes")
        self.centres = list(CENTRES) if "circles" in self.tools else []
        self.circle_action = [[[self.names.index(f"circles:{c}:{s}:{o}") if o < s else 0 for o in range(max(SPACINGS))]
                               for s in SPACINGS] for c in self.centres]
        self.dot_action = ([[[self.names.index(f"dots:{s}:{a}:{b}") if a < s and b < s else 0
                              for b in range(max(SPACINGS))] for a in range(max(SPACINGS))] for s in SPACINGS]
                           if "dots" in self.tools else [])
        self.fractal_action = [i for i, a in enumerate(self.actions) if a[1] == "fractal"]
        self.fractals = len(self.fractal_action)
        self.mask_id = self.models[0].mask_id
        self.pass_layers = state["length"] * state["layers"]  # token-layers of one pass
        self.temperature = temperature

    def prompts(self, images, labels):
        prefix, _ = maskgit_inputs(images, labels, self.hint_rows, self.codebook)
        return prefix.to(DEVICE)

    def empty(self, n):
        return torch.full((n, self.rows * self.cols), self.mask_id, device=DEVICE)

    @torch.inference_mode()
    def apply(self, prefix, body, action, generator=None, reference=None):
        """One pass for every row: row i uses action[i] (an index into self.actions, or -1
        for none). Returns the new body codes. With reference (real body codes), each tool
        sees the reference instead of the body around its cells (training contexts: drawn
        tiles that are near-misses of the real digit)."""
        body = body.clone()
        drawn = body != self.mask_id
        for t, model in enumerate(self.models):
            rows = torch.nonzero(torch.as_tensor([a >= 0 and self.tool_of[a] == t for a in action.tolist()],
                                                 device=DEVICE)).squeeze(1)
            if len(rows) == 0:
                continue
            cells = ~drawn[rows] if self.tools[t] == "holes" else self.masks[action[rows]]
            seen = body if reference is None else reference
            ids = torch.cat((prefix[rows], seen[rows].masked_fill(cells, self.mask_id)), 1)
            logits = model(ids)[:, prefix.shape[1]:].float() / self.temperature
            sampled = torch.multinomial(logits.softmax(-1).flatten(0, 1), 1, generator=generator).view(len(rows), -1)
            body[rows] = torch.where(cells, sampled, body[rows])
        return body

    def images(self, hint_images, body):
        """Body codes -> full images (blank tiles are the empty tile)."""
        codes = body.masked_fill(body == self.mask_id, self.blank).cpu().numpy()
        tiles = self.codebook[codes.reshape(-1, self.rows, self.cols)]
        drawn = tiles.reshape(-1, self.rows, self.cols, 2, 2).transpose(0, 1, 3, 2, 4).reshape(
            -1, 2 * self.rows, 2 * self.cols)
        full = hint_images.copy()
        full[:, self.hint_rows:] = drawn
        return full


def write_run(out, info, split, prompts, full, truth, steps, plans, drawn, layers, meta_extra):
    """A run folder for evaluate.py, fid_range.py and tool_profile.py. drawn: tiles drawn per
    image (the rest stayed blank); layers: token-layers of one pass (tokens x layers)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "meta.json", {
        "kind": "specialists", "ordering": "specialists", "hint_rows": info["hint_rows"],
        "dataset": info["dataset"], "size": info.get("size", 28), "split": split,
        "body_tokens": (info.get("size", 28) - info["hint_rows"]) // 2 * (info.get("size", 28) // 2),
        "count": len(prompts), "plan": "none", "guess": [], "fill_zero": False, "compile": False,
        "passes_per_step": 1, "hardware": device_info(), **meta_extra})
    with (out / "samples.jsonl").open("w") as handle:
        for p, s, plan, d in zip(prompts, steps, plans, drawn):
            handle.write(json.dumps({
                "index": p["index"], "label": p["label"], "generated_hex": "", "generated_tokens": int(d),
                "sequential_steps": int(s), "decode_token_layers": int(s) * layers, "prefill_token_layers": 0,
                "rule_steps": 0, "wall_seconds": None, "mean_exit_depth": None, "stop_reason": "picker",
                "tools": plan}) + "\n")
    save_grid(out / "samples.png", list(zip(truth[:50], full[:50])))
    np.save(out / "generated.npy", full)


def draw(args):
    """Draws a fixed plan with the specialists (a check of the tools, not the picker)."""
    info = read_json(Path(args.data) / "info.json")
    box = Toolbox(args.tools, args.data, args.temperature)
    images, _ = split_images(info, args.split)
    prompts = test_cases(Path(args.data), args.split)[:args.count]
    truth = images[np.array([p["index"] for p in prompts])]
    labels = np.array([p["label"] for p in prompts])
    plan = [box.names.index(name) for name in args.plan.split(",")]
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(args.seed)
    full, drawn = [], []
    for i in range(0, len(prompts), 250):
        prefix = box.prompts(truth[i:i + 250], labels[i:i + 250])
        body = box.empty(len(prefix))
        for a in plan:
            body = box.apply(prefix, body, torch.full((len(prefix),), a, device=DEVICE), generator)
        full.append(box.images(truth[i:i + 250], body))
        drawn.extend((body != box.mask_id).sum(1).tolist())
    full = np.concatenate(full)
    write_run(args.out, info, args.split, prompts, full, truth, [len(plan)] * len(prompts),
              [args.plan.split(",")] * len(prompts), drawn, box.pass_layers,
              {"tools_folder": args.tools, "temperature": args.temperature,
               "decode": {"plan": args.plan, "temperature": args.temperature}})
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("train")
    p.add_argument("--tool", required=True, choices=TOOLS)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--context", choices=("real", "drawn", "picker"), default="real",
                   help="drawn/picker: fine-tuning rounds 2/3 (need --init; picker needs --pickers)")
    p.add_argument("--init", help="folder prefix of the earlier round's specialists (PREFIX_ring, ...; "
                   "PREFIX_TOOL must exist for the trained tool)")
    p.add_argument("--pickers", nargs="+", default=[], help="trained picker folders (round 3)")
    p.add_argument("--real-share", type=float, default=0.5, help="share of rows with real-tile contexts")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    p = sub.add_parser("draw")
    p.add_argument("--tools", required=True, help="folder prefix: PREFIX_ring, PREFIX_lines, PREFIX_holes")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plan", required=True, help="comma-separated action names, e.g. ring:0,lines:45:3:0,holes")
    p.add_argument("--split", choices=SPLITS, default="valid")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"train": train, "draw": draw}[args.action](args)


if __name__ == "__main__":
    main()
