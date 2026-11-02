# Pickmatic5000: specialist general pattern tools

An image is drawn in a few passes. The drawing is done by **pattern tools**: each is its
own specialist network, trained to fill one pattern of positions (ring, lines, circles,
dots, fractal, or "holes" = everything still blank) in one pass. A **picker** decides, for
every request and every pass, which tools to use and with which settings. It is never told
a recipe; it learns from a score of the finished image.

Works with portable PyTorch on NVIDIA and AMD GPUs. Tested so far on 28x28 handwritten
digits.

## How a drawing is made

1. The request comes in. For digits that is the digit to draw and the top 8 pixel rows of a
   real image.
2. The picker chooses this pass's tools and settings, or stops.
3. The chosen specialists draw their tiles (2x2-pixel squares), each seeing the whole image
   so far.
4. Repeat, up to 10 passes. The "holes" tool fills whatever is still blank.

**Today** each pass uses one tool over the whole image, for example "lines at 45 degrees,
every 3rd line". **Next:** the picker writes a list of areas for each pass, each area with
its own tool and settings, so one pass can use several specialists. For example, fine dots
where detail is needed, a fractal pattern in another area and circles in a third. This
spends detail where it matters and lets plain areas stay coarse.

**Two pickers:**
- **step:** sees the image after every pass.
- **plan:** sees only the request.

**Speed control:** an urgency input sets how much each pass costs, so one picker covers
everything from best quality to fastest.

**The score** for the finished image:
- a second classifier's confidence that it shows the requested digit;
- times how real it looks (the realism score of Kynkäänniemi et al. 2019);
- times (1 - urgency) for each pass.

## Results so far

28x28 digits, 1,000 validation digits, RTX 2080 Ti. FID: lower means more realistic.
Baseline: MaskGIT, the same network size trained on the same digits.

| passes | **MaskGIT (baseline)** | diffusion | picker |
|---|---|---|---|
| ~2 | 10.7 | 20.9 | 18.1 |
| ~4 | 7.5 | 11.8 | 10.8 |
| ~8 | 7.0 | 8.2 | **5.4** |

- At about 8 passes the picker beats MaskGIT, and the uncertainty ranges don't overlap.
- It beats our diffusion model at every pass count measured, and at 8 passes it also beats
  diffusion with 20 steps (6.0).
- MaskGIT with the published Halton order (no new training) is still ahead: 3.6 at 8
  steps. That is the target.

Details, ranges and caveats: [docs/SPECIALIST_TOOLS.md](docs/SPECIALIST_TOOLS.md). Full
history: [docs/STATUS.md](docs/STATUS.md).

## Potential

- **Detail where it is needed:** with areas, each region can get its own pattern and
  density, such as fine detail close up and coarse far away. A per-area pattern list drawn
  by separate specialists was not found in prior work.
- **One model for every budget:** urgency trades quality for speed without retraining.
- **Smaller specialists:** each tool learns one narrow job, so it may need far fewer
  parameters than one general network.
- **Real images:** Fashion-MNIST next, then CIFAR-10. Shapes such as circles and fractals
  may matter more there than on digits.
- **Text:** the same tools on a 2D page of text tokens. See future work in
  [docs/SPECIALIST_TOOLS.md](docs/SPECIALIST_TOOLS.md#future-work).

## Running it

```
./run.sh models           # download the model weights from Hugging Face (also done by ./run.sh)
./run.sh                  # set up, draw 1,000 valid digits with both pickers and the baselines, score them
./run.sh pickers --count 100
./run.sh plan "ring:0,lines:45:3:0,holes"    # one fixed tool sequence, no picker
./run.sh --help
```

**Model weights:** the code is on GitHub
([NotANewNick/Pickmatic5000](https://github.com/NotANewNick/Pickmatic5000)); the model
weights (`*.safetensors`, 392 MB) are on Hugging Face
([dontknowmyname/Pickmatic5000](https://huggingface.co/dontknowmyname/Pickmatic5000)).
`./run.sh` downloads them into the same `reference/` paths the first time (`hf download`)
and skips this when they are already there. While the Hugging Face repo is private, the
download needs a login: run `.venv/bin/hf auth login` with your own token (from
https://huggingface.co/settings/tokens), then `./run.sh models` again. (`.venv` is made by
the first `./run.sh models`, which stops with this hint if you are not logged in.)

**Setup:** `./run.sh` calls `experiment/setup_env.sh`, which builds the Python environment
in `.venv/` (PyTorch 2.8.0, NumPy, Pillow, safetensors, huggingface_hub; nothing is installed system-wide). Requirements:
Linux and an NVIDIA GPU (driver 570+) or an AMD GPU with ROCm. Without a GPU it runs on the
CPU (slow; use a small `--count`). Every result records the hardware it ran on.

**Scores:** `summary.md` (correct digit, FID, copy check, passes) and `fid_range.txt` (FID
with its 10-90% range) in the output folder (`work/run-DATE-TIME`). With the defaults the
rows match the table above (`step_u0`, `plan_u0` at about 8 passes; `u0.05` about 4;
`u0.1` about 2). On other hardware the drawings differ a little, so the FIDs do too.

**Included** (`reference/digits-28x28/`; the weights come from Hugging Face, see above):
- `data/`: split settings, tile codebook and the judge (the classifier behind "correct
  digit" and FID). The digits come from `data/raw/mnist`.
- `critics/`: the second classifier of the picker's score.
- `specialists/`: the six pattern tools.
- `baselines/`: MaskGIT (also used for the Halton order) and diffusion.
- `results/tool_picker/`: both pickers and every result above; `results/fair_steps/`: the
  baselines on the test digits.
- Every model is `NAME.safetensors` (weights) + `NAME.json` (settings), converted from the
  original `.pt` files by `experiment/convert_models.py` with identical results.

**Commands** (from the project root, each run with `.venv/bin/python`, with
`PYTHONPATH=experiment`; on old Xeons also `MKL_ENABLE_INSTRUCTIONS=AVX MKL_CBWR=AVX`). `DATA` = `reference/digits-28x28/data`,
`CRITICS` = `reference/digits-28x28/critics`.

| Command | What it does |
|---|---|
| `experiment/specialists.py train --tool lines --data DATA --out OUT_lines` | Train one specialist |
| `experiment/specialists.py draw --tools PREFIX --data DATA --plan "ring:0,lines:45:3:0,holes" --out OUT` | Draw with a fixed tool sequence |
| `experiment/tool_picker.py train --kind step --tools PREFIX --data DATA --critics CRITICS --out OUT` | Train a picker |
| `experiment/tool_picker.py draw --picker OUT --split valid --urgency 0 --out RUN` | Draw with a trained picker |
| `experiment/baselines.py generate maskgit --model MODEL --data DATA --split valid --steps 8 [--order halton] --out RUN` | MaskGIT baseline (also `train`) |
| `experiment/baselines.py generate diffusion --model reference/digits-28x28/baselines/D_diffusion --data DATA --split valid --steps 8 --cfg 1 --out RUN` | Diffusion baseline (`--cfg 1` = no guidance, as in all results; the default is 2) |
| `experiment/evaluate.py RUN ... --data DATA --out summary.md` | FID, correct digit and copy check |
| `experiment/fid_range.py RUN ... --data DATA` | FID with its resampling range |

`PREFIX` is a folder prefix: `reference/digits-28x28/specialists/S2` loads `S2_ring`,
`S2_lines` and the others. `experiment/studies/` holds the scripts that made the results
(records; they expect the `work/p2` workspace).

## Prior work and credit

Every method this builds on, or is compared with, is in [REFERENCES.md](REFERENCES.md),
including MaskGIT, the Halton scheduler, GtR, MOSAIK and the learned-order pickers. The
project started as tests of Amazon's ALoDLM; that code has been removed.

Code written with help from an AI coding assistant (Claude); the ideas and designs are
NotANewNick's.

## Licence

Copyright NotANewNick. The code and the models (`reference/`) are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Creative Commons
Attribution-NonCommercial 4.0 International): you may share and adapt them for
non-commercial use if you credit NotANewNick. Full text: [LICENSE](LICENSE).

Not covered: the data in `data/raw/`. MNIST (`data/raw/mnist/`) is by Yann LeCun, Corinna
Cortes and Christopher J. C. Burges; Fashion-MNIST (`data/raw/fashion/`) is by Zalando
Research (Xiao, Rasul and Vollgraf), MIT licence. Each keeps its own terms; see
[REFERENCES.md](REFERENCES.md).
