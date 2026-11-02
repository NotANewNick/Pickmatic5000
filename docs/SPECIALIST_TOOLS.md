# Pattern tools with a learned picker

An image is drawn in a few passes by **pattern tools**: each is its own specialist network,
the same size as MaskGIT (width 384, 8 layers), trained only to fill one pattern of
positions in a single pass. A **picker** chooses, for each request and each pass, which
tools to use. In the results below each pass used one tool over the whole image. Next, the
picker writes a list of areas per pass, each with its own tool and settings, so one pass can
use several specialists. The picker is never told a recipe and learns only from a score:

reward = second classifier's confidence in the right digit × realism score
(Kynkäänniemi et al. 2019) × (1 − urgency)^passes

Urgency is an input, so one picker covers "best quality" (urgency 0) through "fast"
(urgency 0.2). The cap is 10 passes.

**Tools (6):**
- **ring:** the outer rings.
- **lines:** angles 0-165° in 15° steps, spacing 2/3/4.
- **circles:** concentric, 3 centres.
- **dots:** every k-th tile of every k-th row.
- **fractal:** Sierpinski triangle and carpet.
- **holes:** everything still blank.

There are two pickers:
- **step:** looks at the image after each pass;
- **plan:** sees only the request.

## Results

28x28 digits, 1,000 validation digits, RTX 2080 Ti. FID: lower is more realistic, with the
10-90% resampling range in brackets. Baseline: MaskGIT with its own schedule, same size and
same training data.

| passes | **MaskGIT (baseline)** | plan picker | step picker |
|---|---|---|---|
| ~1.5 | 39.0 (1 step) | 36.3 | 37.0 |
| ~2 | 10.7 (2 steps) | 18.8 (2.2 passes) | 18.1 (2.3 passes) |
| ~4 | 7.5 (4 steps) | 10.8 (3.9 passes) | 11.0 (4.0 passes) |
| ~8 | 7.0 (6.2-8.0) | **5.4 (4.8-6.0)** | **5.6 (4.9-6.2)** |

- **At about 8 passes the picker beats MaskGIT:** 5.4 against 7.0, and the ranges don't
  overlap. MaskGIT at 6 steps (6.7) is no better. Digit correct: 97.8% against 98.5%.
- **Against our diffusion model** (DDIM, the same 1,000 validation digits, 2026-11-01):

  | passes | picker | diffusion |
  |---|---|---|
  | ~2 | 18.1 (15.9-19.5), 2.3 passes | 20.9 (19.1-22.2), 2 steps |
  | ~4 | 10.8 (9.4-12.0) | 11.8 (10.5-13.3) |
  | ~8 | 5.4 (4.8-6.0) | 8.2 (7.0-9.8) |
  | 20 steps | | 6.0 (5.3-7.7) |

  (Diffusion ranges: `results/tool_picker/fid_range_diffusion.txt`.)

  - **At 8 passes the picker wins clearly:** the ranges are apart, and it also beats diffusion
    at 20 steps.
  - **At 2 and 4 passes it is ahead, but the ranges overlap.**
  - **Diffusion gets the digit right more often:** 98.9% at 8 steps, against 97.8% for the
    picker.
  - The test-digit numbers were 6.9 at 8 steps and 10.7 at 4. The validation digits score
    worse for every model; MaskGIT went from 6.2 to 7.0.
- **Strongest known trick:** MaskGIT with the published Halton order (Besnier et al. 2025)
  is ahead. It scores 3.6 at 8 steps, 4.1 at 4 and 4.8 at 3, on the same model with no new
  training. Lifting the specialists to that level is the open target.

All rows: `reference/digits-28x28/results/tool_picker/`; the six specialists:
`reference/digits-28x28/specialists/`. Full history:
[STATUS.md](STATUS.md).

## Potential

- **Quality with many passes:** with no recipe given, the picker found sequences that
  beat plain MaskGIT at equal passes. So separate pattern specialists, picked per pass,
  can do better than one general network.
- **One model for every budget:** the same picker trades quality for speed through the
  urgency input. No retraining is needed for a different budget.
- **Room left:**
  - The pickers ran 60 minutes each.
  - The third training round (specialists retrained on the pickers' own sequences) has not
    reached the pickers yet.
  - New tools (triangle dots, concentric triangles) are written on the `triangles` branch,
    not trained yet.
- **Halton order inside the specialists:** Halton MaskGIT shows that the tile order
  matters a lot at few passes. The same fixed order could be used within a specialist's own
  pass, or as a tool.
- **Smaller specialists:** each tool learns one narrow job, so it may need far fewer
  parameters than MaskGIT. That could give the same quality at lower compute.
- **Real images:** digits are only the first test. Shapes such as circles, triangles and
  fractals may matter more on faces and photos (Fashion-MNIST next, then CIFAR-10).

## Future work

### Area lists (NotANewNick's design, 2026-11-01)

At each pass the picker writes a list of areas, each with:
- a box;
- a tool;
- that tool's settings.

Each area is drawn by its tool's specialist, in the same pass. The first version allows at
most 4 areas, with boxes on a 4x4 region grid. Its cost counts network runs. Pixel-sized and
block-sized tiles per area come later.

### Text on a 2D page

The same tools and picker, but every grid position holds a real text token, laid out as a
page:
- a token grid, such as 32 x 16;
- or a character grid, such as a 64 x 24 terminal.

The tools stay unchanged:
- a horizontal line is a run of consecutive tokens;
- a vertical line is the same column on every row;
- other shapes become other spacing patterns.

The specialists become masked text models, one per tool, or one shared base model with a
small add-on (adapter) per tool. Every tool stays on the menu for every kind of text. The
picker finds out what helps on prose, code, tables and lists, and mixed pages get their own
areas.

**Score:**
- does the code parse?
- are brackets and indentation correct?
- how natural does a separate language model find it?
- the cost per pass, as for images.

**Baselines:**
- block-by-block decoding (LLaDA);
- every-k-th-token decoding (DUS);
- learned unmasking policies (Jazbec et al., ICML 2026).

**First test:** small Python snippets. Code has real 2D structure, and correctness is easy
to check.

Prior work to search before building:
- LayoutLM (2D positions for reading documents);
- PIXEL (text read as rendered images).

### Hardware for the text test

The text pages are 4-30x the digit job per specialist. Estimates, not measured:

| GPU | test with ~7 specialists + pickers | notes |
|---|---|---|
| RTX 2080 Ti, 11 GB | about 1-3 days | 32 x 16 token page, small models (12-25M parameters) |
| 2x AMD MI100, 32 GB each | about half a day to a day | two specialists train at once; fits the 64 x 24 character page; tests the AMD path |

Before using MI100s, check:
- that the current ROCm and PyTorch still support them;
- that they have server-style airflow (the cards have no fans);
- that they reset cleanly when passed through to the unRAID VM.

## Prior work

Pickers that learn their own generation order are published (LO-ARM, LoMDM, Adaptive Order
Policies and others; see [REFERENCES.md](../REFERENCES.md)). So is picking a whole
generation tool per request (CATImage). What looks new here is a set of separate
specialists, each filling one pattern in one pass, picked per pass.
