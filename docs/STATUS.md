# Status (2026-11-02)

RTX 2080 Ti machine. This page says where things stand and what comes next. Older work
(the ALoDLM tests, the row-by-row drawers, the fixed-recipe picker) was removed on
2026-11-02 and is not included in this release.

## Latest (2026-11-01): pattern tools + picker

See [SPECIALIST_TOOLS.md](SPECIALIST_TOOLS.md) (valid digits, RTX 2080 Ti, rows in
`reference/digits-28x28/results/tool_picker/`). Pickers on 6 tools, 60 min each: FID 5.4 at
8.3 passes against MaskGIT 7.0 at 8 steps; worse at 2-4 passes (10.8 vs 7.5 at 4). Halton
MaskGIT: 3.6 at 8 steps, 4.1 at 4.

- **Bug fixed:** `@torch.inference_mode()` had slid onto `halton_order` instead of
  `sample_maskgit`, so MaskGIT drawing kept gradients and ran out of memory at 2+ steps.
- **Bug, not yet fixed:** `tool_picker_queue.sh` lists tools as `ring lines holes` plus
  circles/dots only, so round 3 skipped fractal and the continued pickers (6 tools) could not
  load the 5-tool S3 box. Fix the list, train S3_fractal, then rerun step3/plan3.
- **Stopped 2026-11-02 (power cost):** `M_maskgit_long` (MaskGIT trained as long as all
  specialists together) at epoch 422 of 588. Not in the repo.
- Next: triangles branch (`triangles`, not merged), triangles_train.sh, pickers with 7 tools.

## Baselines at equal steps (2026-10-31)

28x28 digits, 1,000 **test** digits, RTX 2080 Ti, same training digits for every model. Rows in
`reference/digits-28x28/results/fair_steps/`. Cells: correct digit / FID (10-90% resampling
range). Lower FID = more realistic.

| Steps | **MaskGIT (baseline)** | Diffusion (DDIM) |
|---|---|---|
| 1 | 97.6% / 36.4 (35.7-38.3) | 88.0% / 128.8 (121.2-135.5) |
| 2 | 98.5% / 10.9 (9.8-11.9) | 97.3% / 20.9 (20.1-24.0) |
| 3 | 98.3% / 8.3 (7.2-9.3) | 97.7% / 13.6 (12.9-15.2) |
| 4 | 98.3% / 8.0 (6.7-9.8) | 98.1% / 10.7 (9.9-11.8) |
| 8 | 98.7% / 6.2 | 98.3% / 6.9 (5.6-8.1) |
| 10 | | 98.5% / 6.3 (4.9-7.4) |
| 20 | | 98.1% / 5.2 (3.9-6.0) |

- **Diffusion bug, fixed:** `baselines.sample_diffusion` spaced its steps so the last one
  ran at t = 0 and changed nothing. Every diffusion result before 2026-10-31 used one step
  fewer than reported. Now "trailing" spacing (Lin et al. 2024); the old 3-step FID equals
  the new 2-step FID exactly.
- "Diffusion" is our own small model (DDPM training, DDIM sampling), not Stable Diffusion.
- **Prior work for the picker (REFERENCES.md):** picking a whole generation tool per request
  for quality vs cost is published: CATImage (Li et al., TMLR 2025) is the closest and the
  credited baseline. Ours picks the tool for every pass, from the request (and, for the step
  picker, the image so far).
- **Competitor shortlist** (not built yet): Improved MeanFlow (CVPR 2026), Inductive Moment
  Matching (ICML 2025), DMD2 (NeurIPS 2024), Soft-Di[M]O (ICLR 2026); free sampler upgrade
  DPM-Solver++.

## Quality measures

"Correct digit" alone is not enough: per-digit average images and copies of training digits
both score high. `evaluate.py` reports:
- **Correct digit:** the judge (a small CNN, `reference/digits-28x28/data/classifier_mnist_28.safetensors`)
  reads the requested digit.
- **FID** on the judge's 128 features, generated vs the real digits of the same requests.
  Lower is better. `fid_range.py` adds a 10-90% resampling range, and `--real-floor` gives
  the FID of real vs real digits.
- **Copy ratio:** median distance to the nearest training digit, divided by the same for
  the real digits. About 1 is normal; copies score 0.

Rules:
- A method guided by a classifier must be judged by an independent one: the picker's reward
  uses a second classifier (`critics.py`), never the judge.
- Always use the included judge. `evaluate.py` trains a new one when a data folder has none,
  and a new judge gives different FIDs.
- 100-image results are noisy; use 1,000 digits.
- Augmentation (rotate, shift, scale) was tried and dropped: it blurred the digits.

## Next

1. Fix the tool list in `tool_picker_queue.sh`; round 3 with all tools.
2. Area lists: each pass, a list of areas, each with its own tool and settings.
3. Triangles (branch `triangles`), pickers with 7 tools.
4. Fashion-MNIST, then CIFAR-10.
5. AMD runs.
