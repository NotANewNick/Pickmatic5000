# 28x28 digits: everything the results need

- `data/`: `info.json` (split settings: 20,000 training digits drawn with seed 0, 8 hint
  rows, 1,000 test, 5,000 practice and 1,000 valid digits), `codebook_tile.npy` (512 2x2
  tiles, k-means) and `classifier_mnist_28.safetensors` (the judge behind "correct digit" and FID).
  The digits themselves come from `data/raw/mnist`; `common.test_cases` draws each split's
  requests from `info.json`.
- `critics/`: `classifier2.safetensors`, the second classifier in the picker's score
  (`experiment/critics.py`; 98.86% on the real test digits).
- `specialists/`: the six pattern tools (round 2), each `specialist.safetensors` + `specialist.json`.
- `baselines/`: `M_maskgit` (MaskGIT; the Halton rows use the same weights) and
  `D_diffusion` (DDPM training, DDIM sampling), 60 epochs each (`maskgit.*`, `diffusion.*`).
- `results/tool_picker/`: both pickers (`pickers/{step,plan}/picker.*`) and the valid-digit results in README.md
  (`fid_range_diffusion.txt`: the diffusion ranges, recomputed on the CPU from the stored
  runs; they match the ones cited); `results/fair_steps/`: MaskGIT and diffusion on the test
  digits.
- The `summary*.md` tables were made by the older `evaluate.py` and keep its columns. Rule
  steps, rule error, check wrong/missed and exit/bg/stroke depth belong to the removed ALoDLM
  evaluation and are always `-` or 0 here (so is "exit depths" in the footer). All numbers
  are as they were measured.

All trained on an RTX 2080 Ti (each folder's `hardware.json` or settings). The judge and
the second classifier cannot be retrained bit for bit (GPU training is not exact), so keep
these files to reproduce the numbers. `experiment/prepare_data.py` makes the same kind of
data folder for another dataset.

**Model files:** every model is `NAME.safetensors` (the weights only: plain tensors, no pickled
code) plus, where needed, `NAME.json` (the settings to rebuild it: sizes, tool, passes,
training notes). They were converted from PyTorch `.pt` files with
`experiment/convert_models.py`; the drawings and scores are bit for bit the same (checked on
the CPU). New training runs write this format directly.
