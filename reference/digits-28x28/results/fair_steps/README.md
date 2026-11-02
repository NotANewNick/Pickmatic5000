# Fair test at equal steps (1,000 test digits, 28x28, RTX 2080 Ti)

- `fid_range_maskgit.txt`, `summary_maskgit.md`: MaskGIT's own schedule at 1-4 steps
  (`M_maskgit` weights). `runs/maskgit_s8` is the same model at 8 steps.
- `fid_range_diffusion.txt`, `summary_diffusion.md`: diffusion (DDIM, no guidance) at 1-20
  steps, after the step-spacing fix in `baselines.sample_diffusion` (the old spacing ended on
  t = 0 and wasted one step, so every earlier diffusion result used one step fewer than
  reported).

The MaskGIT files also held rows of the old fixed-recipe picker (removed on 2026-11-02; not included in this release).
Those rows were deleted; the MaskGIT rows are unchanged.

Scripts: `experiment/studies/fair_steps.sh`, `experiment/studies/fair_steps_diffusion.sh`.
