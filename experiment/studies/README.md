# Studies

Records of how the results in `reference/digits-28x28/` were made (see docs/STATUS.md and
docs/SPECIALIST_TOOLS.md). Run from the project root. They expect the `work/p2` workspace
(`D=work/p2/data`, `C=work/p2/picker_noplan/critics`, `R=work/p2/runs`); its data, judge,
critics and models are the ones now in `reference/digits-28x28/` (same files). They are not
part of `./run.sh`.

- `baselines_train.sh`: the MaskGIT and diffusion baselines (`baselines/M_maskgit`,
  `baselines/D_diffusion`), from the removed `phase2.sh`.
- `specialists_train.sh`: the first three pattern tools (ring, lines, holes), round 1 on real
  contexts (their round 2 is in `tool_picker_queue.sh`).
- `circles_train.sh`, `dots_train.sh`, `fractal_train.sh`: the circles, dots and fractal
  tools, trained the same way, then fixed plans on valid
  (`results/tool_picker/fixed_plans/`).
- `tool_picker_queue.sh`: MaskGIT and Halton-order MaskGIT on the valid digits, round 2 of
  ring, lines and holes (on each other's drawings), both pickers (step, plan), round 3 and the continued pickers, valid draws at five urgencies. Open bug:
  its tool list misses fractal for round 3 (docs/STATUS.md).
- `tools_chain.sh`: the overnight chain (the three tool scripts, then the picker queue).
- `diffusion_valid.sh`: diffusion at 1-20 steps on the valid digits
  (`results/tool_picker/summary_diffusion.md`, `fid_range_diffusion.txt`).
- `fair_steps.sh`: MaskGIT at 1-4 steps on the test digits (`results/fair_steps/`). A trimmed
  rewrite of the script that ran, not the exact commands (see its header).
- `fair_steps_diffusion.sh`: diffusion at 1-20 steps on the test digits, after the
  step-spacing fix (`results/fair_steps/`).
