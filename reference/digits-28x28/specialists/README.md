# Pattern tools (round 2)

The six specialists the pickers in `../results/tool_picker/pickers/` were trained and scored
with (RTX 2080 Ti, 2026-11-01). Each is its own network, MaskGIT-sized (width 384, 8 layers),
trained to fill one pattern of positions in one pass; round 2 = fine-tuned on drawn context.

Load all six with the prefix: `Toolbox("reference/digits-28x28/specialists/S2", "reference/digits-28x28/data")`
(`experiment/specialists.py`), or pass `--tools reference/digits-28x28/specialists/S2` to
`specialists.py draw` / `tool_picker.py`. Round 3 is not finished (see docs/STATUS.md).
