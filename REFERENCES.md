# References and credits

This project builds on the work below. Third-party code keeps its own license.

## History

The project started as tests of **ALoDLM** (Amazon; arXiv
[2610.04198](https://arxiv.org/abs/2610.04198); code <https://github.com/amazon-science/ALoDLM>,
CC BY-NC 4.0) on small images. That code, its vendored copy and everything built on it were
removed on 2026-11-02 and is not included in this release. No remaining code is derived from it.

## Data and software

- **MNIST**: Y. LeCun, C. Cortes, C. J. C. Burges. Copy in `data/raw/mnist/`,
  downloaded from the PyTorch mirror `ossci-datasets.s3.amazonaws.com/mnist/`. It keeps its
  own terms; it is not covered by this project's licence.
- **PyTorch** 2.8.0, NumPy, Pillow (installed by `experiment/setup_env.sh`, not included).
- Fashion-MNIST: Xiao, Rasul and Vollgraf, 2017, arXiv
  [1708.07747](https://arxiv.org/abs/1708.07747). 70,000 small grayscale clothing
  pictures in 10 classes, made as a drop-in replacement for MNIST. Copy in `data/raw/fashion/`
  (the next test set). By Zalando Research, MIT licence; not covered by this project's licence.

## Baselines we compare against

Masked image generation (`experiment/baselines.py`):
- MaskGIT: Chang et al., CVPR 2022, arXiv [2202.04200](https://arxiv.org/abs/2202.04200). A
  bidirectional transformer predicts all masked tiles at once and keeps the most confident
  each step. The main baseline.
- Halton Scheduler for Masked Generative Image Transformer: Besnier et al., 2025, arXiv
  [2503.17076](https://arxiv.org/abs/2503.17076). Replaces MaskGIT's order with a fixed,
  evenly spread order (`--order halton`), same weights.

Diffusion basics (the diffusion baseline and how we score images):
- DDPM, Denoising Diffusion Probabilistic Models: Ho, Jain and Abbeel, NeurIPS 2020, arXiv
  [2006.11239](https://arxiv.org/abs/2006.11239). Makes images by learning to undo a
  step-by-step noising process.
- DDIM, Denoising Diffusion Implicit Models: Song, Meng and Ermon, ICLR 2021, arXiv
  [2010.02502](https://arxiv.org/abs/2010.02502). Same training as DDPM, but samples in far
  fewer steps.
- Classifier-free guidance: Ho and Salimans, 2022, arXiv
  [2207.12598](https://arxiv.org/abs/2207.12598). Trains one model with and without the
  label and mixes the two predictions when sampling, so no separate classifier is needed.
- Improved DDPM (cosine noise schedule): Nichol and Dhariwal, ICML 2021, arXiv
  [2102.09672](https://arxiv.org/abs/2102.09672). Source of the cosine noise schedule.
- U-Net: Ronneberger, Fischer and Brox, MICCAI 2015, arXiv
  [1505.04597](https://arxiv.org/abs/1505.04597). Encoder-decoder network with skip
  links; the usual denoiser in diffusion models.
- FID (Frechet Inception Distance): Heusel et al., NIPS 2017, arXiv
  [1706.08500](https://arxiv.org/abs/1706.08500). The image-quality score we report.

Fixed patterns filled in parallel (prior work for the tools' patterns):
- Subscale Pixel Network: Menick and Kalchbrenner, ICLR 2019, arXiv
  [1812.01608](https://arxiv.org/abs/1812.01608). Strided sub-images (a point grid) first;
  like the dots tool.
- Checkerboard Context Model for learned image compression: He et al., CVPR 2021, arXiv
  [2103.15306](https://arxiv.org/abs/2103.15306). Anchors first, then the rest in one
  parallel pass; like dots 2:0:0 -> 2:1:1 -> holes.
- Progressive Checkerboards for Autoregressive Multiscale Image Generation: Eigen, 2026, arXiv
  [2602.03811](https://arxiv.org/abs/2602.03811).
- Fast Autoregressive Video Generation with Diagonal Decoding (DiagD): Ye et al., 2025,
  arXiv [2503.14070](https://arxiv.org/abs/2503.14070). Decodes tokens along diagonals;
  lines at 45 degrees are a known idea.

Decoding order, chosen per image:
- AdaNAT: Ni et al., ECCV 2024, arXiv [2409.00342](https://arxiv.org/abs/2409.00342). A
  small policy network, trained with reinforcement learning, picks the generation schedule
  for each image in a masked image model.
- PONDER, Learning to Generate in the Right Order: Erbach et al., 2026, arXiv
  [2610.23531](https://arxiv.org/abs/2610.23531). A learned policy picks where a masked
  image model fills in next and which earlier guesses to redo.
- RandAR: Pang et al., 2024, arXiv [2412.01827](https://arxiv.org/abs/2412.01827). An image
  model that can make tokens in any order, by putting a position token before each one.
- σ-GPT: Pannatier et al., ECML/PKDD 2024, arXiv
  [2404.09562](https://arxiv.org/abs/2404.09562). Tells the model the position of each
  output, so the order can be chosen per sample.
- Adaptive Order Policies for Masked Diffusion: Mohamud et al., 2026, arXiv
  [2606.00295](https://arxiv.org/abs/2606.00295). Learns a small policy for which tokens to
  unmask next (general masked diffusion, not images only).
- Learning Generation Orders for Masked Discrete Diffusion Models via Variational
  Inference: Fox et al., 2026, arXiv [2602.23968](https://arxiv.org/abs/2602.23968). Learns
  the generation order; tested on text (GSM8K), not images.
- Reinforcing the Generation Order of Multimodal Masked Diffusion Models: Ouyang et al.,
  2026, arXiv [2607.08056](https://arxiv.org/abs/2607.08056). A learned module picks the
  token order for text-to-image and image-understanding models.
- Patch Forcing ("Denoising, Fast and Slow: Difficulty-Aware Adaptive Sampling for Image
  Generation"): Schusterbauer et al., CVPR 2026, arXiv
  [2604.19141](https://arxiv.org/abs/2604.19141). Each image patch gets its own noise
  level; easy regions are finished first and then guide the hard ones.

These works choose single tokens or patches per image. The picker here chooses whole
patterns, each drawn by its own specialist network.

Picking a whole generation tool per request (prior work for `experiment/tool_picker.py`),
found 2026-10-31. **The core idea is published**: choose per request which whole generation
tool draws it, trading quality against cost.
- CATImage, Cost-Aware Routing for Efficient Text-To-Image Generation: Li et al., TMLR
  2025, arXiv [2506.14753](https://arxiv.org/abs/2506.14753). Closest: per prompt, picks
  one tool from a mixed pool (autoregressive, 1-4 step distilled models, DeepCache, SDXL
  at several step counts, FLUX) by predicted quality minus lambda x cost. Supervised
  quality predictor; needs every tool run on every training prompt. Credited baseline.
- RouteT2I: Xin et al., ICCV 2025, arXiv [2411.13787](https://arxiv.org/abs/2411.13787).
  Per prompt, a large cloud model or a small edge model.
- DiffServe: Ahmad et al., MLSys 2025, arXiv [2411.15381](https://arxiv.org/abs/2411.15381).
  Cheap model first; a real-vs-generated checker sends weak images to the big model.
- PAK-UCB: ICML 2025, arXiv [2410.13287](https://arxiv.org/abs/2410.13287), and DAK-UCB:
  ICLR 2026, arXiv [2603.23140](https://arxiv.org/abs/2603.23140). Contextual bandits that
  pick a generator per prompt (no speed term).
- ComfyGen, Prompt-Adaptive Workflows for Text-to-Image Generation: arXiv
  [2410.01731](https://arxiv.org/abs/2410.01731). Picks a whole ComfyUI workflow per prompt.
- AdaGen (journal version of AdaNAT): Ni et al., TPAMI, arXiv
  [2603.06993](https://arxiv.org/abs/2603.06993). RL policy for the settings of one generator.
- AdaDiff: AAAI 2025, arXiv [2311.14768](https://arxiv.org/abs/2311.14768). Step count per
  sample for one diffusion model, REINFORCE on quality and speed.
- Automated algorithm selection (the general framing): Kerschke et al., Evolutionary
  Computation 2019, arXiv [1811.11597](https://arxiv.org/abs/1811.11597).

Choosing which tokens to draw or redraw (the "free picker" idea, not yet built):
- Token-Critic: Lezama et al., ECCV 2022, arXiv [2209.04439](https://arxiv.org/abs/2209.04439).
- Draft-and-Revise: Lee et al., NeurIPS 2022, arXiv [2206.04452](https://arxiv.org/abs/2206.04452).
- Learning Unmasking Policies for Diffusion Language Models: ICML 2026, arXiv
  [2512.09106](https://arxiv.org/abs/2512.09106). RL policy for which tokens to unmask;
  reports that "quality minus step cost" collapses to the cheapest action.
- Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies: arXiv
  [2510.05725](https://arxiv.org/abs/2510.05725).
- PRISM, Fine-Tuning Masked Diffusion for Provable Self-Correction: arXiv
  [2510.01384](https://arxiv.org/abs/2510.01384). ReMDM: NeurIPS 2025, arXiv
  [2503.00307](https://arxiv.org/abs/2503.00307).

Few-step competitors to train (shortlist, 2026-10-31; not yet built):
- Improved MeanFlow: CVPR 2026, arXiv [2512.02012](https://arxiv.org/abs/2512.02012);
  MeanFlow: Geng et al., NeurIPS 2025, arXiv [2505.13447](https://arxiv.org/abs/2505.13447).
- Inductive Moment Matching: Zhou, Ermon and Song, ICML 2025, arXiv
  [2503.07565](https://arxiv.org/abs/2503.07565).
- DMD2: Yin et al., NeurIPS 2024, arXiv [2405.14867](https://arxiv.org/abs/2405.14867).
- Soft-Di[M]O: ICLR 2026, arXiv [2509.22925](https://arxiv.org/abs/2509.22925).
- Also: sCM (Lu and Song, ICLR 2025, arXiv [2410.11081](https://arxiv.org/abs/2410.11081)),
  Shortcut Models (Frans et al., ICLR 2025, arXiv [2410.12557](https://arxiv.org/abs/2410.12557)),
  DPM-Solver++ (arXiv [2211.01095](https://arxiv.org/abs/2211.01095)).

DDIM step spacing (`baselines.sample_diffusion`): "trailing" spacing from Lin et al.,
Common Diffusion Noise Schedules and Sample Steps are Flawed, WACV 2024, arXiv
[2305.08891](https://arxiv.org/abs/2305.08891).

Pattern tools and their picker (`experiment/specialists.py`, `experiment/tool_picker.py`):
- Realism score, the picker's fixed realism reward: Kynkäänniemi et al., Improved Precision
  and Recall Metric for Assessing Generative Models, NeurIPS 2019, arXiv
  [1904.06991](https://arxiv.org/abs/1904.06991) (here in a second classifier's features,
  per class, capped at 1).
- The dots tool (a grid of single tiles; coarse-to-fine use is left to the picker):
  multiscale parallel filling from Reed et al., Parallel Multiscale Autoregressive Density
  Estimation, ICML 2017, arXiv [1703.03664](https://arxiv.org/abs/1703.03664).
- Fixed spread-out groups filled in one pass (prior work for the dots and lines tools):
  Halton Scheduler for Masked Generative Image Transformer (Besnier et al., 2025, arXiv
  [2503.17076](https://arxiv.org/abs/2503.17076): a quasi-random spread-out order for
  MaskGIT, no retraining, ImageNet-256 FID 3.74 vs 5.93 by confidence); Plan for Speed:
  Dilated Scheduling for Masked Diffusion Language Models (DUS, ICML 2026, arXiv
  [2506.19037](https://arxiv.org/abs/2506.19037): non-adjacent dilated groups, coarse to
  fine); Locality-aware Parallel Decoding (ICLR 2026, arXiv
  [2507.01957](https://arxiv.org/abs/2507.01957): designed groups for image patches).
- Letting a policy learn the order or the groups itself (prior work for "the picker finds
  its own patterns"): Learning-Order Autoregressive Models (Wang et al., 2025, arXiv
  [2503.05979](https://arxiv.org/abs/2503.05979)); Learning Generation Orders for Masked
  Discrete Diffusion Models via Variational Inference (Fox et al., 2026, arXiv
  [2602.23968](https://arxiv.org/abs/2602.23968)); Unifying Masked Diffusion Models with
  Various Generation Orders and Beyond (LoMDM, ICML 2026, arXiv
  [2602.02112](https://arxiv.org/abs/2602.02112)); Adaptive Order Policies for Masked
  Diffusion (2026, arXiv [2606.00295](https://arxiv.org/abs/2606.00295)); Reinforcing the
  Generation Order of Multimodal Masked Diffusion Models (2026, arXiv
  [2607.08056](https://arxiv.org/abs/2607.08056): an RL-trained order module for
  text-to-image); Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference
  Policies (ICLR 2026, arXiv [2510.05725](https://arxiv.org/abs/2510.05725)); DEMASK
  (arXiv [2604.02560](https://arxiv.org/abs/2604.02560)); Variational Order Inference
  (Li et al., ICLR 2021, arXiv [2110.15797](https://arxiv.org/abs/2110.15797)); Ordered
  AR image generation (Pramanik et al., 2025, arXiv [2504.17069](https://arxiv.org/abs/2504.17069)).
- Separate expert networks per stage of generation: eDiff-I (Balaji et al., 2022, arXiv
  [2211.01324](https://arxiv.org/abs/2211.01324): expert denoisers per noise level); our
  specialists are per pattern instead, picked each pass by the picker.
- RLOO baseline for the picker's REINFORCE: Kool et al., Buy 4 REINFORCE Samples, Get a
  Baseline for Free!, ICLR 2019 workshop (DeepRLStructPred).
- Fine-tuning rounds on the tools' own drawings: SUNDAE (Savinov et al., Step-unrolled
  Denoising Autoencoders for Text Generation, ICLR 2022, arXiv
  [2112.06749](https://arxiv.org/abs/2112.06749)) and scheduled sampling (Bengio et al.,
  NeurIPS 2015, arXiv [1506.03099](https://arxiv.org/abs/1506.03099)).

The 2x2 tile codebook (`common.build_codebook`):
- One token per 2x2 tile from a learned codebook is the idea behind VQ-VAE (van den Oord
  et al., 2017, arXiv [1711.00937](https://arxiv.org/abs/1711.00937)); ours is plain
  k-means on raw pixel tiles, no encoder network.

Prior work for NotANewNick's area-list tools (2026-11-01: the picker writes, each pass, a list of
areas, each with its own tool and settings, drawn by that tool's specialist):
- **Spending detail per region (closest to the goal):**
  - MOSAIK: Hami et al., 2026, arXiv [2608.05450](https://arxiv.org/abs/2608.05450). A
    learned predictor sets a patch size per 64-px region under a token budget, in a diffusion model.
  - Foveated Diffusion: Chao et al., 2026, arXiv [2603.23491](https://arxiv.org/abs/2603.23491).
    Mixed-resolution tokens from a given density mask.
  - D^2iT: Jia et al., CVPR 2025, arXiv [2504.09454](https://arxiv.org/abs/2504.09454).
  - DynaPatch: Li et al., CVPR 2026.
  - Patch Forcing: arXiv [2604.19141](https://arxiv.org/abs/2604.19141).
  - DQ-VAE: arXiv [2305.11718](https://arxiv.org/abs/2305.11718).
  - DPAR: arXiv [2512.21867](https://arxiv.org/abs/2512.21867).
  - Region-Adaptive Sampling (RAS): Liu et al., CVPR 2026,
    arXiv [2502.10389](https://arxiv.org/abs/2502.10389).
- **A fixed pattern, then one fill pass:** GtR (Generation then Reconstruction): Yan et al.,
  ICLR 2026, arXiv [2510.17171](https://arxiv.org/abs/2510.17171). Checkerboard tokens first,
  then the rest in very few passes, plus extra steps on the most detailed tokens. Our
  checkerboard result (dots 2:0:0 -> dots 2:1:1 -> holes) is the same idea.
- **A different sampler or model per hand-drawn region:**
  - ComfyUI Impact Pack Regional Sampler.
  - MultiDiffusion: Bar-Tal et al., ICML 2023, arXiv [2302.08113](https://arxiv.org/abs/2302.08113).
  - Mixture of Diffusers: arXiv [2302.02412](https://arxiv.org/abs/2302.02412).
  - GLIGEN: arXiv [2301.07093](https://arxiv.org/abs/2301.07093).
  - RAPHAEL (an expert per region): arXiv [2305.18295](https://arxiv.org/abs/2305.18295).
- **RL policies choosing which tokens to unmask:**
  - Learning Unmasking Policies for Diffusion LMs: Jazbec et al., ICML 2026,
    arXiv [2512.09106](https://arxiv.org/abs/2512.09106).
  - Mask-GRPO: Luo et al., NeurIPS 2025, arXiv [2510.13418](https://arxiv.org/abs/2510.13418).
  - Token-Critic: Lezama et al., ECCV 2022, arXiv [2209.04439](https://arxiv.org/abs/2209.04439).
  - Learning to Paint (RL over parameterised strokes): Huang et al., ICCV 2019,
    arXiv [1903.04411](https://arxiv.org/abs/1903.04411).
- **Not found in this search:** a picker that writes, each pass, a list of areas, each with its
  own pattern family and settings, each drawn by its own specialist network. (How the picker is
  trained is not part of the claim: reward-trained and supervised pickers are both published.)

## NotANewNick's contributions

The ideas and designs: the pattern-tool design (one specialist network per pattern, each
drawing its whole pattern in one pass), the step and plan pickers with the urgency input,
the fine-tuning rounds across specialists and the area-list design. Also the project's code
(including the baselines' reimplementations and the evaluation scripts) and the results.
Known ideas used here are credited above; they are not claimed as new.

Licence: copyright NotANewNick, CC BY-NC 4.0 for the code and the models (see [LICENSE](LICENSE)
and the README). Third-party data keeps its own terms (above).
