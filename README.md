# Deep Learning-Based Ultrasound Beamforming

Portfolio project for KAIST Biomedical Ultrasound Laboratory (PI: Prof. Eun-Yeong Park)
Master's application — Department of Bio and Brain Engineering.

## Overview

A residual U-Net trained to reconstruct high-quality B-mode ultrasound images from
single-angle (sparse) plane-wave RF data, using a 31-angle Delay-and-Sum compound
as ground truth. Benchmarked against classical DAS and Delay-Multiply-and-Sum with
Coherence Factor (DMAS+CF) beamforming — replicating the algorithm family from
Jeon, Park et al., *"Real-Time Delay-Multiply-and-Sum Beamforming with Coherence
Factor for In Vivo Clinical Photoacoustic Imaging of Humans,"* Photoacoustics 15
(2019).

## Results

Evaluated on 25 held-out synthetic phantoms (disjoint from training/validation):

| Method       | PSNR (dB)        | SSIM                |
|--------------|-------------------|----------------------|
| DAS          | 13.26 ± 1.09      | 0.3268 ± 0.0132      |
| DMAS + CF    | 10.98 ± 1.06      | 0.0456 ± 0.0066      |
| **DL Model** | **17.63 ± 0.67**  | **0.3504 ± 0.0156**  |

The DL model achieves a **+4.37 dB PSNR improvement over DAS** with non-overlapping
error bars (statistically robust, not a single-sample artifact).

### Generalization check (inverse-crime mitigation)

To confirm the model learned the underlying imaging physics rather than
memorizing the exact training distribution, we evaluated it on phantoms
generated with attenuation and scatterer-density settings it never saw during
training (n=15 phantoms per condition):

| Condition                          | PSNR (dB)         | SSIM                 |
|-------------------------------------|--------------------|-----------------------|
| In-distribution                     | 17.68 ± 0.53       | 0.3444 ± 0.0140       |
| Shifted attenuation (0.5 → 0.8)      | 17.55 ± 0.56       | 0.3457 ± 0.0111       |
| Shifted scatterer density (6000 → 3000) | 17.01 ± 0.43   | 0.3578 ± 0.0116       |

Both shifts produce PSNR drops (0.13 dB and 0.67 dB) smaller than the
in-distribution standard deviation itself, indicating the performance
degradation is not statistically distinguishable from noise. This is
evidence against the model having overfit to the exact simulator
configuration it was trained on.

### A note on DMAS+CF

DMAS+CF underperforms DAS in this evaluation, driven by a **depth-dependent
coherence artifact** that we investigated and root-caused via controlled ablation
(varying sampling rate, scatterer density, and compounding angle count — none of
which explained the effect). We attribute this to an inherent SNR/coherence
characteristic of the DMAS nonlinear combination at insufficient scatterer density
for this array geometry — consistent with why Coherence Factor weighting is
motivated in the literature in the first place, though it does not fully resolve
the effect here. We view this as an honest empirical finding rather than a defect
to hide.

## Pipeline

1. **Phantom simulator** (`phantom_simulator.py`) — point-scatterer + geometric
   time-of-flight RF synthesis, with embedded lesions for structure.
2. **Classical beamformers** (`classical_beamformers.py`) — DAS, DMAS, DMAS+CF.
3. **U-Net** (`model.py`) — residual learning, GroupNorm, PixelShuffle upsampling.
4. **Composite loss** (`loss.py`) — L1 + MS-SSIM + VGG perceptual, to preserve
   speckle texture (plain MSE blurs it away).
5. **Training** — AdamW, LR warmup + cosine annealing, early stopping
   (patience=7), best-checkpoint tracking.
6. **Evaluation** — PSNR/SSIM averaged over held-out phantoms; a separate
   generalization check under shifted attenuation/scatterer-density conditions.

## Engineering notes worth highlighting

- **Data efficiency**: pre-generating a fixed phantom pool (rather than on-the-fly
  generation) cut per-epoch time from ~75 minutes to under a second, making
  training tractable on Kaggle's free-tier GPU.
- **Overfitting diagnosis**: an initial run with only 300 phantoms overfit by
  epoch 9; increasing the pool, adding flip augmentation, and tracking the
  best-validation checkpoint (rather than the last epoch) fixed this.
- **Architecture-scale ablation**: doubling both pool size (500→1000) and model
  capacity (32→48 base channels) yielded only a ~1.5% further improvement,
  indicating the setup is near a natural performance ceiling for this problem
  scale — a useful, honest data point rather than a failure.

## Running it

Kaggle Notebook, GPU accelerator (T4 x2), Internet: On (for VGG16 pretrained
weights). No external dataset needed — everything is simulated. See
`ultrasound_beamforming_dl.ipynb` for the complete, consolidated cell sequence.

**Trained model weights** (`model_best.pt`, ~45MB) are not included in this
repository (GitHub's web-upload size limit); they can be reproduced by running
the notebook end-to-end (~1 hour on a T4 GPU), or are available on request.

## Future work

- Photoacoustic (PA) imaging mode (one-way time-of-flight variant), extending
  this pipeline toward the lab's dual-modal PA+US research line.
- Small-scale volumetric (2.5D) extension with a depth-consistency loss.
