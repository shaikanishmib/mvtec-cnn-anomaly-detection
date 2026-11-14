# Notebook 05: PaDiM restart protocol

The failed untracked implementation, its tests, status checklist and pytest scratch
were removed on 2026-11-13. Git returned to clean commit 5e3b520. No tracked file
required reverting; no Notebook 05 or PaDiM experiment artifacts existed. All
Milestones 1–6, raw data, environments and previous outputs remain preserved.

## Statistical method and references

Defard et al., [PaDiM](https://arxiv.org/abs/2011.08785), equations 1–2:
concatenate spatially corresponding pretrained layer1/2/3 embeddings; select a
seeded random subset of channels; estimate a mean and unbiased full covariance
at each position from fit normals only. Add epsilon I (paper default .01).
Score by sqrt((x-mu)^T Sigma^-1 (x-mu)). We compute this through Cholesky
triangular solves without an explicit inverse. Correlations across layers matter.
The covariance assumes approximately Gaussian normal embeddings and spatial
correspondence; shifts, multiple normal modes and small fit sets can hurt.

Architecture reference: [Anomalib PaDiM source](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/padim/torch_model.py),
reviewed 2026-11-13. Independent implementation, no copied third-party code.
Frozen torchvision ResNet18 V1 uses existing PatchExtractor.feature_maps and its
checked cached ImageNet weights/normalization. At 256 square, expected maps are
64x64x64, 128x32x32, 256x16x16. Nearest alignment to 64 square yields 448 channels.
No PatchCore local pooling. Channel selection is uniform without replacement,
shared across images/positions; seeds 2026/2027/2028.

Whole-image PIL bilinear 256-square preprocessing preserves the shared protocol.
This differs from paper 224 center crop and bicubic geometry. Map interpolation
is bilinear to 256 square, Gaussian sigma 4 pixels (reflect without edge repeat),
then bilinear to native size. Image score is maximum smoothed 256-square map.
No test min/max scaling. This is a shared-protocol PaDiM variant, not a direct
reproduction of paper benchmark numbers or its ten-model averaging.

## Gates and bounded selection

First run numerical tests and a real normal-only bottle GPU pilot: first 24 fit
and first 8 development normals, dimension 100, epsilon .01. Record actual raw and
selected dimensions, fit/scoring duration, peak allocated/reserved VRAM, RSS and
score quantiles. Pilot limits: 6 GiB allocated, 16 GiB process RSS, 10 minutes.
No development or test access until a finite-score pilot passes these limits.

After the pilot, compare only dimensions 50/100 and epsilon .01/.1 (four settings)
on bottle/cable/carpet/tile, seed 2026. Inner 20% fit normals supply fixed synthetic
probes through existing make_probes; exclude those sources from Gaussian fitting.
Reserved development normals estimate a higher q95 threshold. Rank mean proxy
image AUROC plus proxy pixel AP divided by two; exact ties prefer lower dimension,
then larger epsilon. Synthetic scores do not establish real-defect performance.
All candidates and failures are retained; no adaptive grid. Sigma stays 4.

Freeze sources, configuration, environment, weights and partition hashes before
calibration/test access. Final category-specific fitting uses all fit normals.
Evaluate 15 categories x three seeds, retaining each result immediately. Image
threshold higher normal q95, strict >; pixel threshold pooled calibration q99.5
at 256 square. Shared exact native pixel metrics and whole-image bootstrap
(2000 resamples; binned pixel intervals disclosed) remain authoritative.
Measure batch-one timing with 50 warmups/200 iterations and CUDA synchronization.
Keep checkpoints/maps/panels local and ignored; publish compact evidence.
