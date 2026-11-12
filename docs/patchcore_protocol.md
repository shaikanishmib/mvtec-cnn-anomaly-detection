# Notebook 03 protocol: bounded PatchCore variant

Preserve all earlier notebooks, results and frozen 70/15/15 normal partitions.
Development categories are bottle, cable, carpet and tile, fixed for object and
texture coverage before PatchCore official test evaluation. No official test
images, masks or labels select a backbone, budget, score, transform or threshold.

Start with cached torchvision ResNet18 IMAGENET1K_V1; inspect cached weight
availability and size without downloading multiple backbones. Freeze all
parameters and BatchNorm statistics. Layer2/layer3 yield 128x32x32 and 256x16x16
features for 256-square images. Pool each locally with 3x3 averaging, bilinearly
align layer3 to layer2 and concatenate 384-dimensional patch embeddings.

Preprocessing exception: keep the previous notebooks' whole-image PIL bilinear
256-square geometry, then use V1 weight transforms with resize/crop size 256.
This retains the official ImageNet mean/std and transform implementation while
avoiding the default classification 224 center crop, which discards borders.
This is explicitly an override of default classification geometry. No test
statistics normalize features. Raw features are the starting policy; L2 feature
normalization requires a separately recorded development hypothesis.

The initial pilot uses first 24 bottle fitting normals, first 8 development
normals, seed 2026 and a 512-patch bank. A random candidate pool capped at 20,000
fitting patches is projected to 32 dimensions for greedy farthest-first coreset
selection. Source indices must be unique. Nearest-neighbor search operates in
the original 384 dimensions with GPU float32 blocked GEMM and direct distance
refinement. Distance blocks are at most 512x2048 (4 MiB float32). No full pairwise
batch allocation. Self-search, float64 fixture comparison and anomaly-map shape
checks must pass before all-category work. Report real resources and estimates.

This is a simplified PatchCore variant: ResNet18 instead of WideResNet50-2,
candidate-pool cap, Gaussian projection for coreset construction, direct feature
concatenation, no paper image-score neighborhood reweighting and no Gaussian
map smoothing. Do not imply paper-equivalent scores. Interpolate continuous
patch distances to native image dimensions for primary localization metrics.

After a feasible pilot, bounded development compares bank budgets 256/1024/2048
and max-patch versus top-1%-patch mean image aggregation. The candidate pool cap
and projection dimension stay fixed. Deterministic fitting embeddings are
cached once and shared between budgets/seeds. Development uses an inner 20%
fitting holdout for fixed labeled synthetic proxies (excluded from bank), plus
the reserved normal development partition. Select the single budget/aggregation
pair maximizing four-category mean (proxy image AUROC + proxy pixel AP)/2;
break exact ties by lower measured batch-one search latency, then smaller bank.
Synthetic proxy labels are not real defects. No adaptive large grid.

Final selection freezes source identities, weights, preprocessing, bank budget,
normalization, image aggregation and calibration before test access. Prefer
seeds 2026/2027/2028 if measured pilot cost permits. Final category-specific banks
use all fitting normals, never calibration/development/test embeddings. Image
thresholds: shared higher normal q95, strict > alarm, target FPR 5%. Pixel
thresholds: pooled normal map q99.5, calibrated separately at 256 square.
Distances are not probabilities and must not be clipped into [0,1]. Any binned
uncertainty approximation must disclose its distance range and tail behavior;
native point metrics remain exact using the existing shared implementation.

Pilot budgets: allocated GPU memory <6 GiB, process RSS <16 GiB, pilot <10 min;
review feasibility if exceeded or complete workload is disproportionate.
Large tensors, weights, maps, banks and licensed panels remain ignored locally.
All starts/results/failures/decisions are immutable registry events. A live pilot
and later explicit artifact-analysis executions must be distinguished.

Sources: Roth et al., CVPR 2022, https://arxiv.org/abs/2106.08265;
torchvision weight metadata and https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18 .
