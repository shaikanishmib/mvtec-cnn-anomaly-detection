# PatchCore variant: measured findings

Frozen ResNet18 ImageNet V1 layer2/layer3 features, local 3x3 pooling,
384-dimensional concatenated embeddings and 1,024 patches per image.
Normal banks contain 2,048 patches selected from at most 20,000 candidates
using 32-dimensional projected farthest-first sampling. Scoring uses full
384-dimensional blocked L2 and top-1%-patch mean image aggregation. All 15
categories completed seeds 2026/2027/2028 without post-test tuning.

The selected development objective was 0.765848; the next candidate was 0.763366.
This small, single-development-seed proxy advantage is not proof of an
optimal architecture. All budgets and both aggregations remain recorded.

| Metric | PatchCore variant | Basic CNN | Autoencoder |
|---|---:|---:|---:|
| image_auroc | 0.967572 | 0.712589 | 0.634623 |
| image_ap | 0.988538 | 0.858954 | 0.827916 |
| pixel_auroc | 0.972944 | 0.785832 | 0.690002 |
| pixel_ap | 0.540154 | 0.235382 | 0.110897 |
| au_pro | 0.920399 | 0.549807 | 0.449350 |
| recall | 0.890435 | 0.401760 | 0.235000 |
| fpr | 0.171806 | 0.098341 | 0.093944 |
| precision | 0.940045 | 0.873299 | 0.789620 |
| f1 | 0.904345 | 0.510593 | 0.314635 |

Recall was 89.04%; test FPR was 17.18% versus the reserved-normal 5% target.
Whole-image bootstrap FPR interval: 14.21%–20.20%.
Ranking, localization AP/AU-PRO and recall improved over both earlier models,
but false alarms increased and inference became slower. The results describe
a quality/cost tradeoff, not unconditional production superiority. Image AP
and precision reflect the benchmark defect prevalence, not a production rate.

Lowest category image AUROC: screw (0.8690).
Lowest category recall: screw (54.06%).
Highest category test FPR: leather (69.79%).
Carpet test FPR was 55.95%; leather was 69.79%.
Texture/object mean FPR: 32.42% / 9.56%.
Normal-score histograms and per-category threshold bootstrap intervals expose
this shift relative to reserved calibration normals. High image AUROC can
coexist with an unsuitable threshold. No threshold was changed after testing.

At separate pixel thresholds, macro recall/precision/F1 were 0.8581 / 0.3289 / 0.4347;
pooled-within-category background pixel FPR averaged 5.87%.
Pixel background includes normal and defective test images. These decisions
use independent pixel calibration and must not be confused with image FPR.
Broad responses beyond defects limit operating-point precision despite high
pixel AUROC. The separate pixel tables include exact CNN/AE operating points.

## Observed failures and memory evidence

- `leather/test/good/001.png` and `carpet/test/good/003.png` are genuine normals with predicted areas over natural texture. Their nearest references are genuine fitting normals. This demonstrates false alarms despite visually similar normal references; it does not prove one cause such as lighting or insufficient bank size.
- `carpet/test/metal_contamination/000.png` is missed at image score 1.146842 against threshold 1.256326; none of its top 11 patch cells intersect the displayed mask. The larger color defect `carpet/test/color/000.png` is detected, with a broad response beyond the native defect boundary.
- `screw/test/manipulated_front/000.png` is missed at 1.788331 against threshold 1.878041. The normal `screw/test/good/014.png` scores 1.878239, just above that threshold; its marked highest-distance cell is in the background. These examples expose both missed fine structure and sensitivity to normal non-defect regions.
- `transistor/test/misplaced/002.png` visibly lacks the component but scores 1.928370 below threshold 2.018568. Its selected query cell matches background in a fitting image. This is consistent with position-agnostic normal-patch matching accepting background at an abnormal location; no positional ablation was run to isolate that mechanism.

The selected panels include 12 false negatives. Their median top-11 patch/mask-cell overlap was 0.3636.
Patch-cell overlap is not native pixel precision and receptive fields exceed
the 8x8 grid cell. Smooth/broad maps also reflect the 32-square feature grid
and interpolation; no counterfactual higher-resolution study was performed.
Minimum fraction of fitting source images represented in a bank: 100.00%.
Image-source coverage and projected candidate radius do not guarantee coverage
of all rare local patterns. The random candidate cap remains an approximation.
Largest selected cached-feature NN replay difference: 0.00000000.
No completed feature extraction was repeated for these reference panels.

Seed-2026 native mask-area analysis (descriptive and category-confounded):

| Defect area | Images | Recall |
|---|---:|---:|
| at most 0.1% | 17 | 0.6471 |
| 0.1% to 1% | 482 | 0.7510 |
| above 1% | 759 | 0.9381 |

Scratch-named defect types: 140 images, weighted seed-2026 recall 0.7286.

The smaller groups have lower recall here; category/defect composition and
feature detail are confounded. These observations do not isolate a resolution
effect. Exact native masks were retained for primary localization metrics.

## Measured cost and uncertainty

All 45 bank fits took 118.91 s, including 62.66 s of coreset construction.
Native evaluation took 68.21 minutes.
The 45 unique final feature caches contain 7.053 GiB on disk; extraction stages totaled 32.80 s once.
Extraction-stage time includes image loading, transfer and cache writes and is
already within fitting/evaluation wall time; do not add it again. Later seeds
reuse compatible features. External ImageNet pretraining cost is excluded.
Mean per-run feature/NN p50: 11.791/3.528 ms.
Combined model p50/p95: 15.339/17.428 ms; end-to-end p50/p95: 33.578/37.723 ms.
These are means of run percentiles, not pooled service percentiles. End-to-end
starts with decoded RGB in memory and excludes disk read/decode.
Peak bank-fitting GPU allocation: 1.149 GiB; batch-one inference allocation: 0.078 GiB.
Peak sampled evaluation process RSS: 5.766 GiB, including native metric scratch work, not isolated serving RAM.
Inference memory scope differs from earlier evaluators, which retained GPU
test-image caches; avoid claiming an isolated memory advantage from those values.
Each bank has 3,145,728 vector bytes; largest serialized bank is 3,198,733 bytes.
One shared frozen feature extractor is 11,179,957 bytes with 2,782,784 parameters and zero trainable parameters.
Bank-only storage must not be compared to complete CNN/AE model storage.
Paired image difference intervals reuse identical whole-image draws across
models/seeds; they are conditional on these fitted models. Seed variation is
reported separately. Pixel intervals target the disclosed 4096-bin transform;
primary point metrics use exact raw native distances.
Maximum absolute run pixel-AP binning bias was 0.000787.

## Limits and retention

This is a simplified PatchCore variant, not a paper-equivalent reproduction:
ResNet18, capped candidate pool/projected selection, direct concatenation,
no paper image-score neighborhood reweighting, no Gaussian smoothing and a
documented whole-image 256 geometry override of classification transforms.
ImageNet transfer differs from the scratch CNN/AE priors. Finite normal
calibration does not guarantee population FPR; texture false alarms are a
major limitation. No independent environment rebuild or deployment claim is
made. All rejected trials, exact comparisons and measured resource costs remain.
The pilot executed live; continuation cells analyze separately completed jobs
without claiming live fitting replay. Licensed panels, features, maps and banks
remain local and ignored. No SuperSimpleNet work was started.
