# PaDiM findings

All 15 categories completed across seeds 2026, 2027 and 2028 (45 category-specific models). The real GPU pilot and 16 bounded development runs preceded the frozen confirmation. No hyperparameter or threshold changed after official test access.

## Selected method

Frozen ImageNet V1 ResNet18 layer1/2/3 features, nearest alignment to 64 square, 100 seeded channels from 448, position-specific unbiased full covariances + 0.01 I, Cholesky Mahalanobis distances, bilinear 256-square maps and Gaussian sigma 4. Image score is maximum smoothed map. The selected four-category synthetic objective was 0.716183. Synthetic proxies are not real defects.

The shared whole-image 256-square bilinear geometry differs from the paper's 224 center crop and bicubic interpolation; we use three separate seeds rather than a ten-model score ensemble. Report these as shared-protocol PaDiM results, not a direct reproduction of the publication.

## Measured macro results

| Metric | Mean | Seed SD | Conditional image-bootstrap 95% interval |
|---|---:|---:|---|
| image_auroc | 0.897933 | 0.007678 | [0.878985, 0.915859] |
| image_ap | 0.946389 | 0.003356 | [0.934185, 0.959208] |
| pixel_auroc | 0.965920 | 0.001314 | [0.962662, 0.969424] |
| pixel_ap | 0.453764 | 0.020768 | [0.435724, 0.472376] |
| au_pro | 0.911929 | 0.003998 | [0.905425, 0.918062] |
| recall | 0.696493 | 0.045293 | [0.678903, 0.714151] |
| fpr | 0.144521 | 0.035187 | [0.115096, 0.174712] |
| precision | 0.925512 | 0.011029 | [0.908106, 0.942371] |
| f1 | 0.755358 | 0.030319 | [0.737451, 0.770298] |

Primary pixel metrics are exact at native resolution. Pixel intervals use whole-image resampling of 4096-bin calibration-scaled histograms; image metric intervals are exact conditional on fitted models and fixed thresholds. Seed SD and calibration-sampling uncertainty are distinct. The summary retains binned-minus-exact point differences as an approximation diagnostic.

## Comparison with all four earlier models

| Model | Image AUROC | Pixel AUROC | Pixel AP | AU-PRO | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Basic CNN | 0.712589 | 0.785832 | 0.235382 | 0.549807 | 0.401760 | 0.098341 |
| Autoencoder | 0.634623 | 0.690002 | 0.110897 | 0.449350 | 0.235000 | 0.093944 |
| PatchCore variant | 0.967572 | 0.972944 | 0.540154 | 0.920399 | 0.890435 | 0.171806 |
| SuperSimpleNet Lite | 0.964783 | 0.952850 | 0.436077 | 0.856502 | 0.892012 | 0.166249 |
| PaDiM | 0.897933 | 0.965920 | 0.453764 | 0.911929 | 0.696493 | 0.144521 |

Paired category/seed/image bootstrap differences for every previous model are retained in paired_image_difference_intervals.csv. These are conditional benchmark comparisons, not a deployment generalization claim.

## Threshold errors and localization limits

The weakest category recalls were screw (0.160), cable (0.275), pill (0.395). The largest normal test FPRs were toothbrush (0.389), hazelnut (0.367), bottle (0.333). The higher empirical normal q95 calibration target does not guarantee a 5% test or population false-alarm rate. Small calibration sets and changed normal variation can produce substantial threshold errors even when ranking is good.

All per-defect results, pixel operating points, category/seed predictions and confusion counts remain available. Mask-area strata and low-recall defect types are retrospective descriptions; repeated images across three seeds are identified as such. Licensed strongest-FP/weakest-FN panels stay local. Spatial Gaussian mismatch, normal multimodality and smoothing of small defects are plausible limitations; no counterfactual isolates their causes.

## Measured resources

Total extraction + Gaussian fit time: 35.81s across all 45 models. Mean category/seed batch-one model p50/p95: 12.930/16.401 ms; end-to-end p50/p95: 20.692/25.042 ms. These are means of per-run quantiles, not pooled quantiles. Timing uses 50 warmups and 200 synchronized iterations; file decode is excluded. Model time contains feature extraction + Gaussian scoring; smoothing/native restoration are postprocessing.

Peak fitting allocated VRAM: 0.384 GiB; peak inference allocated VRAM: 0.232 GiB; peak evaluation process RSS: 7.000 GiB. Maximum Gaussian checkpoint: 165,480,613 bytes. Shared frozen ImageNet weights are stored separately. No fit-time feature cache hides the repeated per-seed extraction cost; existing resized RGB caches may be reused. Native metric sweeps dominate evaluation cost.

## Scope and next work

Milestones 1–6 are preserved. This milestone completes PaDiM only. EfficientAD has not been started. A future family comparison or deployment choice needs its own authorized stage and independent confirmation for claims beyond this benchmark.
