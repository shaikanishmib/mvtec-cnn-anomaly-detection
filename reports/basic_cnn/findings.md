# Frozen Basic CNN findings

All figures below are measured results: 15 categories, three seeds, 45 independently fitted models. The selected policy was frozen before official calibration/test inference; no post-test configuration changes were made.

## Quality and operating point

| Metric | Macro mean | Seed standard deviation |
|---|---:|---:|
| image_auroc | 0.7126 | 0.0215 |
| image_ap | 0.8590 | 0.0126 |
| pixel_auroc | 0.7858 | 0.0249 |
| pixel_ap | 0.2354 | 0.0104 |
| au_pro | 0.5498 | 0.0355 |
| recall | 0.4018 | 0.0167 |
| fpr | 0.0983 | 0.0506 |
| precision | 0.8733 | 0.0447 |
| f1 | 0.5106 | 0.0210 |

Image AUROC 95% whole-image bootstrap interval: [0.6951, 0.7304]. These sampling intervals are conditional on the three trained models; seed spread is separate.

## Every category (three-seed mean)

| Category | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AU-PRO | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 0.7481 | 0.9193 | 0.7809 | 0.2281 | 0.4335 | 0.4444 | 0.0333 |
| cable | 0.5061 | 0.6314 | 0.6907 | 0.1133 | 0.3358 | 0.0797 | 0.0575 |
| capsule | 0.5267 | 0.8570 | 0.5991 | 0.0290 | 0.2006 | 0.1162 | 0.0290 |
| carpet | 0.6061 | 0.8358 | 0.7423 | 0.2157 | 0.4886 | 0.3296 | 0.2143 |
| grid | 0.9287 | 0.9742 | 0.9034 | 0.2426 | 0.8014 | 0.7485 | 0.0476 |
| hazelnut | 0.8387 | 0.9236 | 0.8888 | 0.4265 | 0.7046 | 0.6190 | 0.0000 |
| leather | 0.9173 | 0.9756 | 0.9064 | 0.4013 | 0.8559 | 0.8877 | 0.3021 |
| metal_nut | 0.6290 | 0.8719 | 0.7624 | 0.2982 | 0.5481 | 0.2258 | 0.0909 |
| pill | 0.7535 | 0.9446 | 0.8337 | 0.2064 | 0.7707 | 0.3404 | 0.0385 |
| screw | 0.1989 | 0.6163 | 0.8045 | 0.0061 | 0.3774 | 0.0364 | 0.4959 |
| tile | 0.9026 | 0.9646 | 0.8113 | 0.5644 | 0.6458 | 0.7183 | 0.0707 |
| toothbrush | 0.7056 | 0.8806 | 0.8217 | 0.0856 | 0.4747 | 0.3889 | 0.0278 |
| transistor | 0.6317 | 0.5442 | 0.6526 | 0.0851 | 0.3419 | 0.1250 | 0.0500 |
| wood | 0.9465 | 0.9839 | 0.7174 | 0.3280 | 0.6247 | 0.6667 | 0.0175 |
| zipper | 0.8494 | 0.9613 | 0.8722 | 0.3004 | 0.6435 | 0.2997 | 0.0000 |

The image AP headline must be read against high defect prevalence. Macro recall is only 40.2% and observed FPR 9.8%, despite a normal-calibration q95 target. This is a limited baseline, not a deployment-ready detector. All per-image outcomes and all 73 defect-type results are retained across seeds.

## Failure hypotheses and uncertainty limits

Texture/object summaries, every confusion matrix and representative available TN/TP/FP/FN panels are in Notebook 01. Local panels show original masks, fixed-scale sigmoid maps and calibrated overlays, selected by first path rather than visual quality. Synthetic-to-real gap, limited scratch features, coarse spatial predictions and image-maximum sensitivity are plausible failure explanations, not proven causes. Score saturation/ties are reported explicitly; raw-logit aggregation was not substituted after observing test results.

Pixel point estimates use every native-resolution pixel and exact tied-score sweeps. Pixel confidence intervals target the preregistered 4096-bin approximation, not the exact metric: mean pixel-AP bias is -0.0062 and maximum absolute per-run difference 0.0836. These intervals must not be presented as unbiased exact-pixel AP intervals. Image bootstrap metrics are exact.

No image or pixel run had zero predicted positives; raw records retain the foundation zero-division convention and explicit definedness flags. Pixel correlations are handled by resampling whole images, not independent pixels. Small calibration sets (minimum nine normals), only three seeds, one benchmark and descriptive test exposure limit generalization claims.

## Efficiency and retained evidence

The model has 83,857 parameters and checkpoints up to 343,117 bytes. Mean across runs of model p50 is 3.139 ms, p95 4.629 ms; mean end-to-end p50 is 12.538 ms. These are means of per-run percentiles, not a pooled request percentile. End-to-end excludes file read/decode and uses memory-resident RGB through a native CPU map.

Maximum final-training allocated VRAM was 0.480 GiB. Total recorded final-training wall time was 8.11 minutes including loading/validation. The one-shot evaluation phase wall time was 35.85 minutes; native sorting, bootstrap and IO dominate over fitting.

All 20 bounded development fits remain recorded, including rejected patch-only, perturbation-only and reduced-capacity configurations. Best/last weights, optimizer/RNG states, maps, bootstrap vectors and timing samples stay local; artifact_manifest.json records their SHA256 identities. No pretrained weights, official mask training, threshold optimization on test defects, Autoencoder or other model was used.

Next: a separately requested Notebook 02 can test a different normal-only objective. Do not start it automatically.
