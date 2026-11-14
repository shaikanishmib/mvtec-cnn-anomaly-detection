# EfficientAD-S reduced-budget final benchmark

All 45 frozen category-seed models completed. The initial pilot and four development fits remain separate. No architecture, hyperparameter, score or threshold policy changed using test results.

Frozen identity: `d728cf544ebc6f0c3d012adbca044a7aa5ddf4c0ced2e3210bef19b77a71d321`. Nine full epochs, batch one, deterministic FP32, pinned PDN teacher, dual-head student and AE objectives, no external-image penalty. The selected score is normalized local-map maximum; global/combined maps are retained for analysis only. This is not a paper-equivalent reproduction.

## All nine macro metrics

| Metric | Mean | Seed SD | Conditional image-bootstrap 95% interval |
|---|---:|---:|---|
| image_auroc | 0.816298 | 0.011074 | [0.797444, 0.833745] |
| image_ap | 0.918593 | 0.005765 | [0.906335, 0.931228] |
| pixel_auroc | 0.919887 | 0.001631 | [0.911106, 0.928618] |
| pixel_ap | 0.498476 | 0.007130 | [0.478785, 0.517498] |
| au_pro | 0.818373 | 0.005692 | [0.805768, 0.831457] |
| recall | 0.665639 | 0.010392 | [0.643500, 0.688882] |
| fpr | 0.220081 | 0.010738 | [0.194895, 0.247498] |
| precision | 0.885939 | 0.005699 | [0.869588, 0.901699] |
| f1 | 0.740084 | 0.009945 | [0.719272, 0.758413] |

Image intervals use exact scores with shared whole-image resamples across seeds/models. Pixel intervals use 4,096 signed monotone bins; pixel point estimates are exact native-resolution metrics. Quantization bias is retained in summary.json. These intervals condition on fitted models and thresholds and do not include fitting/calibration uncertainty. Shared metric conventions set precision to zero for zero-alert runs; those cases are explicitly counted rather than treated as a measured positive predictive probability.

## Category behavior and failure evidence

| Category | Image AUROC | Pixel AUROC | Recall | FPR | FP evaluations | FN evaluations | FN with pixel overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 0.9709 | 0.9241 | 0.9418 | 0.0000 | 0 | 11 | 11 |
| cable | 0.6123 | 0.8858 | 0.3913 | 0.0517 | 9 | 168 | 142 |
| capsule | 0.7680 | 0.9018 | 0.5291 | 0.1739 | 12 | 154 | 147 |
| carpet | 0.9357 | 0.9624 | 0.9476 | 0.5833 | 49 | 14 | 14 |
| grid | 0.9502 | 0.9458 | 0.6842 | 0.0000 | 0 | 54 | 51 |
| hazelnut | 0.9035 | 0.9321 | 0.7810 | 0.0167 | 2 | 46 | 44 |
| leather | 0.9988 | 0.9955 | 1.0000 | 0.8021 | 77 | 0 | 0 |
| metal_nut | 0.6357 | 0.9295 | 0.6057 | 0.4242 | 28 | 110 | 81 |
| pill | 0.8870 | 0.9238 | 0.5816 | 0.0385 | 3 | 177 | 160 |
| screw | 0.0572 | 0.9011 | 0.0812 | 0.9106 | 112 | 328 | 178 |
| tile | 0.9977 | 0.9849 | 0.6310 | 0.0000 | 0 | 93 | 93 |
| toothbrush | 0.7130 | 0.9288 | 0.5222 | 0.1111 | 4 | 43 | 25 |
| transistor | 0.8614 | 0.6758 | 0.4167 | 0.0389 | 7 | 70 | 47 |
| wood | 0.9854 | 0.9543 | 0.9778 | 0.0877 | 5 | 4 | 4 |
| zipper | 0.9679 | 0.9526 | 0.8936 | 0.0625 | 6 | 38 | 38 |

Counts are seed-evaluations of the same official images, not independent new images. All 15 categories have genuine original/map/mask panels under ignored outputs/efficientad/final/failure_panels. The committed failure_examples.csv identifies every source and outcome. Where a category has no false positive or false negative, no such example is fabricated.

Observed: 1035 image false-negative evaluations still have at least one pixel alarm overlapping the true defect at the auxiliary normal-calibrated pixel threshold. This demonstrates image aggregation/operating-point misses, not that every localized response is accurate. See mask area strata and per-defect tables for fine scratches and small defects.

Observed texture alarms, leather/carpet results, screw detection, tiny-mask recall and normal peak locations are quantified in the tables. Pixel background alarm area and genuine overlays expose localization spread. Heatmaps have per-panel signed-value colorbars; brightness alone is not comparable across panels. A spatial halo caused by receptive fields or padding is a hypothesis unless established by the displayed case; it is not inferred from a scalar AUROC. Structural classes are reported by the actual dataset defect names. MVTec AD is not MVTec LOCO and cannot establish comprehensive logical-anomaly competence.

## Calibration

Mean observed normal calibration FPR 0.0274; test macro FPR 0.2201. 6 of 45 calibration score sets contain ties. Thresholds use higher 95th percentile and strict greater, always from reserved clean normals. Tied-score Beta intervals are inapplicable; zero-width bootstraps do not prove certainty. Per-run sample sizes, thresholds, bootstrap/leave-one-out sensitivity and order-statistic qualifications are saved.

Bottle response floors are assessed from saved per-seed calibration distributions and native peak positions, without cropping borders or changing score selection after test. The precision correction from development is preserved: exact float32 values are promoted before 17-digit CSV and parsed with round-trip precision.

## Six-model comparison

| Model | image_auroc | image_ap | pixel_auroc | pixel_ap | au_pro | recall | fpr | precision | f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Basic CNN | 0.7126 | 0.8590 | 0.7858 | 0.2354 | 0.5498 | 0.4018 | 0.0983 | 0.8733 | 0.5106 |
| Convolutional Autoencoder | 0.6346 | 0.8279 | 0.6900 | 0.1109 | 0.4494 | 0.2350 | 0.0939 | 0.7896 | 0.3146 |
| PatchCore variant | 0.9676 | 0.9885 | 0.9729 | 0.5402 | 0.9204 | 0.8904 | 0.1718 | 0.9400 | 0.9043 |
| SuperSimpleNet Lite | 0.9648 | 0.9862 | 0.9528 | 0.4361 | 0.8565 | 0.8920 | 0.1662 | 0.9412 | 0.9099 |
| PaDiM | 0.8979 | 0.9464 | 0.9659 | 0.4538 | 0.9119 | 0.6965 | 0.1445 | 0.9255 | 0.7554 |
| EfficientAD-S reduced-budget | 0.8163 | 0.9186 | 0.9199 | 0.4985 | 0.8184 | 0.6656 | 0.2201 | 0.8859 | 0.7401 |

All previous model entries come from their exact stored summaries and matching 15-category/three-seed tables; none were changed. Per-category joins and paired differences are retained. Paired exact-image bootstrap intervals cover six image metrics; paired category resampling covers all nine primary metrics. These are different uncertainty targets. No multiplicity correction is applied, and small macro differences are not evidence of general superiority.

| Model | Fit/train s | Eval s | Model p50/p95 ms | E2E p50/p95 ms | Peak train VRAM GiB | Peak eval RAM GiB | Category checkpoint bytes | Including shared trunk bytes |
|---|---:|---:|---|---|---:|---:|---:|---:|
| Basic CNN | 486.63 | 2123.11 | 3.14/4.63 | 12.54/15.21 | 0.480 | not recorded | 343117 | 343117 |
| Convolutional Autoencoder | 1359.69 | 3522.14 | 2.87/4.98 | 13.27/15.89 | 0.607 | not recorded | 131133 | 131133 |
| PatchCore variant | 118.91 | 4092.58 | 15.34/17.43 | 33.58/37.72 | 1.149 | 5.766 | 3198733 | 14378690 |
| SuperSimpleNet Lite | 5348.89 | 2697.60 | 9.69/17.34 | 28.79/39.42 | 0.709 | 5.518 | 2230175 | 13410132 |
| PaDiM | 35.81 | 1779.22 | 12.93/16.40 | 20.69/25.04 | 0.384 | 7.000 | 165480613 | 176660570 |
| EfficientAD-S reduced-budget | 8575.55 | 2814.06 | 23.07/27.60 | 31.66/36.49 | 0.309 | 5.804 | 32249772 | 32249772 |

Costs are recorded local fitting/evaluation effort, not total pretraining effort. Shared ResNet18 layer1-3 trunk size comes from the exact recorded 11,179,957-byte export; PaDiM reuses the same verified PatchExtractor trunk. The trunk is shared across categories/seeds, not duplicated 45 times. Original weight downloads and persistent feature caches are separate; feature-cache bytes are retained in the cost CSV and are not deployment-model size. Earlier latency scopes differ and CUDA scheduling/power state varied across sessions. Basic CNN/AE evaluation RAM was not recorded. EfficientAD timings conservatively include both raw maps even though only the local map is scored. Per-run percentiles are averaged, not pooled into a new latency distribution.

## Actual resources and budget

Training 8575.55 s; calibration 110.51 s; evaluation 2814.06 s; run pipelines 3.346 hours. The 2.68-hour training and 10.16-hour conservative forecasts remain historical projections, not substituted for these measurements.

Final-checkpoint serialization and replay verification total 9.24 s; teacher preparation 23.59 s; development-map scaling 105.19 s. Per-epoch progress snapshot overhead is included in pipeline wall time but is not separately isolated.

Peak training allocated/reserved VRAM 0.309/0.459 GiB; peak inference allocated/reserved 0.107/0.189 GiB; sampled worker peak RSS 5.804 GiB. Maximum checkpoint 32,249,772 bytes includes teacher, student and AE. Manager process and driver-reserved memory are outside worker RSS/PyTorch allocator scopes.

## Limitations and conclusion

The fixed epoch budget produces 378–2475 optimizer steps per model (68,877 overall), so category differences also involve different update counts. This is disclosed, not retrospectively equalized.

The pinned third-party teacher has repository Apache-2.0 licensing but no independently verified separate weight license/training log. No legitimate external-image penalty input was available. Nine epochs, one-seed/two-category optimistic proxy selection, full-image max floors, small normal calibration sets and exclusion of the global map limit conclusions. Exact test results characterize this reduced variant only. Real-data panels and all failed/interrupted attempts are preserved. Notebook 07 — Final Model Comparison is the next milestone and has not started.

### Post-evaluation visual review: actual observations

The all-category contact sheet and individual screw/leather panels were visually inspected after all 45 evaluations. The following observations refer to the first retained genuine case in each category; full-resolution panels and additional fine-defect zooms remain local. No model, score, border handling or threshold was changed. Heatmaps have separate signed-value colorbars, so brightness alone is not a cross-branch comparison.

| Category / seed | Genuine source | Outcome | Score / cutoff | Visual observation |
|---|---|---|---|---|
| bottle / 2027 | `bottle/test/contamination/000.png` | FN | 0.47813 / 0.47813 | Central contamination responds locally, with bright corner/rim responses; the image is missed at a score tied with its cutoff. |
| cable / 2026 | `cable/test/missing_cable/006.png` | FN | 0.520378 / 0.535737 | The missing-cable interior produces a clear response and overlapping pixel alarms, yet its image maximum is below the cutoff. |
| capsule / 2026 | `capsule/test/faulty_imprint/005.png` | FN | 0.305825 / 0.366305 | A compact response aligns with the faulty imprint on the red half; it remains below the image cutoff. |
| carpet / 2026 | `carpet/test/color/002.png` | FN | 0.315676 / 0.329155 | The color defect receives a local response alongside off-target texture spots. The unselected AE map has strong border responses. |
| grid / 2026 | `grid/test/thread/009.png` | FN | 0.660734 / 0.769681 | A diagonal thread is visible in the local response; the alarm band is broader/more connected than the fragmented ground-truth mask. |
| hazelnut / 2027 | `hazelnut/test/cut/001.png` | FN | 0.389068 / 0.389507 | Two cut regions respond, but the image maximum is only slightly below the calibrated cutoff. |
| leather / 2026 | `leather/test/good/030.png` | FP | 0.797814 / 0.303777 | A genuine normal texture has a bright local hot spot and diffuse pixel alarms despite an empty mask: an actual false positive. |
| metal_nut / 2027 | `metal_nut/test/flip/007.png` | FN | 0.303931 / 0.310598 | The flipped object has broad ring/boundary responses and partial object alarms, but still fails the image cutoff. |
| pill / 2028 | `pill/test/faulty_imprint/011.png` | FN | 0.410777 / 0.510062 | The faulty-imprint area responds together with rim/off-target spots; the image remains below its cutoff. |
| screw / 2028 | `screw/test/scratch_neck/020.png` | FN | 0.362458 / 0.399855 | Neck scratches receive local pixel alarms while the image is missed; the unselected AE map also emphasizes the head/tip. |
| tile / 2026 | `tile/test/glue_strip/003.png` | FN | 1.12063 / 1.27803 | The large glue-strip region responds broadly and overlaps the mask, but its maximum is below the image cutoff. |
| toothbrush / 2026 | `toothbrush/test/defective/014.png` | FN | 0.245262 / 0.261065 | The bristle-top region responds, with partial coverage and separate off-target spots; the image is still missed. |
| transistor / 2028 | `transistor/test/damaged_case/008.png` | FN | 0.276791 / 0.322679 | Upper-case damage receives a local response with broader surrounding responses; the image maximum remains below its cutoff. |
| wood / 2026 | `wood/test/liquid/002.png` | FN | 0.360663 / 0.373221 | Three liquid regions are visible in the local map and alarm overlay, yet the image maximum falls below its cutoff. |
| zipper / 2028 | `zipper/test/squeezed_teeth/001.png` | FN | 0.449607 / 0.463272 | Squeezed upper teeth receive a local response alongside border responses; the image remains below its cutoff. |

Observed versus proposed explanation: localized responses below an image cutoff, normal-texture alarms, corner/border responses and spatial spread beyond masks are directly visible. Padding/receptive fields, insufficient optimization, missing external regularization or acquisition-domain differences are possible explanations; these panels do not causally separate them. A combined/global map looking different does not establish that an alternative score would improve the frozen benchmark. No alternative test-driven score was selected.

The frozen reduced variant met the compute target but is not a reliable universal operating-point detector here. The severe screw failure and carpet/leather false alarms remain central limitations even where pixel ranking or image AP is high. There is no claim of full-paper equivalence or deployment readiness.

### Paired uncertainty: selected exact-image contrasts

Differences below are EfficientAD minus the recorded baseline, using the same 2,000 whole-image draws and fixed models/thresholds. Lower FPR is favorable. These conditional intervals differ from the separately retained 15-category resampling intervals and do not imply unrestricted statistical superiority.

| Baseline | Metric | Difference | Conditional paired 95% interval |
|---|---|---:|---|
| Basic CNN | image_auroc | +0.1037 | [+0.0778, +0.1277] |
| Basic CNN | fpr | +0.1217 | [+0.0942, +0.1522] |
| Basic CNN | f1 | +0.2295 | [+0.2065, +0.2522] |
| Convolutional Autoencoder | image_auroc | +0.1817 | [+0.1482, +0.2148] |
| Convolutional Autoencoder | fpr | +0.1261 | [+0.0965, +0.1567] |
| Convolutional Autoencoder | f1 | +0.4254 | [+0.3972, +0.4540] |
| PatchCore variant | image_auroc | -0.1513 | [-0.1705, -0.1341] |
| PatchCore variant | fpr | +0.0483 | [+0.0135, +0.0852] |
| PatchCore variant | f1 | -0.1643 | [-0.1853, -0.1457] |
| SuperSimpleNet Lite | image_auroc | -0.1485 | [-0.1673, -0.1300] |
| SuperSimpleNet Lite | fpr | +0.0538 | [+0.0207, +0.0867] |
| SuperSimpleNet Lite | f1 | -0.1698 | [-0.1903, -0.1528] |
| PaDiM | image_auroc | -0.0816 | [-0.1046, -0.0598] |
| PaDiM | fpr | +0.0756 | [+0.0374, +0.1145] |
| PaDiM | f1 | -0.0153 | [-0.0376, +0.0067] |
