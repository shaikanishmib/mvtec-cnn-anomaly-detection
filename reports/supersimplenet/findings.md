# SuperSimpleNet Lite: measured findings

All15 categories completed seeds [2026, 2027, 2028] with frozen `noise05` / `top1pct` policy.
This is an intentionally reduced-budget ICPR-style variant: ResNet18 ImageNet V1,
aligned layer2/3 features, widths384/256/32,40 epochs, batch8 and gradient clipping10.
It is not a reproduction of the published WideResNet50-2/300-epoch benchmark.
The original adapted-classifier and unsupervised detached-segmentation routes are explicit.
No official test image or mask entered development, fitting or threshold selection.

## Development evidence

The bounded study retained16 completed trials over four predefined object/texture categories.
An externally interrupted focal-only cable trial retained its first two epochs and restarted
once as v2 because no checkpoint existed. The other completed trials were not repeated.
Reference width fit memory but projected to3.4 hours for45 fits at40 epochs and was excluded
by the prospective90-minute training budget. Lite magnitude, adaptation and loss ablations
used the same fixed .05 held-out synthetic proxies for final selection, avoiding an easier
evaluation task for the stronger-noise candidate. Within-trial curves retained training noise.
Selected mean common-proxy image AUROC / pixel AP / objective: 0.998698 / 0.929798 / 0.954846.
These proxies establish synthetic learnability, not industrial defect accuracy. The small
development set, single development seed and coupled backbone/width reduction limit inference.
There was no full-budget reference-quality comparison.

PatchCore and this model use ImageNet priors, whereas the earlier CNN and
autoencoder trained from scratch. Matching partitions and metrics does not
isolate the causal effect of learned synthetic discrimination: feature geometry,
scoring and training differ as well. No additional unregistered ablation was run.

## Exact-record macro comparison

| Metric | Basic CNN | Autoencoder | PatchCore variant | SuperSimpleNet Lite |
|---|---:|---:|---:|---:|
| image_auroc | 0.712589 | 0.634623 | 0.967572 | 0.964783 |
| image_ap | 0.858954 | 0.827916 | 0.988538 | 0.986239 |
| pixel_auroc | 0.785832 | 0.690002 | 0.972944 | 0.952850 |
| pixel_ap | 0.235382 | 0.110897 | 0.540154 | 0.436077 |
| au_pro | 0.549807 | 0.449350 | 0.920399 | 0.856502 |
| recall | 0.401760 | 0.235000 | 0.890435 | 0.892012 |
| fpr | 0.098341 | 0.093944 | 0.171806 | 0.166249 |
| precision | 0.873299 | 0.789620 | 0.940045 | 0.941235 |
| f1 | 0.510593 | 0.314635 | 0.904345 | 0.909892 |

Image AUROC / pixel AP / recall / test FPR are 0.964783 / 0.436077 / 0.892012 / 0.166249.
The comparison must consider localization, recall and false alarms together with latency.
AP and precision reflect the benchmark defect prevalence. Small macro differences do not
establish statistical superiority. Paired intervals condition on the fitted models and
calibrated thresholds, preserving category/image dependence and matching training seeds.
They exclude fitting and calibration sampling uncertainty; seed standard deviations are separate.
Image FPR whole-image bootstrap95% interval: [0.13969643532670453, 0.19506479392741766].
Pixel operating macro: {'pixel_recall': 0.8091883661565695, 'pixel_fpr': 0.06855046535405314, 'pixel_precision': 0.28818212238089386, 'pixel_f1': 0.38204049446325183}.
Pixel decisions use their own threshold and background pool, distinct from image decisions.

## Focal categories and normal calibration

Leather FPR 0.562500; carpet FPR 0.488095; screw image AUROC 0.822573 and recall 0.591036.
Lowest category image AUROC: screw (0.822573); highest FPR: leather (0.562500).
Reserved normal calibration counts range from 9 to 58 images per category/seed.
Nominal image FPR is5%, using higher q95 with strict greater-than decisions. Under continuous
IID normals, finite order-statistic sampling alone gives expected unseen-normal FPR
between 4.88% and 10.00% for these sample sizes; this is not a target guarantee.
The stability table includes empirical calibration FPR, score concentration, normal test shift,
threshold bootstrap intervals, binomial Wilson test-FPR intervals and IID order-statistic intervals.
Zero empirical false alarms do not imply a zero population FPR. Bootstrap intervals alone
can miss unseen tail values. Correlated pixels are not independent calibration observations.
Calibration distributions contain only original clean normals. No test-tuned threshold was introduced.

Small normal-development brightness changes were evaluated only as a post-freeze diagnostic.
Across altered views, mean absolute image-score change was 0.045173; 3 alarm decisions changed.
Those augmented views never entered fitting or calibration. This sensitivity does not prove
that brightness caused test false positives, and no policy was changed in response.

## Observed failures and fine localization

Each category has deterministic original/mask/map/overlay examples in the full local notebook.
Compact numerical diagnostics and paths remain public; original dataset images are not in Git.
- `leather/test/good/000.png` is FP: image score -0.305979 versus frozen threshold -0.382342; top1% native-pixel overlap with ground truth is 0.0000, and predicted pixel area is 2.18%. Image and pixel decisions are separate.
- `leather/test/good/026.png` is FP: image score -0.000172 versus frozen threshold -0.382342; top1% native-pixel overlap with ground truth is 0.0000, and predicted pixel area is 8.99%. Image and pixel decisions are separate.
- `carpet/test/good/002.png` is FP: image score -0.144011 versus frozen threshold -0.463750; top1% native-pixel overlap with ground truth is 0.0000, and predicted pixel area is 4.61%. Image and pixel decisions are separate.
- `carpet/test/good/009.png` is FP: image score 0.225464 versus frozen threshold -0.463750; top1% native-pixel overlap with ground truth is 0.0000, and predicted pixel area is 15.07%. Image and pixel decisions are separate.
- `carpet/test/metal_contamination/000.png` is FN: image score -0.585691 versus frozen threshold -0.463750; top1% native-pixel overlap with ground truth is 0.0260, and predicted pixel area is 0.51%. Image and pixel decisions are separate.
- `screw/test/good/014.png` is FP: image score 0.351966 versus frozen threshold 0.257154; top1% native-pixel overlap with ground truth is 0.0000, and predicted pixel area is 2.56%. Image and pixel decisions are separate.
- `screw/test/manipulated_front/000.png` is FN: image score -0.214100 versus frozen threshold 0.257154; top1% native-pixel overlap with ground truth is 0.0843, and predicted pixel area is 0.00%. Image and pixel decisions are separate.
- `transistor/test/bent_lead/003.png` is FN: image score 0.065329 versus frozen threshold 0.137275; top1% native-pixel overlap with ground truth is 0.5319, and predicted pixel area is 1.31%. Image and pixel decisions are separate.
- `transistor/test/good/014.png` is FP: image score 0.393375 versus frozen threshold 0.137275; top1% native-pixel overlap with ground truth is 0.0000, and predicted pixel area is 1.38%. Image and pixel decisions are separate.

Native defect-size strata (seed2026; retrospective, no selection):

| Size group | Images | Recall | Median native mask fraction |
|---|---:|---:|---:|
| at most 0.1% | 17 | 0.764706 | 0.000861 |
| 0.1% to 1% | 482 | 0.778008 | 0.004050 |
| above 1% | 759 | 0.942029 | 0.031327 |

A dense head can respond outside the actual defect or miss a defect even if synthetic
discrimination is strong. The displayed examples and defect/background logit differences
quantify that behavior; they do not isolate a single cause. The64-grid head interpolates
32/16-grid backbone activations and supplies no new measured fine detail. Smoothing and
coarse pretrained features are plausible limitations, but no higher-resolution counterfactual
was run. Structural and scratch-type results remain in the per-defect tables, including misses.
Synthetic-to-real transfer is tested by the final benchmark, not assumed from proxy AUROC.

## Measured compute and storage

Final active fit wall seconds: 5348.89; training-loop seconds: 5139.45; native evaluation seconds: 2697.60.
Mean category/seed model p50/p95: 9.689/17.339ms; decoded-RGB-to-native-CPU-map p50/p95: 28.793/39.422ms.
Feature extraction / learned-head p50: 8.292/1.185ms.
Peak allocated training / inference VRAM: 0.709/0.082GiB; peak sampled evaluation RSS: 5.518GiB.
Largest category-head checkpoint: 2230175bytes; shared frozen extractor: 11179957bytes once.
Compatible per-image feature cache: 5354 images, 31.372GiB; reused across seeds and not needed for inference.
Timing uses float32, batch1,50 warmups and200 measurements; decoded input is memory-resident,
file decode excluded. Gaussian/native interpolation and CPU map transfer appear in end-to-end
cost rather than model-only time. GPU peaks use allocator counters; process RSS sampling is20Hz.
Power context is recorded. Different job times and thermal conditions limit tiny latency comparisons.
Final checkpoints support exact epoch recovery; interrupted downtime is excluded from active training.
Pixel bootstrap quantization bias is reported explicitly, while primary native metrics remain exact.

## Conclusions and limits

This study evaluates a fixed, affordable learned feature-discrimination baseline and preserves
its weak settings and failure cases. Results apply to these splits, widths, epochs, preprocessing
and calibration policy. They do not establish paper equivalence or universal production quality.
Future changes need a new preregistered study rather than post-test tuning. No Notebook05,
PaDiM or EfficientAD work was started.
