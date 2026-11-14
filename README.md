# MVTec AD CNN Anomaly Detection

PyTorch research comparing normal-only industrial anomaly detection and defect
localization across all 15 original MVTec AD categories. The scientific notebooks
are the canonical experiment narrative; compact, append-only records preserve
both successful and unsuccessful experiments. The Basic CNN, convolutional autoencoder, PatchCore variant, SuperSimpleNet Lite and PaDiM baselines are complete;
the remaining model families and engineering application are planned.

## Research roadmap

| Notebook | Focus | Status |
|---|---|---|
| [00](notebooks/00_dataset_eda.ipynb) | Dataset EDA and integrity | Complete real-data EDA and verified CUDA kernel |
| [01](notebooks/01_basic_cnn.ipynb) | Basic CNN, synthetic-supervised segmentation | Complete: bounded development and all 15 categories across three seeds |
| [02](notebooks/02_convolutional_autoencoder.ipynb) | Convolutional autoencoder | Complete: bounded development and all 15 categories across three seeds |
| [03](notebooks/03_patchcore.ipynb) | PatchCore variant | Complete: bounded development and all 15 categories across three coreset seeds |
| [04](notebooks/04_supersimplenet.ipynb) | SuperSimpleNet Lite | Architecture, GPU pilots and bounded development verified; complete benchmark on all 15 categories × three seeds; reduced-budget ICPR-style variant |
| [05](notebooks/05_padim.ipynb) | PaDiM | Complete: verified GPU pilot, bounded development and all 15 categories across three seeds |
| [06](notebooks/06_efficientad.ipynb) | EfficientAD | Complete: pilot, four bounded development fits and 45 frozen reduced-budget evaluations; all 15 categories and three seeds |
| 07 | Cross-model comparison | Planned |

The **EfficientAD-S reduced-budget variant** completed 45 fresh models on all 15
categories and seeds 2026/2027/2028, nine epochs each. Both student heads and the
autoencoder are trained; the selected image/pixel map is the normalized local
teacher/student discrepancy. The global map is retained for analysis only.
There is no external-image penalty or paper-equivalence claim.

| EfficientAD macro metric | Measured mean |
|---|---:|
| Image AUROC | 0.816298 |
| Image AP | 0.918593 |
| Pixel AUROC | 0.919887 |
| Pixel AP | 0.498476 |
| AU-PRO | 0.818373 |
| Recall | 0.665639 |
| FPR | 0.220081 |
| Precision | 0.885939 |
| F1 | 0.740084 |

Measured training took **2.38 hours**; completed run pipelines took **3.35 hours**,
within the 12-hour target (the historical conservative forecast was 10.16 hours).
Mean per-run model p50/p95 is **23.07/27.60 ms**; decoded-image end-to-end is
**31.66/36.49 ms**. Peak sampled evaluation RSS is **5.80 GiB**, training CUDA
allocated/reserved **0.309/0.459 GiB**, and checkpoints are **32.25 MB** including
the teacher, student and AE. Timing includes the full two-map eager forward;
upstream pretraining effort is not measured.

The compute target was met, but operating-point behavior is unreliable across
categories: screw image AUROC **0.0572**, carpet FPR **0.5833**, and leather FPR
**0.8021**. Leather ranking remains strong despite false alarms. Tile and wood
have high image ranking; localized responses can still miss the image threshold.
Normal calibration FPR averages **0.0274**, versus test FPR **0.2201** at the frozen
higher-95th-percentile, strict-greater threshold. Six calibration runs have ties;
bottle's score floor and small-sample uncertainty remain explicit.

See the [final measured report](docs/efficientad_final_results.md),
[genuine visual review](docs/efficientad_visual_review.md),
[exact six-model table](reports/efficientad/model_macro_comparison.csv),
[paired uncertainty](reports/efficientad/paired_category_difference_intervals.csv),
[development history](docs/efficientad_development_results.md),
[fidelity audit](docs/efficientad_fidelity_audit.md),
[locked configuration](configs/efficientad_reduced_frozen.json) and
[recovery instructions](docs/efficientad_final_recovery.md).
All 121 tests and the deep artifact/notebook validator passed. Dataset-derived
panels and the full image-output notebook stay local; Git retains numeric results,
aggregate plots, all histories and provenance. No prior model results were changed.
Notebook 07 remains unstarted.

## Dataset

Use the complete original [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
from MVTec's official download form. Attribute Bergmann et al., CVPR 2019 and the
extended IJCV dataset paper. Data is licensed **CC BY-NC-SA 4.0** and must not be
published in this repository. Follow [acquisition and checksums](docs/dataset.md).

The user-provided complete dataset is already present; it was not downloaded
again or modified. Automatic root discovery supports categories directly under
`data/` or within a wrapper such as `data/mvtec_ad/`. All 15 categories passed
decoding and mask checks: **5,354 images**, comprising 3,629 training normals,
467 test normals and 1,258 test defects, each with a valid mask. See
[per-category counts](reports/category_counts.csv) and
[data/manifest identities](reports/dataset_validation.json). No exact duplicate
images were found. Original archive authenticity cannot be independently checked
without the source archive/checksum; this limitation is recorded.

## Reproduce the foundation

Use PowerShell and CPython 3.12.10. This project's `.venv` is isolated and contains
the verified CUDA-enabled scientific/Jupyter environment:

```powershell
cd C:\Users\shaik\mvtec-cnn-anomaly-detection
python -m venv .venv
New-Item -ItemType Directory -Force data,checkpoints,outputs | Out-Null
.\.venv\Scripts\python.exe -m pip install pip==25.0.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt --cache-dir .cache/pip
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name mvtec-research --display-name "MVTec research (CUDA, Python 3.12)"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe scripts/execute_eda.py
```

[Environment instructions](docs/environment.md) explain the Windows/Python-specific
artifact hash lock, direct pins, JupyterLab/kernel setup and actual CUDA checks.
The lock covers 116 external packages with downloaded-wheel SHA256 hashes.
Launch `.venv/Scripts/python.exe -m jupyterlab` and select **MVTec research (CUDA,
Python 3.12)**. A second independent environment rebuild has not been performed.

The execution script verifies the registered kernel's interpreter and saves the
full executed notebook to `outputs/eda/00_dataset_eda.executed.ipynb`. The tracked
copy preserves real numeric outputs and aggregate plots; panels containing
dataset pixels remain local. All real-data sections run; integrity failures stop
execution. Re-run `scripts/prepare_data.py` to verify the frozen split manifests.

## Methodology

Read the [research protocol](docs/research_protocol.md) before fitting. Reserve
approximately 70/15/15 of official normal training images for fitting,
development and calibration, respectively, using frozen seed-2026 partitions.
Synthetic anomalies are proxy labels only. Official test labels and masks never
train models or choose settings/thresholds. Descriptive test-image EDA is disclosed
and must not be described later as untouched/unseen test-label evaluation.

Report image AUROC/AP; pixel AUROC/AP and normalized AU-PRO over FPR ≤0.30 when
localization exists; recall, FPR, precision, F1 and confusion matrices at thresholds
calibrated on held-out normals only. Include per-category and unweighted macro
results, uncertainty, synchronized latency and peak GPU memory. No performance
numbers are reported before actual experiments. See the
[registry policy](experiments/README.md) and [configuration template](configs/experiment_template.json).

## Verified progress and compute limits

The setup includes the protocol, completed 20-section EDA, project-local dataset
validation, deterministic partitions, metric references and an append-only
experiment registry. Seed 2026 assigns 2,551 fitting, 539 development and 539
calibration normals; all 1,725 official test files remain separate. See
[environment audit](reports/environment.json),
[notebook execution evidence](reports/eda_execution.json),
[EDA findings](reports/eda_findings.json) and [validation history](reports/validation.md).

The RTX 4060 Laptop GPU reports 8,188 MiB VRAM (about 8 GB), driver 610.91.
Installed PyTorch **2.11.0+cu128** and torchvision **0.26.0+cu128** detect the GPU.
GPU matrix multiplication, a synthetic CNN forward/backward check and torchvision
CUDA NMS passed in the research kernel. See [GPU evidence](reports/cuda_verification.json).
No pretrained weights have been downloaded. The scratch Basic CNN completed a
bottle pilot and 19 further controlled development fits on four predeclared
categories. The frozen width-16, mixed-synthetic, weighted BCE+Dice policy then
trained **45 fresh models: all 15 categories × seeds 2026/2027/2028**, eight epochs
each. Synthetic edits/probes used fitting normals only; official development
remained normal. Reserved normal calibration set thresholds before one official
test evaluation per category/seed. No post-test configuration changes were made.

## Basic CNN measured baseline

Unweighted category macro means, averaged over three seeds:

| Metric | Mean |
|---|---:|
| Image AUROC | 0.7126 |
| Image AP | 0.8590 |
| Pixel AUROC | 0.7858 |
| Pixel AP | 0.2354 |
| AU-PRO, FPR ≤0.30 | 0.5498 |
| Recall | 0.4018 |
| False-positive rate | 0.0983 |
| Precision | 0.8733 |
| F1 | 0.5106 |

The 40.2% recall and 9.8% FPR expose a limited baseline; the normal-calibration
5% image-FPR target did not guarantee test FPR. Performance varies sharply by
category, including poor screw detection. Image AP must be read against defect
prevalence. Pixel point estimates use exact native-resolution evaluation; pixel
bootstrap intervals use a disclosed 4,096-bin approximation with measured AP bias.
All failures, rejected configurations and seed variation are retained.

The model has **83,857 parameters**, checkpoints about **335 KiB**, maximum final
training allocation **0.480 GiB**, and mean across runs of model p50 **3.139 ms**
(p95 **4.629 ms**). Mean end-to-end p50 is **12.538 ms**, from memory-resident RGB
to native CPU map; file read/decode is excluded. These are means of per-run
percentiles, not a pooled service benchmark. Final training took **8.11 minutes**
including loading/validation; exact native evaluation took **35.85 minutes**.

Read [Notebook 01](notebooks/01_basic_cnn.ipynb),
[all-category results](reports/basic_cnn/all_category_means.csv),
[measured findings](reports/basic_cnn/findings.md),
[development comparison](reports/basic_cnn/development_macro.csv),
[preregistration](docs/basic_cnn_protocol.md),
[frozen configuration](configs/basic_cnn_frozen.json) and
[reproduction instructions](docs/basic_cnn_reproduction.md).
The notebook shows the exact architecture, generator, loss and training loop,
full experiment story, calibration, evaluation and per-category failure analysis.
Public replay preserves genuine compact outputs; licensed image/map panels and
large artifacts remain local. New reproductions use new IDs, never overwrite v1.

## Repository layout and next stage

`notebooks/` holds the research narrative; `src/anomaly_detection/` contains small
auditable utilities; `configs/`, `experiments/`, `reports/`, `scripts/`, `tests/`
and `docs/` hold configuration, evidence and reproduction support. `data/`,
`checkpoints/` and `outputs/` are local ignored storage; scripts recreate outputs.

Notebooks 01, 02 and 03 are complete. SuperSimpleNet requires a separate request. Each future notebook
must preserve its full experimental story, tried configurations, actual results,
calibration, evaluation, failures and interpretations. After the research
comparison, a separate engineering
stage will add reusable PyTorch inference, FastAPI, a browser interface, tests,
benchmarks and deployment. That application is not implemented in this milestone.

<!-- AE_FINDINGS_START -->
## Convolutional autoencoder measured baseline

The frozen shallow depth-3, width-16, latent-8 autoencoder used MSE, 256-square RGB,
20 fixed epochs, squared channel-mean residual maps and top-1% mean image scores.
All 15 categories completed seeds 2026/2027/2028. Calibration used only the
reserved normals. There was no post-test tuning or retraining.

| Metric | Autoencoder | Basic CNN | AE minus CNN |
|---|---:|---:|---:|
| image_auroc | 0.634623 | 0.712589 | -0.077966 |
| image_ap | 0.827916 | 0.858954 | -0.031038 |
| pixel_auroc | 0.690002 | 0.785832 | -0.095830 |
| pixel_ap | 0.110897 | 0.235382 | -0.124485 |
| au_pro | 0.449350 | 0.549807 | -0.100457 |
| recall | 0.235000 | 0.401760 | -0.166760 |
| fpr | 0.093944 | 0.098341 | -0.004397 |
| precision | 0.789620 | 0.873299 | -0.083679 |
| f1 | 0.314635 | 0.510593 | -0.195958 |

Image recall was 23.50% with test FPR 9.39%,
against a calibration image-FPR target of 5%. This is an empirical normal
quantile, not a guarantee under test distribution shift. Category/seed calibration
counts and threshold bootstrap intervals are retained in each evaluation record.
Two runs emitted no image alarms; primary macro precision follows the shared zero-division convention. Defined-run-only precision was 0.826347, reported separately.
The highest category image AUROC was hazelnut (0.9588);
the lowest was metal_nut (0.2388).

Texture image AUROC/recall: 0.6926 / 0.2865;
object image AUROC/recall: 0.6056 / 0.2092.
These are measured category summaries, not evidence that one architecture
mechanism alone causes the differences.

Mean per-run model-stage p50 was 2.871 ms versus CNN 3.139 ms;
end-to-end p50 was 13.272 ms versus 12.538 ms.
Both exclude file read/decode; AE model timing includes residual construction.
These are means of run percentiles, not pooled service latency.
Final training took 22.66 minutes including loading/validation.
Exact native evaluation took 58.70 minutes.
Peak allocated training VRAM was 0.607 GiB;
the model has 31,499 parameters and largest weight checkpoint 131,133 bytes.

## Failure evidence

Deterministically selected panels contain 15 false-negative examples; in 1,
mean residual inside the displayed defect mask was no larger than outside it.
The table and panel identify each image. Low inside/outside contrast, when
present, documents weak local separation. Other false negatives can have
higher local error but still fail image top-k aggregation or the normal threshold.
Selected false negatives had median top-1%-error/mask overlap 0.0518
and median overlap enrichment over mask-area fraction 2.222.
These display-resolution diagnostics distinguish dominant non-defect residuals
from errors at the actual defect; tiny mask area also limits achievable overlap.
0 selected false-negative masks disappeared under nearest-neighbor 256-square display resizing;
all primary localization metrics still used their original native masks.
Across the selected panels, median reconstructed/original neighbor-detail ratio was 0.434.
The lower neighbor-detail measure is consistent with blur in these selected reconstructions; it does not establish that blur alone causes missed defects.
The displayed normals had median residual on high-gradient pixels 0.006548
versus 0.001377 off those pixels.
These deliberately high-scoring normals expose normal-structure residuals;
they are not a random sample of normal reconstruction quality.

Directly inspected seed-2026 panels show the following concrete cases:

- `carpet/test/color/000.png`: the dark color defect is visibly retained in the reconstruction. Its mean in-mask residual is 0.002432 versus 0.010088 outside; normal weave errors dominate this false negative.
- `capsule/test/crack/000.png`: the small crack has weak residuals relative to normal printing and silhouette errors. Only 0.003049 of top-1%-error pixels overlap the displayed mask. Normal `good/002.png` is a false positive with visible printing/edge residuals.
- `metal_nut/test/good/009.png`: normal rims and surface detail produce a false-positive image score of 0.070304. The bent false negative `bent/000.png` retains its coarse altered contour and scores only 0.031789.
- `screw/test/scratch_head/003.png`: the image alarm is correct, but the overlay also marks normal threads; only 0.001524 of top-error pixels overlap the tiny displayed defect mask. Image detection success does not establish accurate localization.

Seed-2026 defect-size analysis (native mask area; descriptive, after freeze):

| Defect area | Images | Recall |
|---|---:|---:|
| at most 0.1% | 17 | 0.2353 |
| 0.1% to 1% | 482 | 0.2573 |
| above 1% | 759 | 0.2411 |

The smallest-group recall minus above-1%-group recall was -0.0058. Size groups mix categories and defect types; the table is descriptive rather than a matched resolution ablation.
256-square resizing and top-k dilution can suppress tiny defects in principle;
no higher-resolution counterfactual was trained after test to measure that effect.
See defect_type_size_summary.csv for scratch-specific counts and outcomes.

## Limits and retention

No pretrained features, skip connections, smoothing or combined loss were used.
The bounded development comparison selected MSE/squared residuals over the
four fixed object/texture categories. Proxy scores were modest and do not
guarantee transfer to real defects. Twenty epochs limit compute and do not
prove convergence for every category. Three seeds do not remove sampling
uncertainty. Small normal calibration partitions can yield unstable thresholds.
Pixel confidence intervals use fixed-bin whole-image bootstrap; point metrics
are exact at native resolution and quantization bias is separately reported.
Mean binned-minus-exact pixel AUROC bias was -0.008928,
with maximum absolute run difference 0.047090.
This is material: those intervals describe the binned approximation and must
not be treated as coverage intervals for the exact pixel AUROC point estimate.
Compare recall, FPR, pixel AP/AU-PRO and latency together before claiming an
improvement. This research benchmark does not establish production suitability.

The complete local notebook keeps original/reconstruction/error/overlay/mask
panels. Public outputs contain compact measured tables and aggregate plots.
Registry records are append-only, including all rejected variants. The pilot
executed live; later notebook cells analyze separately recorded jobs and
explicitly do not claim live training replay. At the autoencoder milestone, no PatchCore work had started.

Read [Notebook 02](notebooks/02_convolutional_autoencoder.ipynb), [results](reports/autoencoder/all_category_means.csv), [findings](reports/autoencoder/findings.md), [experiment registry](experiments/), [protocol](docs/autoencoder_protocol.md) and [reproduction](docs/autoencoder_reproduction.md).
<!-- AE_FINDINGS_END -->

<!-- PATCHCORE_FINDINGS_START -->
## PatchCore variant measured baseline

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
remain local and ignored. At the PatchCore milestone, no SuperSimpleNet work had started.

Read [Notebook 03](notebooks/03_patchcore.ipynb), [results](reports/patchcore/all_category_means.csv), [findings](reports/patchcore/findings.md), [protocol](docs/patchcore_protocol.md) and [reproduction](docs/patchcore_reproduction.md).
<!-- PATCHCORE_FINDINGS_END -->

<!-- SUPERSIMPLENET_FINDINGS_START -->
## SuperSimpleNet Lite: completed benchmark

Notebook04 evaluates an explicitly reduced-budget ICPR-style SuperSimpleNet.
Frozen ResNet18 ImageNet V1 layer2/3 features feed a learned adapter,
segmentation discriminator and auxiliary image detector. Normal-only fitting
uses Perlin-masked feature-space Gaussian noise, not RGB defect corruption.
The bounded four-category study selected noise standard deviation0.05,
margin-plus-focal loss,40 epochs and top1% smoothed-map image scoring.
All15 categories completed seeds2026/2027/2028 before official evaluation.
No post-test retraining or threshold changes were made.

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

Mean model p50/p95: **9.689/17.339ms**; decoded-RGB-to-native-CPU-map p50/p95: **28.793/39.422ms** (file decode excluded).
Peak allocated training VRAM: **0.709GiB**. Final training-loop time: **85.66minutes**; native evaluation: **44.96minutes**.
Shared frozen extractor: **11,179,957bytes** once; largest category-head checkpoint: **2,230,175bytes**.

These are means of run percentiles rather than pooled serving percentiles.
Ranking, recall, false-positive behavior, localization and latency must be
considered together. Paired image intervals preserve matching categories,
seeds and image draws; they condition on fitted models and thresholds.
Small macro differences do not establish statistical superiority.

Image/pixel thresholds use reserved unmodified normal q95/q99.5 distributions.
Finite calibration sample sizes do not guarantee the nominal5% unseen-normal FPR.
The reports include category/seed threshold distributions, Wilson intervals,
order-statistic uncertainty, real failure panels, fine-defect results and
post-freeze normal brightness sensitivity. Pixel bootstrap intervals use a
disclosed fixed-bin approximation; primary native point metrics remain exact.

The publication uses a larger backbone, longer training and a learned primary
image head. Our reduced widths,40-epoch budget and selected map aggregation are
explicit differences. Strong synthetic-proxy performance does not guarantee
synthetic-to-real transfer. All rejected settings and the interrupted cable
attempt remain recorded. Final fitting has exact epoch/RNG recovery.
Public outputs are genuine and compact; complete original/mask/overlay panels,
weights and caches remain local and ignored. No Notebook05 work was started.

Read [Notebook04](notebooks/04_supersimplenet.ipynb), [all-category results](reports/supersimplenet/all_category_means.csv), [measured findings](reports/supersimplenet/findings.md), [frozen policy](configs/supersimplenet_frozen.json), [experiment registry](experiments/04-development-selection-v1.json) and [reproduction guide](docs/supersimplenet_reproduction.md).
Image-quality differences versus PatchCore remain unresolved by the paired conditional intervals; localization is weaker. Median inference is faster, but end-to-end p95 is higher and texture false positives persist. Read the [visually checked panel analysis](reports/supersimplenet/observed_panel_analysis.md) and [matched frozen cases](reports/supersimplenet/matched_frozen_cases.csv).
<!-- SUPERSIMPLENET_FINDINGS_END -->

## Milestone 7 — PaDiM

Notebook 05 completes the frozen ResNet18 spatial-Gaussian model on all 15 categories across seeds 2026/2027/2028. The real normal-image GPU pilot passed before the 16-run bounded development study. Configuration: 100 seeded channels, covariance +0.01 I, sigma-4 smoothed Mahalanobis maps.

Macro image AUROC **0.897933**, pixel AUROC **0.965920**, pixel AP **0.453764**, AU-PRO **0.911929**; normal-q95 operating recall **0.696493**, FPR **0.144521**. Threshold transfer remains a material limitation; inspect category results and confusion counts alongside ranking.

See [findings](reports/padim/findings.md), [full summary](reports/padim/summary.json), [all runs](reports/padim/all_runs.csv), [four-model comparison](reports/padim/model_macro_comparison.csv), [protocol](docs/padim_protocol.md) and [reproduction](docs/padim_reproduction.md). Original masks are evaluated at native resolution; only pixel bootstrap intervals use a disclosed histogram approximation. Checkpoints/maps/panels remain ignored locally. EfficientAD remains planned.
