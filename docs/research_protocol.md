# Research protocol — foundation v1

Freeze this protocol and partition manifest before Notebook 01. Amendments receive
a dated Git commit and rationale; prior experiments retain their original rules.
No model has been trained or evaluated in this repository at initialization.

## Question and coverage

How do a Basic CNN, convolutional autoencoder, PatchCore, SuperSimpleNet, PaDiM
and EfficientAD trade detection/localization quality against normal false alarms,
defect recall, latency and GPU memory under a shared normal-only protocol?
Run category-specific fitting and report all 15 categories: bottle, cable,
capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush,
transistor, wood and zipper. Do not omit difficult or failed categories.

## Provenance and integrity

Use the original [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
not AD 2 or LOCO AD. Cite Bergmann et al., CVPR 2019, and the extended IJCV
dataset paper linked on that page. Data is CC BY-NC-SA 4.0; retain attribution
and license with local copies. Dataset licensing is separate from source code;
do not redistribute images, masks, archives or derived image panels in Git.
Record retrieval UTC time, official source URL, archive name/size and SHA256
locally under data/. If a publisher checksum exists compare it; a self-computed
hash identifies a download but does not independently authenticate it.

Extract to data/mvtec_ad/<category>/{train/good,test/<type>,ground_truth/<type>}.
Decode every image/mask, verify dimensions and binary masks, detect missing and
orphan masks, and create a sorted relative-path SHA256 inventory. Retain its hash
and counts in compact reports; keep the full manifest under data/ or outputs/.
Missing categories, corruption or duplicates block training until investigated.
Do not silently repair or remove official data. Hash equality detects exact
copies only; inspect near duplicates or acquisition groups before fitting.

## Normal partitions and test isolation

Partition official train/good independently per category using seed 2026, sorted
relative paths and a seeded shuffle. Allocate floor(0.15*N) each to development
and calibration, all remaining images to fitting (approximately 70/15/15).
Persist assignments and hashes; never reshuffle to improve results. Fit includes
all learned normalization, feature statistics, memory banks and augmentations.
Development supports reconstruction/normal validation and clearly labeled
synthetic proxy tasks. Calibration is untouched until model/config selection
is frozen; it contains real normals only. Do not refit on held-out normals in
the primary comparison. Separate any all-train literature reproduction.

Only normal fit images may train models. Synthetic anomalies (e.g. CutPaste,
noise or feature perturbations) may be generated from fit normals with seeds
and algorithm settings recorded. Development synthetic proxies use only
development normals; synthetic labels are never real benchmark labels. A Basic
CNN cannot learn real defect classes from normal-only examples: Notebook 01 must
declare a one-class representation/distance objective or a synthetic proxy
classification objective before fitting. External anomaly images are excluded.

Official test images, defect labels and masks are for fixed descriptive EDA and
final evaluation only, never gradients, normalization, early stopping, choosing
resolution/augmentation/architecture/hyperparameters or optimizing thresholds.
EDA visualizes test annotations, so claim only **no test-label tuning**, not
unseen/untouched test labels. Record any earlier exposure or tuning; if tuning
occurs, mark results exploratory and disclose contamination. Do not relabel
normals with invented defects. Supervised mask-based studies need a separate
protocol and are outside this project stage.

## Preprocessing and scores

Primary proposed input is RGB float [0,1], whole-image bilinear resizing to
256x256 with antialiasing. This is a preregistered starting point, not an EDA
performance finding. Record geometric distortion; no test-driven changes.
Pretrained backbones may require documented ImageNet normalization and specific
resolution; record this exception, pretrained weights/source/checksum and a
secondary matched-resolution comparison where feasible. Never download weights
without a declared experiment need. Masks use nearest-neighbor transforms only.
For evaluation restore continuous maps to native image dimensions by bilinear
interpolation, then compare against original binary masks. Never resize binary
masks with bilinear interpolation. Record borders, smoothing and interpolation.

Higher scores always mean more anomalous. Reconstruction models use maximum
per-pixel channel-mean squared error as the primary image score; feature-map
methods use maximum map value. A CNN with no localization uses its declared
one-class distance or synthetic-logit score and reports pixel metrics N/A.
Published model-specific aggregation may be a registered exception, documented
before test access. Never rescale scores with test-set minimum/maximum or labels.

## Metrics and calibration

Report image AUROC and average precision (AP, stepwise precision-recall integral,
not trapezoidal PR AUC). At a frozen category threshold report recall TP/(TP+FN),
FPR FP/(FP+TN), precision TP/(TP+FP), F1 and [[TN,FP],[FN,TP]]. Class 1 is defect.
Choose threshold as the `higher` empirical 95th percentile of held-out normal
calibration scores; alarm iff score > threshold. Fix target image FPR=5% before
evaluation. This controls empirical calibration behavior, not guaranteed
population FPR. Ties and small samples can make thresholds conservative. Do not
maximize test F1. If synthetic development anomalies inform selection, report
their proxy status; calibration remains real-normal only.

For localization report pooled-within-category pixel AUROC, pixel AP and AU-PRO
normalized over background FPR [0,0.30]. Use 8-connected ground-truth components,
equal weighting per defect region, background pixels from all test images,
linear interpolation at 0.30 and trapezoidal integration /0.30. Normal images
have no annotation files: all-zero evaluation masks are semantic normal ground
truth, not fabricated anomaly annotations. Components come only from real masks.
Full-resolution evaluation must be memory-bounded and checked against the exact
small-fixture reference. Do not average per-image AUROC with single-class masks.
Pixel threshold metrics, if included, require a separately preregistered normal
calibration rule; the image threshold does not apply automatically to pixels.
Undefined metrics are null with reasons, never silently zero or omitted.

Report each category, then unweighted macro means over all 15, documenting
missing/failed values and coverage (no success-only averages). Report object
versus texture summaries secondarily. Pooled-image results are supplementary.
Use at least seeds 2026, 2027, 2028 for stochastic final models when feasible;
record deterministic flags and nondeterministic operations. Bootstrap test
images within category with fixed seed (2,000 resamples), resampling whole masks
with their images for pixel metrics. Report 95% percentile intervals and invalid
single-class replicates; seed variation and sampling uncertainty are distinct.
Bootstrap macro results by independent within-category image resampling. Small
test/calibration sets limit recall/FPR precision; no claim of independence among
pixels or production-level guarantees. Training seed spread is not a confidence
interval over the benchmark population.

## Selection, runtime and confirmation

Select within model family on normal development objective and registered
synthetic proxy recall at fixed normal-development FPR, balancing proxy quality,
speed and memory. Do not rank configurations using official test scores. Target
batch-1 feasibility within 8 GB VRAM; start small and record OOMs. Across frozen
families the final test comparison describes quality/recall/FPR/runtime/memory
Pareto tradeoffs; selecting a deployment candidate from it uses test information
and needs independent confirmation data before generalization claims.

Measure fit time and batch-1 inference latency separately: preprocessing,
transfer, model, postprocessing and end-to-end. Record CPU/GPU, power mode,
precision, batch/input size, 50 warmup and 200 timed iterations, synchronized
CUDA timing, median/p95 and peak allocated/reserved VRAM after reset. CPU and
GPU results are separate. No latency is claimed before measurement.

Development runs may vary declared settings and are preserved. Final confirmation
freezes code SHA, environment, weights, partitions, score/map rules and thresholds
before evaluating the official test set once per registered final run. Post-test
changes are exploratory and require disclosure; repeated peeking is not a new
untouched confirmation.

## Retention and milestones

Use unique experiment IDs and append-only JSON start/result/failure events in
experiments/. Record family, all categories, hypothesis, exact preprocessing and
hyperparameters, seeds, UTC times, config hash, code SHA/dirty state, package lock,
hardware, dataset/partition hashes, fit details, metrics, calibration, checkpoint
path/hash, interpretation and decision. Keep large files ignored and register
their retention location. Notebooks show executed functions/configs and model
construction/evaluation steps; no opaque single-call experiment notebooks.
Preserve failures, tracebacks and unsuccessful settings. Corrections append new
events referencing old IDs; never silently delete or overwrite evidence.

After each meaningful completed milestone run relevant tests, validate notebook
structure/execution, inspect staged content for secrets/data/large files, check
links and origin, commit with a factual message and push origin/main. A blocked
dataset milestone may publish verified setup only with explicit incomplete EDA
status. Do not publish benchmark claims without actual registered experiments.
