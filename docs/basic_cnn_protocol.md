# Basic CNN preregistration — 2026-11-12

This extends the foundation protocol before the first training run. No pretrained
weights, real defects, official masks, calibration files or test files enter any
training/development experiment. This is a synthetic-supervised **segmentation**
baseline, not a supervised classifier of real defect types or another planned
architecture. Exact limits and configurations are in configs/basic_cnn_plan.json.

## Data and selection gate

User requirement: synthetic anomalies occur **only on fitting-partition images**.
For controlled development, sort fitting paths per category and deterministically
shuffle with seed 2026; reserve floor(20%) as an internal synthetic probe source.
The other 80% receives gradient updates. Donors remain within the same inner
partition/category. Official development images remain real normals, unmodified.
The probe is explicitly synthetic and is never a benchmark annotation. Outer
manifests and their SHA256 identities remain unchanged.

Pilot bottle first. Categories bottle, metal_nut (objects), carpet and wood
(textures) are selected before any new test evaluation. Five predefined variants
cover capacity, mixed/patch-only/perturbation-only defects and weighted BCE versus
BCE+Dice; no learning-rate grid. Maximum eight epochs, minimum three, patience
three on mean(probe segmentation objective, official-normal development BCE).
Proxy masks are generated once with a separate seed, independent of candidate
training settings. Score selection is macro proxy recall at the official-normal
development 95th-percentile score threshold; tie-break by proxy pixel AP and
parameter count. Proxy outcomes do not estimate real-defect performance.

Training objectives use balanced normal/synthetic image sampling, positive pixel
weight 8 and AdamW (lr .001, weight decay .0001). Float32 is primary; no AMP unless
a separately logged comparison establishes stable gradients/output agreement.
Eight-epoch pilot curves, elapsed time, peak memory and learning behavior must
be reported before all-category training. If it cannot learn, investigate first;
at most two logged pilot repairs, never unbounded tuning. Limit training to 60
wall-clock minutes and peak allocated GPU memory to 6 GiB. OOM produces a failure
record and retained configuration; any batch/resolution change gets a new ID.

## Architecture, synthetic labels and image scores

Three small Conv/GroupNorm/ReLU encoder blocks downsample by 8. A shallow
segmentation head combines shallow and repeated coarse features and emits one
logit per image pixel. Nearest-repeat upsampling supports deterministic gradient
flow and uses no pretrained backbone. GroupNorm avoids running batch statistics
that might encode the synthetic mixture. Input: RGB bilinear-antialiased whole
image resize to 256x256, float [0,1], fixed centering to [-1,1] inside the network.

Two fitting-only generators blend a shifted donor-normal patch or procedural
localized color/noise into soft irregular elliptical supports. Random location,
scale, angle and blending avoid a fixed global/border cue. Pixels outside support
remain byte-for-byte unchanged in float tensors. Masks are derived from actual
nonzero channel differences, not merely the nominal support. They describe
synthetic edits, never real defects. This still has a synthetic-to-real domain
gap; it is not claimed to generate realistic benchmark labels.

Anomaly probability is sigmoid(logit); the primary image score is its maximum.
No test-based normalization, smoothing or aggregation selection is permitted.

## Frozen confirmation

After bounded selection, freeze the full policy, source/configuration hashes,
software lock, selected variant, epoch count and seeds **before test inference**.
Final models are freshly initialized separately for each category and seed
2026/2027/2028, using all original fitting normals. The fixed final epoch count is
the rounded median best epoch over four development categories, clipped to 3–8;
there is no category-specific post-test early stopping or adjustment. Internal
probe normals may now rejoin fitting because final selection is already frozen.
Official development/calibration normals never receive gradients.

Primary image threshold: normal-calibration higher 95th percentile, alarm iff
score > threshold. A higher 99th-percentile policy is a predetermined sensitivity
report, never a selection opportunity. Separate pixel threshold: conservative
99.5th percentile of pooled normal-calibration model-resolution pixel scores,
using 65,536 fixed bins; report sample counts and quantization. Restoring maps to
native size may change pixel FPR. Calibration uncertainty includes normal-score
bootstrap intervals, small sample limitations and no population-FPR guarantee.

Each category/seed test evaluation is executed once and kept immutable. Native
continuous maps use bilinear interpolation against untouched original masks.
Image metrics use existing utilities; exact native pixel AUROC/AP/AU-PRO require
a memory-bounded evaluator cross-checked against the existing small reference.
Report operating-point confusion counts, per-defect results, all 15 categories
and macro means, texture/object summaries, three-seed variation and image
bootstrap uncertainty. Any approximate bootstrap pixel implementation must state
its binning separately from exact primary metrics. No tuning follows test access.

Keep append-only start/result/failure registry events, every curve, exact configs,
checkpoint hashes and output identities. Retain best/last meaningful checkpoints
locally; publish compact numeric/aggregate results and full notebook narrative.
Licensed image/map panels remain local, with their artifacts documented. Future
models and engineering are outside this milestone.

Byte identities refer to checked-out artifacts: .gitattributes fixes source and
CSV line endings to LF, and generated JSON/frozen Windows records to CRLF. This
retains their original measured SHA256 values across fresh Git checkouts; Git's
internal normalized JSON blobs may use different line endings.
