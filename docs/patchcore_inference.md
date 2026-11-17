# PatchCore inference foundation

This package adapts the frozen **simplified ResNet18 PatchCore** research model.
It does not fit a coreset, download weights, recalibrate thresholds, run an HTTP
service or start the dashboard. Existing benchmark files remain unchanged.

## Architecture and category policy

`anomaly_detection.inference.PatchCorePredictor` owns one frozen extractor and
one category-specific memory bank. Construct it once and reuse it. All 15 original
MVTec AD categories are supported: bottle, cable, capsule, carpet, grid, hazelnut,
leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood and zipper.

Deployment v1 selects **seed 2026 for every category**, the lowest preregistered
seed. This rule was established without ranking seeds by test results. All 45
original banks were found and checksum-verified; 15 are selected. Do not replace
a category with its best test seed. Manifest v1 is an exclusive registration of
existing artifacts, not a new experiment or a replacement research freeze.

The extractor is the original frozen ResNet18 ImageNet V1 layer1–3 trunk. Scoring
uses pooled layer2/layer3 features (384 channels, 32 × 32 patches), exhaustive
float32 L2 search against 2,048 stored vectors, and the mean of the largest
`ceil(1024 * .01) = 11` patch distances. There is no paper neighborhood reweighting,
feature L2 normalization, Gaussian map smoothing or probability conversion.
Fitting caches are not inference dependencies.

## Exact geometry, precision and decisions

Inputs are converted to RGB without EXIF orientation changes, resized by PIL
bilinear to the whole 256-square image, converted from uint8 to float32 /255,
then processed by the original torchvision V1 normalization/identity 256 geometry.
The model map is bilinearly resized from 32 to 256, then on CPU from 256 to native
size, both with `align_corners=False`. A direct 32-to-native resize is different
and is intentionally not used. Scores are reduced from the patch map, not the
native map or its visualization. Research functions are reused unchanged.

Image and pixel cutoffs come from the original reserved-normal calibration JSONs.
Both decisions use strict `>`; ties are normal. The image score is a raw L2
distance, **not a probability**. Do not clip it into [0,1]. Native pixel decisions
use the saved pixel cutoff even though it was calibrated on 256-square maps;
this preserves the research convention and its interpolation limitations.

Inference uses full FP32, frozen BatchNorm and no gradients. The package scopes
cuDNN deterministic/TF32-off flags and restores them after each forward, serializing
these contexts across package instances. It rejects CUDA matmul TF32 when enabled;
set `torch.backends.cuda.matmul.allow_tf32 = False` before inference. Other libraries
must not mutate global backend/thread flags concurrently. The adapter does not
silently choose a device or tune CPU thread counts. A worker should own and reuse
its predictor; this foundation does not establish service-level concurrency limits.

The original cached features were produced in batches of 16 with cuDNN TF32
permitted. Full-FP32 batch-one inference consequently has small numerical
differences. This is disclosed precision behavior, not a new scoring policy.
Validation uses historical-reference `atol=.005, rtol=.01` and much tighter
CPU/CUDA `atol=rtol=.0002`. The initial overly strict historical-reference test
failed and is retained in the engineering recovery history. Every selected
sample's **image decision** must agree exactly; pixel decisions near a cutoff
are not promised bitwise equal across numerical backends.

## Artifact requirements and trust

`configs/patchcore_deployment_v1.json` records the model family, fixed seed,
configuration, preprocessing identity, software versions, shared extractor,
15 bank identities, calibration identities and original source records. All
paths are relative to an explicit `artifact_root`; path traversal and symlinks
escaping that root are rejected. SHA256 and byte sizes are checked before
deserialization. `torch.load(..., weights_only=True)` is used. Category, seed,
configuration, final-fit provenance, bank shape/dtype/finiteness and threshold
agreement with the calibration JSON are also checked.

Treat the manifest from the trusted Git release as the trust root; checksums do
not authenticate an attacker-controlled manifest. Pass its expected SHA256 to
pin it independently (see the verified value in the engineering validation JSON).
Loading verifies artifact identities once; do not mutate files or predictor
weights after initialization. The package does not rehash files per image.

Shared extractor: `checkpoints/patchcore/shared-resnet18-features.pt`, 11,179,957
bytes, SHA256 `2192b8a31687274b2abcb7cc59cd35e1c3267ce09e9acbdb70f55291854b9949`.
Original torchvision ImageNet V1 checkpoint SHA256:
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Each bank checksum is individually listed in the manifest and the availability
table. The shared trunk is about 10.66 MiB and a category bank about 3.05 MiB;
the original full classification checkpoint is not required at runtime.

Weights, banks, raw MVTec images, maps and caches stay outside Git. To reproduce
on another machine, restore the legitimately retained artifacts at the manifest's
relative paths and verify their exact SHA256 values. Restore the calibration
JSONs and frozen config from this repository. No download is automatic. Do not
substitute a newly initialized ResNet or a regenerated bank. If the original
bank is unavailable, inference is blocked until the legitimate artifact is
restored; this milestone does not authorize retraining it. Code licensing alone
does not establish commercial rights to ImageNet weights or MVTec data.

## Environment and Python API

Use the existing Python 3.12 `.venv` and locked requirements. The validated
environment is PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.2.6,
Pillow 11.2.1 on Windows 11 with an RTX 4060 Laptop GPU. CPU operation uses the
same installation; CUDA is optional for CPU inference. A CPU-only wheel build
has not separately been benchmarked. Do not recreate the working environment.
The package is included by the existing setuptools `src` package discovery.

Dependencies are recorded in `requirements-research-cuda.txt` and
`requirements.lock.txt`; environment setup is documented in `docs/environment.md`.
The existing editable installation already exposes the new module. If restoring
a checkout into an already prepared compatible environment, register the local
package without downloading dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m anomaly_detection.inference --help
```

Runtime artifact layout is determined by the manifest: the frozen policy JSON,
shared `checkpoints/patchcore/shared-resnet18-features.pt`, selected
`checkpoints/patchcore/03-final-b2048-<category>-s2026-v1/bank.pt`, and original
`reports/patchcore/evaluation/03-test-b2048-<category>-s2026-v1-calibration.json`.
Result/prediction/training records are additionally required for research-parity
validation, not ordinary prediction. See `docs/patchcore_reproduction.md` for
the original legitimate weight provenance. Restoring a release requires the
byte-identical saved bank/trunk files; regenerating serialized weights does not
guarantee the registered checksum.

```python
from pathlib import Path
import torch
from anomaly_detection.inference import PatchCorePredictor

root = Path.cwd()  # repository / restored artifact root
torch.backends.cuda.matmul.allow_tf32 = False
# Optional explicit CPU policy; measured timings below use four threads.
torch.set_num_threads(4)
predictor = PatchCorePredictor(
    root / 'configs/patchcore_deployment_v1.json', root,
    category='bottle', device='cuda',  # or 'cpu'; no implicit fallback
    expected_manifest_sha256='58be2eafb16ac012ac18e0feb69f11b2f510e7417464eaba5bd9797b423272fa',
)
prediction = predictor.predict(root / 'your_input.png', visualize=True)
print(prediction.to_dict())
raw_native_map = prediction.anomaly_map
native_pixel_alarms = prediction.pixel_decisions
heatmap_rgb = prediction.heatmap
overlay_rgb = prediction.overlay
```

Accepted input forms: image path, encoded bytes, a PIL image, or H × W × 3 uint8
RGB NumPy array. Grayscale/RGBA PIL or encoded inputs are explicitly converted
to RGB; arrays must already be RGB uint8. Limits are 32 MiB encoded and 32 million
decoded pixels; animated/multipage and corrupt images are rejected. Errors use
`InputError`, `ArtifactError` or clear device/category `ValueError`s. Inference
requires a caller-supplied category; automatic category recognition is not part
of this model.

## Prediction schema and visualization

The frozen `Prediction` record returns category, seed, device, manifest SHA256,
raw score, cutoff, `is_anomaly`, pixel cutoff, native float32 map, 256-square map,
32-square patch distances and native boolean pixel decisions. With visualization
enabled it also returns uint8 RGB heatmap/overlay and display scaling metadata.
`to_dict()` produces JSON-compatible scalar metadata and shapes; native arrays
are included only with `include_maps=True`. Keeping maps as arrays avoids large
JSON payloads. Later API transport/schema design remains a separate milestone.

Heatmaps use per-image min/max scaling and a fixed RGB ramp. Overlay alpha is
.45 by default. Raw maps and image scores are never normalized or changed.
Constant maps have a finite constant display. Per-image colors are unsuitable
for comparing anomaly severity across images; use raw values/cutoffs instead.

Local command, without a server:

```powershell
.\.venv\Scripts\python.exe -m anomaly_detection.inference --manifest configs/patchcore_deployment_v1.json --artifact-root . --category bottle --input your_input.png --device cpu --visualize --output-dir outputs/local-inspection/new-case
```

The output directory must be new. The command writes metadata JSON, the raw
native `.npy` map and optional heatmap/overlay PNGs. Keep licensed outputs local.
It never overwrites an existing output directory or the source image.

## Measured validation and runtime

See `reports/engineering/patchcore/integration.json`, `parity.csv` and
`validation.json`. Lexicographically first normal and first defect were selected
for each category, independent of score: 30 real images, 60 CPU/CUDA predictions
and 60 repeats. All reference image decisions match, every repeat is identical,
and all 15 loaders pass. Maximum score error against saved research predictions:
0.000194788. Maximum 256-map error: 0.00220728. Maximum CPU/CUDA native-map error:
0.00000464917. These results cover representative samples, not every test image
or a guarantee that no near-threshold decision can differ on another machine.

Batch one, four CPU threads, 20 warmups and 100 measurements per timing series:

| Category/device | Model wall p50/p95 ms | Decoded E2E p50/p95 ms | File E2E p50/p95 ms |
| --- | ---: | ---: | ---: |
| bottle / CPU | 81.89 / 90.56 | 95.80 / 106.23 | 130.94 / 146.87 |
| bottle / CUDA | 20.63 / 23.55 | 34.67 / 40.41 | 68.36 / 76.33 |
| carpet / CPU | 86.17 / 96.72 | 102.14 / 113.34 | 162.60 / 180.69 |
| carpet / CUDA | 22.15 / 25.28 | 41.27 / 44.57 | 102.13 / 108.43 |

GPU CUDA-event model p50/p95: bottle 20.39/23.26 ms; carpet 21.88/25.06 ms.
Model wall includes feature extraction and NN search on resident input, with
explicit synchronization. CUDA events isolate device computation. Decoded E2E
also includes preprocessing, transfer, CPU-native map interpolation and score;
file E2E additionally opens/decodes a warm-cache PNG. Visualization, constructor
checksum/loading time and cold-disk behavior are excluded. Record construction
latency separately if integrating a short-lived process. CPU and CUDA runs are
sequential and are not claims about throughput or request concurrency.

Peak observed process RSS was 1,366,896,640 bytes (~1.27 GiB) in the mixed-device
validation process; this includes retained references and the CUDA context, not
a minimum serving requirement. GPU peak allocated/reserved: 69,377,024 /
90,177,536 bytes (~66.16/86 MiB). These are PyTorch allocator counters, not total
driver memory. OS/CPU identifiers and exact measurements are retained in JSON.
Windows power/scheduling variability remains; no claim of identical historical
research timing or hardware-independent speed is made.

## Verification and recovery

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp
.\.venv\Scripts\python.exe scripts/validate_patchcore_engineering.py
```

The real-image validator refuses to overwrite `integration.json`; reuse completed
measurements. `scripts/validate_patchcore_inference.py` is the bounded original
measurement command, not a routine application startup operation. The exclusive
manifest builder likewise refuses to recreate registered deployment v1.
`research_protection.json` binds all prior tracked research files except the
authorized rolling README/progress/checkout metadata. Unit/integration tests
cover all category artifacts, preprocessing, distances, score direction, strict
thresholds, native maps, visualization, invalid inputs, checksum/path traversal,
manifest mismatch, reproducibility and CPU/CUDA agreement.

## Limits and next milestone

This is a research inference foundation, not a production alarm system. Nominal
5% normal-only calibration did not transfer reliably in the completed benchmark;
leather/carpet false alarms and screw misses remain. Thresholds have not been
changed to hide those limitations. Selection of PatchCore is conditional on the
MVTec comparison, not new external validation. No domain-drift monitoring, service
resource limits, authentication, deployment packaging, concurrent serving SLA,
GPU fallback policy or automatic threshold adaptation is claimed. FastAPI and
the interactive dashboard are deliberately deferred to later milestones.
