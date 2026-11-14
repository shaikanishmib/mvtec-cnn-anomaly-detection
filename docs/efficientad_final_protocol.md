# Frozen EfficientAD final execution protocol

Continuation of 664e106dbdd25104ea4e3c3be192fc8be7a0b772. No model-selection
change. Configuration SHA256 d728cf544ebc6f0c3d012adbca044a7aa5ddf4c0ced2e3210bef19b77a71d321.
All 15 categories, seeds 2026/2027/2028, fresh models, nine full shuffled epochs.
Training follows the existing development recipe exactly: full fitting-only
teacher moments, dual-head student and auxiliary AE, hard/AE/STAE losses,
tensor photometric augmentation, constant Adam, deterministic FP32, no penalty.
All development/provenance limitations remain. No pilot/development run is final.

The selected anomaly map and image score are the normalized local teacher/student
map and its maximum at 256 resolution. Full local/global raw maps are retained
for analysis; global/combined scores cannot replace the selected final score.
Map q90/q99.5 use only development normals. Scoring uses the existing CPU float32
map normalization helper, then promotes exact scores to float64 before 17-digit
CSV serialization, with round-trip reading. Image threshold: reserved clean
calibration normals, higher 95th percentile, strict greater. Tied calibration
scores invalidate continuous-score Beta uncertainty; no tuning from calibration.

Primary image metrics use shared binary_metrics. Primary pixel AUROC/AP/AU-PRO
use shared exact_pixel_metrics on native-resolution signed maps, bilinear
align_corners=False, 8-connected equal-weight regions, normalized PRO area at
FPR<=.30. No sigmoid, clipping, smoothing or border crop is introduced. Test
scores and masks are used only after training/map scaling/calibration complete.

Conditional uncertainty: existing 2,000 whole-image bootstrap draws, same seed
2026+category index and sorted image order as previous models, fixed thresholds.
Image bootstrap is exact; pixel bootstrap uses 4,096 monotone arctangent bins
centered/scaled by normal calibration maps (median and IQR). Signed scores are
not clipped; histogram binning is used only for uncertainty. Quantization bias
versus exact primary metrics is reported. Optional pixel operating diagnostics
use higher .995 quantile of calibration map pixels, fixed before test, and do
not change any of the nine primary metrics or image-level thresholds.

Latency per model: 50 warmups and 200 measurements on a decoded calibration RGB
image. Separate CPU preprocessing, H2D, full eager GPU raw-map computation,
D2H/CPU normalization/native-map postprocessing, and end-to-end without file
decode. A separate 50-measurement file-decode-through-CPU-map series is retained.
Model timing is full two-map computation, conservative for local-only deployment.
No CUDA graph, TF32 or mixed-precision change is adopted.

Durability: immutable attempt IDs; atomic optimizer/model/RNG snapshot after each
epoch; checkpoint and calibrated training-stage metadata before official test.
Completion requires artifact hashing, exact score round-trip, checkpoint replay,
metric reconciliation and registry result. A completed result is never overwritten.
Completed attempts are verified before skipping. Corrupt completed artifacts are
preserved and a new attempt is recorded; valid trained state can be reused for
evaluation recovery. Checkpoints, raw maps, bootstrap vectors and image panels
remain ignored. Compact metrics, histories, image scores and identities go in Git.

One manifest-driven manager holds an exclusive live-PID lock and launches one
project-.venv worker at a time. Restart command:
`.\.venv\Scripts\python.exe scripts/run_efficientad_final.py --all`.
Inspect docs/milestone_status.md, reports/efficientad/final/benchmark_manifest.json
and registry first. Resource gate checks each completed run against the published
category-specific conservative estimate, 12 h target and >=20 GiB free disk.
Material overrun (after >=3 runs, ratio >1.25; or any run >2x conservative estimate)
stops before another worker. Model policy and coverage never change automatically.

Readiness: exact project interpreter, torch 2.11.0+cu128, CUDA RTX 4060 Laptop,
all frozen identities and 15 categories confirmed; ~502 GiB free, no competing
runner, zero existing final artifacts. Ten focused core/research tests passed.
Final-evaluator fixture tests must pass before the first official run.
