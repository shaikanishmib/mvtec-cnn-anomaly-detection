# SuperSimpleNet research reproduction

Use this standalone repository and its existing locked Python3.12 CUDA environment. Never reference the separate personal project. Original MVTec AD data, ImageNet weights, caches and learned checkpoints stay local and ignored. The unchanged seed-2026 manifests define fitting, development, reserved normal calibration and official test partitions for every model and training seed.

The reference research sources are the [ICPR paper](https://arxiv.org/html/2408.03143v2) and [official implementation pinned at98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d](https://github.com/blaz-r/SuperSimpleNet/tree/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d). MIT attribution is retained in `docs/third_party/SuperSimpleNet_LICENSE.txt`. This project targets the original adapted-classifier path rather than the later JIMS default. The selected model is explicitly a reduced-budget Lite variant, not paper equivalence.

Read the chronological notebook and `reports/supersimplenet/protocol.md` first. `pilot_report.json`, `reference_feasibility.json`, `cuda_verification.json`, the common-proxy tables and `development_findings.md` explain the measured gate. The single new pretrained acquisition was official WideResNet50-2 ImageNet V1 for reference feasibility; the existing V2 weights were not substituted. Lite uses the existing official ResNet18 V1 weights. The extractor rejects missing or mismatched weight hashes and does not automatically download models.

The final policy is `configs/supersimplenet_frozen.json`, checked against `freeze_identity.json`. Its source and weight hashes must match before fitting or evaluation. No official test image/mask may influence a new experiment's configuration or calibration. Re-running these scripts is not a new blind confirmation after reading existing results.

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/supersimplenet/pytest-new-reproduction
.venv/Scripts/python.exe scripts/benchmark_supersimplenet.py fit
.venv/Scripts/python.exe scripts/benchmark_supersimplenet.py evaluate
```

Fitting skips completed immutable results. An interrupted final fit restores its model, AdamW optimizer, scheduler, shuffle/mask/noise generators and global Torch RNG state from an atomic epoch checkpoint; no completed epoch needs repeating. It retains pre-resume histories and interruption reservations. A final checkpoint that exists without a reconciled result deliberately stops for technical reconciliation rather than overwriting evidence. Development before this recovery wrapper lost one externally interrupted two-epoch cable attempt; its history is preserved and only that unavailable state restarted as v2.

The evaluator requires **all45 fits** to be present before any test evaluation. Each category/seed reserves one immutable evaluation. Calibration uses only original unmodified reserved normals: higher q95 image and pooled higher q99.5 pixel thresholds with strict greater-than decisions. The selected primary image score is the top1% mean of the smoothed256 dense map. The learned image head is still trained and computed, but is auxiliary to this selected score. No test-derived min/max normalization is used. A technical evaluation failure must be preserved and explicitly reconciled; the evaluator does not silently overwrite a partial trial.

Cached features are per original image, extractor source and weight hash. They align32/16 layer grids to64 **before** concatenation/pooling. Existing PatchCore caches are incompatible and are not substituted. Compatible SSN embeddings are reused across splits, trials and seeds; completed dataset cache builds are reused. Separately labeled correctness replays and the required50-warmup/200-iteration latency benchmark intentionally repeat forward passes and are not counted as new dataset extraction. Caches are not an inference dependency.

After frozen evaluation, the analysis commands read retained predictions rather than retrain:

```powershell
.venv/Scripts/python.exe scripts/summarize_supersimplenet.py
.venv/Scripts/python.exe scripts/analyze_supersimplenet.py
.venv/Scripts/python.exe scripts/diagnose_supersimplenet_brightness.py
.venv/Scripts/python.exe scripts/write_supersimplenet_findings.py
.venv/Scripts/python.exe scripts/publish_supersimplenet_notebook.py
.venv/Scripts/python.exe scripts/validate_supersimplenet.py
```

Some analysis/acquisition/registry-writing scripts intentionally use exclusive creation; do not rerun them over completed evidence. Name and register a new recovery when required. The adapter diagnostic is CPU-only and was recorded separately after freeze; it reuses held-out fitting-normal embeddings and does not alter selection. Post-freeze brightness diagnostics use development normals, never augmented calibration distributions or a new threshold policy.

Primary image/native pixel metrics and AU-PRO reuse the established shared implementations. Image bootstrap draws are paired by category and seed across models. Pixel uncertainty uses4096 fixed sigmoid bins and reports quantization bias against exact native point metrics. Bootstrap intervals condition on the fitted models and thresholds; calibration tail sampling and seed variation are reported separately. The confidence analysis does not justify a universal production false-positive guarantee.

Notebook pilot cells contain actual live execution. Development/final continuation cells execute analysis of separately recorded jobs and expose their actual implementation. Session-specific execution counts are preserved without renumbering. The complete local notebook is `outputs/supersimplenet/04_supersimplenet.full.executed.ipynb`; public Notebook04 removes only explicitly marked large original-image panel outputs. Earlier notebooks and model records remain unchanged. Do not start Notebook05 automatically.
