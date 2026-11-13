# Reproduce Notebook 03

Use this repository's existing Python 3.12 CUDA environment and complete validated
MVTec AD data. All partition manifests and their original hashes remain frozen.
Do not use another repository or a different split. `psutil==7.2.2` was already in
the locked environment and is now declared directly for RSS measurement; no
packages or pretrained weights were downloaded for this milestone.

The original live pilot is
`outputs/patchcore/03_patchcore.pilot.executed.ipynb`. The complete local notebook is
`outputs/patchcore/03_patchcore.complete.executed.ipynb`. Public Notebook 03 strips
only dataset-pixel panels, retaining actual compact outputs. The continuation
explicitly analyzes separately completed fits/evaluations rather than claiming
they ran live in its kernel. Its cells expose actual extraction, coreset, NN,
calibration and evaluation code, all development hypotheses and measured decisions.

The ResNet18 ImageNet V1 cached file has full SHA256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
The extractor refuses to download missing weights silently. Use the recorded
official torchvision URL and record a new acquisition if weights are absent.
Frozen BatchNorm and parameters are essential. Whole-image PIL 256-square
geometry plus official V1 transforms with resize/crop 256 preserve borders;
this is an explicit override of default classification center cropping.

The original command sequence was:

```powershell
.\.venv\Scripts\python.exe scripts/build_patchcore_notebook.py
.\.venv\Scripts\python.exe scripts/run_patchcore.py development
.\.venv\Scripts\python.exe scripts/select_patchcore.py
.\.venv\Scripts\python.exe scripts/run_patchcore.py final-fit
.\.venv\Scripts\python.exe scripts/run_patchcore.py final-evaluate
.\.venv\Scripts\python.exe scripts/export_patchcore_backbone.py
.\.venv\Scripts\python.exe scripts/summarize_patchcore.py
.\.venv\Scripts\python.exe scripts/analyze_patchcore.py
.\.venv\Scripts\python.exe scripts/publish_patchcore_notebook.py
.\.venv\Scripts\python.exe scripts/write_patchcore_findings.py
.\.venv\Scripts\python.exe scripts/validate_patchcore.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Starts, results and decisions are append-only. Existing IDs cannot be reused;
assign new linked IDs for a new reproduction or failed/interrupted trial.
Selection/freeze files open exclusively. Do not rerun a completed scientific
experiment just to regenerate notebook outputs. Final drivers skip completed
jobs; an incomplete reserved job requires investigation and a linked new event.

Cached compatible embeddings are verified by hashes before reuse. They live in
`outputs/patchcore/features/`, keyed by exact image paths/hashes, transform,
extractor source and weights. Synthetic proxies have separate keyed caches.
Final banks use only complete category-specific fitting partitions. The source
preserves intermediate indices, projected coverage and source-image provenance.
Three seeds vary candidate pool/projection/initial center, not CNN extraction.
Banks and the single shared feature-extractor checkpoint remain ignored locally.
Git checkout byte identities follow `.gitattributes`, including CRLF frozen JSON.

GPU float32 GEMM searches use 512-query/2048-bank blocks and direct nearest-distance
refinement, with TF32 disabled and CUBLAS_WORKSPACE_CONFIG=:4096:8. This is a flat
exhaustive backend within float32 numerical precision; almost-tied indices can
be precision-limited. It is not an approximate ANN index or full QxMxD broadcast.
The coreset itself is approximate because of candidate capping and projection.

Primary native metrics reuse the verified evaluator's exact implementation.
`prepare_patchcore_evaluator.py` documents the adaptation without modifying earlier
evaluators. Never regenerate frozen source after confirmation. Native scores are
raw L2 distances; separate image q95 and pixel q99.5 calibration use only reserved
normals. Bootstrap histograms transform d/(d+s), with s from normal calibration,
not test extrema or labels. Image bootstrap comparisons are paired across models
and seeds using the same official image draws. Pixel CI targets and quantization
bias are disclosed separately from exact native point estimates.

Latency includes actual preprocessing/transfer, feature extraction, NN, native
map generation and CPU transfer from a decoded memory-resident RGB image. Disk
read/decode is excluded, as in prior notebooks. Fifty warmups precede 200 timed,
synchronized batch-one runs. Means of run p50/p95 are not pooled percentiles.
RSS uses 20-Hz sampling; native metric scratch allocations contribute to evaluation
RSS and are not serving-memory estimates. Earlier evaluators retained GPU test
images during latency measurement; memory scopes must be compared carefully.

No official test data may select a new budget, backbone, score, transform or
threshold after this confirmation. No SuperSimpleNet work is authorized here.
