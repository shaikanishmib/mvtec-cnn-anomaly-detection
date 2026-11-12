# Reproduce Notebook 02

Use the existing project `.venv` CUDA environment and complete validated dataset
under this repository's `data/`. Do not use a different dataset root or recreate
partitions. The loader checks every frozen partition manifest against the stored
SHA256. Existing Notebook 01 notebooks, histories and results are preserved.

The original live three-epoch bottle pilot is in
`outputs/autoencoder/02_convolutional_autoencoder.executed.ipynb`. Its builder
contains the actual architecture, training loop, registry reservation, plots and
local reconstruction panels. Running it again with the existing ID intentionally
fails at the append-only reservation. Assign a new experiment ID for a new pilot.

The separately run bounded development jobs use:

```powershell
.\.venv\Scripts\python.exe scripts/run_autoencoder.py development
```

All 20 configuration/category pairs use seed 2026 and 20 fixed epochs. The inner
fitting holdout supplies fixed synthetic probes; no altered images train the AE.
Completed jobs are not silently rerun. Failed or interrupted IDs cannot be reused;
register a new ID with a link to the previous event and explain the reason.

Selection uses `scripts/select_autoencoder.py`. Its output files open exclusively
and cannot overwrite the existing freeze. The frozen policy and compute report
must be inspected before final training. The original command sequence was:

```powershell
.\.venv\Scripts\python.exe scripts/select_autoencoder.py
.\.venv\Scripts\python.exe scripts/run_autoencoder.py final-train
.\.venv\Scripts\python.exe scripts/run_autoencoder.py final-evaluate
.\.venv\Scripts\python.exe scripts/summarize_autoencoder.py
.\.venv\Scripts\python.exe scripts/publish_autoencoder_notebook.py
.\.venv\Scripts\python.exe scripts/validate_autoencoder.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Final train and evaluation skip completed immutable runs. An interrupted reserved
evaluation must be investigated and continued with a new linked event; do not
change the trained policy or hide prior output. Do not use the published test
results to revise the confirmation configuration.

The public Notebook 02 preserves live pilot outputs followed by separately
executed artifact-analysis cells. It does not claim that retained training jobs
ran live in the analysis kernel. Printed source is the actual implementation;
all configuration changes, curves, selection, thresholds, metrics and failure
examples are visible. The final local notebook retains licensed panels at
`outputs/autoencoder/02_convolutional_autoencoder.complete.executed.ipynb`.
The public notebook strips only dataset-pixel panel outputs. Raw arrays,
checkpoints, optimizer state and images stay ignored. Compact experiment events,
tables and artifact identities are published.

The evaluator is derived from the verified CNN evaluator to share exact native
pixel metrics and whole-image uncertainty conventions. `prepare_ae_evaluator.py`
documents that derivation; do not regenerate frozen source after confirmation.
`prepare_ae_summary.py` documents the corresponding aggregation derivation.
Calibration consumes only reserved normals: image q95 and pooled pixel q99.5.
Image maps use squared RGB-mean residual and fixed top-1% aggregation. Pixel maps
are bilinearly restored to native resolution for exact point metrics.

Latency starts with an already decoded RGB image in memory and ends with a
native residual map on CPU. Fifty warmups precede 200 synchronized batch-one
measurements. The model stage includes residual-map construction. Report both
median/p95 forward-stage and end-to-end costs; disk read/decode is excluded for
both CNN and AE. Byte-size checkpoint comparisons exclude separately stored
optimizer/RNG files for final models. The pilot checkpoint additionally contains
optimizer/RNG state, so its bytes are not directly comparable to final weights.
Per-run inference allocation is measured with the evaluator's resident test-image
cache still present; it is not an isolated deployment-process memory estimate.
