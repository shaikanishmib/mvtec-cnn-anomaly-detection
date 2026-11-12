# Reproducing Notebook 01

Use this repository's existing MVTec AD data, isolated CUDA `.venv` and registered
`mvtec-research` kernel. Installation is documented in environment.md. No pretrained
models, extra frameworks, dataset downloads or separate project files are needed.

## Read and replay the recorded research

```powershell
cd C:\Users\shaik\mvtec-cnn-anomaly-detection
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_basic_cnn.ipynb
# Or, on the original research machine with retained artifacts:
.\.venv\Scripts\python.exe scripts/execute_basic_cnn_notebook.py
```

The notebook defaults to replay: it checks and displays **actual saved** metrics,
curves, exact configurations and decisions without silently fitting or repeating
official test inference. Architecture, generator, loss and the full training
controller are executable notebook definitions injected from the exact shared
source. Calibration and complete native evaluation source are visible, including
the actual functions/configurations used. This is not a one-call hidden model.

On this research machine, all best/last checkpoints, optimizer/RNG states,
model-resolution maps, bootstrap vectors, histograms and timing samples remain
local. The full executed notebook contains synthetic fitting examples and all 15
category outcome panels. The public notebook retains genuine numeric/aggregate
outputs; original image/mask panels and large artifacts are deliberately omitted.

A fresh clone has the public numeric history but not ignored checkpoints/maps.
Missing local visual artifacts are explicitly reported, never fabricated. A
numeric-only replay is not recorded as a full visual reproduction. Generate new
registered runs as below to obtain local maps, or restore the original local
artifacts and verify their recorded SHA256 identities.

## Run new, separately registered reproductions

In Notebook 01, enable only the work intended:

- `EXECUTE_NEW_PILOT=True`: one new bottle pilot with the original configuration.
- `EXECUTE_NEW_DEVELOPMENT=True`: the bounded five-configuration/four-category
  search. A separately reproduced bottle pilot is reused if enabled.
- `EXECUTE_NEW_CONFIRMATION=True`: 45 fresh fits/evaluations under the original
  frozen policy (15 categories, seeds 2026/2027/2028).

The UTC `RUN_PREFIX` creates new experiment IDs. Existing v1 registry records,
checkpoints, decisions, frozen configuration and public results are not replaced.
New measured results are displayed separately; reproductions do not constitute a
new untouched test confirmation. Do not change the selected policy after seeing
these test results. All fitting/development/calibration/test restrictions still
apply. The original research's active training wall time and separately measured
evaluation time are recorded; exact native-pixel sorting can dominate runtime.

Original phase commands, shown for chronology, were `run_basic_cnn.py pilot`,
`run_basic_cnn.py development`, `select_basic_cnn.py`, `run_basic_cnn.py final-train`
and `run_basic_cnn.py final-evaluate`. Their v1 names are reserved; use the notebook's
new-ID branches for another run rather than deleting records to rerun these IDs.
The selection script refuses to overwrite the existing freeze.

## Verify retained original artifacts

```powershell
.\.venv\Scripts\python.exe scripts/validate_basic_cnn.py --verify-calibration
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-new-basic-cnn-run
```

The artifact validator checks all original model/source/configuration identities,
normal-only gradient/probe sources, exact coverage, per-image metric arithmetic,
native histogram totals and ranking bounds, bootstrap intervals and timing
percentiles. Optional calibration verification re-infers **normal calibration
images only**, never official test images, and compares thresholds/scores without
changing them. The full original artifact check requires the retained local files;
new-ID reproductions have their own identities.

Pixel point estimates are exact at native resolution. Pixel confidence intervals
target a disclosed 4,096-bin image-bootstrap approximation; its measured bias is
reported, especially for AP. Raw legacy precision uses the foundation utility's
zero-division convention; no-alarm cases are explicitly flagged and public
operating tables show undefined precision as N/A with coverage. Image F1 remains
zero when all true defects are missed. No result is dropped to improve averages.

Per-image CSV scores are round-trippable float32 decimals. Restore float32 before
strict threshold comparisons; interpreting rounded decimals directly as float64
can change an exact tie. The validator checks scores against retained float32
map maxima, so publication formatting cannot silently change confusion counts.

Notebook 02 and all other model families remain separate future work.
