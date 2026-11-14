# Reproducing Notebook 05

Use the existing exact interpreter from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/padim/pytest-fresh-reproduction
.\.venv\Scripts\python.exe scripts/summarize_padim.py
.\.venv\Scripts\python.exe scripts/analyze_padim.py
.\.venv\Scripts\python.exe scripts/validate_padim.py
```

The aggregate scripts reconcile existing saved predictions/checkpoints; they do
not re-evaluate the official test set. The validator checks the frozen source
hashes, result/checkpoint/map identities and full category/seed coverage.
Raw MVTec AD images/masks and the original licensed data remain local. Restore
the original validated dataset and all local checkpoints/maps/bootstrap vectors
if reproducing artifact analysis on another machine. Consult the artifact manifest
for exact paths/checksums. The external ImageNet V1 checkpoint identity is frozen
in configs/padim_frozen.json; no automatic new weight acquisition occurs.

## Fresh experiment sequence

The completed registry is append-only. Do not run these commands over completed
IDs. For an independent reproduction, use a separate checkout and a new registered
run namespace, retaining the committed evidence. These commands document the
original ordered phases; they are not an invitation to overwrite results:

```powershell
New-Item -ItemType Directory -Force reports/padim,outputs/padim | Out-Null
.\.venv\Scripts\python.exe scripts/environment.py reports/padim/environment.json
.\.venv\Scripts\python.exe scripts/run_padim.py pilot
.\.venv\Scripts\python.exe scripts/run_padim.py development
.\.venv\Scripts\python.exe scripts/run_padim.py freeze
.\.venv\Scripts\python.exe scripts/run_padim.py evaluate
```

Pilot and development never load official test images. The final phase uses all
fit normals, saves category Gaussian checkpoints, calibrates on reserved normals
and then evaluates each test category once per seed. It preserves each completed
evaluation immediately and skips existing completed results on restart. An
incomplete reserved run fails closed to preserve evidence; investigate and create
a separately documented recovery ID rather than delete or overwrite that run.
No configuration changes are permitted after the prospective freeze.

## Notebook publication

`scripts/build_padim_notebook.py` constructs the 20-section notebook;
`scripts/publish_padim_notebook.py` executes it using `mvtec-research`, which must
resolve to the project's exact interpreter. The full local execution is retained
at outputs/padim/05_padim.full.executed.ipynb. The publication script refuses to
overwrite that local execution. Use an explicit new execution filename for a
subsequent independent notebook replay. Public outputs contain aggregate plots
and tables only; original image panels stay local and ignored.

Measured fit time includes feature extraction and Gaussian estimation. Latency
excludes file decoding and uses 50 warmups/200 synchronized batch-one iterations.
Model time includes extraction and Mahalanobis scoring; end-to-end also includes
RGB preprocessing/transfer, smoothing, native restoration and CPU map transfer.
Feature extraction is repeated per seed/category in the recorded final workload;
there is no PaDiM embedding cache masking final fit cost. Existing shared resized
uint8 image caches may be reused. Laptop power/thermal state can change latency.

The full covariance storage is O(H W d²); the selected 100-dimensional 64-square
Gaussian checkpoint is large relative to small learned heads. Fit scratch is
spatially blocked and inference solves multiple right-hand sides. All raw-distance
primary pixel metrics are exact at native resolution; only pixel bootstrap
intervals use calibration-scaled 4096-bin histograms. Image bootstrap intervals
are exact conditional on fixed models and calibration thresholds. Training seed
spread, calibration sampling uncertainty and dataset sampling are distinct.
