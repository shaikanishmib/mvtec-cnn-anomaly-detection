# Frozen benchmark recovery and publication

Inspect docs/milestone_status.md, reports/efficientad/final/benchmark_manifest.json,
reports/efficientad/final/resource_gate.json and experiments/06-final-*.json first.
The frozen configuration and evaluator identity must remain unchanged.

Use only project Python. Safe recovery wrapper:

```powershell
.\.venv\Scripts\python.exe scripts/resume_efficientad_final.py --status
.\.venv\Scripts\python.exe scripts/resume_efficientad_final.py
```

The wrapper detects a live manager and does not duplicate it. If a terminal died
while its child worker survived, it waits for that worker rather than launching
a competing fit. The manager rechecks completed artifact hashes; interrupted
training restores the latest atomic epoch model/optimizer/NumPy/torch CPU/CUDA RNG
snapshot. Interrupted evaluation reuses the verified trained checkpoint. A corrupt
completed attempt is retained and a new attempt ID is used. Never delete or edit
completed outputs or change the frozen model to recover from resource trouble.

The rare gap between publishing a verified result JSON and appending its registry
event can be reconciled without repeating fitting/evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/reconcile_efficientad_registry.py
```

After all 45 are complete, execute in this order:

```powershell
.\.venv\Scripts\python.exe scripts/summarize_efficientad_final.py
.\.venv\Scripts\python.exe scripts/write_efficientad_final_findings.py
.\.venv\Scripts\python.exe scripts/publish_efficientad_final_notebook.py
.\.venv\Scripts\python.exe scripts/qualify_efficientad_baseline.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp
.\.venv\Scripts\python.exe scripts/validate_efficientad_final.py
```

The notebook publisher executes only appended saved-artifact replay cells and
retains all prior pilot/development sources/outputs. Original images, maps,
checkpoints, caches and dataset-derived panels/full notebook remain ignored.
All-category numeric evidence, aggregate plots and compact histories/scores are
reviewed for Git. Inspect genuine panels before writing qualitative conclusions;
distinguish measured behavior from proposed explanations. Reconcile summary
metrics and prior-model exact comparisons; do not tune from the final test.

Finally update README and milestone status with actual findings, audit staged
content and line-ending-dependent SHA identities, commit, push and verify remote
main SHA. Do not declare completion before tests, deep artifact and notebook
validators pass. Notebook 07 is the next separately authorized milestone.
