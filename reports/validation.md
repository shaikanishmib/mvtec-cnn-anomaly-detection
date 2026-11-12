# Foundation validation — Milestone 1 history

This is a setup/protocol milestone, not completed dataset research or a model
benchmark. No official MVTec files were available in this repository.

- Notebook: 59 valid cells, 20 ordered sections, 19 executable code cells. Every
  code cell has explanatory Markdown before and after it; all cells compile.
- Missing-data execution: all 19 code cells ran without cell errors; all 15
  categories were explicitly missing. Dataset counts, actual plots, image/mask
  validation and official-file leakage checks remain unexecuted.
- Committed notebook outputs are cleared. Local execution evidence is under
  ignored `outputs/eda/`; compact status is in `eda_execution.json`.
- Tests use tiny **synthetic fixtures**, covering all 15 category names, full
  image/mask decoding, missing/corrupt/mismatched/nonbinary/orphan masks, illegal
  anomalous training files, duplicates, deterministic disjoint normal partitions,
  project-local roots, known metric values, calibration ties and immutable records.
- First test attempt: 10 passed, 27 fixture setup errors because the sandbox
  denied the default pytest temporary directory. No test assertions failed.
  Authorized rerun: 37 passed, one nonfatal pytest cache-permission warning.
  Final run: **37 passed, 0 failures, 0 warnings**, using a fresh project-local
  temporary directory and disabling cache:
  `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-final-20261112 --tb=short`.
  Use a fresh `--basetemp` subdirectory for subsequent sandboxed runs.
- CPU PyTorch and torchvision import successfully; `pip check` reports no broken
  requirements. GPU visible to nvidia-smi, but PyTorch CUDA unavailable.
- No models trained/downloaded, no test-label optimization, no dataset published.

Re-run `scripts/validate_notebook.py`, `scripts/execute_eda.py`, and `pytest` with
the documented interpreter. `scripts/audit_milestone.py` checks the concrete
staged files, repository/branch and local documentation links before commit.

## Milestone 2 — real data and CUDA, 2026-11-12

The user supplied complete data; no dataset was downloaded or changed. Independent
preflight and notebook scans decoded 5,354 original images and all 1,258 paired
defect masks, with no corrupt/missing/orphan masks or exact duplicate images.
All 15 categories and original splits passed. Archive download timestamp and
publisher checksum remain unknown; local hashes establish file identities only.

- Notebook: all 19 code cells in 20 sections executed successfully in the
  registered isolated CUDA kernel. Each code section retains explanatory Markdown
  before/after. Genuine tables and aggregate plots remain published; the complete
  image-containing executed copy stays local. All four generated figures were
  visually inspected, covering every category and mask overlay.
- Seed-2026 manifests: 2,551 fit, 539 development, 539 calibration, 1,725 test.
  Paths and exact image contents are disjoint. A second full scan in the notebook
  and shuffled-input recomputation reproduced all four frozen manifest texts and
  SHA256 identities exactly. Test-mask references remain test-only.
- CUDA: PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128 detected the RTX 4060.
  Matrix multiplication, finite synthetic CNN forward/backward gradients and
  torchvision CUDA NMS passed in the Jupyter kernel. Peak smoke-check allocated
  memory was 10,416,128 bytes (9.93 MiB), reserved 25,165,824 bytes (24 MiB).
  No optimizer, real model training, pretrained weights or benchmark claims.
- Environment: isolated `.venv`, CPython 3.12.10, JupyterLab 4.4.3; `pip check`
  clean. The artifact hash lock covers 116 external packages on Windows/Python
  3.12. Global CPU packages and all other project environments were left alone.
- Tests: **44 passed, 0 failed, 0 errors, 0 skipped**, no warnings. Includes all
  previous synthetic fixtures, nested/direct/ambiguous root discovery, real
  manifest consistency/isolation/reproducibility, full EDA evidence and actual GPU
  checks. JUnit XML stays local; its identity and summary are published in
  `milestone2_validation.json`.
- Setup issues retained: the initial pip network attempt was sandbox-blocked and
  succeeded after authorization; pip-report JSON parsing initially hit Windows'
  default encoding and was corrected to explicit UTF-8. No GPU compatibility
  failure or failed scientific/model experiment occurred.

Remaining methodological limits: exact hashes do not exclude near duplicates or
shared acquisition groups; the isolated environment was installed and tested but
not rebuilt a second time; small calibration/test samples limit FPR precision.
Test examples were viewed descriptively, with no test-label tuning. Notebook 01
has not been created or started. The earlier missing-data milestone remains in Git
history and is explicitly distinguished above.
