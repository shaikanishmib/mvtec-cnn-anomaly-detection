# Foundation validation

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
