# CUDA research environment

Milestone 2 uses **this repository's isolated `.venv`** with CPython 3.12.10,
PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA runtime 12.8. The NVIDIA
RTX 4060 Laptop GPU has 8,188 MiB VRAM and driver 610.91. No standalone CUDA
Toolkit, source compilation, model weights or changes to other environments were
needed. The former CPU snapshot `requirements-audit.lock.txt` is historical;
its environment audit remains in Milestone 1's Git history.

This matched pair is published for Windows/Linux on the
[official PyTorch version page](https://pytorch.org/get-started/previous-versions/).
It was installed from the official CUDA 12.8 wheel index. The driver-visible CUDA
capability differs from the wheel's runtime version; real GPU operations establish
readiness.

## Reproduce with PowerShell

Use CPython 3.12.10 from a fresh checkout. The artifact lock targets Windows x64 /
CPython 3.12 and records every external research dependency, its exact version
and SHA256 of the downloaded wheel. PyTorch entries use direct official URLs.
Do not use the old CPU snapshot or an inherited system-site-packages environment.

```powershell
cd C:\Users\shaik\mvtec-cnn-anomaly-detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.0.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt --cache-dir .cache/pip
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name mvtec-research --display-name "MVTec research (CUDA, Python 3.12)"
.\.venv\Scripts\python.exe scripts/verify_cuda.py
.\.venv\Scripts\python.exe -m jupyterlab notebooks/00_dataset_eda.ipynb
```

Select **MVTec research (CUDA, Python 3.12)** in JupyterLab. The browser editor,
kernel, automated execution and future research use this `.venv`. Execution
verifies that the registered kernel's interpreter matches its own. Activation is
optional; explicit paths avoid PowerShell policy and PATH ambiguity. Verify
`include-system-site-packages = false` in pyvenv.cfg.

The initial direct inputs were `requirements-cuda.txt` (official PyTorch index)
then `requirements-research-cuda.txt` (PyPI). `scripts/freeze_environment.py`
combined retained local pip reports with installed metadata to create the hash
lock. Original installation reports stay ignored under outputs/. See
[lock identity](../reports/environment_lock.json). A second independent rebuild
was not performed; this installed isolated environment and kernel are verified.

## Verify data, notebook and CUDA

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/execute_eda.py
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-new-run
```

Use a fresh project-local `--basetemp` for sandboxed tests. Preparation only reads
original data and refuses to replace a different frozen manifest. Execution keeps
the full notebook and licensed image panels locally; the published copy retains
real numeric outputs and aggregate plots only.

[CUDA evidence](../reports/cuda_verification.json) records matrix multiplication,
finite CNN forward/backward gradients, torchvision CUDA NMS and peak memory.
Inputs are synthetic tensors; there are no optimizer steps or model training.
This smoke check is not a model memory/latency benchmark. Full software/hardware
facts are in the [environment audit](../reports/environment.json).
