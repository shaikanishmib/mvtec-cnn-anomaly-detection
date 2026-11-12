# Environment and reproduction

Verified interpreter: CPython 3.12.10, Windows; see
[environment audit](../reports/environment.json). The existing packages pass
`pip check`. No package was installed or upgraded for this milestone.
`requirements-audit.lock.txt` captures every installed distribution; pyproject
pins the direct dependencies actually used. This snapshot includes CPU-only
torch 2.15.0+cpu / torchvision 0.29.2+cpu. Availability of these exact wheels on
public indexes has not been verified. Do not confuse a captured environment
with an independently rebuilt one.

The local `.venv` uses `--system-site-packages` to reuse the audited installation
without changing it. It is **not isolated** from subsequent global changes.
Before each run compare `pip freeze` to the snapshot. For an independent rebuild,
use CPython 3.12.10 and an isolated venv; retain the original package wheels and
hashes, install the lock, then run tests and `pip check`. The captured lock pins
versions but is not a wheel-hash supply-chain lock. A clean rebuild remains an
explicit verification task, not an achieved result.

```powershell
cd C:\Users\shaik\mvtec-cnn-anomaly-detection
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe scripts/execute_eda.py
```

JupyterLab is not installed in the audited environment. To use a browser-based
editor, install `jupyterlab==4.4.3` into a separate environment and launch it with
`python -m jupyterlab`. Register the research interpreter from this repository:

```powershell
python -m venv .venv-jupyter
.\.venv-jupyter\Scripts\python.exe -m pip install jupyterlab==4.4.3
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name mvtec-research --display-name "MVTec research (Python 3.12)"
$env:JUPYTER_PATH = (Resolve-Path .venv/share/jupyter).Path
.\.venv-jupyter\Scripts\python.exe -m jupyterlab
# Select "MVTec research (Python 3.12)" in the browser.
```

For training, create a **separate isolated CUDA environment** instead of replacing
global packages. The [official PyTorch matched pair](https://pytorch.org/get-started/previous-versions/)
in `requirements-cuda.txt` uses CUDA 12.8 wheels:

```powershell
python -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu128
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements-research-cuda.txt
.\.venv-cuda\Scripts\python.exe -m pip check
.\.venv-cuda\Scripts\python.exe scripts/environment.py
.\.venv-cuda\Scripts\python.exe -m pytest -q
.\.venv-cuda\Scripts\python.exe -m pip freeze > requirements-cuda-resolved.lock.txt
```

The companion file pins the full direct scientific/Jupyter stack. Resolve
compatibility and freeze its transitive lock before Notebook 01. The CUDA file
is an alternative specification, not compatible with the CPU pyproject torch
pins; do not install both. CUDA setup has not been downloaded or tested here.
The NVIDIA driver's reported CUDA capability is not the PyTorch runtime. Verify
imports and a real CUDA allocation with `scripts/environment.py` before training.
No local CUDA toolkit installation is required merely to use binary wheels.
