# MVTec AD CNN Anomaly Detection

PyTorch research comparing normal-only industrial anomaly detection and defect
localization across all 15 original MVTec AD categories. The scientific notebooks
are the canonical experiment narrative; compact, append-only records preserve
both successful and unsuccessful experiments. No benchmark results exist yet.

## Research roadmap

| Notebook | Focus | Status |
|---|---|---|
| [00](notebooks/00_dataset_eda.ipynb) | Dataset EDA and integrity | Written; missing-data execution verified; real-data EDA blocked |
| 01 | Basic CNN, declared one-class or synthetic proxy objective | Planned |
| 02 | Convolutional autoencoder | Planned |
| 03 | PatchCore | Planned |
| 04 | SuperSimpleNet | Planned |
| 05 | PaDiM | Planned |
| 06 | EfficientAD | Planned |
| 07 | Cross-model comparison | Planned |

## Dataset

Use the complete original [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
from MVTec's official download form. Attribute Bergmann et al., CVPR 2019 and the
extended IJCV dataset paper. Data is licensed **CC BY-NC-SA 4.0** and must not be
published in this repository. Follow [acquisition and checksums](docs/dataset.md).

Extract under `data/mvtec_ad/<category>/` with `train/good`, `test/<type>` and
`ground_truth/<type>` directories. Data, checkpoints, weights, executed notebooks
and image panels are Git-ignored. No dataset was available at initialization;
category counts, image statistics and observed plots are therefore unavailable.

## Reproduce the foundation

Use PowerShell and CPython 3.12.10. The verified local environment reuses installed
packages without modifying them:

```powershell
cd C:\Users\shaik\mvtec-cnn-anomaly-detection
python -m venv --system-site-packages .venv
New-Item -ItemType Directory -Force data,checkpoints,outputs | Out-Null
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe scripts/execute_eda.py
```

[Environment instructions](docs/environment.md) explain the audited package lock,
direct dependency pins, JupyterLab/kernel setup and alternative pinned CUDA
specification. The inherited environment is not an independently rebuilt isolated
environment; that verification remains outstanding. Launch Jupyter Notebook or
JupyterLab in a browser and select the research interpreter. No VS Code is needed.

The execution script uses its own interpreter and saves the executed notebook to
`outputs/eda/00_dataset_eda.executed.ipynb`. The tracked notebook has cleared
outputs to avoid distributing dataset images. Missing data is explicitly reported
and skipped; successful missing-data execution is **not completed dataset EDA**.

## Methodology

Read the [research protocol](docs/research_protocol.md) before fitting. Reserve
approximately 70/15/15 of official normal training images for fitting,
development and calibration, respectively, using frozen seed-2026 partitions.
Synthetic anomalies are proxy labels only. Official test labels and masks never
train models or choose settings/thresholds. Descriptive test-image EDA is disclosed
and must not be described later as untouched/unseen test-label evaluation.

Report image AUROC/AP; pixel AUROC/AP and normalized AU-PRO over FPR ≤0.30 when
localization exists; recall, FPR, precision, F1 and confusion matrices at thresholds
calibrated on held-out normals only. Include per-category and unweighted macro
results, uncertainty, synchronized latency and peak GPU memory. No performance
numbers are reported before actual experiments. See the
[registry policy](experiments/README.md) and [configuration template](configs/experiment_template.json).

## Verified progress and compute limits

The setup includes the protocol, a 20-section EDA notebook, project-local dataset
validation, deterministic partitions, metric references and an append-only
experiment registry. See [environment audit](reports/environment.json) and
[notebook execution status](reports/eda_execution.json) for machine-readable facts.

The RTX 4060 Laptop GPU reports 8,188 MiB VRAM (about 8 GB), driver 610.91.
Installed PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu import successfully;
PyTorch CUDA is unavailable. CPU EDA works. CUDA-enabled training, model latency,
and memory feasibility remain unverified. No model weights have been downloaded
and no model has been trained.

## Repository layout and next stage

`notebooks/` holds the research narrative; `src/anomaly_detection/` contains small
auditable utilities; `configs/`, `experiments/`, `reports/`, `scripts/`, `tests/`
and `docs/` hold configuration, evidence and reproduction support. `data/`,
`checkpoints/` and `outputs/` are local ignored storage; scripts recreate outputs.

Next: acquire and validate MVTec AD, verify the CUDA environment, then begin
Notebook 01 — Basic CNN. After the research comparison, a separate engineering
stage will add reusable PyTorch inference, FastAPI, a browser interface, tests,
benchmarks and deployment. That application is not implemented in this milestone.
