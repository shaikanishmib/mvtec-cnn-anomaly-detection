# MVTec AD CNN Anomaly Detection

PyTorch research comparing normal-only industrial anomaly detection and defect
localization across all 15 original MVTec AD categories. The scientific notebooks
are the canonical experiment narrative; compact, append-only records preserve
both successful and unsuccessful experiments. The Basic CNN baseline is complete;
the remaining model families and engineering application are planned.

## Research roadmap

| Notebook | Focus | Status |
|---|---|---|
| [00](notebooks/00_dataset_eda.ipynb) | Dataset EDA and integrity | Complete real-data EDA and verified CUDA kernel |
| [01](notebooks/01_basic_cnn.ipynb) | Basic CNN, synthetic-supervised segmentation | Complete: bounded development and all 15 categories across three seeds |
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

The user-provided complete dataset is already present; it was not downloaded
again or modified. Automatic root discovery supports categories directly under
`data/` or within a wrapper such as `data/mvtec_ad/`. All 15 categories passed
decoding and mask checks: **5,354 images**, comprising 3,629 training normals,
467 test normals and 1,258 test defects, each with a valid mask. See
[per-category counts](reports/category_counts.csv) and
[data/manifest identities](reports/dataset_validation.json). No exact duplicate
images were found. Original archive authenticity cannot be independently checked
without the source archive/checksum; this limitation is recorded.

## Reproduce the foundation

Use PowerShell and CPython 3.12.10. This project's `.venv` is isolated and contains
the verified CUDA-enabled scientific/Jupyter environment:

```powershell
cd C:\Users\shaik\mvtec-cnn-anomaly-detection
python -m venv .venv
New-Item -ItemType Directory -Force data,checkpoints,outputs | Out-Null
.\.venv\Scripts\python.exe -m pip install pip==25.0.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt --cache-dir .cache/pip
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name mvtec-research --display-name "MVTec research (CUDA, Python 3.12)"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe scripts/execute_eda.py
```

[Environment instructions](docs/environment.md) explain the Windows/Python-specific
artifact hash lock, direct pins, JupyterLab/kernel setup and actual CUDA checks.
The lock covers 116 external packages with downloaded-wheel SHA256 hashes.
Launch `.venv/Scripts/python.exe -m jupyterlab` and select **MVTec research (CUDA,
Python 3.12)**. A second independent environment rebuild has not been performed.

The execution script verifies the registered kernel's interpreter and saves the
full executed notebook to `outputs/eda/00_dataset_eda.executed.ipynb`. The tracked
copy preserves real numeric outputs and aggregate plots; panels containing
dataset pixels remain local. All real-data sections run; integrity failures stop
execution. Re-run `scripts/prepare_data.py` to verify the frozen split manifests.

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

The setup includes the protocol, completed 20-section EDA, project-local dataset
validation, deterministic partitions, metric references and an append-only
experiment registry. Seed 2026 assigns 2,551 fitting, 539 development and 539
calibration normals; all 1,725 official test files remain separate. See
[environment audit](reports/environment.json),
[notebook execution evidence](reports/eda_execution.json),
[EDA findings](reports/eda_findings.json) and [validation history](reports/validation.md).

The RTX 4060 Laptop GPU reports 8,188 MiB VRAM (about 8 GB), driver 610.91.
Installed PyTorch **2.11.0+cu128** and torchvision **0.26.0+cu128** detect the GPU.
GPU matrix multiplication, a synthetic CNN forward/backward check and torchvision
CUDA NMS passed in the research kernel. See [GPU evidence](reports/cuda_verification.json).
No pretrained weights have been downloaded. The scratch Basic CNN completed a
bottle pilot and 19 further controlled development fits on four predeclared
categories. The frozen width-16, mixed-synthetic, weighted BCE+Dice policy then
trained **45 fresh models: all 15 categories × seeds 2026/2027/2028**, eight epochs
each. Synthetic edits/probes used fitting normals only; official development
remained normal. Reserved normal calibration set thresholds before one official
test evaluation per category/seed. No post-test configuration changes were made.

## Basic CNN measured baseline

Unweighted category macro means, averaged over three seeds:

| Metric | Mean |
|---|---:|
| Image AUROC | 0.7126 |
| Image AP | 0.8590 |
| Pixel AUROC | 0.7858 |
| Pixel AP | 0.2354 |
| AU-PRO, FPR ≤0.30 | 0.5498 |
| Recall | 0.4018 |
| False-positive rate | 0.0983 |
| Precision | 0.8733 |
| F1 | 0.5106 |

The 40.2% recall and 9.8% FPR expose a limited baseline; the normal-calibration
5% image-FPR target did not guarantee test FPR. Performance varies sharply by
category, including poor screw detection. Image AP must be read against defect
prevalence. Pixel point estimates use exact native-resolution evaluation; pixel
bootstrap intervals use a disclosed 4,096-bin approximation with measured AP bias.
All failures, rejected configurations and seed variation are retained.

The model has **83,857 parameters**, checkpoints about **335 KiB**, maximum final
training allocation **0.480 GiB**, and mean across runs of model p50 **3.139 ms**
(p95 **4.629 ms**). Mean end-to-end p50 is **12.538 ms**, from memory-resident RGB
to native CPU map; file read/decode is excluded. These are means of per-run
percentiles, not a pooled service benchmark. Final training took **8.11 minutes**
including loading/validation; exact native evaluation took **35.85 minutes**.

Read [Notebook 01](notebooks/01_basic_cnn.ipynb),
[all-category results](reports/basic_cnn/all_category_means.csv),
[measured findings](reports/basic_cnn/findings.md),
[development comparison](reports/basic_cnn/development_macro.csv),
[preregistration](docs/basic_cnn_protocol.md),
[frozen configuration](configs/basic_cnn_frozen.json) and
[reproduction instructions](docs/basic_cnn_reproduction.md).
The notebook shows the exact architecture, generator, loss and training loop,
full experiment story, calibration, evaluation and per-category failure analysis.
Public replay preserves genuine compact outputs; licensed image/map panels and
large artifacts remain local. New reproductions use new IDs, never overwrite v1.

## Repository layout and next stage

`notebooks/` holds the research narrative; `src/anomaly_detection/` contains small
auditable utilities; `configs/`, `experiments/`, `reports/`, `scripts/`, `tests/`
and `docs/` hold configuration, evidence and reproduction support. `data/`,
`checkpoints/` and `outputs/` are local ignored storage; scripts recreate outputs.

Notebook 01 is complete. Notebook 02 requires a separate request. Each future notebook
must preserve its full experimental story, tried configurations, actual results,
calibration, evaluation, failures and interpretations. After the research
comparison, a separate engineering
stage will add reusable PyTorch inference, FastAPI, a browser interface, tests,
benchmarks and deployment. That application is not implemented in this milestone.
