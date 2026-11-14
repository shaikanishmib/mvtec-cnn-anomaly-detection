"""Build an explicitly partial, saved-artifact replay notebook; never fit models."""
from pathlib import Path
import nbformat as n
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'notebooks/06_efficientad.ipynb').exists():
    existing=n.read(ROOT/'notebooks/06_efficientad.ipynb',4)
    if existing.metadata.get('efficientad_development_replay'):
        raise RuntimeError('Notebook contains preserved development history; use saved-artifact replay, not the pilot-only builder')
sections=[
('Motivation','EfficientAD combines efficient local descriptors with global feature reconstruction. This notebook currently documents the reference review, implementation and real GPU pilot. Development and the frozen benchmark remain pending.'),
('Paper review','Read Batzner et al., WACV 2024, using https://arxiv.org/html/2303.14535v3 after CVF returned 403. Algorithms 1–2 and Tables 6–8 define this implementation. See docs/efficientad_protocol.md for the source links and detailed derivation.'),
('Teacher provenance','The teacher is the public third-party PDN-S checkpoint from nelson1425/EfficientAD, revision fcab5146f84ae17597044ad5ddf1656ccf805401. Repository Apache-2.0; no separate weight model card. Its accompanying code describes ImageNet WideResNet101_2 distillation; that history cannot be independently attested from tensors.'),
('Architecture','PDN convolutions use channels 3→128→256→256→384, kernels 4,4,3,4 and average pooling after the first two. Paper padding gives 64×64 output and a 33-pixel local receptive field. The student outputs 768 channels. The global autoencoder has a 64×1×1 bottleneck. Medium capacity adds wider layers and 1×1 convolutions; it has not been measured here.'),
('Loss mathematics','Normalize teacher channels with fitting-only population moments. D=(t−s1)²; Lhard=mean(D≥its .999 quantile). Lpenalty=mean(s1(P)²), Lae=mean((t−a)²), Lstae=mean((a−s2)²). Sum all four; no external P is available, so this pilot omits penalty. Lstae updates both branches.'),
('Map mathematics','Local=mean_channels((t−s1)²); global=mean_channels((a−s2)²). Bilinearly resize to 256×256, normalize each as .1(M−q90)/(q99.5−q90), then average equally. Image score=max(combined map). Larger scores indicate anomalies. Final development scaling and reserved-normal threshold calibration are pending.'),
('Environment and replay','The following code is saved-artifact replay of a separately executed real-data GPU pilot. It does not launch training. Live execution command: `.\\.venv\\Scripts\\python.exe scripts/pilot_efficientad.py`; its append-only start event prevents accidental repeated fitting.'),
('Dataset partitions','The existing frozen 70/15/15 normal partitions are unchanged. Pilot: bottle, seed 2026, first eight sorted fitting normals; no official test files, labels, masks or calibration normals were opened. Selected fitting images were checked against their SHA256 manifest.'),
('Teacher preparation','Downloaded only the 10.8 MB checkpoint and repository license, not ImageNet. All teacher parameters are frozen. Pilot teacher moments and map scaling use the eight fitting normals solely for engineering checks. These statistics cannot be used as final benchmark calibration.'),
('Core implementation','The next cell exposes reusable model code, including each scientific loss, normalization and map operation. The notebook is intended to make the process inspectable rather than hide it behind a single training API.'),
('Predeclared pilot','Before training, docs/milestone_status.md recorded 32 batch-one Adam steps at lr 1e−4, weight decay 1e−5, hard quantile .999. Fixed brightness 1.1 gives a repeatable AE learning probe; paper training uses randomized brightness/contrast/saturation. This is a disclosed engineering variant.'),
('Actual pilot results','Replay the actual compact report below. The run verified finite losses and gradients, changes in student and AE parameters, decreased fixed-image hard and AE losses, finite 256×256 maps and exact checkpoint inference replay.'),
('Learning curves','These 32 steps are a short engineering trace, not evidence of convergence or anomaly quality. The image cycles every eight steps and dropout operates during training; fixed-image eval-mode before/after losses provide the cleaner learning check.'),
('Compute feasibility','The pilot step time can project an approximate 70,000-step paper-budget fit. Multiply by 45 for the requested all-category three-seed benchmark, then add data loading, evaluation and thermal margin. This extrapolation is not a measured full training runtime. Deterministic batch-one latency has a substantial p95 tail and needs further characterization.'),
('Bounded development plan','Proposed cap: 12 fits total, bottle and carpet, 2,000 steps each, seed 2026. Questions cover mean versus hard loss, AE contribution, map normalization, aggregation and capacity/budget. Penalty contrast requires legitimate documented external inputs and replaces another trial. No development fits have executed or been selected/rejected.'),
('Model selection and calibration','Pending. No frozen final configuration or calibrated threshold exists. Later use development normals and fitting-derived synthetic proxies for selection, reserved calibration normals for strict score>threshold. Report sample size, nominal/observed FPR and uncertainty; no retrospective test tuning.'),
('Frozen benchmark and metrics','Pending: zero final category-seed evaluations out of target 45. Image AUROC/AP, pixel AUROC/AP, AU-PRO, recall, FPR, precision and F1 are unavailable. No final metrics can be inferred from this normal-only pilot.'),
('Failure analysis','Pending genuine test examples from all 15 categories after freeze. No false positive/negative claims are made. Later inspect scratches, textures, halos, logical/structural anomalies and localized responses missed by max aggregation alongside original images and masks; keep image panels local.'),
('Six-model comparison','Pending EfficientAD evaluation. Previous Basic CNN, convolutional autoencoder, PatchCore, SuperSimpleNet Lite and PaDiM records remain unchanged. Later comparisons must match categories and report paired uncertainty, nine metrics, compute, memory and storage without unsupported superiority claims.'),
('Validation, limitations and recovery','Focused core tests and deep pilot identity validation are provided. This notebook is a pilot-stage research record, not a completed milestone. No external penalty, only eight normals, fixed augmentation and 32 steps limit conclusions. Full development, calibration, benchmark, failure analysis and comparison remain. Notebook 07 is not started. Recovery instructions and current Git status are in docs/milestone_status.md.')]
nb=n.v4.new_notebook(metadata={'kernelspec':{'display_name':'MVTec research (CUDA, Python 3.12)','language':'python','name':'mvtec-research'}})
cells=[n.v4.new_markdown_cell('# 06 — EfficientAD\n\nPilot-stage research notebook. Full benchmark is gated and has not started.')]
codes={7:"from pathlib import Path\nimport json, sys\nROOT = Path.cwd() if (Path.cwd()/'reports').exists() else Path.cwd().parent\nassert Path(sys.executable).resolve() == ROOT/'.venv/Scripts/python.exe'\npilot = json.loads((ROOT/'reports/efficientad/pilot.json').read_text())\nprint('Saved-artifact replay:', pilot['experiment_id'], pilot['completed_utc'])",
10:"import inspect\nfrom anomaly_detection import efficientad\nprint(inspect.getsource(efficientad))",
12:"print(json.dumps({k:v for k,v in pilot.items() if k!='history'}, indent=2))",
13:"import matplotlib.pyplot as plt\nfor key in ('hard','autoencoder','student_autoencoder'):\n    plt.plot([r['step'] for r in pilot['history']], [r[key] for r in pilot['history']], label=key)\nplt.yscale('log'); plt.xlabel('Optimizer step'); plt.ylabel('Loss'); plt.legend(); plt.show()",
14:"seconds = pilot['seconds_per_step']\nprint({'estimated_70000_step_hours_per_model':seconds*70000/3600, 'estimated_45_model_training_hours':seconds*70000*45/3600, 'latency_ms':pilot['model_latency_ms']})"}
for i,(title,body) in enumerate(sections,1):
    cells.append(n.v4.new_markdown_cell(f'## {i}. {title}\n\n{body}'))
    if i in codes:
        cells.append(n.v4.new_code_cell(codes[i]))
        interpretation={7:'Replay confirms the real pilot identity; it does not reproduce training or independently remeasure CUDA.',10:'Each operation is visible above. The optional penalty accepts external images, but none were supplied in the recorded pilot.',12:'Hard loss fell from 17.1074 to 9.3763; AE loss fell from .98938 to .95656. This establishes short-run branch learning only. Model p50/p95 were 33.47/145.27 ms; these are measured on this laptop with deterministic FP32 settings.',13:'The trace retains all 32 steps, including fluctuations, rather than presenting only favorable endpoints. It cannot establish anomaly discrimination or convergence.',14:'At .1568 seconds per pilot step, linear projection is about 3.049 hours per 70k-step fit and 137.19 hours for 45 fits, excluding overhead. A full run is not yet validated; latency tails and sustained thermal behavior remain uncertain.'}[i]
        cells.append(n.v4.new_markdown_cell(interpretation))
nb.cells=cells
n.write(nb,ROOT/'notebooks/06_efficientad.ipynb')

if __name__=='__main__':
    import os
    os.environ['IPYTHONDIR']=str(ROOT/'.cache/ipython-efficientad')
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter-efficientad')
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=180,kernel_name='mvtec-research',resources={'metadata':{'path':str(ROOT)}}).execute()
    n.write(nb,ROOT/'notebooks/06_efficientad.ipynb')
