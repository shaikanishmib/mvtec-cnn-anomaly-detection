"""Build an explicit, executable PaDiM research narrative with compact outputs."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
n=nbf.v4.new_notebook()
n.metadata.kernelspec=dict(display_name='MVTec research (CUDA, Python 3.12)',language='python',name='mvtec-research')
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
n.cells=[md('''# Notebook 05 — PaDiM

This notebook combines a live normal-only mathematical demonstration with explicit
artifact analysis of independently executed, registered pilot, development and
confirmation jobs. Loading a completed result does not rerun or claim to rerun
its benchmark. The original data are licensed MVTec AD; image panels stay local.
''')]


def section(number,title,before,source,after,local=False):
    n.cells.append(md(f'## {number}. {title}\n\n'+before))
    cell=code(source)
    if local:cell.metadata['contains_dataset_images']=True
    n.cells.extend([cell,md(after)])


section(1,'Research question and execution scope',
'''Can spatial normal-feature distributions provide useful detection and localization
without learned anomaly heads or nearest-neighbor memory? We assess ranking,
normal false alarms, defect recall, native localization, latency and storage.
The four earlier models and all their published results remain preserved.''',
'''from pathlib import Path
import os,sys,json,inspect,time
ROOT=Path.cwd()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt,torch
from IPython.display import display,Markdown
from anomaly_detection.data import CATEGORIES,sha256,autodiscover_data_root
from anomaly_detection.cnn_research import read_partition,load_resized,require_fitting
from anomaly_detection.padim import PaDiMExtractor,SpatialGaussian,aligned_embedding,anomaly_maps
from anomaly_detection.padim_research import extract,score
from anomaly_detection.basic_cnn import seed_everything
torch.set_num_threads(4)
seed_everything(2026)
torch.backends.cuda.matmul.allow_tf32=False
report=ROOT/'reports/padim'
summary=json.loads((report/'summary.json').read_text())
frozen=json.loads((ROOT/'configs/padim_frozen.json').read_text())
print('Completed category-specific models:',summary['models'])
print('Seeds:',summary['seeds'])''',
'''Final models use only the frozen fit-normal partition. The benchmark is a
comparison under this repository's protocol; it does not reproduce the paper's
different crop, interpolation, data partition or ten-model ensemble exactly.''')

section(2,'Clean restart and retained foundation',
'''The unsuccessful Milestone 7 attempt was removed before rebuilding. Inspection
identified only untracked PaDiM files and scratch artifacts; no shared tracked
modifications needed reverting. The audit records each precise cleanup action.''',
"display(Markdown((report/'restart_audit.md').read_text()))",
'''Git returned to the Milestone 6 baseline before any new implementation. Data,
the existing virtual environment, notebooks 00–04, shared metrics, checkpoints,
reports and all earlier experiment records were retained throughout this work.''')

section(3,'Exact environment and GPU verification',
'''All jobs used the existing project interpreter. The saved report captures the
actual CUDA build, GPU and package versions; this live cell checks the active
kernel independently. No environment recreation or PyTorch reinstall occurred.''',
'''print('Kernel executable:',sys.executable)
assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
assert torch.cuda.is_available()
print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))
display(pd.Series(json.loads((report/'environment.json').read_text())))''',
'''Allocator VRAM counters describe tensors/reservations, not total device
occupancy. Process RSS includes CPU arrays and native evaluation work. GPU
timings synchronize CUDA, and latency excludes reading compressed image files.''')

section(4,'Paper, architecture and statistical assumptions',
r'''Defard et al. [PaDiM](https://arxiv.org/abs/2011.08785), equations 1–2, fit a
Gaussian at each spatial position. For embeddings $x_p^n$, use
$\mu_p=N^{-1}\sum_n x_p^n$ and
$\Sigma_p=(N-1)^{-1}\sum_n(x_p^n-\mu_p)(x_p^n-\mu_p)^T+\epsilon I$.
Score $d_p(x)=\sqrt{(x_p-\mu_p)^T\Sigma_p^{-1}(x_p-\mu_p)}$.
The [Anomalib architecture source](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/padim/torch_model.py)
was also reviewed. This implementation was written independently.''',
'''print(inspect.getsource(SpatialGaussian))
# Analytic two-dimensional illustration, unrelated to benchmark fitting/selection.
axis=np.linspace(-4,4,160)
xx,yy=np.meshgrid(axis,axis);grid=np.stack([xx,yy],-1)
covariance=np.array([[4.,1.8],[1.8,1.]])+.01*np.eye(2)
distances=np.sqrt(np.einsum('...i,ij,...j->...',grid,np.linalg.inv(covariance),grid))
fig,ax=plt.subplots(figsize=(5,4))
ax.contour(xx,yy,distances,levels=[1,2,3,4],colors=['#8dc6db','#56a3c3','#367b9f','#244c6e'])
for point,color in [(np.array([1.5,1.]),'green'),(np.array([1.5,-1.]),'red')]:
    distance=np.sqrt(point@np.linalg.solve(covariance,point))
    ax.scatter(*point,color=color,label=f'Same Euclidean radius; Mahalanobis={distance:.2f}')
ax.set(xlabel='Feature channel A',ylabel='Feature channel B',title='Correlation changes what counts as unusual',aspect='equal')
ax.legend(fontsize=7);fig.tight_layout();plt.show();plt.close(fig)''',
'''Full covariance captures correlations among shallow and deep channels. A
large distance signals an unusual direction after accounting for normal
variation. Scores are distances, not calibrated probabilities. Spatial alignment,
Gaussian assumptions and limited fit samples constrain the method.''')

section(5,'Frozen normal partitions and data provenance',
'''The existing seed-2026 70/15/15 normal partition manifests remain unchanged.
Fit estimates all Gaussians; development selects configuration using synthetic
proxies; reserved normals calibrate thresholds only after selection is frozen.''',
'''display(pd.DataFrame([dict(category=c,**{p:len(read_partition(p,c)) for p in ['fit','development','calibration','test']}) for c in CATEGORIES]))
fit_rows=read_partition('fit','bottle');dev_rows=read_partition('development','bottle')
require_fitting(fit_rows)
assert not set(fit_rows.path)&set(dev_rows.path)
print('Dataset validation SHA256:',sha256(ROOT/'reports/dataset_validation.json'))''',
'''Earlier descriptive EDA saw test annotations. We therefore claim no test-label
tuning, not unseen test labels. Each category retains all official test normals
and defects; difficult categories are not omitted from macro results.''')

section(6,'Preprocessing and paper fidelity',
'''Use whole-image RGB PIL bilinear resize to 256 square, followed by the existing
V1 weight transform with identity resize/crop geometry and ImageNet mean/std.
The paper used 224 center cropping and bicubic interpolation. This exception
preserves shared geometry and keeps borders for model comparison.''',
'''images=load_resized(fit_rows.iloc[:24],256)
extractor=PaDiMExtractor(100,2026).cuda()
print('Input uint8:',tuple(images.shape))
print('Official weight transform with geometry override:',extractor.transform)
print('Weight SHA256:',extractor.weight_sha256)
assert extractor.weight_sha256==frozen['weight_sha256']''',
'''There is no learned normalization from development, calibration or test
statistics. Whole-image resizing can distort aspect ratios. This shared geometry
and the held-out normal partitions prevent direct equality with paper numbers.''')

section(7,'Frozen ResNet18 multiscale features',
'''Layer1, layer2 and layer3 encode successively coarser context. We reuse the
verified frozen extractor; all parameters and BatchNorm statistics remain fixed.
Nearest alignment repeats coarse cells at the finest spatial grid.''',
'''with torch.inference_mode():
    raw=extractor.feature_maps(images[:1].cuda().float()/255)
    aligned=aligned_embedding(raw)
print('Layer shapes:',[tuple(x.shape) for x in raw])
print('Concatenated shape:',tuple(aligned.shape))
print(inspect.getsource(aligned_embedding))
assert not any(p.requires_grad for p in extractor.parameters())''',
'''The measured 448-channel embedding joins 64, 128 and 256 channels at a
64-square grid. There is no PatchCore 3x3 local pooling. One Gaussian belongs
to each of 4096 positions, preserving correlations across all selected levels.''')

section(8,'Seeded dimensionality reduction',
'''Uniform channel sampling without replacement reduces covariance storage and
fitting cost. A single subset is fixed for every image and location of a model.
Only the channel subset changes across the three final model seeds.''',
'''features=extract(extractor,images)
print('Selected fit tensor:',tuple(features.shape))
print('First selected channel indices:',extractor.indices[:20].cpu().tolist())
assert extractor.indices.unique().numel()==100
print(inspect.getsource(PaDiMExtractor))''',
'''Persistent covariance storage scales as positions times dimension squared.
With dimension 100 this is about 164 MB in float32, plus means. Inference cost
is independent of the number of fit normals once Gaussian parameters are saved.''')

section(9,'Live normal-only Gaussian estimation',
'''This cell performs an explicit normal-only replay using the same first 24 fit
images as the pilot. It is a mathematical demonstration, not a new final fit or
benchmark. Spatial blocking bounds covariance construction and solve workspace.''',
'''torch.cuda.synchronize();started=time.perf_counter()
gaussian=SpatialGaussian(.01).fit(features,'cuda')
torch.cuda.synchronize()
print('Live demonstration fitting seconds:',time.perf_counter()-started)
print('Mean / Cholesky:',tuple(gaussian.mean.shape),tuple(gaussian.cholesky.shape))
print('Minimum Cholesky diagonal:',float(gaussian.cholesky.diagonal(dim1=-2,dim2=-1).min()))''',
'''The sample covariance uses the unbiased N-1 denominator and fixed additive
0.01 identity regularization. Cholesky factors avoid an explicit matrix inverse.
Regularization also makes low-sample and constant-feature covariances solvable.''')

section(10,'Live held-out normal scoring and map construction',
'''Eight development normals are scored here through triangular solves, bilinear
upsampling and fixed sigma-4 Gaussian smoothing. Multiple right-hand sides avoid
repeating a full covariance tensor for every image in a scoring batch.''',
'''dev_features=extract(extractor,load_resized(dev_rows.iloc[:8],256))
development_maps=score(gaussian,dev_features)
live_scores=development_maps.flatten(1).max(1).values.numpy()
print('Map shape:',tuple(development_maps.shape))
print('Image maxima:',live_scores)
assert torch.isfinite(development_maps).all()
print(inspect.getsource(anomaly_maps))''',
'''Higher distances mean more anomalous. Image scores are maxima of the smoothed
256-square map, fixed before real-test access. Native interpolation is applied
only to continuous maps; the original binary masks remain at native resolution.''')

section(11,'Numerical correctness evidence',
'''Tests compare independent float64 covariance solves, constant singular inputs,
multiple right-hand sides, CPU/GPU parity, seeded selection, serialization,
category checking and smoothing. A tiny synthetic native evaluator fixture also
checks image-resize restoration and exact ranking/region metrics independently.''',
'''print((ROOT/'tests/test_padim.py').read_text())
print('Recorded suite evidence:',json.loads((report/'validation.json').read_text())['tests'])''',
'''These tests establish implementation correctness on known fixtures. The real
normal-image pilot separately establishes pretrained feature extraction and GPU
resource feasibility. Neither test suite nor synthetic proxy quality establishes
benchmark detection performance by itself.''')

section(12,'Actual real-data GPU pilot record',
'''The original registered pilot ran before development and final evaluation.
This cell reads that immutable measurement; the live demonstration timings above
are separate executions. The pilot required finite scores and bounded resources.''',
'''pilot=json.loads((report/'runs/05-pilot-bottle-s2026-v1-result.json').read_text())
display(pd.Series(pilot))
np.testing.assert_allclose(live_scores,pilot['image_scores'],rtol=1e-5,atol=1e-5)
assert pilot['passed']''',
'''The pilot used 24 actual bottle fitting normals and eight development normals.
Its raw and selected feature shapes, extraction/fitting/scoring duration, RSS,
VRAM and score statistics are recorded rather than estimated. Passing this gate
authorized the bounded study under the predeclared research protocol.''')

section(13,'All bounded development experiments',
'''Four configurations vary dimension 50/100 and epsilon 0.01/0.1 over bottle,
cable, carpet and tile. The inner 20% fit holdout supplies identical synthetic
probes; reserved development normals supply diagnostic q95 thresholds. Probe
sources never enter the fitted Gaussian used for this development comparison.''',
'''development=[json.loads(p.read_text()) for p in sorted((report/'runs').glob('05-dev-*-result.json'))]
display(pd.DataFrame([dict(category=r['category'],**r['configuration'],proxy_image_auroc=r['proxy_image_auroc'],proxy_pixel_ap=r['proxy_pixel_ap'],objective=r['selection_objective'],proxy_recall=r['proxy_recall'],normal_development_fpr=r['development_fpr']) for r in development]))
display(pd.DataFrame(frozen['development']))
print('All registered candidates:',len(development))''',
'''The selected mean synthetic objective favors dimension 100 and epsilon 0.01.
Cable and tile proxy performance is weaker than bottle; these results remain
in the record. No new configuration was added in response to final-test scores.''')

section(14,'Prospective configuration freeze',
'''Before calibration or test access, freeze dimensions, covariance policy,
smoothing, image aggregation, source identities, weight checksum, environment,
dataset validation and all three seeds. The running evaluator checks these hashes
before each category/seed so accidental source changes stop execution.''',
'''from anomaly_detection.padim_evaluation import verify_freeze
verified,identity=verify_freeze()
assert identity==summary['frozen_configuration_sha256']
print(json.dumps(verified,indent=2))''',
'''The recorded Git base was the completed Milestone 6 commit with new PaDiM
files still uncommitted. Exact source hashes identify the evaluated implementation
despite that dirty worktree. Publication happens only after artifact validation.''')

section(15,'Final fitting and checkpoint provenance',
'''Every final Gaussian uses all fit normals from one category and its own seeded
channel subset. Checkpoints retain category, configuration, Gaussian parameters
and fit-source identity. This artifact-analysis cell loads a completed checkpoint;
it does not refit it or re-evaluate official test images.''',
'''first=json.loads((report/'evaluation/05-test-bottle-s2026-v1-result.json').read_text())
assert sha256(ROOT/first['checkpoint'])==first['checkpoint_sha256']
restored,saved=SpatialGaussian.load(ROOT/first['checkpoint'],'bottle')
print('Saved category:',saved['category'],'configuration:',saved['configuration'])
print('Fit provenance:',saved['provenance'])
runs=pd.read_csv(report/'all_runs.csv')
display(runs[['category','seed','training_seconds','checkpoint_bytes','train_peak_vram_gib']])
del restored''',
'''All 45 checkpoints are retained locally under checkpoints/padim and excluded
from Git. Gaussian storage is larger than small learned heads, but inference
requires no stored normal-image memory bank. Shared ImageNet weights are reused.''')

section(16,'Normal-only calibration and threshold uncertainty',
'''Thresholds are set from reserved real normals after the model is fixed. Image
alarms use the higher empirical q95 and strict greater-than. Pixels use a separate
pooled q99.5 at 256 square. A nominal target is not a population FPR guarantee.''',
'''calibration=first['calibration']
assert float(np.quantile(calibration['scores'],.95,method='higher'))==calibration['image_threshold']
display(runs[['category','seed','calibration_images','image_threshold','pixel_threshold','recall','fpr']])
fig,ax=plt.subplots(figsize=(8,3))
selected=runs[runs.seed.eq(2026)]
ax.bar(selected.category,selected.fpr);ax.axhline(.05,color='red',linestyle='--',label='Nominal target')
ax.set(ylabel='Normal test FPR',title='Calibration transfer at fixed thresholds');ax.tick_params(axis='x',rotation=65);ax.legend();fig.tight_layout();plt.show();plt.close(fig)''',
'''Calibration samples can be small and unrepresentative of test normals. Report
actual false alarms with recall, even when AUROC is high. Bootstrap intervals
condition on the fitted models and frozen thresholds and exclude calibration
sampling uncertainty; seed standard deviation measures a different variation.''')

section(17,'Complete native benchmark and uncertainty',
'''Each official test category was evaluated once for each registered seed.
Ranking metrics use raw distances and original-resolution masks. Exact pixel
point metrics use a memory-bounded native sweep; pixel bootstrap intervals use
4096 calibration-scaled bins and resample whole images, not independent pixels.''',
'''display(pd.DataFrame(summary['macro']).T)
display(pd.read_csv(report/'all_category_means.csv'))
display(pd.read_csv(report/'macro_by_seed.csv'))
display(pd.DataFrame(summary['pixel_bootstrap_quantization_bias']).T)
assert len(runs)==45 and runs.groupby('category').size().eq(3).all()''',
'''Macro means weight categories equally. Pixel AP remains prevalence-sensitive;
AU-PRO gives equal weight to 8-connected real defect regions over background FPR
0–0.30. The binned-minus-exact diagnostic quantifies the approximation in pixel
uncertainty; the primary point estimates are always exact native metrics.''')

section(18,'Comparison against all four completed models',
'''Use precise stored results for Basic CNN, autoencoder, PatchCore variant and
SuperSimpleNet Lite. Paired image intervals use identical category/seed/image
resamples and fixed fitted models. Inspect detection, localization, recall, FPR
and runtime together rather than treating AUROC as the sole outcome.''',
'''comparison=pd.read_csv(report/'model_macro_comparison.csv',index_col=0)
display(comparison)
display(pd.read_csv(report/'paired_image_difference_intervals.csv'))
display(pd.read_csv(report/'model_latency_comparison.csv'))
fig,ax=plt.subplots(figsize=(10,3.6))
comparison.loc[['image_auroc','pixel_auroc','pixel_ap','au_pro']].T.plot.bar(ax=ax)
ax.set(ylim=(0,1.05),ylabel='Macro score');ax.tick_params(axis='x',rotation=15);fig.tight_layout();plt.show();plt.close(fig)''',
'''These comparisons describe this benchmark and its implementation/budget
choices. A deployment choice informed by these test results would need independent
confirmation data. Geometry and training budgets differ from original papers,
so literature scores are context rather than interchangeable baselines.''')

section(19,'Retrospective failure and small-defect analysis',
'''The following diagnostic tables are generated only after all confirmations.
They retain threshold errors, mask-area strata and defect types. This analysis
cannot change hyperparameters or thresholds. The displayed example policy uses
strongest FP and weakest FN margins at seed 2026, with successful controls.''',
'''examples=pd.read_csv(report/'failure_examples.csv')
display(examples[['category','outcome','defect','score_minus_threshold','defect_fraction']])
display(pd.read_csv(report/'small_defect_summary.csv'))
display(pd.read_csv(report/'texture_object.csv'))
display(pd.read_csv(report/'defect_type_size_summary.csv').sort_values('recall').head(20))
fig,ax=plt.subplots(figsize=(10,3))
category_means=pd.read_csv(report/'all_category_means.csv').set_index('category')
category_means[['recall_mean','fpr_mean']].plot.bar(ax=ax);ax.set(ylabel='Rate',ylim=(0,1.05));fig.tight_layout();plt.show();plt.close(fig)''',
'''These retrospective strata do not establish causality. Small defects may be
diluted by receptive fields and smoothing; object misalignment may violate
spatial Gaussian assumptions. No resolution, alignment or smoothing counterfactual
was run, so those explanations remain hypotheses rather than measured causes.''')

n.cells.append(md('''### Local qualitative failure panels

These licensed original-image panels remain only in the full local executed
notebook and ignored output directory. Public output is cleared for this cell.
Cases are selected by the already declared error-margin policy, with no new
threshold or model selection. Each map uses its own raw distance color scale.'''))
panel=code('''from PIL import Image
data_root=autodiscover_data_root(ROOT)
chosen=[]
for category in ['bottle','cable','screw']:
    group=examples[examples.category.eq(category)]
    for outcome in ['FP','FN','TP']:
        subset=group[group.outcome.eq(outcome)]
        if not len(subset) and outcome=='FN':subset=group[group.outcome.eq('TN')]
        if len(subset):chosen.append(subset.iloc[0])
fig,axes=plt.subplots(len(chosen),4,figsize=(12,2.8*len(chosen)),squeeze=False)
for row,axs in zip(chosen,axes):
    with Image.open(data_root/row.path) as image:rgb=np.asarray(image.convert('RGB'))
    if isinstance(row['mask'],str):
        with Image.open(data_root/row['mask']) as image:truth=np.asarray(image)>0
    else:truth=np.zeros(rgb.shape[:2],bool)
    small=np.load(ROOT/row.maps_path,mmap_mode='r')[int(row.row_index)]
    native=torch.nn.functional.interpolate(torch.from_numpy(np.array(small))[None,None],size=rgb.shape[:2],mode='bilinear',align_corners=False).numpy()[0,0]
    axs[0].imshow(rgb);axs[0].set_title(f'{row.category}: {row.outcome} ({row.defect})',fontsize=9)
    axs[1].imshow(truth,cmap='gray');axs[1].set_title('Original binary annotation')
    axs[2].imshow(native,cmap='inferno');axs[2].set_title(f'Image margin {row.score_minus_threshold:.2f}')
    axs[3].imshow(rgb);axs[3].imshow(np.ma.masked_where(native<=row.pixel_threshold,native),cmap='autumn',alpha=.6);axs[3].set_title('Frozen pixel threshold overlay')
    for ax in axs:ax.axis('off')
fig.tight_layout();fig.savefig(ROOT/'outputs/padim/failure-panels.png',dpi=110)
plt.show();plt.close(fig)''')
panel.metadata['contains_dataset_images']=True
n.cells.extend([panel,md('''The same original mask geometry is used for scoring and visual inspection.
The overlay threshold is separately calibrated on normal pixels. A visually
bright map is not a probability estimate, and these selected cases are qualitative
examples rather than a representative random sample of benchmark outcomes.''')])

n.cells.append(md('''### Observed panel review after confirmation

Bottle's false positive concentrates on the normal interior; its contamination
control responds at the foreign object. Cable's missing-wire false negative has
no frozen-threshold overlay, while the bent-wire true positive broadly covers
the displaced wires. Screw's scratch-neck false negative does not alarm; its
manipulated-front true positive responds near the damaged tip, with extra border
responses. Per-image color scales make bright maps incomparable across examples.
See the [visual review](../reports/padim/observed_panel_analysis.md). These are
selected observations and do not isolate the causes of aggregate errors.'''))

section(20,'Resources, limitations and reproducibility',
'''Latency uses 50 warmups and 200 synchronized batch-one float32 iterations.
Feature extraction and Gaussian scoring form model time; smoothing, native
interpolation and CPU transfer are postprocessing. Fit and evaluation resources
are recorded separately. Local licensed images and large arrays are not published.''',
'''display(pd.Series({k:v for k,v in summary.items() if k!='macro'}))
display(Markdown((report/'findings.md').read_text()))
display(Markdown((ROOT/'docs/padim_reproduction.md').read_text()))
del extractor,gaussian,features,dev_features,raw,aligned
torch.cuda.empty_cache()''',
'''PaDiM now has a complete verified record under the shared protocol. Remaining
limitations include held-out calibration size, pretrained domain transfer,
Gaussian/spatial assumptions, variant fidelity and conditional uncertainty.
EfficientAD remains unstarted; this notebook makes no production-readiness claim.''')

nbf.validate(n)
(ROOT/'notebooks/05_padim.ipynb').write_bytes(nbf.writes(n).encode('utf-8'))
print('Built Notebook 05; execute after all result artifacts are validated')
