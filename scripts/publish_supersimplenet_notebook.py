"""Execute artifact analysis separately; preserve genuine pilot session outputs."""
import os,json,asyncio,sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];report=ROOT/'reports/supersimplenet'
development_only='--development' in sys.argv
sys.path.insert(0,str(ROOT/'scripts'))
from run_supersimplenet import development_identifier
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
tail=nbf.v4.new_notebook();tail.metadata.kernelspec=dict(display_name='MVTec research (CUDA, Python 3.12)',language='python',name='mvtec-research')
tail.cells=[md('''# Continued record: bounded development and frozen confirmation
The preceding pilot contains genuine live training. This continuation **executes artifact analysis** of separately recorded training/evaluation jobs. It does not claim those jobs trained live in this kernel. Execution counts restart for this separate analysis session and are preserved as produced by the kernels. Earlier pending statements describe the chronological pilot state; completed phases below supersede them.

## Implementation exposed as code
The following definitions expose the feature alignment, adapter, synthetic generation, heads, losses, inference, cache and actual fitting loop used by the recorded jobs. The original modules and exact source hashes remain the authoritative job implementation. In this notebook demonstration only, extractor file-root lookup is expressed as the already verified `ROOT` variable. No training job is rerun here.

For a dense grid of size H×W, embedding width D, segmentation width h_s and detector width h_d, learned-head convolution cost is proportional to HW[D² + D h_s + 25(D+1)h_d], in addition to the frozen CNN. Global pooling is linear in grid size. Persistent head weights scale with the bracketed term; feature activations scale with HW(D+h_s+h_d) at batch one. Training doubles the adapted batch before perturbation. Inference never constructs synthetic masks and has no dependence on the number of fitting images or a patch bank. Upsampling and native-map transfer add output-resolution costs and remain in end-to-end timing.
'''),code('''from pathlib import Path
import sys,json,math
ROOT=Path.cwd()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt,torch
from PIL import Image
from IPython.display import display,Markdown
from anomaly_detection.data import CATEGORIES,autodiscover_data_root,sha256
torch.set_num_threads(4)
report=ROOT/'reports/supersimplenet'
frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())
summary=json.loads((report/'summary.json').read_text())
display(pd.DataFrame([dict(category=c,**{p:len(__import__('anomaly_detection.cnn_research',fromlist=['read_partition']).read_partition(p,c)) for p in ['fit','development','calibration','test']}) for c in CATEGORIES]))
print('Frozen weights:',frozen['weight_sha256'])
from anomaly_detection.ssn_research import model_for
inspection=model_for(frozen['configuration'])
display(pd.DataFrame([dict(component=name,parameters=sum(p.numel() for p in part.parameters()),trainable=sum(p.numel() for p in part.parameters() if p.requires_grad)) for name,part in [('adapter',inspection.adapter),('segmentation',inspection.seg),('detection convolution',inspection.det),('detection linear',inspection.fc)]]))
print('Shared frozen extractor parameters:',json.loads((report/'backbone_storage.json').read_text())['parameters'])
del inspection
'''),code((ROOT/'src/anomaly_detection/supersimplenet.py').read_text(encoding='utf-8').replace('Path(__file__).resolve().parents[2]', 'ROOT')),
md('''### Actual data provenance, cache and end-to-end optimization
`require_fitting` checks fitting-normal provenance before cache construction. Each category owns its own learned heads and immutable checkpoint. Frozen extraction is cached per original image/weight/source identity. Held-out fitting sources create synthetic probes; clean development normals set diagnostic thresholds. Reserved calibration does not enter development. The explicit loop retains every epoch, including weak or unstable outcomes.'''),
code((ROOT/'src/anomaly_detection/ssn_research.py').read_text(encoding='utf-8').replace('from .','from anomaly_detection.'))]
tail.cells.extend([md('''### Durable final fitting and interruption recovery
The externally interrupted focal-only cable trial stopped after two epochs without a checkpoint. Its partial history and interruption record remain; only that incomplete trial restarted as v2. Completed trials were never retrained. Final fitting adds an atomic epoch checkpoint with model, optimizer, scheduler and RNG states. A CPU test verifies resumed optimization exactly matches uninterrupted optimization. The computation inside each epoch is the same tested `train_epoch` used in development; this wrapper changes recovery and recording only.
'''),code((ROOT/'src/anomaly_detection/ssn_final.py').read_text(encoding='utf-8').replace('from .','from anomaly_detection.')),
md('''### Additional measured CUDA verification — artifact replay
The complete forward/backward pass and checkpoint/RNG recovery were also verified on real fitting-normal cached features. Repeating extraction and replaying the stored compatible cache produced exactly the same tensor in this check. This correctness experiment did not inform candidate selection or access reserved calibration/test images.'''),
code("print(json.dumps(json.loads((report/'cuda_verification.json').read_text()),indent=2))")])
for candidate in ['reference_noise','noise05','frozen_adapter','focal_only']:
    for category in ['bottle','cable','carpet','tile']:
        identifier=development_identifier(candidate,category)
        if candidate=='focal_only' and category=='cable':
            tail.cells.extend([md('''## Interrupted attempt 04-dev-focal_only-cable-s2026-v1
**Hypothesis:** removing the balanced margin tests whether it impedes short-budget segmentation learning on cable. The reserved run was externally interrupted after two epochs and before a checkpoint was saved. This artifact replay preserves those actual epochs; it does not treat the interruption as evidence against the configuration. [Interruption record](../experiments/04-dev-focal_only-cable-s2026-v1-interrupted.json).
'''),code("display(pd.read_csv(report/'runs/04-dev-focal_only-cable-s2026-v1-history.csv'))"),
md('''**Decision:** preserve the two-epoch history and restart only this incomplete trial as v2 with the same seed/configuration. Completed configurations and extracted features are reused. The independently completed v2 trial follows below.''')])
        r=json.loads((report/f'runs/{identifier}-result.json').read_text())
        common=json.loads((report/f'04-common-{candidate}-{category}-s2026-v1.json').read_text())
        decision=json.loads((ROOT/f'experiments/{identifier}-decision.json').read_text())
        best=max(common['candidates'],key=lambda k:common['candidates'][k]['objective']);measured=common['candidates'][best]
        tail.cells.extend([md(f'''## Experiment {identifier}: hypothesis before the recorded job
{r['configuration']['hypothesis']}. Category **{category}**, fixed seed2026, 40 epochs, final checkpoint. Fitting/probe sources are disjoint. This artifact-analysis cell reads the completed [immutable experiment](../experiments/{identifier}-result.json), actual history and common-proxy evaluation. Training-curve proxies use the training magnitude; the final configuration comparison uses identical .05 proxies for every candidate.
'''),code(f'''r=json.loads((report/'runs/{identifier}-result.json').read_text())
common=json.loads((report/'04-common-{candidate}-{category}-s2026-v1.json').read_text())
print(json.dumps(r['configuration'],indent=2));display(pd.DataFrame(common['candidates']).T)
print('Resources:',r['resources'],'adapter relative weight drift:',r['adapter_relative_weight_drift'])
history=pd.DataFrame(r['history'])
fig,axes=plt.subplots(1,2,figsize=(10,2.8))
history.plot(x='epoch',y=['train_loss','margin_loss','segmentation_focal','detection_focal'],ax=axes[0])
history.plot(x='epoch',y=['proxy_image_auroc','proxy_pixel_ap'],marker='.',ax=axes[1]);plt.tight_layout();plt.show();plt.close(fig)
'''),md(f'''**Measured result and interpretation:** common-proxy best aggregation `{best}` has image AUROC **{measured['auroc']:.6f}**, pixel AP **{measured['pixel_ap']:.6f}**, objective **{measured['objective']:.6f}**. The fixed training loop took **{r['resources']['seconds']:.2f}s**. Final balanced margin diagnostic was **{r['history'][-1]['margin_loss']:.6f}**; this remains a diagnostic even when omitted from the focal-only objective. Adapter relative weight drift was **{r['adapter_relative_weight_drift']:.6f}**. {decision['interpretation']}. Decision: **{decision['decision']}**. These synthetic measurements do not establish real-defect quality.
''')])
tail.cells.extend([md('''## Common-proxy selection and measured execution gate
The registered small study ended after its predefined16 trials. Reference width was rejected by the pilot compute gate, not by official defect scores. Compare common noise/masks and every planned image aggregation. Within .005 of the best objective, prefer lower training time then parameters. Freeze all model, training, preprocessing, scoring and calibration settings before any official test access. Seed count follows measured training cost and is explicit below.
'''),code('''display(pd.read_csv(report/'development_macro.csv'))
display(pd.read_csv(report/'development_common_proxy.csv'))
print(json.dumps(frozen,indent=2))
print(json.dumps(json.loads((report/'compute_budget.json').read_text()),indent=2))
assert sha256(ROOT/'configs/supersimplenet_frozen.json')==summary['frozen_configuration_sha256']
print('Source identities verified:',all(sha256(ROOT/p)==h for p,h in frozen['source_sha256'].items()))
'''),md('''## Adapter behavior: post-freeze descriptive analysis
These measurements reuse clean cached held-out fitting-normal features on CPU. They compare the original seeded adapter with the trained checkpoint: relative weight drift, spatial feature variation and RMS magnitude. No pretrained extraction or fitting is repeated. The frozen-adapter control must replay its initial weights exactly. This analysis does not change selection; the adapter/loss ablations at .015 do not isolate their effects at the selected .05 magnitude.
'''),code("display(pd.read_csv(report/'adapter_diagnostics.csv')[['candidate','category','relative_weight_drift','trained_to_initial_weight_norm_ratio','initial_adapter_spatial_std','trained_adapter_spatial_std','trained_to_initial_spatial_std_ratio']])"),md('''## Threshold calibration and frozen evaluation implementation
Image threshold is reserved-normal higher q95 with strict `>` decisions. The separate pixel threshold is higher pooled q99.5 of unmodified normal256 maps. The learned detector and dense map are separate scores. Calibration is saved before test images load; no synthetic features enter either calibration distribution. No test-derived normalization is used.

The implementation below reuses shared image/native-pixel metrics and AU-PRO. Primary point metrics are exact on original mask dimensions. Only pixel bootstrap intervals use4096 fixed sigmoid bins; the quantization bias is reported. Image intervals and paired differences use identical whole-image draws across category-matched models. They condition on fitted models and fixed thresholds; they do not include model fitting or threshold estimation uncertainty.
'''),code((ROOT/'src/anomaly_detection/ssn_evaluation.py').read_text(encoding='utf-8').replace('from .','from anomaly_detection.')),
md('''## Independent all-category fitting: actual loss curves
Every category has independent adapter/segmentation/detection parameters. All registered fits finished before the first official test evaluation. These are recorded final-epoch histories, not new training in this kernel. Weak categories and seeds remain visible.'''),code('''runs=pd.read_csv(report/'all_runs.csv');display(runs[['category','seed','training_seconds','train_peak_vram_gib','checkpoint_bytes']])
fig,axes=plt.subplots(3,5,figsize=(15,8))
for ax,category in zip(axes.flat,CATEGORIES):
    for seed in frozen['seeds']:
        h=pd.read_csv(report/f'runs/04-final-{category}-s{seed}-v1-history.csv')
        ax.plot(h.epoch,h.train_loss,label=str(seed))
    ax.set_title(category);ax.set_xlabel('Epoch');ax.set_ylabel('Total training loss')
axes.flat[0].legend();plt.tight_layout();plt.show();plt.close(fig)
'''),md('''## All-category native-resolution benchmark
Macro averages give each category equal weight. Per-category/seed records, confusion matrices, per-defect-type scores and pixel operating points remain available. Image AP/precision depend on the benchmark prevalence. Pixel background includes background from normal and defective images. No configuration or threshold was changed after these results.
'''),code('''display(pd.DataFrame(summary['macro']).T)
display(pd.read_csv(report/'all_category_means.csv'))
display(runs[['category','seed','tn','fp','fn','tp','pixel_tn','pixel_fp','pixel_fn','pixel_tp']])
display(pd.read_csv(report/'pixel_operating_category_means.csv'))
display(pd.read_csv(report/'per_defect_results.csv'))
'''),md(r'''## Calibration stability and uncertainty
The q95 normal order statistic is not a distribution-free guarantee of5% false alarms. Small sample sizes make even the maximum calibration score uncertain. Under independent continuous identical normal distributions, if the chosen zero-based order index is $k$, the unseen-normal tail probability has a $\mathrm{Beta}(n-k,k+1)$ distribution across calibration samples, with mean $(n-k)/(n+1)$. The table includes this diagnostic interval; actual distribution shift or tied scores violate its assumptions. Empirical tail bootstrap can miss unseen extremes. Pixel counts are spatially correlated and must not be treated as millions of independent calibration observations.
'''),code('''stability=pd.read_csv(report/'calibration_stability.csv');display(stability)
fig,axes=plt.subplots(1,3,figsize=(12,3))
for ax,category in zip(axes,['leather','carpet','screw']):
    result=json.loads((report/f'evaluation/04-test-{category}-s2026-v1-result.json').read_text())
    images=pd.read_csv(report/f'evaluation/04-test-{category}-s2026-v1-images.csv')
    ax.hist(result['calibration']['calibration_scores'],bins=12,alpha=.6,density=True,label='Calibration normal')
    ax.hist(images.loc[images.label.eq(0),'score'],bins=12,alpha=.5,density=True,label='Test normal')
    ax.axvline(result['calibration']['image_threshold'],color='k',label='Frozen q95');ax.set_title(category)
axes[0].legend(fontsize=7);plt.tight_layout();plt.show();plt.close(fig)
'''),md('''## Normal and defective inference, masks, heatmaps and overlays — local outputs
Panel choices are deterministic: first path for each available FP/FN/TP outcome, highest-score normal, and first scratch when available. Each category is represented. Original dataset images, native masks and large panels are retained only in the fully executed local notebook. Public cell sources and compact numerical diagnostics remain. Heatmap color limits are per-image; brightness across panels does not compare absolute scores. Overlays use the frozen pixel threshold. A segmentation threshold and the primary image aggregation can disagree; neither is retroactively changed.
'''),code('''examples=pd.read_csv(report/'failure_examples.csv');data=autodiscover_data_root(ROOT)
for category in CATEGORIES:
    selected=examples[examples.category.eq(category)]
    fig,axes=plt.subplots(len(selected),4,figsize=(10,2.3*len(selected)),squeeze=False)
    for row,axs in zip(selected.itertuples(),axes):
        with Image.open(data/row.path) as image:rgb=np.asarray(image.convert('RGB'))
        if isinstance(row.mask,str):
            with Image.open(data/row.mask) as image:truth=np.asarray(image)>0
        else:truth=np.zeros(rgb.shape[:2],bool)
        small=np.load(ROOT/row.maps_path,mmap_mode='r')[row.row_index]
        native=torch.nn.functional.interpolate(torch.from_numpy(np.array(small))[None,None],size=rgb.shape[:2],mode='bilinear',align_corners=False).numpy()[0,0]
        axs[0].imshow(rgb);axs[1].imshow(truth,cmap='gray');axs[2].imshow(native,cmap='inferno')
        axs[3].imshow(rgb);axs[3].imshow(np.ma.masked_where(native<=row.pixel_threshold,native),cmap='autumn',alpha=.6)
        axs[0].set_title(f'{category} {row.outcome}: {row.defect}',fontsize=8)
        axs[1].set_title('Native ground truth');axs[2].set_title(f'Map; image margin {row.score_minus_threshold:.3f}');axs[3].set_title('Frozen pixel overlay')
        for ax in axs:ax.axis('off')
    fig.tight_layout();fig.savefig(ROOT/f'outputs/supersimplenet/failures-{category}.png',dpi=100);plt.show();plt.close(fig)
'''),md('''## Observed failures and fine-defect analysis
The numerical diagnostics quantify defect/background map separation and top1% native-pixel overlap in the displayed cases. They do not prove a causal architectural explanation. Small-mask strata and scratch-type results are retrospective descriptions, never selection data. A coarse feature grid and smoothing can limit detail, but no higher-resolution counterfactual was run to isolate that effect.
'''),code('''display(examples)
display(pd.read_csv(report/'small_defect_summary.csv'))
types=pd.read_csv(report/'defect_type_size_summary.csv');display(types[types.defect.str.contains('scratch')])
display(pd.read_csv(report/'texture_object.csv'))
'''),md('''## Comparison with all three completed models
These tables read precise stored results, not rounded prompt values. The macro table preserves each model's original seed coverage; paired intervals instead use matching seeds, categories and image draws. Differences are conditional on fitted models and thresholds. Inspect recall/FPR, dense localization and practical latency together; no superiority claim follows from a small AUROC difference.
'''),code('''display(pd.read_csv(report/'model_macro_comparison.csv',index_col=0))
display(pd.read_csv(report/'paired_image_difference_intervals.csv'))
display(pd.read_csv(report/'paired_prediction_disagreements.csv'))
display(pd.read_csv(report/'model_latency_comparison.csv'))
for family in ['basic_cnn','autoencoder','patchcore']:
    matched=pd.read_csv(report/f'{family}_category_comparison.csv')
    display(matched[matched.category.isin(['leather','carpet','screw'])])
'''),md('''## Compute, storage, limitations and conclusions
Latency uses50 warmups and200 batch-one float32 measurements from memory-resident decoded RGB through a native CPU map and image scalar. File decode is excluded consistently with earlier models. CNN feature extraction and learned heads are timed separately; model-only cost excludes postprocessing. Dense feature caches accelerate fitting and are not an inference dependency. Shared frozen extractor storage is counted once; category heads are separate. Process RSS is sampled20Hz and CUDA peaks use allocator counters, not total device occupancy. Runs were sequential on one laptop GPU; power context and measured variability are retained.
'''),code('''print(json.dumps({k:v for k,v in summary.items() if k not in ['macro','seed_budget_decision']},indent=2))
display(Markdown((report/'findings.md').read_text()))
''')])
if development_only:
    stop=next(i for i,c in enumerate(tail.cells) if c.cell_type=='markdown' and c.source.startswith('## Threshold calibration'))
    tail.cells=tail.cells[:stop]
    tail.cells[1].source=tail.cells[1].source.replace("summary=json.loads((report/'summary.json').read_text())", "summary={'frozen_configuration_sha256':sha256(ROOT/'configs/supersimplenet_frozen.json')}")
    tail.cells.append(md('''## Development milestone status
The complete bounded development record and prospective freeze are now verified. Official test evaluation, final calibration, failure analysis and model-quality comparison are pending. No final benchmark result is claimed. The measured resource gate above determines category/seed coverage before final fitting starts.'''))
for cell in tail.cells:
    cell.metadata['execution_session']='recorded-job-analysis'
    if cell.cell_type=='code' and 'failures-{category}.png' in cell.source:cell.metadata['local_dataset_pixels']=True
out=ROOT/'outputs/supersimplenet';os.environ['IPYTHONDIR']=str(out/'ipython')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
session='development' if development_only else 'analysis'
executed=out/f'04_supersimplenet.{session}.executed.ipynb'
if executed.exists():raise FileExistsError('Preserve prior executed notebook; use a new named recovery if needed')
try:NotebookClient(tail,timeout=900,kernel_name='mvtec-research',resources={'metadata':{'path':str(ROOT)}}).execute()
finally:executed.write_bytes(nbf.writes(tail).encode())
pilot=nbf.read(out/'04_supersimplenet.pilot.executed.ipynb',as_version=4)
for cell in pilot.cells:cell.metadata['execution_session']='architecture-live-pilot'
pilot.cells.extend(tail.cells);nbf.validate(pilot)
full=out/('04_supersimplenet.development.full.executed.ipynb' if development_only else '04_supersimplenet.full.executed.ipynb')
full.write_bytes(nbf.writes(pilot).encode())
for cell in pilot.cells:
    if cell.metadata.get('local_dataset_pixels'):cell.outputs=[]
(ROOT/'notebooks/04_supersimplenet.ipynb').write_bytes(nbf.writes(pilot).encode())
print('Published genuine compact outputs; large original-image panels remain local')
