"""Append executed artifact analysis to the preserved live pilot notebook.

Later training/evaluation ran via the visible library code; this is explicitly
artifact analysis, not a claim to have trained every model inside this kernel.
"""
from pathlib import Path
import os
import nbformat as nbf
from nbclient import NotebookClient
import ast
import csv

root=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
def function_source(relative,name):
    text=(root/relative).read_text()
    node=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name)
    return ast.get_source_segment(text,node)
with (root/'reports/autoencoder/development_macro.csv').open(newline='') as stream:
    development_rows=list(csv.DictReader(stream))
best_by_variant={variant:max([row for row in development_rows if row['variant']==variant],key=lambda row:float(row['objective']))
                 for variant in ['base','deeper','narrow','wide','mse']}
development_markdown=['## Measured ablation interpretations','',
    'Both maps for every category/configuration are reported above. The following',
    'compact table shows each variant’s higher-scoring map on the fixed proxy objective.',
    'It does not select a separate map per category.','',
    '| Variant | Map | Proxy image AUROC | Proxy pixel AP | Objective | Parameters |',
    '|---|---|---:|---:|---:|---:|']
for variant,row in best_by_variant.items():
    development_markdown.append(f"| {variant} | {row['map_kind']} | {float(row['auroc']):.6f} | {float(row['pixel_ap']):.6f} | {float(row['objective']):.6f} | {int(float(row['parameters'])):,} |")
base_objective=float(best_by_variant['base']['objective'])
development_markdown+=['',
    f"The deeper model raised the proxy objective by {float(best_by_variant['deeper']['objective'])-base_objective:.6f} over base,",
    'but depth also reduced latent spatial area fourfold and increased parameter count.',
    'This trial cannot separate context, compression and parameter-count effects.','',
    f"The narrow bottleneck raised the objective by {float(best_by_variant['narrow']['objective'])-base_objective:.6f};",
    f"the wide bottleneck changed it by {float(best_by_variant['wide']['objective'])-base_objective:.6f}.",
    'More capacity did not establish a compelling advantage on this proxy.','',
    f"MSE/squared residuals ranked first, only {float(development_rows[0]['objective'])-float(development_rows[1]['objective']):.6f} above the runner-up.",
    'The four-category, single-development-seed comparison does not establish a',
    'statistically certain architecture ranking. The frozen rule still selects',
    'one policy for confirmation; rejected trials remain visible. No specific',
    'combined-loss or smoothing hypothesis was supported strongly enough to expand',
    'this initial bounded study. Real-defect transfer remains a separate question.']
tail=nbf.v4.new_notebook()
tail.metadata.kernelspec=dict(display_name='MVTec research (CUDA, Python 3.12)',language='python',name='mvtec-research')
tail.cells=[md("""# Continued chronology — bounded development and frozen confirmation

The preceding pilot cells were executed live in the initial notebook kernel.
The following cells execute analysis of retained artifacts from separately run
training/evaluation jobs. **They do not replay training as if it ran live here.**
Training/evaluation source, configs, all curves, failures and provenance are
included below. Raw panels remain local. The initial 'pending' text records the
state at that chronological point; final coverage below supersedes it.
"""),code("""from pathlib import Path
import sys,json,inspect
ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt,torch
from PIL import Image
from anomaly_detection.data import CATEGORIES,TEXTURES,autodiscover_data_root,sha256
from anomaly_detection.cnn_research import read_partition,load_resized
from anomaly_detection.ae_research import fit,reconstruct
from anomaly_detection.ae_evaluation import load_checkpoint,calibrate_model,evaluate_frozen,benchmark_model
from anomaly_detection.autoencoder import anomaly_map,image_scores
from anomaly_detection.metrics import binary_metrics,calibrate_normal
from anomaly_detection.pixel_metrics import exact_pixel_metrics
torch.set_num_threads(4)
report=ROOT/'reports/autoencoder'
freeze=json.loads((ROOT/'configs/autoencoder_frozen.json').read_text())
summary=json.loads((report/'summary.json').read_text())
display(pd.read_csv(report/'development_macro.csv'))
"""),md("""## Bounded training implementation

The source below is the actual loop used for the 20 development fits and 45 final
fits. It trains on unmodified normal fitting images, monitors normal development
loss, records every epoch, and reserves immutable start/result/failure events.
Development excludes an inner fitting holdout from gradients and generates fixed
synthetic probes from that holdout only. Final fits use every fitting normal,
with no calibration/test gradient access. All variants ran exactly 20 epochs;
there is no best-test-epoch selection. The held-out probes are proxies, not
industrial defects. Full optimizer/RNG state is retained locally.

Chronology disclosure: the scalar selection rule was documented after the first
base bottle development result existed, while the remaining jobs were running.
Categories and five ablations were fixed before development. The final policy
was frozen before final fits; no official autoencoder test results informed it.
"""),code((root/'src/anomaly_detection/ae_research.py').read_text().replace('from .','from anomaly_detection.').replace('ROOT = Path(__file__).resolve().parents[2]','')),md("""## Hypotheses, configurations and complete development ledger

Depth 4 tests greater context/compression; latent 2 tests stricter capacity and
latent 32 tests whether extra capacity helps normal reconstruction at the cost
of reconstructing edited regions. MSE tests emphasis on large residuals against
L1. Only one factor changes from the shallow base. A fixed four-category mean of
(proxy image AUROC + pixel AP)/2 chooses variant/map, with recall then parameter
count as tie breakers. Both absolute and squared maps are reported for every fit.
Raw L1 and MSE values have different units and are not ranked against each other.
"""),code("""development=pd.read_csv(report/'development_comparison.csv')
display(development)
events=[json.loads(p.read_text()) for p in sorted((ROOT/'experiments').glob('02-*.json'))]
display(pd.DataFrame([dict(experiment=e['experiment_id'],**e['hyperparameters']) for e in events
                      if e['experiment_id'].startswith('02-dev-') and e['status']=='started']))
shapes=[]
for variant in ['base','deeper','narrow','wide','mse']:
    event=next(e for e in events if e['experiment_id']==f'02-dev-{variant}-bottle-s2026-v1-start')
    c=event['hyperparameters']
    architecture=ConvAutoencoder(**{k:c[k] for k in ['width','depth','latent_channels']})
    with torch.inference_mode():
        x=torch.zeros(1,3,256,256)
        encoded=architecture.encoder(x);z=architecture.bottleneck(encoded);y=architecture.decoder(z)
    shapes.append(dict(variant=variant,input=list(x.shape),encoded=list(encoded.shape),
        bottleneck=list(z.shape),output=list(y.shape),parameters=sum(p.numel() for p in architecture.parameters()),
        scalar_compression=x.numel()/z.numel()))
display(pd.DataFrame(shapes))
print('Depth 4 also reduces bottleneck spatial area fourfold; this ablation does not isolate depth from compression.')
recomputed=development.groupby(['variant','map_kind']).objective.mean().sort_values(ascending=False)
print('Recomputed selection objective:',recomputed.to_string())
assert recomputed.index[0]==(freeze['selected_variant'],freeze['configuration']['map_kind'])
display(pd.DataFrame([dict(id=e['experiment_id'],status=e['status'],hypothesis=e['hypothesis'],decision=e['decision']) for e in events]))
failures=[e for e in events if e['status']=='failed']
print('Failed trials:',len(failures))
for e in failures: print(e['experiment_id'],e['interpretation'])
"""),md("""## Every development learning curve

Each panel includes all epochs from one variant/category. Training and normal
development use the same reconstruction loss within a panel. Cross-panel loss
magnitudes for MSE versus L1 are not comparable. No curves are omitted based on
their appearance. Fixed epochs bound compute; a low reconstruction loss alone
does not imply useful anomaly detection.
Loss axes are logarithmic to keep late-epoch changes visible.
"""),code("""fig,axes=plt.subplots(5,4,figsize=(16,16))
for i,variant in enumerate(['base','deeper','narrow','wide','mse']):
    for j,category in enumerate(['bottle','cable','carpet','tile']):
        h=pd.read_csv(report/f'runs/02-dev-{variant}-{category}-s2026-v1-history.csv')
        axes[i,j].plot(h.epoch,h.train_loss,label='fit')
        axes[i,j].plot(h.epoch,h.normal_development_loss,label='normal dev')
        axes[i,j].set_title(variant+' / '+category);axes[i,j].legend();axes[i,j].set_yscale('log')
plt.tight_layout();plt.show()
"""),md("""## Development decision and measured compute gate

The following freeze and budget were written before all-category training and
official test evaluation. The fixed proxy rule determines selection; real test
results cannot revise it. There was no evidence-based need to add a combined
loss, smoothing or resolution grid in this initial bounded baseline. Their
absence is not a claim that they would fail. Fixed 256-square inputs can erase
small details and distort aspect ratios, a limitation retained in the benchmark.
"""),code("""print(json.dumps(freeze,indent=2))
print(json.dumps(json.loads((report/'compute_budget.json').read_text()),indent=2))
selected=pd.read_csv(report/'development_macro.csv').iloc[0]
print('Selected using proxy objective:',selected.variant,selected.map_kind,selected.objective)
print('Proxy advantage is not a real-defect performance guarantee.')
"""),md("""## Threshold calibration and uncertainty

The API below accepts only a category's reserved normal calibration partition.
It takes no official test scores, labels or masks. Image alarms use the shared
higher empirical 95th percentile and strict > comparison, target FPR 5%.
Localization uses a separately calibrated pooled normal-pixel q99.5, rounded
up to a 65,536-bin boundary. Image and pixel thresholds are not interchangeable.
Bootstrap intervals resample whole normal images for the image threshold.
Small calibration counts and distribution shift prevent a population guarantee.
"""),code(function_source('src/anomaly_detection/ae_evaluation.py','calibrate_model')),md("""## Native-resolution metrics and measured inference

The evaluator freezes checkpoint/configuration identities, writes calibration
before reading official test rows, then restores continuous maps by bilinear
interpolation to native resolution. Ground-truth masks are never interpolated
for primary metrics. Exact native pixel AUROC/AP/AU-PRO use the same shared
implementation as Notebook 01. AU-PRO weights 8-connected regions equally and
integrates background FPR through 0.30. Image scores remain at model resolution.
The implementation and latency scope are printed for review. Each latency run
uses 50 warmups and 200 synchronized timed batch-one iterations; disk decode
is excluded. Residual-map construction is included in the model stage.
Inference allocation also includes the evaluator's resident test-image cache,
as in Notebook 01; it is not an isolated deployment-process memory estimate.
"""),code("__file__ = str(ROOT/'src/anomaly_detection/ae_evaluation.py')\n"+(root/'src/anomaly_detection/ae_evaluation.py').read_text().replace('from .','from anomaly_detection.')+'\n'+(root/'src/anomaly_detection/pixel_metrics.py').read_text()),md("""## All-category results and calibration operating points

These are retained one-shot frozen evaluations, not a fresh test inference pass.
All 15 categories and three seeds are included. Unweighted category macro
averages give each category equal weight. Seed standard deviation describes
training variation; whole-image bootstrap intervals describe sampling uncertainty.
Pixel intervals use 4,096 fixed residual bins; exact point estimates and measured
quantization bias are reported separately. Pixels are not independent samples.
Undefined precision when there are no alarms follows the shared zero convention
and is explicitly counted in the summary.
"""),code("""runs=pd.read_csv(report/'all_runs.csv')
display(pd.read_csv(report/'all_category_means.csv'))
display(pd.DataFrame(summary['macro']).T)
display(runs[['category','seed','calibration_images','image_threshold','recall','fpr','precision','f1','tn','fp','fn','tp']])
print(json.dumps(summary,indent=2))
display(pd.read_csv(report/'texture_object.csv'))
"""),md("""## Matched comparison with Basic CNN

Values come from exact stored CSV/JSON results, not rounded reference values.
Both families use the same official partitions, 256-square inputs, three seeds,
native metric implementation and normal calibration policy. CNN used synthetic
segmentation supervision; the autoencoder trains only to reconstruct unedited
normals. Map/aggregation policies differ and are disclosed, so this comparison
measures the complete frozen pipelines. Higher AUROC alone is not sufficient:
compare recall, FPR, localization AP/AU-PRO and latency together. Image AP depends
on the benchmark's high defect prevalence and is not production precision.
Training cost reflects the frozen 20 AE epochs versus eight CNN epochs.
"""),code("""comparison=pd.read_csv(report/'cnn_macro_comparison.csv',index_col=0)
display(comparison)
display(pd.read_csv(report/'cnn_category_comparison.csv'))
display(pd.read_csv(report/'cnn_category_cost_comparison.csv'))
cnn=json.loads((ROOT/'reports/basic_cnn/summary.json').read_text())
display(pd.DataFrame({'Basic CNN':{k:cnn[k] for k in ['mean_model_p50_ms','mean_end_to_end_p50_ms','peak_training_vram_gib','parameters','max_checkpoint_bytes','total_training_seconds']},
                      'Autoencoder':{k:summary[k] for k in ['mean_model_p50_ms','mean_end_to_end_p50_ms','peak_training_vram_gib','parameters','max_checkpoint_bytes','total_training_seconds']}}))
for metric,row in comparison.iterrows():
    print(metric, 'AE minus CNN:',format(row['AE minus CNN'],'.6f'))
"""),md("""## Failure analysis: deterministic examples from every category (local)

For seed 2026, choose the first manifest-ordered FP, FN and TP when available,
plus the highest-scoring normal, deduplicating identical paths. This makes both success and failure selection
verifiable. Columns show the native original resized for display, reconstruction,
residual, predicted overlay at the separately calibrated pixel threshold and
ground-truth mask. Native primary metrics were already frozen; the following
descriptive panels cannot change model settings. Masks here use nearest-neighbor
display resizing only. Reconstructions are recomputed from retained checkpoints;
scores and outcomes come from the retained final evaluation.
Residuals and overlays use the retained maps; reconstructions are checkpoint
replays. Their map differences are measured rather than assumed exactly zero.
Residual display uses a fixed [0, 0.05] color scale for every final squared-error
panel; this display clipping does not change scores or predictions.
"""),code("""data_root=autodiscover_data_root(ROOT)
analysis=[]
for category in CATEGORIES:
    identifier=f"02-test-{freeze['selected_variant']}-{category}-s2026-v1"
    result=json.loads((report/f'evaluation/{identifier}-result.json').read_text())
    training=json.loads((report/f"runs/{result['training_experiment']}-result.json").read_text())
    model,_=load_checkpoint(ROOT/training['checkpoint'])
    table=pd.read_csv(report/f'evaluation/{identifier}-images.csv')
    retained_maps=np.load(ROOT/result['local_maps'],mmap_mode='r')
    choices=[]
    for outcome in ['FP','FN','TP']:
        subset=table[table.outcome.eq(outcome)]
        if len(subset):choices.append(subset.iloc[0])
    choices.append(table[table.label.eq(0)].sort_values('score',ascending=False).iloc[0])
    unique={}
    for row in choices:
        if row.path not in unique:unique[row.path]=row
    choices=list(unique.values())
    fig,axes=plt.subplots(len(choices),5,figsize=(15,3*len(choices)),squeeze=False)
    for i,row in enumerate(choices):
        with Image.open(data_root/row.path) as image:
            rgb=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
        x=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).cuda().float()/255
        with torch.inference_mode():
            reconstruction=model(x)
            replay_residual=anomaly_map(x,reconstruction,freeze['configuration']['map_kind'])[0].cpu().numpy()
        residual=np.asarray(retained_maps[int(row.name)]).copy()
        retained_score=image_scores(torch.from_numpy(residual[None])).item()
        assert np.isclose(retained_score,row.score,atol=1e-7)
        if isinstance(row['mask'],str):
            with Image.open(data_root/row['mask']) as image:mask=np.asarray(image.resize((256,256),Image.Resampling.NEAREST))>0
        else:mask=np.zeros((256,256),bool)
        pred=residual>result['calibration']['pixel_threshold']
        overlay=rgb.astype(float)/255
        overlay[pred]=.5*overlay[pred]+np.array([.5,0,0])
        arrays=[rgb,reconstruction[0].cpu().permute(1,2,0).numpy(),residual,overlay,mask]
        for j,a in enumerate(arrays):
            axes[i,j].imshow(a,vmin=0,vmax=(.05 if j==2 else (1 if j else 255)));axes[i,j].axis('off')
        axes[i,0].set_title(f'{category} {row.outcome} score={row.score:.4f}')
        inside=float(residual[mask].mean()) if mask.any() else None
        outside=float(residual[~mask].mean())
        top_count=int(np.ceil(residual.size*.01))
        top_indices=torch.from_numpy(residual).flatten().topk(top_count).indices.numpy()
        top_overlap=float(mask.ravel()[top_indices].mean()) if mask.any() else None
        original_float=rgb.astype(float)/255
        restored=reconstruction[0].cpu().permute(1,2,0).numpy()
        original_detail=float(np.abs(np.diff(original_float,axis=0)).mean()+np.abs(np.diff(original_float,axis=1)).mean())
        reconstruction_detail=float(np.abs(np.diff(restored,axis=0)).mean()+np.abs(np.diff(restored,axis=1)).mean())
        gradient=np.zeros((256,256))
        gradient[1:]+=np.abs(np.diff(original_float,axis=0)).mean(2)
        gradient[:,1:]+=np.abs(np.diff(original_float,axis=1)).mean(2)
        edges=gradient>np.quantile(gradient,.75)
        analysis.append(dict(category=category,path=row.path,outcome=row.outcome,image_score=row.score,
            threshold=result['calibration']['image_threshold'],mask_pixels=int(mask.sum()),
            residual_inside_mask=inside,residual_outside_mask=outside,
            top1pct_defect_overlap=top_overlap,
            top1pct_defect_enrichment=top_overlap/float(mask.mean()) if mask.any() else None,
            checkpoint_replay_map_max_abs_difference=float(np.abs(replay_residual-residual).max()),
            original_neighbor_detail=original_detail,reconstruction_neighbor_detail=reconstruction_detail,
            reconstruction_detail_ratio=reconstruction_detail/max(original_detail,1e-12),
            residual_on_edges=float(residual[edges].mean()) if edges.any() else None,
            residual_off_edges=float(residual[~edges].mean()),
            predicted_pixel_fraction=float(pred.mean())))
    for ax,title in zip(axes[0],['Original / outcome','Reconstruction','Residual [0,.05]','Pixel alarm overlay','Ground truth']):
        ax.set_xlabel(title);ax.xaxis.set_visible(True)
    plt.tight_layout();plt.show()
pd.DataFrame(analysis).to_csv(report/'failure_examples.csv',index=False)
"""),md("""## Evidence-based interpretation and limitations

The following table ties false negatives and normal false positives to actual
scores and residuals. Inside/outside-mask means are descriptive at display
resolution, not substitutes for native localization metrics. A false negative
means the top-1% residual score failed the normal-calibrated image threshold;
its reconstruction and mask panel distinguish small residuals from dilution of
small defects. A false positive documents a normal structure the model failed
to reconstruct sufficiently. Blurry output and normal edge errors should be
assessed directly in these retained panels; error does not prove semantic
understanding. Texture/object differences are measured above. The architecture
has no skip bypass, but can still copy anomaly-like patterns learned from normal
local appearance. No claim is made that every false negative has one cause.
"""),code("""examples=pd.read_csv(report/'failure_examples.csv')
display(examples)
display(pd.read_csv(report/'small_defect_summary.csv'))
display(runs.groupby('category').agg(image_auroc_seed_std=('image_auroc','std'),
    recall_seed_std=('recall','std'),fpr_seed_std=('fpr','std'),
    minimum_threshold=('image_threshold','min'),maximum_threshold=('image_threshold','max')))
fn=examples[examples.outcome.eq('FN')]
fp=examples[examples.outcome.eq('FP')]
print('Displayed false negatives:',len(fn),'displayed false positives:',len(fp))
if len(fn): display(fn[['category','mask_pixels','image_score','threshold','residual_inside_mask','residual_outside_mask']])
print('Measured texture/object differences:')
display(pd.read_csv(report/'texture_object.csv'))
thresholds=[]
for path in sorted((report/'evaluation').glob('*-result.json')):
    r=json.loads(path.read_text());c=r['calibration']
    thresholds.append(dict(category=r['category'],seed=r['seed'],n=c['images'],threshold=c['image_threshold'],
        low=c['image_threshold_bootstrap_95ci'][0],high=c['image_threshold_bootstrap_95ci'][1],test_fpr=r['fpr']))
display(pd.DataFrame(thresholds))
print('Small normal samples, wide threshold intervals and category-dependent test FPR limit deployment claims.')
"""),md("""## Preservation and reproduction

All meaningful experiment starts/results/failures and the selection decision are
append-only JSON in [experiments](../experiments/). Configuration identities,
checkpoint/map/history hashes and compact real outputs are retained under
[reports/autoencoder](../reports/autoencoder/). Raw images, checkpoints, maps,
optimizer state and this notebook's full executed image panels remain ignored
locally. The public notebook strips only dataset-pixel panel outputs. It keeps
actual tables, plots and execution evidence. Training ran via
`scripts/run_autoencoder.py`; notebook analysis explicitly reads those artifacts.
Use new experiment IDs for reproductions; do not overwrite the v1 registry.
No post-test tuning/retraining or PatchCore work is part of this milestone.
Dataset attribution and license are retained in
[the project dataset documentation](../docs/dataset.md); MVTec image panels
remain local under the existing CC BY-NC-SA 4.0 retention policy.
""")]
extra=[md("""## Final learning curves: all categories and all seeds

Every final model trained for the frozen 20 epochs. Solid lines show fitting
loss; dashed lines show normal-development loss, which did not select epochs or
change the frozen policy. All seeds are drawn for each category. Continued loss
decline at epoch 20 would indicate that the budget limited convergence, not
permission to tune after final results.
"""),code("""fig,axes=plt.subplots(3,5,figsize=(18,10))
for ax,category in zip(axes.flat,CATEGORIES):
    for seed in freeze['seeds']:
        h=pd.read_csv(report/f"runs/02-final-{freeze['selected_variant']}-{category}-s{seed}-v1-history.csv")
        line=ax.plot(h.epoch,h.train_loss,label=str(seed))[0]
        ax.plot(h.epoch,h.normal_development_loss,'--',color=line.get_color())
    ax.set_title(category);ax.legend(fontsize=7);ax.set_yscale('log')
plt.tight_layout();plt.show()
"""),md("""## Image ranking and calibration shift plots

These descriptive curves use seed 2026 throughout, never the best seed. Each
category's ROC and PR curve uses the retained official image scores. All-seed
metrics remain the primary tables. The calibration plots compare real calibration
normals against official test normals, with the frozen image threshold marked;
they reveal distribution shift without recalibrating on test data.
"""),code("""from sklearn.metrics import roc_curve,precision_recall_curve
for plot_kind in ['ROC','PR','normal score shift']:
    fig,axes=plt.subplots(3,5,figsize=(18,10))
    for ax,category in zip(axes.flat,CATEGORIES):
        identifier=f"02-test-{freeze['selected_variant']}-{category}-s2026-v1"
        t=pd.read_csv(report/f'evaluation/{identifier}-images.csv')
        c=json.loads((report/f'evaluation/{identifier}-calibration.json').read_text())
        if plot_kind=='ROC':
            fpr,tpr,_=roc_curve(t.label,t.score);ax.plot(fpr,tpr);ax.plot([0,1],[0,1],'--')
            ax.set_xlabel('FPR');ax.set_ylabel('Recall')
        elif plot_kind=='PR':
            precision,recall,_=precision_recall_curve(t.label,t.score);ax.plot(recall,precision)
            ax.axhline(t.label.mean(),linestyle='--');ax.set_xlabel('Recall');ax.set_ylabel('Precision')
        else:
            cal=np.asarray(c['calibration_scores'])
            normal=t.loc[t.label.eq(0),'score'].to_numpy()
            low=min(cal.min(),normal.min());high=max(cal.max(),normal.max())
            edges=np.linspace(low,max(high,low+1e-9),13)
            ax.hist(cal,bins=edges,alpha=.5,label='calibration normals')
            ax.hist(normal,bins=edges,alpha=.5,label='test normals')
            ax.axvline(c['image_threshold'],color='red');ax.legend(fontsize=6)
        if plot_kind in ['ROC','PR']:ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_title(category)
    fig.suptitle(plot_kind);plt.tight_layout();plt.show()
""")]
tail.cells[-1:-1]=extra
dev_panels=[md("""## Reconstruction examples for every development variant (local)

The first normal development image in manifest order is shown for every one of
the 20 development fits. These retained tensors come from the original fits,
not a new training run or favorable example search. Columns are original,
reconstruction, absolute residual (fixed [0,.2]) and squared residual (fixed
[0,.05]). Rows follow base, deeper, narrow, wide, MSE. Licensed panels stay local.
"""),code("""development_panels=[]
for category in ['bottle','cable','carpet','tile']:
    fig,axes=plt.subplots(5,4,figsize=(12,13))
    for i,variant in enumerate(['base','deeper','narrow','wide','mse']):
        saved=torch.load(ROOT/f'outputs/autoencoder/02-dev-{variant}-{category}-s2026-v1/development-reconstructions.pt',weights_only=True)
        x=saved['original'][:1].float()/255
        y=saved['reconstruction'][:1]
        values=[x[0].permute(1,2,0).numpy(),y[0].permute(1,2,0).numpy(),
                anomaly_map(x,y,'absolute')[0].numpy(),anomaly_map(x,y,'squared')[0].numpy()]
        for j,a in enumerate(values):
            axes[i,j].imshow(a,vmin=0,vmax=(.2 if j==2 else (.05 if j==3 else 1)))
            axes[i,j].axis('off')
        axes[i,0].set_title(category+' / '+variant)
    plt.tight_layout();plt.show()
""")]
position=next(i for i,cell in enumerate(tail.cells) if cell.cell_type=='markdown' and cell.source.startswith('## Development decision'))
tail.cells[position:position]=dev_panels+[md('\n'.join(development_markdown))]
for cell in tail.cells:
    if cell.cell_type=='code' and ('analysis=[]' in cell.source or 'development_panels=[]' in cell.source):cell.metadata['local_dataset_pixels']=True
os.environ['IPYTHONDIR']=str(root/'outputs/autoencoder/ipython')
(root/'outputs/autoencoder/ipython').mkdir(parents=True,exist_ok=True)
NotebookClient(tail,timeout=1200,kernel_name='mvtec-research',resources={'metadata':{'path':str(root)}}).execute()
pilot=nbf.read(root/'outputs/autoencoder/02_convolutional_autoencoder.executed.ipynb',as_version=4)
pilot.cells.extend(tail.cells)
pilot.metadata['execution_provenance']='Initial pilot executed live; continuation separately executed artifact analysis, not live retraining.'
(root/'outputs/autoencoder/02_convolutional_autoencoder.complete.executed.ipynb').write_bytes(nbf.writes(pilot).encode('utf-8'))
for cell in pilot.cells:
    if cell.metadata.get('local_dataset_pixels'):cell.outputs=[]
(root/'notebooks/02_convolutional_autoencoder.ipynb').write_bytes(nbf.writes(pilot).encode('utf-8'))
