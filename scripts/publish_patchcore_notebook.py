"""Execute artifact analysis, preserving the actual live pilot separately."""
from pathlib import Path
import json,os,asyncio
import nbformat as nbf
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
tail=nbf.v4.new_notebook()
tail.metadata.kernelspec=dict(display_name='MVTec research (CUDA, Python 3.12)',language='python',name='mvtec-research')
tail.cells=[md("""# Continued chronological record: development and confirmation
The preceding pilot was executed live. The continuation executes analysis of
separately recorded bank-building and official-evaluation jobs. It **does not
claim those jobs ran live in this kernel**. Their actual implementation is
included below as executable function definitions. Pending statements above
describe the historical pilot state; the completed phases below supersede them.
## 16. Development implementation and provenance
Hypothesis: a larger normal coreset improves local coverage enough to justify
its additional storage/search cost. No MVTec gradients update the pretrained CNN.
Thus there are no training-loss/epoch curves; coverage, diagnostics and resource
curves document this nonparametric fit. Deterministic features are cached once.
"""),code("""from pathlib import Path
import sys,json,time,math
ROOT=Path.cwd()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt,torch
from PIL import Image
from matplotlib.patches import Rectangle
from anomaly_detection.patchcore import PatchExtractor,nearest_neighbors,pixel_maps,image_scores,load_bank
from anomaly_detection.cnn_research import read_partition,inner_fit_split,load_resized,make_probes
from anomaly_detection.data import CATEGORIES,TEXTURES,autodiscover_data_root,sha256
from anomaly_detection.tracking import config_hash
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
report=ROOT/'reports/patchcore'
frozen=json.loads((ROOT/'configs/patchcore_frozen.json').read_text())
summary=json.loads((report/'summary.json').read_text())
print('Frozen weight SHA256:',frozen['weight_sha256'])
print('Cached weight files (availability/size only, not multiple backbone experiments):')
print([(p.name,p.stat().st_size) for p in (Path(torch.hub.get_dir())/'checkpoints').glob('*.pth')])
print('ResNet18 is the measured backbone; WideResNet50-2 was not benchmarked.')
print('CUDA:',torch.cuda.get_device_name(0))
counts=[]
for category in CATEGORIES:
    counts.append(dict(category=category,**{p:len(read_partition(p,category)) for p in ['fit','development','calibration','test']}))
display(pd.DataFrame(counts))
"""),md("""### Actual cache and bank construction
This is the source used for the recorded jobs. `require_fitting` rejects
non-fitting rows before a memory is built. The inner fitting holdout supplies
fixed synthetic probes during development and never enters that development
bank. Final banks use the complete fitting partition. Feature cache hashes
include original rows, weights, transform policy and extractor source identity.
"""),code((root/'src/anomaly_detection/patchcore_research.py').read_text(encoding='utf-8').replace('from .','from anomaly_detection.')+'\n'+(root/'src/anomaly_detection/patchcore_runs.py').read_text(encoding='utf-8').replace('from .','from anomaly_detection.'))]
for budget in [256,1024,2048]:
    for category in ['bottle','cable','carpet','tile']:
        identifier=f'03-dev-b{budget}-{category}-s2026-v1'
        result=json.loads((root/f'reports/patchcore/runs/{identifier}-result.json').read_text())
        decision=json.loads((root/f'experiments/{identifier}-decision.json').read_text())
        tail.cells.extend([md(f"""### Experiment {identifier}: hypothesis
A {budget}-patch coreset balances representative coverage against storage and
NN cost on **{category}**. Hold weights, layers, preprocessing, candidate cap,
projection and seed fixed. Compare both max and top-1% aggregation. This cell
reads the actual registered job; it does not rebuild its bank or reuse test labels.
[Immutable experiment](../experiments/{identifier}-result.json).
"""),code(f"""r=json.loads((report/'runs/{identifier}-result.json').read_text())
print(json.dumps(r['configuration'],indent=2))
display(pd.DataFrame(r['metrics']).T)
print('Coverage:',r['candidate_coverage'])
print('Coreset seconds:',r['coreset_seconds'],'search p50 ms:',r['development']['search_p50_ms'])
print('Normal-development patch distance quantiles:',r['development']['normal_development_patch_distance_quantiles'])
print('Fitting cache reused:',r['feature_cache']['cache_hit'])
"""),md(f"""### Measured result, interpretation and decision
Max/top-1% proxy image AUROC: **{result['metrics']['max']['auroc']:.6f} /
{result['metrics']['top1pct']['auroc']:.6f}**; proxy pixel AP:
**{result['metrics']['max']['pixel_ap']:.6f}**. Coreset construction took
**{result['coreset_seconds']:.3f} s**. {decision['interpretation']}
Decision: **{decision['decision']}**. Synthetic proxy performance does not
establish transfer to real defects. Raw configurations and rejected trials remain.
""")])
tail.cells.extend([md("""### Development coverage/cost curves and measured selection
Every category and budget is shown. Coverage is a projected candidate-pool
diagnostic, not a guarantee for every original embedding. The candidate cap
can miss rare normals. Search timings are measured, so small non-monotonicities
are not removed. The predefined objective selects one global budget/aggregation.
"""),code("""dev=pd.read_csv(report/'development_comparison.csv')
display(dev);display(pd.read_csv(report/'development_macro.csv'))
records=[]
for budget in [256,1024,2048]:
    for category in ['bottle','cable','carpet','tile']:
        r=json.loads((report/f'runs/03-dev-b{budget}-{category}-s2026-v1-result.json').read_text())
        records.append(dict(budget=budget,category=category,radius=r['candidate_coverage']['projected_radius'],
            coreset_seconds=r['coreset_seconds'],search_ms=r['development']['search_p50_ms']))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for category,group in pd.DataFrame(records).groupby('category'):
    for ax,key in zip(axes,['radius','coreset_seconds','search_ms']):
        ax.plot(group.budget,group[key],marker='o',label=category);ax.set_xlabel('Bank patches');ax.set_ylabel(key)
axes[0].legend();plt.tight_layout();plt.show();plt.close(fig)
ranked=dev.groupby(['budget','aggregation']).objective.mean().sort_values(ascending=False)
assert ranked.index[0]==(frozen['configuration']['bank_budget'],frozen['configuration']['aggregation'])
print('Recomputed choice:',ranked.to_string())
"""),md("""### All-budget synthetic reconstruction-free map examples (local)
For each development category, recreate the original fixed fitting-derived
synthetic RGB probe with seed 9901 and verify its mask against the retained
mask. No features are extracted again. Read the original saved distance maps.
Rows: budgets 256/1024/2048; columns: edited RGB, raw map, synthetic mask.
All maps in one category share a display limit derived from normal development
maps. These are proxy edits, never official defects. Licensed pixels stay local.
"""),code("""development_panels=[]
for category in ['bottle','cable','carpet','tile']:
    _,probe_rows=inner_fit_split(read_partition('fit',category))
    images=load_resized(probe_rows,256).cuda()
    probes,masks=make_probes(images,probe_rows)
    saved=[]
    for budget in [256,1024,2048]:
        saved.append(np.load(ROOT/f'outputs/patchcore/03-dev-b{budget}-{category}-s2026-v1/development_maps.npz'))
    limit=max(float(np.quantile(s['normal'],.995)) for s in saved)
    fig,axes=plt.subplots(3,3,figsize=(10,10))
    index=len(probe_rows)
    for i,s in enumerate(saved):
        assert np.array_equal(s['masks'],masks[:,0].cpu().numpy())
        values=[probes[index].cpu().permute(1,2,0).numpy(),pixel_maps(torch.from_numpy(s['proxy'][index:index+1]))[0].numpy(),s['masks'][index]]
        for j,a in enumerate(values):axes[i,j].imshow(a,vmin=0,vmax=limit if j==1 else 1);axes[i,j].axis('off')
        axes[i,0].set_title(category+' / bank '+str([256,1024,2048][i]))
    plt.tight_layout();plt.show();plt.close(fig)
"""),md("""## 17. Candidate decision and compute gate
Read the immutable selection and compute report written before final fits/test
access. The selected proxy advantage is small and single-development-seed
evidence does not establish an optimal architecture. No additional backbone,
normalization or smoothing search was justified within this bounded study.
## 18. Threshold calibration
Only reserved normals calibrate image higher-q95 and independent pooled pixel
higher-q99.5 thresholds; alarm iff score > threshold. Distances are unbounded,
not probabilities. The 4,096-bin pixel bootstrap uses d/(d+s), where s is median
normal-calibration patch distance, to avoid clipping high distances. Exact native
point metrics use raw distances. Threshold and image-metric bootstrap resample
whole images, not independent pixels. Small samples and shift limit FPR control.
## 19. Frozen model and evaluation implementation
The source below is the actual frozen calibration, search-timing and native
evaluation implementation. These definitions execute here; historical official
inference is not repeated. Existing exact metric and bootstrap helpers are reused.
"""),code("""print(json.dumps(frozen,indent=2))
print(json.dumps(json.loads((report/'compute_budget.json').read_text()),indent=2))
"""),code("__file__=str(ROOT/'src/anomaly_detection/patchcore_evaluation.py')\n"+(root/'src/anomaly_detection/patchcore_evaluation.py').read_text(encoding='utf-8').replace('from .','from anomaly_detection.')),
md("""## 20. All-category evaluation: frozen hypothesis
Hypothesis: one independently sampled normal coreset per category/seed transfers
to real defects with acceptable recall, false alarms and latency. All 15
categories and seeds 2026/2027/2028 are retained. Deterministic features are shared
across seeds; no test-driven bank refit occurs. This is a 45-bank confirmation
cohort, not 45 independent test datasets. The following reads measured results.
"""),code("""runs=pd.read_csv(report/'all_runs.csv')
display(pd.read_csv(report/'all_category_means.csv'))
display(pd.DataFrame(summary['macro']).T)
display(runs[['category','seed','calibration_images','image_threshold','recall','fpr','precision','f1','tn','fp','fn','tp']])
display(pd.read_csv(report/'texture_object.csv'))
print(json.dumps(summary,indent=2))
"""),md("""### Confirmation interpretation
Assess all categories, recall/FPR and localization together; AP depends on the
dataset's defect prevalence. The final measured conclusions below discuss weak
categories and practical cost. Seed standard deviation is distinct from the
conditional whole-image bootstrap. Pixel intervals target a disclosed binned
approximation; report its measured bias against exact point metrics.
## 21. Genuine images, maps, overlays, masks and nearest normal references (local)
Seed 2026 throughout. Show first manifest-ordered FP, FN, TP where available and
highest-scoring normal, deduplicating paths. No best seed/example selection.
Heatmaps/overlays use retained raw-distance maps. Cached test embeddings supply
the query; no completed feature extraction is repeated. For defects, mark the
highest-distance grid cell intersecting the display mask; for normals, mark the
highest-distance cell. The reference is its actual bank nearest neighbor's
source fitting image/cell. Grid cells are 8x8 input pixels, **not the full CNN
receptive field**. The display limit is twice the normal-calibrated pixel
threshold, with no test-score normalization. Primary metrics use native masks.
"""),code("""data_root=autodiscover_data_root(ROOT)
failure_examples=[]
for category in CATEGORIES:
    identifier=f"03-test-{frozen['selected_variant']}-{category}-s2026-v1"
    r=json.loads((report/f'evaluation/{identifier}-result.json').read_text())
    fit=json.loads((report/f"runs/{r['training_experiment']}-result.json").read_text())
    bank,saved=load_bank(ROOT/fit['checkpoint'],category,'cuda')
    rows=read_partition('fit',category)
    table=pd.read_csv(report/f'evaluation/{identifier}-images.csv')
    maps=np.load(ROOT/r['local_maps'],mmap_mode='r')
    patches=np.load((ROOT/r['local_maps']).parent/'patch_distances.npy',mmap_mode='r')
    embeddings=np.load(ROOT/r['test_cache']['path'],mmap_mode='r')
    choices=[]
    for outcome in ['FP','FN','TP']:
        subset=table[table.outcome.eq(outcome)]
        if len(subset):choices.append(subset.iloc[0])
    choices.append(table[table.label.eq(0)].sort_values('score',ascending=False).iloc[0])
    unique={}
    for row in choices:
        if row.path not in unique:unique[row.path]=row
    choices=list(unique.values());fig,axes=plt.subplots(len(choices),5,figsize=(15,3*len(choices)),squeeze=False)
    for i,row in enumerate(choices):
        with Image.open(data_root/row.path) as image:rgb=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
        if isinstance(row['mask'],str):
            with Image.open(data_root/row['mask']) as image:mask=np.asarray(image.resize((256,256),Image.Resampling.NEAREST))>0
        else:mask=np.zeros((256,256),bool)
        heat=np.array(maps[int(row.name)]);coarse=np.array(patches[int(row.name)])
        assert np.isclose(image_scores(torch.from_numpy(coarse[None]),frozen['configuration']['aggregation']).item(),row.score,atol=1e-6)
        q=torch.from_numpy(np.array(embeddings[int(row.name)])).cuda()
        replay,neighbors=nearest_neighbors(q,bank)
        cell_mask=torch.nn.functional.adaptive_max_pool2d(torch.from_numpy(mask.astype(np.float32))[None,None],(32,32))[0,0].bool().numpy()
        choice=int(np.where(cell_mask,coarse,-np.inf).argmax()) if cell_mask.any() else int(coarse.argmax())
        source_index=int(saved['indices'][int(neighbors[choice])]);source_image=source_index//1024;source_cell=source_index%1024
        normal_path=rows.path.iloc[source_image]
        with Image.open(data_root/normal_path) as image:reference=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
        alarm=heat>r['calibration']['pixel_threshold'];overlay=rgb.astype(float)/255
        overlay[alarm]=.5*overlay[alarm]+np.array([.5,0,0])
        for j,a in enumerate([rgb,heat,overlay,mask,reference]):
            axes[i,j].imshow(a,vmin=0,vmax=2*r['calibration']['pixel_threshold'] if j==1 else (255 if j in [0,4] else 1));axes[i,j].axis('off')
        for j,cell,color in [(0,choice,'cyan'),(4,source_cell,'yellow')]:
            axes[i,j].add_patch(Rectangle(((cell%32)*8,(cell//32)*8),8,8,fill=False,edgecolor=color,linewidth=1))
        axes[i,0].set_title(f'{category} {row.outcome} score={row.score:.3f}')
        axes[i,4].set_title('Nearest fitting normal grid cell')
        top=torch.from_numpy(coarse).flatten().topk(11).indices.numpy()
        failure_examples.append(dict(category=category,path=row.path,outcome=row.outcome,score=row.score,
            image_threshold=r['calibration']['image_threshold'],pixel_threshold=r['calibration']['pixel_threshold'],
            mask_pixels=int(mask.sum()),mean_distance_inside=float(heat[mask].mean()) if mask.any() else None,
            mean_distance_outside=float(heat[~mask].mean()),predicted_pixel_fraction=float(alarm.mean()),
            top_patch_defect_overlap=float(cell_mask.ravel()[top].mean()) if mask.any() else None,
            nearest_reference=normal_path,query_cell=choice,reference_cell=source_cell,nearest_distance=float(replay[choice]),
            cached_replay_max_abs_difference=float(np.abs(replay.cpu().numpy().reshape(32,32)-coarse).max())))
    plt.tight_layout();plt.show();plt.close(fig)
pd.DataFrame(failure_examples).to_csv(report/'failure_examples.csv',index=False,lineterminator=chr(10))
"""),md("""## 22. Failure analysis: observed diagnostics
Read actual FP/FN maps and nearest normal references above. Coarse-grid overlap
is a descriptive diagnostic, not native pixel precision. High normal distances
can reflect incomplete bank coverage or feature mismatch; the reference panels
and coverage table provide evidence, not proof of one cause. Local features may
miss structural arrangements and downsampling can limit fine scratches, but a
controlled ablation would be needed to isolate those mechanisms. Native mask
size groups and defect-type recall document actual outcomes without refitting.
"""),code("""display(pd.read_csv(report/'failure_examples.csv'))
display(pd.read_csv(report/'memory_coverage.csv'))
display(pd.read_csv(report/'small_defect_summary.csv'))
display(pd.read_csv(report/'defect_type_size_summary.csv'))
thresholds=[]
for path in sorted((report/'evaluation').glob('*-result.json')):
    r=json.loads(path.read_text());c=r['calibration']
    thresholds.append(dict(category=r['category'],seed=r['seed'],n=c['images'],threshold=c['image_threshold'],
        low=c['image_threshold_bootstrap_95ci'][0],high=c['image_threshold_bootstrap_95ci'][1],test_fpr=r['fpr']))
display(pd.DataFrame(thresholds))
"""),md("""## 23. Runtime, GPU/CPU memory and storage
Fitting time covers cache access, coreset construction and bank saving; no CNN
training occurs. Unique extraction-stage time includes image loading, GPU work,
transfer and cache writes, and is deduplicated by cache identity. RSS is sampled
at 20 Hz; short spikes may be missed. Evaluation RSS includes exact native metric
scratch work and differs from isolated inference memory. CUDA allocation uses
allocator counters. Bank bytes exclude the single shared frozen extractor.
Batch-one latency: 50 warmups, 200 synchronized iterations, already decoded RGB
through native CPU map/image score, disk decode excluded. Feature extraction and
NN are measured separately; model is their combined elapsed stage. Compare p50,
p95 and end-to-end, not just cached-embedding search throughput.
"""),code("""display(runs[['category','seed','coreset_seconds','extraction_p50_ms','nn_p50_ms','model_p50_ms','model_p95_ms','end_to_end_p50_ms','end_to_end_p95_ms','cpu_peak_rss_gib','train_peak_vram_gib','inference_peak_vram_gib','checkpoint_bytes']])
print(json.dumps(summary['shared_backbone_storage'],indent=2))
print('Unique feature caches:',summary['deterministic_feature_caches'],'unique extraction-stage seconds:',summary['unique_extraction_stage_seconds'])
"""),md("""## 24. Exact stored comparison: CNN, AE and PatchCore variant
Same categories, partitions, normal calibration policy and native exact metrics.
Different pretrained priors, objectives and score aggregations make this a
complete-pipeline comparison, not an isolated backbone causal study. The image
bootstrap differences are paired by the same official images and draws, shared
across fitted seeds; they are conditional on these fitted models. Pixel primary
metrics are exact; binned uncertainty transforms differ across families, so
paired pixel confidence intervals are deliberately not claimed.
"""),code("""display(pd.read_csv(report/'model_macro_comparison.csv'))
display(pd.read_csv(report/'paired_image_difference_intervals.csv'))
display(pd.read_csv(report/'paired_prediction_disagreements.csv'))
for name in ['cnn','ae']:
    display(pd.read_csv(report/f'{name}_category_comparison.csv'))
    display(pd.read_csv(report/f'{name}_category_costs.csv'))
"""),md("""## 25. Conclusions, limitations and preservation
Measured interpretation follows after all diagnostics are reconciled. Do not
declare superiority from one AUROC. Consider recall, FPR, localization, latency,
RSS/VRAM, bank/model storage, pretraining and category-specific weaknesses.
This simplified variant is not a paper reproduction. Candidate sampling,
projected selection, feature geometry and finite calibration constrain claims.
All experiment starts/results/failures/decisions are append-only in
[the registry](../experiments/); [compact results](../reports/patchcore/) link
actual local feature/bank/map identities. The initial pilot executed live;
later cells explicitly analyze separately completed jobs. Licensed image
panels remain in the local executed notebook and are stripped from public
outputs. Dataset attribution/license: [project documentation](../docs/dataset.md).
No SuperSimpleNet work is part of this milestone.
""")])
plots=[md("""### Ranking curves and normal calibration shift
Seed 2026 is fixed for display, not selected as the best seed. All three seeds
remain in the primary summaries. Curves read retained scores only. Normal-score
histograms use common within-category bins and show the frozen threshold; they
do not recalibrate from test normals. Recall and false-alarm behavior matter
alongside AUROC and pixel localization.
"""),code("""from sklearn.metrics import roc_curve,precision_recall_curve
for kind in ['ROC','PR','Normal-score shift']:
    fig,axes=plt.subplots(3,5,figsize=(17,10))
    for ax,category in zip(axes.flat,CATEGORIES):
        identifier=f"03-test-{frozen['selected_variant']}-{category}-s2026-v1"
        table=pd.read_csv(report/f'evaluation/{identifier}-images.csv')
        calibration=json.loads((report/f'evaluation/{identifier}-calibration.json').read_text())
        if kind=='ROC':
            x,y,_=roc_curve(table.label,table.score);ax.plot(x,y);ax.plot([0,1],[0,1],'--');ax.set_xlabel('FPR');ax.set_ylabel('Recall')
        elif kind=='PR':
            y,x,_=precision_recall_curve(table.label,table.score);ax.plot(x,y);ax.axhline(table.label.mean(),linestyle='--');ax.set_xlabel('Recall');ax.set_ylabel('Precision')
        else:
            cal=np.asarray(calibration['calibration_scores']);test=table.loc[table.label.eq(0),'score'].to_numpy()
            low=min(cal.min(),test.min());high=max(cal.max(),test.max())
            bins=np.linspace(low,max(high,low+1e-9),13)
            ax.hist(cal,bins=bins,alpha=.5,label='calibration normals');ax.hist(test,bins=bins,alpha=.5,label='test normals')
            ax.axvline(calibration['image_threshold'],color='red');ax.legend(fontsize=6)
        if kind in ['ROC','PR']:ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_title(category)
    fig.suptitle(kind);plt.tight_layout();plt.show();plt.close(fig)
""")]
tail.cells[-1:-1]=plots
pixel_text="""### Separate pixel-threshold operating points
These are already-recorded native pixel decisions at independently calibrated
normal pixel q99.5 thresholds. They do not reuse image thresholds. Per-category
rates are macro-averaged across seeds. Confusion counts remain per category/seed;
no independence-based pixel confidence interval is claimed.
"""
pixel_code="""import json,pandas as pd
from pathlib import Path
pixel_report=globals().get('ROOT',Path.cwd())/'reports/patchcore'
display(pd.read_csv(pixel_report/'pixel_operating_category_means.csv'))
display(pd.read_csv(pixel_report/'pixel_operating_model_comparison.csv'))
display(pd.DataFrame([json.loads((pixel_report/'summary.json').read_text())['pixel_operating_macro']]))
pixel_runs=pd.read_csv(pixel_report/'all_runs.csv')
display(pixel_runs[['category','seed','pixel_threshold','pixel_recall','pixel_fpr','pixel_precision','pixel_f1','pixel_tn','pixel_fp','pixel_fn','pixel_tp']])
"""
pixel_cell=code(pixel_code);pixel_cell.metadata['pixel_operating_analysis']=True
pixel_md=md(pixel_text);pixel_md.metadata['pixel_operating_analysis']=True
tail.cells[-1:-1]=[pixel_md,pixel_cell]
for cell in tail.cells:
    if cell.cell_type=='code' and ('development_panels=[]' in cell.source or 'failure_examples=[]' in cell.source):cell.metadata['local_dataset_pixels']=True
folder=root/'outputs/patchcore'
os.environ['IPYTHONDIR']=str(folder/'ipython')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
NotebookClient(tail,timeout=1200,kernel_name='mvtec-research',resources={'metadata':{'path':str(root)}}).execute()
notebook=nbf.read(folder/'03_patchcore.pilot.executed.ipynb',as_version=4)
pilot=json.loads((root/'reports/patchcore/03-pilot-resnet18-bottle-s2026-v1-result.json').read_text())
annotation=(f"\n\nRecorded pilot gate: extraction {pilot['extraction_seconds']:.3f} s, coreset "
    f"{pilot['coreset_seconds']:.3f} s, search p50 {pilot['search_p50_ms']:.3f} ms, "
    f"allocated VRAM {pilot['resources']['peak_gpu_allocated_bytes']/2**30:.3f} GiB, "
    f"peak sampled RSS {pilot['resources']['cpu_peak_rss_bytes']/2**30:.3f} GiB. "
    "Correctness checks passed; the recorded decision was to proceed with bounded development. "
    "See [pilot gate](../reports/patchcore/pilot_gate.json), committed before development selection.")
for cell in notebook.cells:
    if cell.cell_type=='markdown' and cell.source.startswith('### Pilot interpretation'):cell.source+=annotation
notebook.cells.extend(tail.cells)
notebook.metadata['execution_provenance']='Live pilot followed by separately executed artifact analysis; bank fitting and official inference are not claimed live in continuation kernel.'
(folder/'03_patchcore.complete.executed.ipynb').write_bytes(nbf.writes(notebook).encode('utf-8'))
for cell in notebook.cells:
    if cell.metadata.get('local_dataset_pixels'):cell.outputs=[]
(root/'notebooks/03_patchcore.ipynb').write_bytes(nbf.writes(notebook).encode('utf-8'))
