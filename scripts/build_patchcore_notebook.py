"""Live implementation and CUDA pilot; preserves all outputs honestly."""
from pathlib import Path
import os,asyncio
import nbformat as nbf
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
n=nbf.v4.new_notebook()
n.metadata.kernelspec=dict(display_name='MVTec research (CUDA, Python 3.12)',language='python',name='mvtec-research')
n.cells=[md(r"""# Notebook 03 — PatchCore

## 1. Motivation and hypotheses
Reconstruction models can copy defects and blur normal structure. PatchCore
instead compares local pretrained features against a memory of normal patches.
Hypothesis: local nearest-neighbor distances separate fitting-derived synthetic
edits from normal appearance without a learned reconstruction objective.
This first execution is a live small normal-only CUDA pilot; development and
all-category confirmation remain pending. Earlier notebooks/results are preserved.

## 2. Architecture and mathematics
For frozen CNN activations $F_2(x),F_3(x)$, apply local average pooling and align
the coarser map to the finer grid. Concatenation gives $z_{h,w}\in\mathbb R^{384}$.
The fitting-normal memory is $M=\{z_i\}$. A coreset approximately minimizes
$\max_{z\in M}\min_{m\in M_C}\|Pz-Pm\|_2$, using random projection $P$ and greedy
farthest-first traversal of a capped candidate pool. Search uses original
384-dimensional vectors: $a_{h,w}=\min_{m\in M_C}\|z_{h,w}-m\|_2$.
Continuous patch distances are interpolated into a pixel map. Image scores use
maximum patch distance or a fixed top-1% mean; they are not pixel probabilities.

This is a **simplified PatchCore variant**, not paper equivalence: ResNet18,
direct concatenation, capped candidate-pool projected coreset, no neighborhood
image-score reweighting and initially no Gaussian smoothing. Sources:
[Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265) and
[torchvision weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18).
"""),code("""from pathlib import Path
import sys,json,time,copy,traceback,subprocess,math
ROOT=Path.cwd()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt,torch,psutil
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.basic_cnn import seed_everything,synthetic_batch
from anomaly_detection.cnn_research import read_partition,require_fitting,load_resized
from anomaly_detection.tracking import record,config_hash
from anomaly_detection.patchcore_research import Resources
torch.set_num_threads(4)
assert torch.cuda.is_available()
seed_everything(2026)
torch.backends.cuda.matmul.allow_tf32=False
print(torch.__version__,torch.cuda.get_device_name(0))
print('GPU GiB:',torch.cuda.get_device_properties(0).total_memory/2**30)
print('RAM GiB:',psutil.virtual_memory().total/2**30)
"""),md("""## 3. Dataset and frozen partition verification
The existing loader checks manifest hashes and normal-only provenance. No
official test image or mask is accessed by this pilot. The first 24 fitting
normals form the pilot bank; first eight development normals measure transfer.
## 4. Environment and GPU verification
Use the existing locked CUDA environment. Both ResNet18 and WideResNet50-2
weights are cached; start with the smaller ResNet18 and download neither.
Actual shapes, resources and correctness are measured below. Weight hashes and
the source/environment identities are recorded before fitting.
"""),code("""fit_rows=read_partition('fit','bottle').iloc[:24].copy()
require_fitting(fit_rows)
dev_rows=read_partition('development','bottle').iloc[:8].copy()
print('Pilot fitting normals:',len(fit_rows),'development normals:',len(dev_rows))
metadata=json.loads((ROOT/'reports/dataset_validation.json').read_text())
print(json.dumps(metadata['partitions'],indent=2))
"""),md("""## 5. Preprocessing
Default ResNet18 classification weights resize the short edge to 256 and center
crop 224. For full-image localization, retain the previous models' PIL bilinear
256-square geometry, then call the official weight transform with resize/crop
256. This documented geometry override keeps all borders and official ImageNet
mean/std; it does not use test statistics. Aspect ratios can be distorted.
## 6. Pretrained extraction
All parameters and BatchNorm statistics are frozen, including after `.train()`.
## 7. Layer selection
At 256 square: layer1 is 64x64x64, layer2 is 128x32x32 and layer3 is 256x16x16.
Layer2+layer3 yields 1,024 patches/image; using layer1 would quadruple patch
count and search cost. The initial pair balances detail and semantic context.
## 8. Multi-scale aggregation
Pool 3x3 locally, align by bilinear interpolation and concatenate channels.
## 9. Patch embeddings
Flatten spatial locations, not channels: B,C,H,W → (BHW),C.
## 10. Normal memory
Only category-specific fitting normals supply bank vectors.
## 11. Coreset
Random candidate cap 20,000; Gaussian projection to 32 dimensions; greedy
farthest-first selection. Keep original full-dimensional vectors for scoring.
## 12. Nearest-neighbor search
GPU blocked exhaustive L2, at most 512x2048 temporary distances (4 MiB).
Direct subtraction refines selected distances. Index choices near numerical ties
remain float32-precision limited. This is an efficient PyTorch flat-search
equivalent, avoiding an additional FAISS installation on this Windows environment.
## 13. Pixel maps
Interpolate continuous distances; masks are never bilinearly transformed.
## 14. Image scores
Compare max-patch and top-1%-patch mean on permitted development evidence.
The following is the actual implementation, executed as definitions here.
"""),code((root/'src/anomaly_detection/patchcore.py').read_text(encoding='utf-8')),
md("""## 15. Initial verification and measured pilot
Hypothesis: 24 fitting normals and a 512-patch coreset fit safely, exact-search
fixtures agree with float64 reference distances, self-search returns zero and
the score map has the declared shape. These checks precede all-category work.
The pilot is a feasibility experiment, not an official detection benchmark.
"""),code("""identifier='03-pilot-resnet18-bottle-s2026-v1'
config=dict(backbone='resnet18',weights='IMAGENET1K_V1',layers=['layer2','layer3'],
    resolution=256,bank_budget=512,candidate_limit=20000,projection_dim=32,
    query_chunk=512,bank_chunk=2048,seed=2026,normalize=False)
local=ROOT/'outputs/patchcore'/identifier
local.mkdir(parents=True,exist_ok=True)
report=ROOT/'reports/patchcore';report.mkdir(parents=True,exist_ok=True)
entry=dict(experiment_id=identifier+'-start',model_family='PatchCoreVariant',categories=['bottle'],
    hypothesis='A cached frozen ResNet18 with a 512-patch projected coreset supports bounded exact search and finite anomaly maps.',
    preprocessing='Whole PIL RGB bilinear 256 + official V1 transform with identity resize/crop 256 geometry',
    hyperparameters=config,seed=2026,
    environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),ram_bytes=psutil.virtual_memory().total,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p:sha256(ROOT/p) for p in ['src/anomaly_detection/patchcore.py','src/anomaly_detection/patchcore_research.py']},
        lock_sha256=sha256(ROOT/'requirements.lock.txt'),partitions=metadata['partitions']),
    training={},metrics={},calibration={},paths=dict(local=local.relative_to(ROOT).as_posix()),
    interpretation='Reserved before normal-only pilot; no official test access',decision='Measure feasibility before expansion',status='started')
record(ROOT/'experiments',entry)
try:
    extractor=PatchExtractor().cuda()
    print('Weight SHA256:',extractor.weight_sha256)
    print('Transform:',extractor.transform)
    with Resources() as resource:
        fitting=load_resized(fit_rows,256).cuda().float()/255
        development=load_resized(dev_rows,256).cuda().float()/255
        tick=time.perf_counter()
        features=extractor(fitting)
        torch.cuda.synchronize();extraction_seconds=time.perf_counter()-tick
        shapes=[list(f.shape) for f in extractor.feature_maps(fitting[:1])]
        embeddings=patch_embeddings(features)
        tick=time.perf_counter()
        indices,coverage=projected_coreset(embeddings,config['bank_budget'],config['seed'])
        bank=embeddings[indices].clone()
        torch.cuda.synchronize();coreset_seconds=time.perf_counter()-tick
        self_distances,_=nearest_neighbors(bank[:16],bank)
        assert self_distances.eq(0).all()
        g=torch.Generator().manual_seed(3)
        fixture_bank=torch.randn(37,16,generator=g).cuda()
        fixture_query=torch.randn(13,16,generator=g).cuda()
        actual,_=nearest_neighbors(fixture_query,fixture_bank,7,11)
        reference=torch.cdist(fixture_query.double(),fixture_bank.double(),compute_mode='donot_use_mm_for_euclid_dist').min(1).values
        torch.testing.assert_close(actual.double(),reference,rtol=1e-6,atol=1e-6)
        dev_features=extractor(development)
        tick=time.perf_counter()
        distances,_=nearest_neighbors(patch_embeddings(dev_features),bank)
        torch.cuda.synchronize();search_seconds=time.perf_counter()-tick
        patch_maps=distances.reshape(len(development),32,32)
        maps=pixel_maps(patch_maps)
        assert maps.shape==(8,256,256) and torch.isfinite(maps).all() and maps.ge(0).all()
        for _ in range(20):nearest_neighbors(patch_embeddings(dev_features[:1]),bank)
        times=[]
        for _ in range(50):
            tick=time.perf_counter();nearest_neighbors(patch_embeddings(dev_features[:1]),bank)
            torch.cuda.synchronize();times.append((time.perf_counter()-tick)*1000)
        altered=development[:1].clone()
        altered[:,:,96:128,96:128]=1-altered[:,:,96:128,96:128]
        altered_distances,_=nearest_neighbors(patch_embeddings(extractor(altered)),bank)
        altered_map=pixel_maps(altered_distances.reshape(1,32,32))
    fitting_counts={c:len(read_partition('fit',c)) for c in CATEGORIES}
    result=dict(configuration=config,weight_sha256=extractor.weight_sha256,
        weight_url=ResNet18_Weights.IMAGENET1K_V1.url,
        extractor_parameters=sum(p.numel() for p in extractor.parameters()),
        trainable_parameters=sum(p.numel() for p in extractor.parameters() if p.requires_grad),
        feature_shapes=shapes,embedding_shape=list(embeddings.shape),patches_per_image=1024,
        extraction_seconds=extraction_seconds,coreset_seconds=coreset_seconds,
        search_8_images_seconds=search_seconds,search_p50_ms=float(np.median(times)),
        search_p95_ms=float(np.quantile(times,.95)),coverage=coverage,
        resources=resource.result,bank_bytes=bank.numel()*bank.element_size(),
        feature_bytes_per_image=1024*384*4,
        all_category_fitting_images=fitting_counts,
        expected_feature_cache_bytes={c:n*1024*384*4 for c,n in fitting_counts.items()},
        projected_three_seed_final_extraction_seconds=extraction_seconds/24*sum(fitting_counts.values()),
        coreset_cost_caveat='Pilot bank 512; candidate cap and selected bank budget affect final cost. Deterministic extraction runs once, not three times.',
        correctness=dict(float64_fixture=True,zero_self_distances=True,finite_nonnegative_maps=True,map_shape=list(maps.shape)))
    torch.save(dict(bank=bank.cpu(),indices=indices.cpu(),config=config),local/'pilot-bank.pt')
    torch.save(dict(normal=development.cpu(),maps=maps.cpu(),altered=altered.cpu(),altered_map=altered_map.cpu()),local/'pilot-panels.pt')
    with (report/(identifier+'-result.json')).open('x') as stream:json.dump(result,stream,indent=2)
    event=copy.deepcopy(entry)
    event.update(experiment_id=identifier+'-result',status='completed',metrics=result,
        decision='Pilot correctness passed; assess measured workload before bounded development',
        interpretation='Feasibility evidence only; synthetic map example is not a real-defect benchmark.')
    record(ROOT/'experiments',event)
    print(json.dumps(result,indent=2))
except BaseException as exc:
    (local/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    event=copy.deepcopy(entry)
    event.update(experiment_id=identifier+'-failure',status='failed',decision='preserve and investigate',interpretation=str(exc))
    record(ROOT/'experiments',event)
    raise
"""),md("""### Pilot interpretation and decision
Use the measured output above to decide feasibility. Self-distance and float64
fixture checks test implementation correctness, not detection quality. Coreset
coverage is measured only in projected candidate space; it does not guarantee
coverage of every original patch. The synthetic edit below is a map sanity check.
Licensed image panels remain local; public outputs retain measurements.
"""),code("""fig,axes=plt.subplots(2,2,figsize=(8,8))
limit=float(torch.cat([maps[:1],altered_map]).max())
for i,(images,heatmaps) in enumerate([(development[:1],maps[:1]),(altered,altered_map)]):
    axes[i,0].imshow(images[0].cpu().permute(1,2,0));axes[i,0].axis('off')
    axes[i,1].imshow(heatmaps[0].cpu(),vmin=0,vmax=limit);axes[i,1].axis('off')
axes[0,0].set_title('Real development normal')
axes[1,0].set_title('Synthetic edit, not official defect')
plt.tight_layout();plt.show()
"""),md("""## 16. Development experiments
Predeclared bottle/cable/carpet/tile; bank budgets 256/1024/2048; max versus
top-1%-patch mean. One frozen backbone/layer pair, one blocked search backend.
Normal development diagnostics and explicitly synthetic inner-fitting holdout
probes support selection. All variants, timings and decisions must be retained.
## 17. Candidate selection
Four-category mean (proxy image AUROC + proxy pixel AP)/2, then latency/bank size
for exact ties. No official test-label tuning. Selection is pending.
## 18. Threshold calibration
Reserved normal partition only; image higher q95 and separate pixel q99.5.
Report counts, strict > decisions, test FPR and uncertainty. Pending freeze.
## 19. Frozen configuration
Freeze complete source, weight, coreset, preprocessing, scoring and calibration
identities before confirmation. No freeze or final evaluation is claimed yet.
## 20. All-category evaluation
Pending the measured gate, then all 15 categories, preferably three coreset seeds.
Cache compatible deterministic features instead of extracting them repeatedly.
## 21. Anomaly maps and overlays
Final original/heatmap/overlay/native-ground-truth panels are pending evaluation.
## 22. Failure analysis
Investigate small scratches, texture, structure, memory coverage and threshold
instability using observed data. Do not infer causes from unseen results.
## 23. Runtime and GPU/CPU memory
Pilot measures extraction, coreset, search, RSS sampling and CUDA peaks. Final
latency will use 50 warmups/200 batch-one timed runs, plus separate end-to-end cost.
## 24. Comparison with CNN and Autoencoder
Pending final results. Read exact stored summaries and category-matched scores;
do not substitute rounded references or select a winner on AUROC alone.
## 25. Conclusions and limitations
Only implementation/pilot evidence exists at this chronological point. The
variant omits paper reweighting and smoothing, caps coreset candidates and uses
ImageNet transfer with a localization geometry override. The public notebook
removes only licensed image-panel outputs; the full executed pilot stays local.
No SuperSimpleNet work is part of this milestone.
""")]
n.cells[-2].metadata['local_dataset_pixels']=True
# The panel is before the final pending section; locate it explicitly.
for c in n.cells:
    if c.cell_type=='code' and 'fig,axes=plt.subplots' in c.source:c.metadata['local_dataset_pixels']=True
folder=root/'outputs/patchcore';folder.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(folder/'ipython');(folder/'ipython').mkdir(exist_ok=True)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
NotebookClient(n,timeout=900,kernel_name='mvtec-research',resources={'metadata':{'path':str(root)}}).execute()
(folder/'03_patchcore.pilot.executed.ipynb').write_bytes(nbf.writes(n).encode('utf-8'))
for c in n.cells:
    if c.metadata.get('local_dataset_pixels'):c.outputs=[]
(root/'notebooks/03_patchcore.ipynb').write_bytes(nbf.writes(n).encode('utf-8'))
