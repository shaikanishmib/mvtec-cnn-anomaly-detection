"""Execute an honest architecture/diagnostic notebook; never fabricate outputs."""
from pathlib import Path
import os,asyncio,nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
n=nbf.v4.new_notebook()
n.metadata.kernelspec=dict(display_name='MVTec research (CUDA, Python 3.12)',language='python',name='mvtec-research')
n.cells=[md(r'''# Notebook 04 — SuperSimpleNet

## Research questions and execution status
Can learned discrimination of spatially masked feature perturbations improve real defect detection, calibration and inference cost? Does the feature adapter help? Can a smaller extractor preserve useful discrimination?

This chronological architecture and pilot record precedes development selection and official test access. Cells explicitly labeled **artifact replay** load genuine earlier measurements; subsequent diagnostic cells execute live. The complete benchmark is **not complete** at this point. Earlier models, records and partitions are preserved.

## Original architecture and sources
Read the [ICPR paper, arXiv v2](https://arxiv.org/html/2408.03143v2) alongside the [official source pinned at 98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d](https://github.com/blaz-r/SuperSimpleNet/tree/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d). The source's later JIMS default uses unadapted classifier features; here the original ICPR adapted-classifier path is explicit. MIT attribution is retained under `docs/third_party/`.

Basic CNN learned normal/synthetic RGB discrimination. The autoencoder minimized reconstruction error. PatchCore stored normal local features and searched nearest neighbors. SuperSimpleNet instead learns dense and image-level discrimination after frozen pretrained feature extraction, with synthetic perturbations inserted into adapted features rather than RGB images. Inference needs no memory-bank search and no synthetic generator.

## Mathematical flow
Let $F_2,F_3$ be frozen ImageNet features. First align both to $64\times64$, concatenate, and apply average pooling: $z=P_3[U_2F_2(x);U_4F_3(x)]$. A learned pointwise affine adapter gives $a=W_a*z+b_a$.

Training duplicates $a$ and independently perturbs **both** copies: $\tilde a=a+M\odot\epsilon$, $\epsilon\sim\mathcal N(0,0.015^2 I)$. $M=1[p>0.2]$ for Perlin gradient noise $p$, with approximately half of masks forced empty. Dense target is $M$ and image target is $y=1[\max M>0]$. Clean and altered features share the same normal source distribution. No real defects enter training.

The segmentation branch is $s=W_2*\operatorname{LeakyReLU}(\operatorname{BN}(W_1*\tilde a))$. The detector computes $d=\operatorname{ReLU}(\operatorname{BN}(W_5*[\tilde a;\operatorname{stopgrad}(s)]))$ and a linear scalar from max/mean pooling of both $d$ and $s$. Detaching every segmentation route prevents classifier gradients from corrupting localization, while classification still updates the adapter through its feature input.

With focal $\ell_f(p,t)=-(1-p_t)^4\log p_t$, optimize $L=L_{\rm margin}+L_f(s,M)+L_f(c,y)$. The margin averages normal and anomaly pixels **separately**, using $\max(0,s+0.5)$ and $\max(0,0.5-s)$. Empty-class means contribute zero. AdamW uses adapter LR $10^{-4}$, head LR $2\cdot10^{-4}$, group-specific decay, and fixed scheduler fractions .8/.9. Final pixel map is bilinear-upsampled raw segmentation logits followed by Gaussian smoothing $\sigma=4$. The detector produces a separate raw image logit; max/top1% map rules are development candidates, not interchangeable probabilities.

Complexity is fixed per image: frozen CNN plus pointwise adapter/segmentation and a 5x5 classifier convolution. Unlike PatchCore, inference does not scale with normal memory size. Dense 64-grid heads and feature-cache IO can still be expensive.

## Fidelity and intentional differences
Reference widths are 1536/1024/128 with WideResNet50-2 **ImageNet V1**. Lite uses ResNet18 V1 and 384/256/32. We preserve established whole-image 256 geometry and normal-only partitions, use reduced epoch/batch/seed budgets, and add gradient clipping at10. Our independently seeded Perlin implementation is not RNG-identical to the official one. Raw logits avoid official test-derived min/max normalization. These reduced-budget models are not claimed to reproduce paper results. See the complete preregistered [protocol](../reports/supersimplenet/protocol.md).
'''),code('''from pathlib import Path
import sys,json,time,inspect
ROOT=Path.cwd()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt,torch,psutil
from anomaly_detection.supersimplenet import *
from anomaly_detection.ssn_research import experiment,feature_cache,model_for,predict
from anomaly_detection.cnn_research import read_partition,require_fitting,load_resized
from anomaly_detection.data import CATEGORIES,sha256
torch.set_num_threads(4)
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32=False
print(torch.__version__,torch.cuda.get_device_name(0))
print('VRAM GiB:',torch.cuda.get_device_properties(0).total_memory/2**30)
print('RAM GiB:',psutil.virtual_memory().total/2**30)
'''),md('''## Architecture illustration
Only the training path injects noise. Calibration and inference follow the clean adapted-feature path. All masks below are synthetic feature-space targets.'''),code('''from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(12,3.3));ax.set(xlim=(0,12),ylim=(0,3));ax.axis('off')
boxes=[(.1,1.3,1.6,'RGB256\\nfrozen CNN'),(2,1.3,1.6,'Align64 + pool\\nadapter'),(4,1.3,1.6,'Train only:\\nPerlin + noise'),(6,1.8,1.6,'Segmentation\\n1x1 head'),(6,.3,1.6,'Detection 5x5\\n+ max/mean'),(8.5,1.8,2.7,'Upsample + Gaussian\\npixel anomaly logits'),(8.5,.3,2.7,'Linear detector\\nimage anomaly logit')]
for x,y,w,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,.65,boxstyle='round,pad=.06',fc='#e7eff8',ec='#365776'));ax.text(x+w/2,y+.325,t,ha='center',va='center',fontsize=9)
for a,b in [((1.7,1.6),(2,1.6)),((3.6,1.6),(4,1.6)),((5.6,1.6),(6,2.1)),((5.6,1.6),(6,.65)),((7.6,2.1),(8.5,2.1)),((7.6,.65),(8.5,.65)),((6.8,1.8),(6.8,.95))]:
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#365776'))
ax.text(7.05,1.3,'detach',fontsize=8);fig.tight_layout()
fig.savefig(ROOT/'reports/supersimplenet/architecture.svg');plt.show()
'''),md('''## Dataset isolation and preprocessing
The shared loader verifies seed-2026 manifests. Fitting and development contain normals only. Pilot fitting/probe sources are disjoint; development normals are never perturbed in their clean diagnostic distribution. Reserved calibration and official test data are not opened in this notebook stage. PIL whole-image RGB bilinear resizing to256 is followed by the correct ImageNet V1 normalization.
'''),code('''fit=read_partition('fit','bottle');dev=read_partition('development','bottle')
require_fitting(fit)
assert not set(fit.path)&set(dev.path)
print('Categories:',CATEGORIES)
print('Bottle fitting/development:',len(fit),len(dev))
print('Pilot fitting/probe/development counts:',32,8,16)
'''),md('''## Implementation: frozen feature extraction and alignment
The actual reusable implementation is exposed below. Upsampling occurs before pooling; existing PatchCore caches have incompatible geometry and are not reused. Deterministic per-image caches are reused across trials/seeds.'''),code('''print(inspect.getsource(SSNExtractor))
print(inspect.getsource(aggregate_features))
'''),md('''## Feature anomaly implementation and exact targets
Perlin noise uses smooth gradient-lattice interpolation. A binary spatial mask gates independently sampled channel noise; its actual nonempty support determines the image label. The clean cached feature tensor is not modified.'''),code('''print(inspect.getsource(perlin_noise))
print(inspect.getsource(feature_anomalies))
'''),md('''## Adapter, segmentation and detection heads; losses and training step
The source shows exactly where classification gradients stop. Synthetic generation is explicit and cannot occur in ordinary forward/inference. Margin and focal losses are implemented directly.'''),code('''print(inspect.getsource(SuperSimpleHeads))
print(inspect.getsource(focal_loss));print(inspect.getsource(training_loss))
print(inspect.getsource(optimizer_for));print(inspect.getsource(anomaly_maps))
from anomaly_detection.ssn_research import train_epoch
print(inspect.getsource(train_epoch))
'''),md('''## Initial verification and chronological pilot: artifact replay
Five CPU tests passed for feature shapes/freezing, exact perturbation support, losses/gradients, detached classifier routing, checkpoint replay, score orientation and aggregation. The following results were measured by `scripts/pilot_supersimplenet.py`; this cell replays those artifacts and does not claim to retrain them live.

**Hypothesis R:** the reference-width architecture fits safely and is affordable for45 fits. **Hypothesis L:** 20 Lite epochs can learn the reference perturbation on32 fitting normals. Neither hypothesis uses official defect data.'''),code('''reference=json.loads((ROOT/'reports/supersimplenet/reference_feasibility.json').read_text())
pilot=json.loads((ROOT/'reports/supersimplenet/runs/04-pilot-lite-bottle-s2026-v1-result.json').read_text())
print({k:reference[k] for k in ['layer_shapes','aggregated_shape','segmentation_shape','detection_shape','head_parameters','mean_step_seconds','projected_40_epoch_45_fit_training_seconds','training_resources']})
print('Lite parameters:',pilot['parameters'],'resources:',pilot['resources'])
print('Checkpoint replay exact:',pilot['checkpoint_replay_exact'])
display(pd.DataFrame(pilot['development']['candidates']).T)
h=pd.DataFrame(pilot['history']);fig,axes=plt.subplots(1,2,figsize=(10,3))
h.plot(x='epoch',y=['train_loss','margin_loss'],ax=axes[0]);h.plot(x='epoch',y=['proxy_image_auroc','proxy_pixel_ap'],ax=axes[1]);plt.show()
'''),md('''**Measured interpretation and decision:** the reference path fits memory but projects beyond the prespecified 90-minute training budget, so it is rejected for broad study. Lite losses decrease but the margin stays near1, image discrimination is below chance and localization AP remains weak on this tiny proxy. This does not establish useful learning. Do not launch the all-category benchmark.

## Bounded diagnosis: stronger planned perturbation — live training
**Hypothesis N:** the planned .05 standard deviation provides a more learnable signal than .015 in the reduced-width, short-budget pilot. All other settings, seed, 32 normal fitting sources, eight disjoint probe sources and 16 clean development normals are unchanged. This is one predeclared magnitude candidate, not a search over real defects. The following code performs training live and preserves every epoch and checkpoint.'''),code('''config=dict(backbone='resnet18',channels=384,seg_width=256,det_width=32,adapter_trainable=True,
    noise_std=.05,loss='margin_focal',seed=2026,batch_size=8,extraction_batch=8,epochs=20,
    validation_every=2,hypothesis='Larger planned feature perturbations improve learnability under the same bounded Lite pilot')
extractor=SSNExtractor('resnet18').cuda()
diagnostic=experiment('04-pilot-lite-noise05-bottle-s2026-v1','bottle',config,extractor,phase='pilot',pilot=True)
display(pd.DataFrame(diagnostic['development']['candidates']).T)
print('Checkpoint exact:',diagnostic['checkpoint_replay_exact'],'resources:',diagnostic['resources'])
'''),md('''## Synthetic feature diagnostics — live inference
Plot a clean feature channel, planted mask, perturbation magnitude and predicted dense map. These are feature-space diagnostics, not real industrial anomaly evidence. The fixed seeded generator supplies exactly the same proxy support as development scoring.'''),code('''sources=feature_cache(extractor,fit.iloc[32:40],8)
head=model_for(config).cuda().eval()
head.load_state_dict(torch.load(ROOT/diagnostic['checkpoint'],weights_only=True,map_location='cpu')['state_dict'])
with torch.no_grad():
    z=sources.batch(range(8));a=head.adapter(z)
    perturbed,mask,label=feature_anomalies(a,torch.Generator().manual_seed(9901),torch.Generator(device='cuda').manual_seed(9902),noise_std=.05)
    seg,score=head.discriminate(perturbed)
    index=int(torch.where(label>0)[0][0]);delta=(perturbed-torch.cat([a,a])).abs().mean(1)
    maps=anomaly_maps(seg)
    assert torch.isfinite(maps).all() and torch.isfinite(score).all()
    assert torch.equal(delta.ne(0),mask[:,0].bool())
fig,axes=plt.subplots(1,4,figsize=(11,3))
for ax,value,title in zip(axes,[a[index%8,0],mask[index,0],delta[index],maps[index]],['Clean feature channel','Exact synthetic target','Mean absolute perturbation','Predicted map']):
    ax.imshow(value.cpu(),cmap='viridis');ax.set_title(title);ax.axis('off')
plt.show()
display(pd.DataFrame([dict(trial=r['experiment_id'],image_auroc=r['development']['candidates']['head']['auroc'],pixel_ap=r['development']['candidates']['head']['pixel_ap'],last_margin=r['history'][-1]['margin_loss'],seconds=r['wall_seconds']) for r in [pilot,diagnostic]]))
'''),md('''## Gate and next research stages
The measured pilot report and diagnostic decision are stored alongside the notebook. No official test metrics, threshold calibration, frozen final configuration or all-category completeness is claimed here. The next step depends on demonstrated learning on normal-derived synthetic proxies, not a lower loss alone.

The preregistered bounded development study covers bottle/cable/carpet/tile, adaptation, noise and loss behavior, with head/max/top1% aggregation. Calibration will remain separate: higher q95 image and q99.5 pixel normal quantiles, strict `>` decisions, sample counts and uncertainty. Final fitting must precede official test evaluation; no retraining after inspection.

## Comparison and limitations at this stage
No comparison to completed Basic CNN, Autoencoder or PatchCore quality is justified from a tiny synthetic pilot. Prior exact stored results remain unchanged. The future comparison must include matched category scores, precision/recall/FPR/localization, paired uncertainty and measured inference cost, with special attention to leather/carpet false positives and screw detection. Synthetic discrimination may not transfer to real defects; a 64-grid feature representation can lose fine details; tiny held-out sets produce unstable diagnostics; reduced training and widths differ from the publication.

Raw images, weights, dense caches and checkpoints remain local and ignored. This public notebook contains genuine compact outputs. Its training cell executed live; the earlier reference/Lite record was explicitly replayed. The experiment registry is append-only under [experiments](../experiments/), with configurations, hashes and decisions. Full local executed notebook: `outputs/supersimplenet/04_supersimplenet.pilot.executed.ipynb`.
''')]
out=ROOT/'outputs/supersimplenet';out.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(out/'ipython');(out/'ipython').mkdir(exist_ok=True)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    NotebookClient(n,timeout=900,kernel_name='mvtec-research',resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    (out/'04_supersimplenet.pilot.executed.ipynb').write_bytes(nbf.writes(n).encode())
    (ROOT/'notebooks/04_supersimplenet.ipynb').write_bytes(nbf.writes(n).encode())
