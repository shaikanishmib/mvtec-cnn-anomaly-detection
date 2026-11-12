"""Build Notebook 01 once from immutable measured runs and exact executed source.

The notebook is canonical after creation; do not regenerate over manual narrative.
Source injection avoids maintaining a second handwritten implementation.
"""
from pathlib import Path
import inspect
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import nbformat as nbf
import pandas as pd
from anomaly_detection import basic_cnn as architecture, cnn_research as research
from anomaly_detection import cnn_evaluation as evaluation, pixel_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/"notebooks/01_basic_cnn.ipynb"
    if target.exists():raise FileExistsError("Preserve the existing scientific narrative; edit deliberately")
    summary=json.loads((ROOT/"reports/basic_cnn/summary.json").read_text())
    final=pd.read_csv(ROOT/"reports/basic_cnn/all_runs.csv")
    weakest=final.groupby("category").image_auroc.mean().sort_values().head(3)
    trials=pd.read_csv(ROOT/"reports/basic_cnn/development_comparison.csv")
    references=trials[trials.variant.eq("base")].set_index("category")
    trial_notes=[]
    for row in trials.itertuples():
        ref=references.loc[row.category]
        decision="keep selected configuration" if row.variant=="dice" else "reject for final confirmation"
        delta=row.proxy_recall-ref.proxy_recall
        comparison=("Initial reference for this category." if row.variant=="base" else
                    f"Recall difference from the same-category baseline: {delta:+.3f}; this is a synthetic-proxy comparison.")
        trial_notes.append(f"**{row.experiment_id}** — {row.category}, {row.variant}: proxy recall {row.proxy_recall:.3f}, pixel AP {row.proxy_pixel_ap:.3f}, selected epoch {row.best_epoch}, peak {row.peak_vram_gib:.3f} GiB. {comparison} Decision: {decision} under the predefined macro criterion. Exact configuration, complete curve and checkpoint identity are preserved in the linked registry/run artifacts.")
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def section(number,title,before,code,after,private=False):
        md(f"## {number}. {title}\n\n{before}")
        cell=nbf.v4.new_code_cell(code.strip())
        if private:cell.metadata["contains_dataset_images"]=True
        cells.append(cell);md(after)
    def source(obj):return inspect.getsource(obj).rstrip()
    md('''# 01 — Basic CNN anomaly detection

An educational, scratch-trained anomaly segmentation baseline on all 15 MVTec AD
categories. This narrative preserves **20 controlled development fits and 45 fresh
final models**, including rejected configurations and disappointing real results.
It follows [the preregistration](../docs/basic_cnn_protocol.md) and the frozen
policy. Notebook 00 already viewed test annotations descriptively; the claim is
**no test-label tuning**, never unseen test labels.

This notebook execution **replays actual immutable results**; it does not silently
retrain or repeat test evaluation. Exact architecture, generator, loss and training
loop source is injected below from the modules actually executed. Actual driver
commands and configurations are shown alongside verified artifacts. Optional
reproduction flags create new IDs and are off by default. A later repeat is not a
new untouched confirmation. Dataset-pixel panels remain in the local full executed
copy; numeric outputs, curves and aggregate plots are published.''')
    section(1,"Objectives and research hypotheses",
    '''H1: a small CNN can learn local synthetic edits without pretrained features.
H2: diverse patch/perturbation training transfers better to a held-out mixed proxy
than either alone. H3: Dice helps sparse localization. H4: real-defect transfer may
be weak because synthetic masks describe edits, not benchmark annotations.
The five-configuration search and three final seeds were bounded before training.''',
    '''from pathlib import Path
import sys, json, copy, time, traceback, inspect
from datetime import datetime, timezone
ROOT=Path.cwd().resolve()
if ROOT.name=="notebooks":ROOT=ROOT.parent
assert (ROOT/"configs/basic_cnn_frozen.json").is_file()
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from IPython.display import display, Code
from anomaly_detection.data import CATEGORIES, TEXTURES, autodiscover_data_root, sha256
from anomaly_detection.tracking import record, config_hash
from anomaly_detection.metrics import binary_metrics, calibrate_normal
from anomaly_detection.basic_cnn import seed_everything, image_scores
from anomaly_detection.cnn_research import (read_partition, require_fitting, inner_fit_split,
    load_resized, make_probes, development_metrics, code_identity, predict_logits)
from anomaly_detection.cnn_evaluation import evaluate_frozen
from environment import inspect as environment_inspect
from verify_cuda import verify_gpu
OUT=ROOT/"outputs/basic_cnn/notebook";OUT.mkdir(parents=True,exist_ok=True)
REPORT=ROOT/"reports/basic_cnn"
plan=json.loads((ROOT/"configs/basic_cnn_plan.json").read_text())
frozen=json.loads((ROOT/"configs/basic_cnn_frozen.json").read_text())
assert sha256(ROOT/"configs/basic_cnn_frozen.json")==json.loads((REPORT/"freeze_identity.json").read_text())["sha256"]
EXECUTE_NEW_PILOT=False
EXECUTE_NEW_DEVELOPMENT=False
EXECUTE_NEW_CONFIRMATION=False
RUN_PREFIX="01-reproduction-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
display(plan)
print("Replay mode: no new fitting or official-test inference.")''',
    '''The displayed plan is the exact bounded search, not a retrospective grid.
Every subsequent result is read from a recorded experiment with configuration and
artifact identities. Rejected attempts remain first-class scientific evidence.''')
    section(2,"How a convolutional network works",
    r'''A convolutional layer learns local cross-correlations:
$z_{k,i,j}=b_k+\sum_{c,u,v}W_{k,c,u,v}x_{c,i+u,j+v}$. ReLU retains positive
responses. Stride-2 layers expand receptive fields while reducing spatial size.
The same filters scan all locations, sharing parameters instead of memorizing
one dense weight per image pixel. This numeric check exposes one actual output.''',
    '''seed_everything(2026)
convolution=nn.Conv2d(3,4,3,stride=2,padding=1)
x=torch.rand(2,3,32,32)
z=convolution(x)
padded=F.pad(x,(1,1,1,1))
manual=(padded[0,:,:3,:3]*convolution.weight[0]).sum()+convolution.bias[0]
torch.testing.assert_close(z[0,0,0,0],manual)
print("Input:",tuple(x.shape),"stride-2 output:",tuple(z.shape),"ReLU:",tuple(F.relu(z).shape))
print("One output agrees with the explicit filter sum:",float(manual.detach()))''',
    '''This is a local linear filter followed by nonlinearity, not an anomaly score
by itself. Learning requires an objective. Segmentation preserves spatial output;
the image alarm later aggregates the learned map under a frozen rule.''')
    section(3,"Why ordinary supervised classification is inappropriate",
    '''Official training images contain only normals. A normal-versus-real-defect
classifier has no real positive training examples. We instead supervise local
synthetic edits, with explicitly synthetic masks. Official defects and masks are
never smuggled into training, development, calibration or donor selection.''',
    '''bottle_fit=read_partition("fit","bottle")
require_fitting(bottle_fit)
assert bottle_fit["mask"].isna().all()
assert bottle_fit.path.str.contains("/train/good/").all()
print("Eligible bottle fitting normals:",len(bottle_fit))
print("Real defective training examples: none. Synthetic targets are a proxy task.")''',
    '''A synthetic-positive label means that the generator altered an input pixel.
It does not assert what a real industrial defect looks like. Final benchmark
annotations retain their original meaning and are used only after selection.''')
    section(4,"Dataset summary and relevant EDA",
    '''Notebook 00 validated all original files and masks. Native images are square
700–1024 pixels; grid, screw and zipper are grayscale. Category imbalance and
small calibration sets matter more than a single pooled accuracy number.''',
    '''counts=pd.read_csv(ROOT/"reports/category_counts.csv").set_index("category")
eda=json.loads((ROOT/"reports/eda_findings.json").read_text())
display(counts)
print("Images:",int(counts.total.sum()),"training normals:",int(counts.train_normal.sum()))
print("Native resolutions:",eda["native_resolution_by_category"])
print("Grayscale categories:",eda["grayscale_categories"])''',
    '''The 3,629 official training normals are already partitioned. This model does
not gain access to held-out normals merely because they share the original train
directory. Image AP must be read relative to each category's defect prevalence.''')
    section(5,"Environment and actual GPU verification",
    '''Use the registered isolated `.venv` kernel. Float32 is retained because the
pilot is fast and uses little VRAM; AMP is not introduced without a separate
stability comparison. Deterministic initialization and supported deterministic
operations are enforced; independent seeds measure stochastic training variation.''',
    '''assert Path(sys.prefix).resolve()==(ROOT/".venv").resolve()
hardware=environment_inspect();gpu=verify_gpu()
assert gpu["cuda_available"] and gpu["isolated"]
display(hardware);display(gpu)
print("Training precision: float32; cudnn benchmark disabled; deterministic algorithms enabled.")''',
    '''The GPU checks are synthetic plumbing checks, not model training or benchmark
quality. All recorded runs use the same pinned environment. Timing results later
state their power context, warmup, synchronization and input/output boundaries.''')
    section(6,"Exact fitting, development, calibration and test partitions",
    '''Frozen outer manifests are unchanged. During selection, 20% of fitting
normals become an internal proxy source, disjoint from gradient images. Only
fitting normals receive synthetic edits. Official development stays normal;
calibration/test remain closed until selection. Final models rejoin all fitting
normals under the already selected fixed eight-epoch policy.''',
    '''partition_metadata=json.loads((ROOT/"reports/dataset_validation.json").read_text())
for part,entry in partition_metadata["partitions"].items():
    assert sha256(ROOT/entry["path"])==entry["sha256"]
display(pd.read_csv(ROOT/"reports/partition_counts.csv"))
gradient_rows,probe_rows=inner_fit_split(bottle_fit)
assert not set(gradient_rows.sha256)&set(probe_rows.sha256)
print("Pilot bottle gradient normals:",len(gradient_rows),"held-out fitting probe normals:",len(probe_rows))
print("Final fitting counts are the full fit column; no outer development/calibration gradients.")''',
    '''The internal split is a development mechanism within fitting, not a new
official benchmark split. Development proxy masks are synthetic and confined to
fitting images. Outer calibration/test identities remain separate and fixed.''')
    section(7,"Image preprocessing",
    '''Convert to RGB, resize the whole image to 256×256 with PIL's antialiased
bilinear filter, retain uint8 caches and divide by 255 at batching. The network
uses fixed $2x-1$ centering; no means/variances are learned from test data. Cache
identity includes source paths, SHA256 hashes, resolution and transform version.''',
    source(research.load_resized)+'''

example_uint8=load_resized(gradient_rows.iloc[:4],256)
print("Cached dtype/shape:",example_uint8.dtype,tuple(example_uint8.shape))
print("Float range:",float((example_uint8.float()/255).min()),float((example_uint8.float()/255).max()))''',
    '''RGB conversion replicates grayscale channels without inventing color. Both
training generators preserve grayscale coherence. Resizing discards detail;
predicted continuous maps later return to native size, while official masks are
never bilinearly blurred or resized for the primary evaluation.''')
    section(8,"Two fitting-only synthetic defect strategies",
    '''A shifted donor-normal patch or localized procedural color/intensity field
is blended into a soft irregular ellipse. Random position, orientation, scale and
blend strength avoid a fixed border/global cue. Half the training images remain
normal. The target is the **actual nonzero channel difference**, not a nominal
support that might include unchanged pixels. Donors come from the same fitting
batch/category. This simple patch idea is inspired by
[CutPaste](https://arxiv.org/abs/2104.04015), not a reproduction of that method.''',
    source(architecture.synthetic_batch)+'''

normal=example_uint8[:2].cuda().float()/255
examples=[];checks=[]
for strategy in ("patch","perturb"):
    changed,mask=synthetic_batch(normal,torch.Generator(device="cuda").manual_seed(9901),strategy,0.)
    assert torch.equal(mask.bool(),changed.ne(normal).any(1,keepdim=True))
    checks.append({"strategy":strategy,"edited_pixels":int(mask.sum()),"exact_difference_mask":True})
    for i in range(len(normal)):examples.append((strategy,normal[i].cpu(),changed[i].cpu(),mask[i,0].cpu()))
fig,axes=plt.subplots(len(examples),4,figsize=(10,9))
for row,(strategy,original,changed,mask) in enumerate(examples):
    axes[row,0].imshow(original.permute(1,2,0));axes[row,0].set_title("fit normal")
    axes[row,1].imshow(changed.permute(1,2,0));axes[row,1].set_title(strategy)
    axes[row,2].imshow(mask,cmap="gray",vmin=0,vmax=1);axes[row,2].set_title("synthetic mask")
    axes[row,3].imshow(changed.permute(1,2,0));axes[row,3].imshow(np.ma.masked_where(mask==0,mask),cmap="Reds",alpha=.45,vmin=0,vmax=1)
    for ax in axes[row]:ax.axis("off")
plt.tight_layout();plt.savefig(OUT/"fitting_synthetic_examples.png",dpi=100);plt.show()
(REPORT/"synthetic_mask_checks.json").write_text(json.dumps(checks,indent=2)+"\\n")
display(checks)''',
    '''All shown inputs are fitting normals; no official defect or annotation is
used. Exact-mask and unchanged-border tests passed for both strategies, including
grayscale inputs. These synthetic edits still differ from real cracks, missing
parts and subtle surface changes. The local figure is retained; dataset pixels
are intentionally omitted from the published notebook.''',private=True)
    section(9,"Architecture implementation and intermediate feature maps",
    '''Three Conv/GroupNorm/ReLU blocks halve spatial resolution. GroupNorm has no
running batch statistics to encode the synthetic/normal mixture. A light head
combines shallow detail and repeated coarse context, then repeats logits back to
input size. Nearest-repeat upsampling supports deterministic gradients and makes
the limited effective localization resolution explicit. There is no pretrained
backbone, teacher, memory bank or reconstruction objective.''',
    source(architecture.BasicCNN)+'''

seed_everything(2026)
demo=BasicCNN(16).cuda()
with torch.no_grad():
    sample=torch.rand(2,3,256,256,device="cuda")
    shallow=demo.encoder1(2*sample-1);middle=demo.encoder2(shallow);coarse=demo.encoder3(middle)
    prediction=demo(sample)
print("Input, shallow, middle, coarse, logits:",[tuple(t.shape) for t in (sample,shallow,middle,coarse,prediction)])
print("Trainable parameters:",sum(p.numel() for p in demo.parameters()))
print(demo)''',
    '''The measured width-16 model has 83,857 parameters; width 8 has 21,193.
Image logits preserve input shape, but repeated predictions cannot recover detail
lost by downsampling. A probability heatmap is sigmoid(logit); the frozen image
score is its maximum, without test-set normalization or smoothing selection.
These are synthetic-discriminator scores, not calibrated posterior probabilities
of real industrial defects.''')
    section(10,"Loss and gradient flow",
    r'''Weighted BCE uses $-w y\log p-(1-y)\log(1-p)$, with positive weight 8.
The ablation adds $1-(2\sum py+1)/(\sum p+\sum y+1)$ per image. BCEWithLogits
computes the sigmoid/log terms stably; Dice addresses sparse overlap but may
encourage unwanted responses on normals. Its weight is selected only by proxy
objectives. The following small tensor fixture demonstrates real autograd.''',
    source(architecture.segmentation_loss)+'''

demo.zero_grad(set_to_none=True)
fixture_inputs=torch.rand(2,3,64,64,device="cuda",requires_grad=True)
fixture_logits=demo(fixture_inputs)
fixture_mask=torch.zeros_like(fixture_logits);fixture_mask[0,:,20:34,25:40]=1
loss=segmentation_loss(fixture_logits,fixture_mask,8.,1.)
loss.backward()
print("Fixture loss:",float(loss.detach()))
print("Encoder gradient norm:",float(demo.encoder1[0].weight.grad.norm()))
assert torch.isfinite(fixture_inputs.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in demo.parameters())
del demo,sample,shallow,middle,coarse,prediction,fixture_logits,fixture_inputs,fixture_mask,loss
torch.cuda.empty_cache()''',
    '''This is a synthetic tensor calculation, not a claimed benchmark result.
Gradients reach the early encoder through the segmentation head. Actual training
also checks finite losses/gradient norms and clips gradients at norm 10, with
AdamW updates and no unvalidated mixed-precision switch.''')
    section(11,"Actual training loop and immutable logging",
    '''The exact batch loop and experiment controller used by the driver are shown
below as executable definitions, not hidden behind a single model-building call.
Selection uses mean(normal-development BCE, fixed-probe objective), minimum three
epochs and patience three, with at most eight. Every start/result/failure gets a
new append-only registry event; source snapshots and best/last/RNG checkpoints stay
local. Final training deliberately uses a fixed duration instead of test peeking.
Each resized category fits in a GPU uint8 cache, so batching uses explicit seeded
indices with no DataLoader workers or Windows multiprocessing. Cold preprocessing
is included in total run time and separated from epoch timing.''',
    source(research.train_epoch)+"\n\n"+source(research.train_experiment),
    '''Each epoch changes fitting-derived synthetic edits using its recorded RNG,
while the probe is fixed with seed 9901. Official development inputs remain
unmodified normals and never receive gradients. OOM/nonfinite/budget failures
would be recorded rather than silently changing settings. No training OOM occurred.''')
    section(12,"Initial bottle experiment",
    '''The actual pilot command was `.venv/Scripts/python.exe -X utf8
scripts/run_basic_cnn.py pilot`. It ran before the controlled ablations or any new
official test inference. Replay verifies its configuration, curve and checkpoint
identities; the optional branch creates a new experiment ID and never overwrites it.''',
    '''def verified_run(identifier):
    result=json.loads((REPORT/f"runs/{identifier}-result.json").read_text())
    assert result["configuration_sha256"]==config_hash(result["configuration"])
    assert sha256(ROOT/result["history"])==result["history_sha256"]
    checkpoint=ROOT/result["checkpoint"]
    if checkpoint.exists():assert sha256(checkpoint)==result["checkpoint_sha256"]
    else:print("Checkpoint is local-only; reproduce training to obtain:",result["checkpoint"])
    return result
pilot=verified_run("01-pilot-base-bottle-s2026-v1")
if EXECUTE_NEW_PILOT:
    pilot=train_experiment(RUN_PREFIX+"-pilot-bottle","bottle",pilot["configuration"],
        "Registered reproduction of initial synthetic-supervised pilot",phase="pilot")
display(pilot["configuration"])
display({k:pilot[k] for k in ("experiment_id","best_epoch","epochs_completed","parameters","gradient_normal_images","probe_normal_sources","peak_allocated_bytes","training_seconds_including_loading_validation")})
display(pilot["selected_development_metrics"])''',
    '''The pilot showed clear learning: training loss 0.427→0.204; selected epoch 7
had proxy image AUROC 0.932, pixel AP 0.701 and recall 0.897 at the normal-development
q95 threshold. Peak VRAM was 0.51 GiB. This technically justified the bounded
development phase, not a claim of real-defect success or calibrated test recall.''')
    section(13,"Training and permitted validation curves",
    '''Curves below are the actual saved per-epoch values. Weighted training/probe
losses and unweighted normal BCE have different targets, so their scales are not
interchangeable. Proxy AP/recall diagnose learning independently of normal loss;
a constant all-normal predictor would not solve the proxy task.''',
    '''history=pd.read_csv(ROOT/pilot["history"])
display(history)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for key in ("train_loss","probe_loss","normal_development_bce","stopping_objective"):
    axes[0].plot(history.epoch,history[key],marker="o",label=key)
for key in ("proxy_pixel_ap","proxy_recall_at_dev_normal_q95"):
    axes[1].plot(history.epoch,history[key],marker="o",label=key)
for ax in axes:ax.set_xlabel("Epoch");ax.legend(fontsize=8);ax.grid(alpha=.25)
axes[0].set_ylabel("Objective");axes[1].set_ylabel("Synthetic proxy metric")
plt.tight_layout();plt.savefig(OUT/"pilot_curves.png",dpi=100);plt.show()''',
    '''The curves contain nonmonotonic normal responses and proxy operating points,
not just an improving headline. Selection follows the preregistered joint
objective. No extra epochs, test masks or post-test repairs were used to make
these curves or the later confirmation appear stronger.''')
    section(14,"Capacity experiment",
    '''Compare width 8 and width 16 on the same four predetermined categories,
seeds, probe masks, data policy and optimizer. This is a controlled small search,
not a grid selected from favorable official test results.''',
    '''development_reproductions=[]
if EXECUTE_NEW_DEVELOPMENT:
    base={k:plan[k] for k in ("seed","resolution","batch_size","learning_rate","weight_decay","max_epochs","minimum_epochs","patience","normal_probability","positive_weight")}
    for variant in plan["variants"]:
        for category in plan["development_categories"]:
            config=dict(base,**variant)
            if EXECUTE_NEW_PILOT and variant["name"]=="base" and category=="bottle":
                reproduced=pilot
            else:
                reproduced=train_experiment(f"{RUN_PREFIX}-dev-{variant['name']}-{category}",category,config,
                    "Explicit registered reproduction of bounded development; no new policy selection",phase="development")
            development_reproductions.append({"experiment_id":reproduced["experiment_id"],"category":category,"variant":variant["name"],**reproduced["selected_development_metrics"]})
    display(pd.DataFrame(development_reproductions))
development=pd.read_csv(REPORT/"development_comparison.csv")
development_macro=pd.read_csv(REPORT/"development_macro.csv").set_index("variant")
display(development[development.variant.isin(["base","small"])])
display(development_macro.loc[["base","small"]])''',
    '''The smaller model reduced parameter count and memory but had lower macro
proxy recall/AP. This result justifies capacity only on the synthetic development
task. It does not establish that a larger CNN universally detects real anomalies.''')
    section(15,"Synthetic-design experiments",
    '''Patch-only and perturbation-only candidates train against the same fixed
mixed proxy as the baseline. Keeping the evaluation generator independent of each
candidate avoids giving an ablation a custom, easier validation target.''',
    '''display(development[development.variant.isin(["base","patch","perturb"])])
development_macro.loc[["base","patch","perturb"],["proxy_recall","proxy_pixel_ap"]].plot.bar(figsize=(9,4),ylim=(0,1))
plt.ylabel("Synthetic proxy metric");plt.title("Same held-out mixed proxy, four-category macro")
plt.xticks(rotation=0);plt.tight_layout();plt.savefig(OUT/"synthetic_ablation.png",dpi=100);plt.show()''',
    '''Patch-only recall averaged 0.314 versus 0.811 for mixed BCE. Its weak transfer
to the held-out proxy is retained as an unsuccessful configuration, with all
curves/checkpoints available. Mixing edits helps this proxy; real industrial
defects remain a separate distribution and are not used to select the generator.''')
    section(16,"BCE versus BCE plus Dice",
    '''The Dice ablation changes only Dice weight from zero to one. Positive BCE
weight, normal sampling, capacity, optimizer and resolution remain fixed. Early
stopping objectives are used within each candidate; cross-candidate selection
compares the same proxy recall and pixel AP instead of incomparable loss scales.''',
    '''display(development[development.variant.isin(["base","dice"])])
display(development_macro.loc[["base","dice"]])
comparison=development.pivot(index="category",columns="variant",values="proxy_pixel_ap")
display(comparison[["base","dice"]].assign(difference=lambda d:d.dice-d.base))''',
    '''Mixed BCE+Dice ranked first: macro proxy recall 0.849 and pixel AP 0.716.
Four categories are a development convenience, not a large independent sample;
this is not a significance claim or evidence that real-test localization improved.''')
    section(17,"Resolution experiment decision",
    '''Resolution changes were conditional on weak pilot learning plus available
VRAM. The pilot learned useful proxy structure, and the bounded baseline did not
need another search. We therefore explicitly did not run a resolution ablation.
Small-real-defect loss from 256×256 remains a limitation, not a tuned-away result.''',
    '''print("Resolution tested:",plan["resolution"])
print("Resolution experiment:",frozen["resolution_experiment"])
print("Pilot selected proxy pixel AP:",pilot["selected_development_metrics"]["proxy_pixel_ap"])
print("Pilot peak allocated GiB:",pilot["peak_allocated_bytes"]/2**30)
print("Native square widths:",sorted({v["width"] for v in eda["native_resolution_by_category"].values()}))''',
    '''Available VRAM alone is not a reason to expand the search. The baseline keeps
the preregistered 256×256 resolution and reports any small-defect/localization
failures honestly. No resolution was selected by looking at official test masks.''')
    section(18,"Selection, preserved decisions and frozen policy",
    '''All five configurations and all 20 category fits remain visible. Selection
maximizes macro proxy recall, then proxy pixel AP, then favors fewer parameters.
The selected variant's median best epoch is eight. Final runs use fresh models,
all fitting normals, three seeds and this fixed duration; no post-test changes.''',
    '''display(development_macro)
decisions=[]
for identifier in frozen["development_run_ids"]:
    event=json.loads((ROOT/f"experiments/{identifier}-decision.json").read_text())
    decisions.append({"experiment_id":identifier,"decision":event["decision"],"interpretation":event["interpretation"]})
display(pd.DataFrame(decisions))
display(frozen["configuration"])
print("Frozen policy SHA256:",sha256(ROOT/"configs/basic_cnn_frozen.json"))
print("Final categories:",frozen["categories"],"seeds:",frozen["seeds"])
print("Frozen timestamp:",frozen["frozen_at_utc"])''',
    '''Reject means that the configuration lost the bounded comparison, not that it
was erased or never attempted. All-category training was launched only after the
pilot gate and this selection. Each category/seed begins from independent random
weights; no category-specific learned checkpoint is reused for another category.
Initialization can be identical across categories within a seed by design; their
gradient images, updates and learned weights remain separate.

### Experiment-by-experiment interpretation

'''+"\n\n".join(trial_notes))
    section(19,"Image and pixel threshold calibration",
    '''Only reserved real normal calibration images set thresholds. Image alarms
use the higher q95 order statistic and strict score > threshold; q99 is a fixed
sensitivity report. Pixel decisions separately use conservative q99.5 pooled
normal pixels in 65,536 bins at model resolution. Threshold bootstrap intervals
describe finite calibration uncertainty, not guaranteed production FPR.''',
    source(evaluation.calibrate_model)+'''

calibrations=[]
for identifier in pd.read_csv(REPORT/"all_runs.csv").experiment_id:
    path=REPORT/f"evaluation/{identifier}-calibration.json"
    item=json.loads(path.read_text());item["seed"]=int(path.name.split("-s")[-1].split("-")[0])
    calibrations.append({k:item[k] for k in ("category","seed","images","image_threshold","secondary_image_threshold_q99","pixel_threshold","image_threshold_bootstrap_95ci")})
calibration_table=pd.DataFrame(calibrations)
display(calibration_table[calibration_table.seed.eq(2026)].sort_values("category"))
print("Smallest normal calibration sample:",int(calibration_table.images.min()))''',
    '''Toothbrush has only nine calibration normals; the q95 rule becomes the
sample maximum and can still miss population extremes. The pixel threshold is
not the image threshold. Native map interpolation can change pixel FPR relative
to calibration resolution. Neither threshold maximizes real-test F1 or uses masks.''')
    section(20,"Fresh all-category fits and frozen one-shot evaluation",
    '''Actual phase commands were `scripts/run_basic_cnn.py final-train` then
`scripts/run_basic_cnn.py final-evaluate`. The exact evaluator source below shows
freeze/hash guards, calibration before test access, native original masks,
one-pass exact ranking metrics and immutable evaluation reservation. This replay
loads recorded results; the optional reproduction loop gets new IDs and must be
reported as a repeat, not an untouched new confirmation.''',
    '''from anomaly_detection.cnn_evaluation import evaluate_frozen
display(Code(inspect.getsource(evaluate_frozen),language="python"))
reproduced_evaluations={}
if EXECUTE_NEW_CONFIRMATION:
    for seed in frozen["seeds"]:
        for category in CATEGORIES:
            config=dict(frozen["configuration"],seed=seed)
            trained=train_experiment(f"{RUN_PREFIX}-final-{category}-s{seed}",category,config,
                "Explicit registered reproduction of frozen final policy",phase="final")
            reproduced=evaluate_frozen(f"{RUN_PREFIX}-test-{category}-s{seed}",trained,sha256(ROOT/"configs/basic_cnn_frozen.json"))
            reproduced_evaluations[(category,seed)]=reproduced
all_runs=pd.read_csv(REPORT/"all_runs.csv")
assert len(all_runs)==45 and all_runs.groupby("category").size().eq(3).all()
evaluations={}
for identifier in all_runs.experiment_id:
    path=REPORT/f"evaluation/{identifier}-result.json"
    result=json.loads(path.read_text());evaluations[(result["category"],result["seed"])]=result
assert len(evaluations)==45
if reproduced_evaluations:
    reproduced_table=pd.DataFrame([{k:r[k] for k in ("experiment_id","category","seed","image_auroc","image_ap","pixel_auroc","pixel_ap","au_pro","recall","fpr")} for r in reproduced_evaluations.values()])
    display(reproduced_table)
    reproduced_table.to_csv(OUT/f"{RUN_PREFIX}-metrics.csv",index=False,lineterminator="\\n")
display(all_runs[["category","seed","image_auroc","pixel_ap","recall","fpr"]])''',
    '''Each category/seed was evaluated once under the frozen policy. Replaying
saved tables or rendering retained maps is not new inference or tuning. The
selected configuration is retained even when later real-test results are weak;
test-driven repairs would require a separate, explicitly exploratory protocol.''')
    section(21,"Image and exact native pixel metrics with uncertainty",
    '''Image AUROC/AP use the existing metric utilities. Native pixel AUROC/AP and
normalized AU-PRO use one sorted score sweep, equal 8-connected region weights
and background FPR ≤0.30. Chunk boundaries preserve tied scores exactly. Whole-image
bootstrap uses 2,000 replicates: image metrics exact, pixel confidence intervals a
disclosed 4,096-bin approximation. Point estimates do not downsample masks/pixels.''',
    source(pixel_metrics.exact_pixel_metrics)+'''

fixture=binary_metrics([0,0,1,1],[.1,.4,.35,.8],.5)
assert fixture["auroc"]==.75 and np.isclose(fixture["ap"],5/6)
metric_summary=json.loads((REPORT/"summary.json").read_text())
display(pd.DataFrame(metric_summary["macro"]).T)
print(metric_summary["bootstrap_scope"])
print("Confusion ordering: [[TN,FP],[FN,TP]]; AP is stepwise average precision.")
display(metric_summary["metric_edge_cases"])''',
    '''Pixel AP is sensitive to tiny defect prevalence; a high pixel AUROC alone
can hide poor localization. Undefined metrics would be null with reasons; all
categories here contain both classes. Seed standard deviation and conditional
image-bootstrap intervals quantify different uncertainties. Three seeds and one
benchmark still do not establish production generalization. Pixel intervals target
the binned approximation: quantization can materially bias pixel AP even when
AUROC changes little. Exact point estimates remain primary; the summary reports
the measured binned-minus-exact bias rather than presenting these as exact-pixel
confidence intervals. The foundation utility's zero-division convention is kept
in immutable raw records, with no-alarm precision explicitly flagged and defined-
run coverage reported. Public operating tables mark undefined precision N/A;
zero F1 remains valid because real defects exist. No undefined ranking metric is
silently replaced by zero.''')
    section(22,"Confusion matrices and operating points",
    '''Show all 15 seed-2026 image confusion matrices, with no category omitted.
Thresholds were fixed from calibration normals. Compare the predetermined q99
policy to primary q95 without selecting a winner after observing test labels.''',
    '''fig,axes=plt.subplots(3,5,figsize=(13,8))
for ax,category in zip(axes.flat,CATEGORIES):
    matrix=np.asarray(evaluations[(category,2026)]["confusion_matrix"])
    ax.imshow(matrix,cmap="Blues",vmin=0,vmax=max(1,matrix.max()))
    for (r,c),value in np.ndenumerate(matrix):ax.text(c,r,str(value),ha="center",va="center")
    ax.set_title(category);ax.set_xticks([0,1],["normal","defect"],fontsize=7)
    ax.set_yticks([0,1],["normal","defect"],fontsize=7)
fig.supxlabel("Predicted class");fig.supylabel("True class")
plt.tight_layout();plt.savefig(OUT/"all_category_confusions.png",dpi=100);plt.show()
operating=all_runs.copy()
operating.loc[~operating.precision_defined,"precision"]=np.nan
operating.loc[~operating.pixel_precision_defined,"pixel_precision"]=np.nan
display(operating.groupby("category")[["recall","fpr","precision","f1","q99_recall","q99_fpr","pixel_recall","pixel_fpr","pixel_precision","pixel_f1"]].mean())
display(operating.groupby("category")[["precision_defined","pixel_precision_defined"]].sum().rename(columns={"precision_defined":"image_precision_defined_seeds","pixel_precision_defined":"pixel_precision_defined_seeds"}))''',
    '''Threshold-free ranking metrics and calibrated operating points answer
different questions. The q95 policy's observed test FPR need not equal 5%; small
calibration samples and distribution differences can produce more false alarms.
Pixel operating points remain separate and may be poor despite good ranking.''')
    section(23,"Per-category predictions, masks, maps and transparent overlays",
    '''For every category, choose the first path of each available TN/TP/FP/FN
outcome, not the most attractive heatmap. Missing outcome classes are explicitly
listed; examples are never fabricated. Reuse retained seed-2026 predictions with
no additional test inference. Probability colors use fixed [0,1] scale.''',
    '''DATA=autodiscover_data_root(ROOT)
panel_index=[]
for category in CATEGORIES:
    result=evaluations[(category,2026)]
    maps_path=ROOT/result["local_maps"]
    if not maps_path.exists() and (category,2026) in reproduced_evaluations:
        result=reproduced_evaluations[(category,2026)]
        maps_path=ROOT/result["local_maps"]
    if not maps_path.exists():
        print("EXPLICITLY SKIPPED local licensed panel:",category,"prediction artifact is not distributed in Git. Use separately registered reproduction.")
        panel_index.append({"category":category,"figure":None,"absent_outcomes":[],"missing_local_artifact":True})
        continue
    table=pd.read_csv(REPORT/f"evaluation/{result['experiment_id']}-images.csv")
    assert sha256(maps_path)==result["maps_sha256"]
    maps=np.load(maps_path,mmap_mode="r")
    selected=[];absent=[]
    for outcome in ("TN","TP","FP","FN"):
        subset=table[table.outcome.eq(outcome)]
        if subset.empty:absent.append(outcome)
        else:selected.append((outcome,subset.index[0]))
    fig,axes=plt.subplots(len(selected),4,figsize=(10,2.6*len(selected)),squeeze=False)
    for r,(outcome,index) in enumerate(selected):
        row=table.loc[index]
        with Image.open(DATA/row.path) as image:original=np.asarray(image.convert("RGB"))
        with torch.inference_mode():
            native=F.interpolate(torch.from_numpy(np.asarray(maps[index]).copy())[None,None],size=original.shape[:2],mode="bilinear",align_corners=False).numpy()[0,0]
        if isinstance(row["mask"],str):
            with Image.open(DATA/row["mask"]) as mask_image:truth=np.asarray(mask_image)>0
        else:truth=np.zeros(original.shape[:2],dtype=bool)
        axes[r,0].imshow(original);axes[r,0].set_title(f"{outcome}: score {row.score:.3f}")
        axes[r,1].imshow(truth,cmap="gray",vmin=0,vmax=1);axes[r,1].set_title("official mask / normal zeros")
        axes[r,2].imshow(native,cmap="magma",vmin=0,vmax=1);axes[r,2].set_title("probability [0,1]")
        axes[r,3].imshow(original)
        axes[r,3].imshow(np.ma.masked_where(native<=result["calibration"]["pixel_threshold"],native),cmap="Reds",alpha=.55,vmin=0,vmax=1)
        axes[r,3].set_title("pixel-calibrated overlay")
        for ax in axes[r]:ax.axis("off")
    fig.suptitle(category+" / "+result["experiment_id"]+"; absent outcomes: "+(", ".join(absent) if absent else "none"),fontsize=9)
    plt.tight_layout();filename=f"{category}-outcomes.png";plt.savefig(OUT/filename,dpi=90);plt.show()
    panel_index.append({"category":category,"figure":filename,"absent_outcomes":absent,"missing_local_artifact":False,
                        "paths":table.loc[[i for _,i in selected],"path"].tolist()})
(OUT/"panel_index.json").write_text(json.dumps(panel_index,indent=2)+"\\n")
display(pd.DataFrame(panel_index)[["category","figure","absent_outcomes"]])''',
    '''The complete local executed notebook and per-category figures retain every
available error class with raw image, ground truth, heatmap and transparent
thresholded overlay. The public notebook omits licensed dataset-pixel panels;
per-image scores/outcomes, aggregate plots and all analysis remain published.
Zero normal masks express semantic normal ground truth, not synthetic defects.''',private=True)
    section(24,"False-positive, false-negative and defect-type analysis",
    '''Use all categories and defect types, including difficult cases. These are
post-confirmation explanations, not opportunities to change the frozen model.
Geometry loss, local synthetic shortcuts and synthetic-to-real domain gap are
possible explanations, not causal conclusions from visual inspection alone.''',
    '''failures=all_runs[all_runs.seed.eq(2026)].set_index("category")
display(failures[["tn","fp","fn","tp","image_auroc","pixel_ap","recall","fpr","unique_image_scores","fraction_image_scores_equal_one","image_threshold"]].reindex(CATEGORIES))
per_defect=pd.read_csv(REPORT/"per_defect_results.csv")
defect_summary=per_defect.groupby(["category","defect"]).agg(images=("defective_images","first"),recall=("recall","mean"),image_auroc=("auroc","mean"),image_ap=("ap","mean"))
display(defect_summary.sort_values("recall").head(20))
display(pd.read_csv(REPORT/"texture_object.csv"))
print("All defect-type results:",REPORT.relative_to(ROOT)/"per_defect_results.csv")''',
    '''A missing false-positive/negative example means no such event occurred in
that category/seed, not that it was hidden. Real defects can be less conspicuous
than the synthetic proxy; high normal scores can suppress recall through normal
calibration. Eight epochs and scratch features may underfit structure or overfit
edit cues. These hypotheses motivate later models, without retroactive tuning.
The score-tie table also exposes float32 sigmoid saturation and any threshold at
one: strict score > threshold can then produce no alarms. Sigmoid scores are not
calibrated real-defect probabilities. A future raw-logit aggregation study needs
its own preregistration; it is not retroactively substituted here.''')
    section(25,"Runtime, memory and model size",
    '''Measurements use batch 1, float32, 50 warmups and 200 synchronized timed
iterations on a calibration-normal input. Preprocess, transfer, forward and
postprocess are timed separately. End-to-end starts with memory-resident RGB
and ends with a native probability map on CPU; disk read/decode is excluded.
Power scheme/GPU context and raw timing samples are retained per experiment.
Timing uses GPU sigmoid/interpolation; exact metric restoration uses CPU float32
interpolation. Inference allocated-memory fields include the resident evaluation
input cache, so they are process footprints, not isolated model-only allocations.''',
    '''efficiency=all_runs.groupby("category")[["training_seconds","model_p50_ms","model_p95_ms","end_to_end_p50_ms","end_to_end_p95_ms","train_peak_vram_gib","checkpoint_bytes","parameters"]].mean()
display(efficiency.reindex(CATEGORIES))
print("Maximum training VRAM GiB:",float(all_runs.train_peak_vram_gib.max()))
print("Parameter count:",int(all_runs.parameters.iloc[0]))
print("Latency boundary:",evaluations[("bottle",2026)]["latency"]["scope"])
display(evaluations[("bottle",2026)]["latency"]["power_context_at_end_of_timing"])''',
    '''Means of per-run p50/p95 values are summaries of measured run percentiles,
not percentiles of a pooled production workload. Small checkpoints and low VRAM
do not compensate for missed defects. Laptop clock/power variability and excluded
disk decode limit portability of latency numbers. No architecture/config changed
after these measurements or real-test results became visible.''')
    section(26,"All 15 categories, three seeds and macro reporting",
    '''Report every category with seed variation and unweighted macro means.
Object/texture summaries are supplementary. AP is compared with test prevalence;
pixel AP with pixel foreground prevalence. Confidence intervals resample whole
images and preserve spatial dependence instead of pretending pixels independent.''',
    '''category_means=pd.read_csv(REPORT/"all_category_means.csv").set_index("category")
display(category_means.reindex(CATEGORIES))
display(pd.read_csv(REPORT/"macro_by_seed.csv"))
category_means[["image_auroc_mean","image_ap_mean","pixel_auroc_mean","pixel_ap_mean","au_pro_mean"]].plot.bar(figsize=(13,5),ylim=(0,1))
plt.ylabel("Metric");plt.title("Frozen Basic CNN: every category, mean over three seeds")
plt.legend(fontsize=8,ncol=3);plt.tight_layout();plt.savefig(OUT/"all_category_metrics.png",dpi=100);plt.show()
prevalence=[]
for category in CATEGORIES:
    r=evaluations[(category,2026)]
    prevalence.append({"category":category,"image_defect_prevalence":counts.loc[category,"test_defective"]/(counts.loc[category,"test_normal"]+counts.loc[category,"test_defective"]),
                       "pixel_foreground_prevalence":r["positive_pixels"]/(r["positive_pixels"]+r["negative_pixels"]),
                       "image_AP_mean":category_means.loc[category,"image_ap_mean"],"pixel_AP_mean":category_means.loc[category,"pixel_ap_mean"]})
display(pd.DataFrame(prevalence))''',
    '''No success-only average, excluded category or favorable seed is substituted
for the full comparison. Three-seed spread describes model variation; the 2,000
image-bootstrap replicates describe sampling uncertainty conditional on these
models. Neither creates independent new inspection data. All raw operating-point
counts and per-defect outcomes remain available for independent review.''')
    main_metrics=summary["macro"]
    conclusion=f'''**Observed frozen macro means over 15 categories and three seeds:** image
AUROC {main_metrics['image_auroc']['mean']:.4f}, image AP {main_metrics['image_ap']['mean']:.4f},
pixel AUROC {main_metrics['pixel_auroc']['mean']:.4f}, pixel AP {main_metrics['pixel_ap']['mean']:.4f},
AU-PRO {main_metrics['au_pro']['mean']:.4f}. Calibrated recall is
{main_metrics['recall']['mean']:.4f} and FPR {main_metrics['fpr']['mean']:.4f}.
The lowest mean image-AUROC categories are {', '.join(f'{c} ({v:.3f})' for c,v in weakest.items())}.
These outcomes are retained without changing the selected configuration.

The pilot establishes synthetic learning, not reliable real-defect detection.
The gap between proxy success and benchmark outcomes must be interpreted together
with category prevalence, false alarms, misses, weak small-defect localization and
the limited scratch representation. These are observational failure hypotheses,
not proven causal mechanisms. No pretrained backbone was used.

**Next recommendation:** a separately requested convolutional autoencoder study
can test a different normal-only learning objective; later pretrained-feature
methods test representation transfer. Neither is implemented here. Do not start
Notebook 02 automatically. Preserve the rejected CNN configurations and all final
results as the reference for later comparisons.'''
    section(27,"Conclusions, limits and next-model recommendations",
    '''Assess hypotheses against actual results rather than a favorable headline.
The complete machine-readable record covers configuration search, calibration,
frozen metrics, uncertainty, efficiency and every category/seed. This notebook
does not advertise a deployment-ready detector or silently begin another model.''',
    '''display(metric_summary)
print("Completed development fits:",20,"fresh final models:",len(all_runs))
print("Configuration changed after test:",metric_summary["configuration_changed_after_test"])
print("Artifact manifest SHA256:",metric_summary["artifact_manifest_sha256"])
print("Local full notebook: outputs/basic_cnn/01_basic_cnn.executed.ipynb")
print("Notebook 02 and other architectures: not started.")''',conclusion)
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"MVTec research (CUDA, Python 3.12)","language":"python","name":"mvtec-research"},
        "language_info":{"name":"python","version":"3.12.10"},
        "research":{"mode":"replay of immutable measured experiments; reproduction flags default false",
                    "source_injection":"exact module source, no handwritten duplicate implementation"}})
    nbf.validate(notebook);nbf.write(notebook,target)
    print("Created",target.name,len(cells),"cells")


if __name__=="__main__":main()
