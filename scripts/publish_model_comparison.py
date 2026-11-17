"""Generate figures, written analysis and Notebook 07 from canonical saved data."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json
import os
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib-comparison'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf
from anomaly_detection.model_comparison import MODELS,NAMES,METRICS,category_macro

OUT=ROOT/'reports/final_comparison'


def markdown_table(frame, digits=4):
    """Dependency-free, compact Markdown export."""
    def fmt(v):
        if isinstance(v,(float,np.floating)):
            return 'not recorded' if pd.isna(v) else f'{v:.{digits}f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def publish():
    f=pd.read_csv(OUT/'all_runs.csv',float_precision='round_trip')
    macro=category_macro(f)
    cat=pd.read_csv(OUT/'category_summary.csv')
    stats=pd.read_csv(OUT/'paired_comparisons.csv')
    cal=pd.read_csv(OUT/'calibration_summary.csv').set_index('model').reindex(MODELS)
    cost=pd.read_csv(OUT/'compute_summary.csv').set_index('model').reindex(MODELS)
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    labels=['Basic CNN','Conv. AE','PatchCore*','SSN Lite','PaDiM','EfficientAD*']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,
                         'savefig.dpi':160,'svg.hashsalt':'milestone9'})
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),bbox_inches='tight')
        plt.close(fig)
    for metric in ('image_auroc','pixel_ap','au_pro','recall','fpr'):
        matrix=cat.pivot(index='category',columns='model',values=metric).reindex(columns=MODELS)
        fig,ax=plt.subplots(figsize=(9,7))
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='magma_r' if metric=='fpr' else 'viridis')
        ax.set_xticks(range(6),labels,rotation=20,ha='right');ax.set_yticks(range(15),matrix.index)
        for i in range(15):
            for j in range(6):
                val=matrix.iloc[i,j]
                ax.text(j,i,f'{val:.2f}',ha='center',va='center',color='white' if (val<.6 if metric!='fpr' else val>.35) else 'black',fontsize=8)
        ax.set_title(metric.replace('_',' ').title()+(' (lower is better)' if metric=='fpr' else ' (higher is better)')+'; seed means')
        fig.colorbar(im,ax=ax,shrink=.7);save(fig,'heatmap_'+metric)
    rng=np.random.default_rng(9009);draws=rng.integers(0,15,(10000,15))
    for group,metrics in [('ranking',METRICS[:5]),('operating',METRICS[5:])]:
        fig,axes=plt.subplots(1,len(metrics),figsize=(3*len(metrics),4),sharey=True)
        for ax,metric in zip(axes,metrics):
            means=[];los=[];his=[]
            for model in MODELS:
                values=cat.query('model == @model')[metric].to_numpy()
                lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975]);point=values.mean()
                means.append(point);los.append(point-lo);his.append(hi-point)
            ax.errorbar(means,range(6),xerr=[los,his],fmt='o',capsize=3)
            ax.set_title(metric.replace('_',' '));ax.set_xlim(0,1.02);ax.grid(axis='x',alpha=.2)
            ax.set_yticks(range(6),labels);ax.invert_yaxis()
        fig.suptitle('Category-macro means; 95% category-cluster bootstrap intervals',y=1.02)
        fig.tight_layout();save(fig,group+'_metrics')
    fig,ax=plt.subplots(figsize=(9,5))
    for i,m in enumerate(MODELS):
        ax.scatter(cost.loc[m,'end_to_end_p50_ms'],macro.loc[m,'image_auroc'],s=70)
        ax.annotate(labels[i],(cost.loc[m,'end_to_end_p50_ms'],macro.loc[m,'image_auroc']),xytext=(5,6 if i%2 else -15),textcoords='offset points')
    ax.set(xlabel='Median of per-run decoded-RGB end-to-end p50 (ms)',ylabel='Category-macro image AUROC',title='Quality and latency: batch one FP32; file decode excluded')
    ax.grid(alpha=.2);save(fig,'quality_latency')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(6)
    axes[0].bar(x-.18,cal.calibration_fpr,.36,label='Calibration normals')
    axes[0].bar(x+.18,cal.fpr,.36,label='Test normals')
    axes[0].axhline(.05,color='black',linestyle='--',label='Nominal target .05')
    axes[0].set_xticks(x,labels,rotation=25,ha='right');axes[0].legend(fontsize=8);axes[0].set_ylabel('Category-seed macro FPR')
    for i,m in enumerate(MODELS):
        v=f.query('model == @m');axes[1].scatter(v.calibration_n,v.fpr,alpha=.45,s=12,label=labels[i])
    axes[1].set(xlabel='Reserved calibration normal count',ylabel='Frozen test FPR',title='Category variation; three seeds share test images')
    axes[1].legend(fontsize=7,ncol=2);fig.tight_layout();save(fig,'calibration_transfer')
    fig,ax=plt.subplots(figsize=(9,4))
    for i,m in enumerate(MODELS):
        values=f.query('model == @m').groupby('category').image_auroc.std().sort_index()
        ax.scatter(np.arange(15)+(i-2.5)*.10,values,s=15,label=labels[i])
    ax.set_xticks(range(15),sorted(f.category.unique()),rotation=45,ha='right');ax.set_ylabel('Within-category image AUROC SD (three seeds)');ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();save(fig,'seed_stability')
    macro_table=macro.rename(index=NAMES).reset_index().rename(columns={'model':'Method'})
    macro_md=markdown_table(macro_table)
    (OUT/'nine_metric_table.md').write_text(macro_md+'\n',encoding='utf-8')
    # Exact source tables remain authoritative; report values are rounded for reading.
    top=stats.query("model_a == 'patchcore' and model_b == 'supersimplenet'")
    top_md=markdown_table(top[['metric','difference','ci_low','ci_high','p_holm_135']])
    cost_md=markdown_table(cost[['training_hours','evaluation_hours','model_p50_ms','model_p95_ms',
        'end_to_end_p50_ms','end_to_end_p95_ms','deployment_mib_one_category','train_peak_vram_gib',
        'inference_peak_vram_gib','cpu_peak_rss_gib']].rename(index=NAMES).reset_index())
    cal_md=markdown_table(cal.rename(index=NAMES).reset_index())
    category_rows=[]
    for c in sorted(f.category.unique()):
        q=cat.query('category == @c').set_index('model')
        category_rows.append(dict(category=c,image_leader=NAMES[q.image_auroc.idxmax()],
            image_auroc=q.image_auroc.max(),pixel_ap_leader=NAMES[q.pixel_ap.idxmax()],
            pixel_ap=q.pixel_ap.max(),image_auroc_range=q.image_auroc.max()-q.image_auroc.min(),
            efficientad_fpr=q.loc['efficientad','fpr']))
    category_table=pd.DataFrame(category_rows)
    category_table.to_csv(OUT/'category_findings.csv',index=False,lineterminator='\n',float_format='%.17g')
    texture=f.assign(kind=np.where(f.category.isin(['carpet','grid','leather','tile','wood']),'texture','object')).groupby(['model','kind'])[list(METRICS)].mean().reset_index()
    texture.to_csv(OUT/'texture_object_summary.csv',index=False,lineterminator='\n',float_format='%.17g')
    instability=f.groupby(['model','category']).image_auroc.std().reset_index(name='image_auroc_seed_sd')
    instability=instability.loc[instability.groupby('model').image_auroc_seed_sd.idxmax()]
    instability.to_csv(OUT/'largest_seed_variability.csv',index=False,lineterminator='\n',float_format='%.17g')
    sig=stats.query('p_holm_135 < .05')
    report=f'''# Final six-model research comparison

## Executive summary

This saved-artifact analysis audits 270 final evaluations: six methods, all 15
original MVTec AD categories and seeds 2026/2027/2028. No training, inference or
threshold refitting occurs in Milestone 9. Simplified PatchCore is the primary
engineering **research prototype candidate**: its image AUROC is 0.9676 and its
three localization point estimates exceed those of the close image-ranking
alternative, SuperSimpleNet Lite. That is an explicit tradeoff, not proof of
universal superiority or production readiness. The latter is faster and has
slightly higher calibrated F1. The reduced EfficientAD variant met its compute
budget but transferred thresholds poorly, especially on screw/leather/carpet.

## Research question and methodology

Which of six recorded CNN-based approaches offers the strongest evidence for a
subsequent engineering prototype under quality, false-alarm, cost and provenance
constraints? Category means first average three seeds; macro means weight the
15 categories equally. This is a comparison on one public benchmark, not an
external prospective validation of the selected model. Original test-set EDA
was part of the foundation; no claim that the dataset had never been inspected
is made. Frozen policies were selected without official test-label tuning.

All methods share the original seed-2026 train-normal split: 2,551 fitting,
539 development and 539 calibration images; test has 1,725 images, including
467 normals and 1,258 defects. Per-category calibration counts range 9â€“58.
Path identities, hashes, normal-only membership, disjointness and test labels
are checked against original manifests. All 270 result files agree with their
original all-run summaries, frozen configurations and saved image predictions.
`source_provenance.json` binds original sources and local artifact hashes;
`prediction_reconciliation.csv` documents float32 restoration when historical
CSV decimal precision otherwise changes ties. Original metrics are preserved.

## Six methods and protocol differences

* Basic CNN: scratch width-16 synthetic-supervised segmentation, eight epochs,
  BCE/Dice, maximum sigmoid image score; 83,857 parameters. Score saturation
  and ties affect its operating point.
* Convolutional AE: scratch depth-three, width-16, latent-eight reconstruction,
  20 MSE epochs; squared RGB residual map and top-1% mean image score.
* Simplified PatchCore: frozen ImageNet ResNet18 layer2/3 features, 384 dimensions,
  20,000 candidate cap, 32-dimensional projection, 2,048-vector greedy coreset;
  top-1% distance image score. This omits the paper's neighborhood reweighting,
  uses a smaller backbone and does not reproduce the complete paper pipeline.
* SuperSimpleNet Lite: frozen ResNet18 plus adapter/segmentation/detection heads,
  40 feature-noise training epochs, margin/focal objective; primary score is
  top-1% smoothed dense logits. Its detection head is auxiliary. It is a variant.
* PaDiM: ResNet18 aligned layer1â€“3 features, 100 of 448 channels selected per seed,
  per-position full Gaussian covariance, epsilon .01, sigma-four smoothing,
  maximum-map image score. Fitting is covariance estimation, not gradient training.
* EfficientAD-S reduced-budget: pinned frozen PDN-S teacher, dual-head student,
  auxiliary AE, nine full epochs, batch one; hard .999 feature quantile.
  Development-normal map normalization and **local-only maximum** scoring are
  frozen. Both heads/AE are trained, but the global map is not the primary score.
  External-image penalty is absent; this is not paper-equivalent EfficientAD.

Whole RGB images are resized to 256 square. Scratch methods use [0,1]; pretrained
methods use their recorded normalization. All image scores increase with anomaly.
Strict score > cutoff applies; image cutoff is the higher .95 normal quantile.
Pixel thresholds target .995: early methods use histogram upper edges, later
methods use their recorded pooled conventions. Thresholds are not comparable in
absolute units. Primary pixel AUROC/AP use exact native-resolution rankings;
AU-PRO weights 8-connected regions equally through FPR .30, normalized by .30.
Earlier conditional pixel-bootstrap intervals use binned approximations, not
the primary exact pixel values. This analysis never treats individual pixels as
independent experimental replications. Seed 2026 represents optimization,
coreset selection or channel selection depending on model, not identical noise.

## Overall nine-metric benchmark

{macro_md}

Ranking metrics and frozen operating-point metrics answer different questions.
PatchCore leads image ranking and all three localization macro point estimates.
SSN Lite has slightly higher recall/F1 and slightly lower FPR. PaDiM localizes
strongly despite weaker image ranking: a maximum score is vulnerable to isolated
background responses. EfficientAD pixel AP exceeds SSN/PaDiM despite much weaker
image AUROC. Basic CNN/AE have lower FPR partly because they miss many defects;
low FPR alone is not evidence of a useful detector. AP also reflects the high
defect prevalence of the official test split and should not be transferred to
a low-prevalence factory stream without further validation.

## Every-category findings

{markdown_table(category_table)}

All five requested category heatmaps are in `figures/heatmap_*.png`.
`texture_object_summary.csv` separates five textures from ten objects; neither
subset should be replaced by an image-count-weighted aggregate. Winners above
are descriptive seed means, not 15 separate claims of statistical superiority.
The heatmaps expose rank changes and instances where strong image ranking
coexists with weaker pixel AP. `within_category_seed_sd.csv` and
`seed_stability.png` retain instability rather than hiding it inside one mean.
In particular, EfficientAD screw image ranking collapses, while leather's high
image ranking coexists with severe false alarms. Carpet is another threshold
transfer failure. Native masks are not used for any new model-selection tuning.

This is not solely an EfficientAD problem: PatchCore's leather/carpet test FPR
is 0.6979/0.5595 and SSN Lite's is 0.5625/0.4881 despite excellent image ranking.
Their frozen cutoffs do not provide an acceptable universal factory alarm policy.
PaDiM has lower leather/carpet FPR (0.0729/0.0833), but screw recall is only
0.1597. AE screw AUROC is 0.9573 while recall is 0.0644 and pixel AP 0.0263:
strong image ranking alone does not establish threshold transfer or localization.
See the full category tables rather than assuming one architecture wins everywhere.

Texture/object category-macro results (three seeds averaged; five/ten categories):

{markdown_table(texture[['model','kind','image_auroc','pixel_ap','recall','fpr']])}

Largest within-category image-AUROC seed SD for each architecture:

{markdown_table(instability)}

## Paired statistical uncertainty

The predeclared unit is the category. Each paired difference averages its three
seed evaluations before a 10,000-draw category-cluster bootstrap (seed 9009).
This preserves the dependence of repeated evaluations on the same test images.
Intervals estimate sensitivity to category composition among this benchmark's
categories; they do not capture all within-image, training or acquisition-domain
uncertainty. Exact paired sign randomization uses all 32,768 category sign
patterns and assumes exchangeable differences under its null. Holm correction
is across all 135 pairs-by-metrics tests. Effect sizes include mean/median
differences, paired dz and category wins/ties/losses. FPR preference is reversed
for wins only; signed differences retain A minus B.

PatchCore minus SuperSimpleNet Lite:

{top_md}

There are {len(sig)} Holm-adjusted p<.05 comparisons in the full 135-test table.
Inspect `paired_comparisons.csv` for every pair, including effects and intervals.
A positive point estimate with an interval crossing zero is uncertain under this
category resampling scheme. An interval excluding zero and an adjusted test
not rejecting are different evidence statements, not a contradiction. No
equivalence or universal superiority claim follows from close macro means.

## Calibration and threshold transfer

{cal_md}

These are descriptive averages of per-category/seed rates. A nominal 5% cutoff
does not promise 5% official-test FPR. With nine calibration normals, a higher
.95 quantile is the maximum, making empirical strict-exceedance FPR zero; that
does not establish zero future risk. Order statistics are discrete and ties
reduce empirical exceedances further. Observed calibration rates are optimistic
for future observations because the same normals determine the cutoff.
`calibration_runs.csv` includes sample counts and exact binomial test-FPR
intervals conditional on each already fitted model. These intervals are not
pooled across seeds. `threshold_seed_variability.csv` reports within-model,
within-category cutoff ranges, not cross-model scale comparisons.

Normal score distribution summaries are retained in the canonical rows. Large
calibration-to-test gaps demonstrate failed threshold transfer, not a proven
cause. Limited samples, normal-distribution shift, image aggregation, ties,
texture/border responses are plausible mechanisms with different supporting
evidence. EfficientAD's bottle cutoff ties and actual normal-leather hot spots
are directly recorded. No oracle cutoff, test-fitted calibration or post-test
policy replacement appears in primary results or recommendations.

## Compute, memory and storage

{cost_md}

Training/evaluation times are sums over 45 runs; latency entries are medians of
the 45 **per-run percentiles**, not percentiles of pooled timings. All series use
50 warmups and 200 timed batch-one FP32 measurements, explicit synchronization
in the original evaluators, on the same RTX 4060 Laptop GPU. Model-only timing
includes different scoring work: extraction+nearest neighbors for PatchCore,
extraction+Mahalanobis for PaDiM; smoothing is postprocessing for PaDiM. The most
comparable scatter uses decoded RGB to native CPU map, excluding file decode.
Only EfficientAD retains a separate decode-inclusive series; it is not mixed
with these points. Windows scheduling, power and temperature varied across
historical sessions; these are measured costs, not controlled causal speedups.

PatchCore/PaDiM costs exclude original backbone pretraining; bank/Gaussian fitting
can reuse features. PatchCore's `original_extraction_seconds` is retained per run
but must not be summed as if shared cached work were repeated 45 times. SSN
uses cached features during fitting but includes its backbone in inference.
Category checkpoint size plus the shared 10.66 MiB truncated ResNet trunk is
shown for pretrained ResNet methods; raw feature caches are not serving assets.
EfficientAD checkpoints include its teacher. CPU RSS for CNN/AE was not recorded
and remains missing. Later CPU peaks include native-metric sorting and are not
minimum serving RAM; allocated VRAM differs from reserved memory/driver totals.
No claims about energy or fair total pretraining cost are possible.

## Genuine failure evidence

`failure_source_inventory.csv` links existing selected-case tables. Existing
small-defect and per-defect tables retain their original seed/sample scope; they
are not silently treated as all-45-run independent observations. Original
Notebook 01 supplies CNN failure panels; no new images or maps are synthesized.
Notebook 07 replays existing EfficientAD panels locally across all 15 categories,
then clears licensed image outputs in its public copy.

The original AE failure table measures reconstruction detail ratios and residual
overlap: blurred reconstructions and background residuals are observed. PatchCore
retains nearest-reference cases and lower recall on tiny defects; finite bank
coverage is a plausible mechanism, not an isolated causal intervention. SSN's
saved overlap and defect-size records expose real-defect misses despite synthetic
feature training; synthetic-to-real mismatch remains an interpretation. PaDiM's
saved maps show broad responses; per-position Gaussian and alignment limitations
are architectural assumptions rather than experimentally isolated causes.

EfficientAD's existing visual review records neck-scratch responses in missed
screw images, a normal-leather hot spot with diffuse alarms, broader grid-thread
alarm bands than the mask, bottle corner/rim responses and cutoff ties, and
missing-cable/flip responses below their image cutoffs. Local pixel response is
not equivalent to successful image detection or precise segmentation. Fine
scratches/tiny anomalies and halos persist across methods. Structural examples
in original MVTec AD do not establish comprehensive logical-anomaly coverage.
See `../../docs/efficientad_visual_review.md` and original notebooks for observed
panels. No claim that changing a branch/threshold would fix them is tested here.

Milestone 9 also inspected existing local panels directly. In AE's carpet panel,
the dark anomalous patch is reproduced in the blurred reconstruction and has
little residual response; the normal texture has widespread residual speckle.
In PatchCore's screw panel, a tip defect has a local alarm but its image is FN,
and a normal example has a bright background corner. SSN Lite's screw panel
shows a normal thread-region FP and a manipulated-front FN with no pixel alarm.
PaDiM's existing panel shows a normal bottle center hot spot, a narrow bottle
contamination mask with a much wider alarm, and a screw neck-scratch FN with
off-target responses. These are selected cases, not frequency estimates.
Their paths and SHA256 values are in `visual_evidence_provenance.json`; images
remain local. Architectural explanations remain hypotheses.

The saved seed-2026 size summaries contain only 17 defects at most 0.1% of the
image: PatchCore recalls 0.6471, SSN Lite 0.7647 and AE 0.2353 in that stratum.
These are small, selected-seed descriptive counts, not a new statistical test
or the all-seed macro benchmark. PaDiM's size table explicitly pools repeated
seed evaluations; its 51 rows in that stratum are 17 unique images × three seeds.

## Transparent engineering recommendation

Shortlist the two strongest image-AUROC methods, PatchCore and SSN Lite. Among
them PatchCore has higher pixel AUROC/AP/AU-PRO point estimates and almost equal
image ranking; this supports selecting it for a **quality-first research
prototype** without inventing a weighted composite. Its slower nearest-neighbor
inference, memory-bank storage and nontrivial FPR remain costs. Pairwise category
intervals and Holm results qualify every superiority claim. The selected model
is chosen after comparison on this test benchmark: deployment requires a new,
independent validation set and a defined false-alarm cost.

Alternatives: SSN Lite for a faster pretrained quality/latency balance (smaller
learned heads, slightly stronger frozen F1); PaDiM for minimal measured Gaussian
fitting cost and strong localization if its large category covariance files and
weaker image detection are acceptable. Scratch CNN/AE offer small artifacts
and faster inference but insufficient quality here. Reduced EfficientAD is not
the low-latency default on these measurements and is unsuitable as a universal
thresholded detector in this recorded configuration.

Provenance is a selection constraint: ResNet18 official torchvision ImageNet V1
weights are pinned to SHA256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Code licensing does not automatically resolve weight/data commercial rights.
EfficientAD teacher comes from third-party `nelson1425/EfficientAD`, revision
`fcab5146f84ae17597044ad5ddf1656ccf805401`, checkpoint SHA256
`b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337`.
Repository Apache-2.0 does not establish a separate weight license or complete
teacher-distillation history. These uncertainties prevent production clearance.

## Conclusions and limitations

Pretrained-feature methods outperform the scratch baselines descriptively here.
Transfer representations are a plausible reason; architecture, training budget,
loss, aggregation and smoothing also differ, so this is not a causal pretraining
ablation. AE reconstruction did not reliably separate defects from normal fine
detail. PatchCore is strong without category gradient fitting; SSN Lite is
competitive with learned feature adaptation. PaDiM preserves strong pixel
ranking while image maxima and calibration are less favorable. Reduced
EfficientAD's short training, missing external penalty, selected local score and
teacher provenance constrain interpretation; no claim about the full paper's
potential follows from this reduced result.

Limitations include one dataset, small normal-calibration sets, shared test
images, only three seeds, variant architectures, historical timing/power
differences, missing early RSS, unequal pretraining resources, simplified
category-level uncertainty and multiple-comparison assumptions. No result
automatically generalizes beyond original MVTec AD. No engineering application
has been started.

## Reproducibility and references

Use the existing project interpreter. Canonical analysis needs compact published
source records; the deep identity audit additionally requires retained local
checkpoints/maps. No downloads are performed. Commands from repository root:

```powershell
.\\.venv\\Scripts\\python.exe scripts/build_model_comparison.py
.\\.venv\\Scripts\\python.exe scripts/publish_model_comparison.py
.\\.venv\\Scripts\\python.exe scripts/execute_model_comparison.py
.\\.venv\\Scripts\\python.exe scripts/validate_model_comparison.py
.\\.venv\\Scripts\\python.exe -m pytest -q --basetemp=.pytest-tmp
```

See original Notebooks 00â€“06, frozen JSONs, `analysis_plan.md`, the provenance
manifest, protocol/reproduction documents under `docs/`, and these primary
references: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad),
[PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html),
[PaDiM](https://arxiv.org/abs/2011.08785),
[SuperSimpleNet](https://arxiv.org/abs/2408.03143),
[EfficientAD](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html).
Paper links identify methods; reported experimental claims derive from this
repository's saved artifacts, not imported paper benchmark numbers.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    make_notebook(report)


def make_notebook(report):
    nb=nbf.v4.new_notebook()
    nb.metadata.kernelspec=dict(display_name='Python (mvtec-research)',language='python',name='mvtec-research')
    nb.metadata.language_info=dict(name='python',version='3.12')
    cells=[nbf.v4.new_markdown_cell('# Final six-model research comparison\n\nMilestone 9 performs live analysis of saved results only. No training, inference or calibration is rerun. Licensed panel outputs remain local. The accompanying report supplies full interpretation and provenance.')]
    setup="""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd() if (Path.cwd()/'reports').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.model_comparison import validate_coverage, category_macro, paired_statistics, MODELS, METRICS
D = ROOT/'reports/final_comparison'
runs = pd.read_csv(D/'all_runs.csv',float_precision='round_trip')
assert validate_coverage(runs)
print('Saved-artifact analysis only:',len(runs),'records; no model execution')
"""
    sections=[
      ('Introduction and objectives', 'Compare detection, localization, frozen threshold transfer and measured cost without redefining historical policies. The recommendation is a research decision requiring new deployment validation.', setup),
      ('Methods and architectural variants','All six methods retain their recorded names and limitations; simplified PatchCore, SSN Lite and reduced EfficientAD are not paper-equivalent implementations.', "display(runs[['model','score_direction','latency_scope']].drop_duplicates())"),
      ('Dataset and shared partitions','The same fitting/development/calibration/test manifests supply all methods. Normal-only calibration and fixed category coverage are checked from saved identities.', "display(pd.DataFrame(json.loads((D/'audit_summary.json').read_text())['partition_counts'].items(),columns=['partition','images']))"),
      ('Source and integrity audit','The deep audit hashes original run artifacts and reconciles 270 primary image evaluations from existing prediction CSVs. Pixel metrics remain the exact recorded native-resolution values.', "display(json.loads((D/'audit_summary.json').read_text()))\ndisplay(pd.read_csv(D/'prediction_reconciliation.csv').groupby('model').agg(max_error=('max_metric_error','max'),all_flags_match=('prediction_flags_match','all')))"),
      ('Canonical result construction','One row represents model/category/seed. Source result, prediction, configuration and calibration identities are retained; equal categories, not large image counts, determine macro values.', "display(runs[['model','category','seed','source_result_sha256']].head(8))\nassert not runs.duplicated(['model','category','seed']).any()"),
      ('Overall nine-metric comparison','Ranking and calibrated operating points are separate. A lower false-positive rate can coexist with very poor recall, so none of the nine metrics alone decides a candidate.', "display(category_macro(runs).round(6))\ndisplay(Image(filename=str(D/'figures/ranking_metrics.png')))\ndisplay(Image(filename=str(D/'figures/operating_metrics.png')))"),
      ('Category image ranking','Every category has three paired seeds per architecture. Category heatmaps show descriptive rankings, not 15 independent declarations of superiority.', "display(Image(filename=str(D/'figures/heatmap_image_auroc.png')))\ndisplay(pd.read_csv(D/'category_findings.csv'))"),
      ('Category localization','Pixel AP penalizes diffuse or off-target responses differently from pixel AUROC. AU-PRO weights connected defect regions rather than treating each pixel as an independent replicate.', "display(Image(filename=str(D/'figures/heatmap_pixel_ap.png')))\ndisplay(Image(filename=str(D/'figures/heatmap_au_pro.png')))"),
      ('Objects, textures and seed instability','Five textures and ten objects are summarized separately. Three seeds share official test images; their variation measures experimental factors, not three independent test datasets.', "display(pd.read_csv(D/'texture_object_summary.csv').round(4))\ndisplay(Image(filename=str(D/'figures/seed_stability.png')))"),
      ('Paired statistical uncertainty','Resample 15 paired category means after averaging seeds. Exact sign randomization assumes category exchangeability; Holm adjustment spans all 135 contrasts. No pixel-level independence is assumed.', "computed = paired_statistics(runs)\nstored = pd.read_csv(D/'paired_comparisons.csv',float_precision='round_trip')\nimport numpy as np\nassert np.allclose(computed.difference,stored.difference,atol=1e-14)\ndisplay(stored.query(\"model_a == 'patchcore' and model_b == 'supersimplenet'\")[['metric','difference','ci_low','ci_high','p_holm_135']])\nprint('Holm p<.05 contrasts:',int((stored.p_holm_135<.05).sum()),'of',len(stored))"),
      ('Calibration sample size and ties','Nominal .05 is a quantile policy, not a guaranteed test FPR. Small calibration sets and strict ties make achieved calibration exceedance rates discrete and often below nominal.', "display(pd.read_csv(D/'calibration_summary.csv').round(5))\ndisplay(Image(filename=str(D/'figures/calibration_transfer.png')))"),
      ('Threshold transfer by category','Compare recall and FPR at the frozen cutoff only. Test normal binomial intervals condition on an existing model; repeated-seed intervals are not pooled as independent observations.', "display(Image(filename=str(D/'figures/heatmap_recall.png')))\ndisplay(Image(filename=str(D/'figures/heatmap_fpr.png')))\ndisplay(pd.read_csv(D/'threshold_seed_variability.csv').head(15))"),
      ('Compute and storage','Latencies are medians of per-run percentiles, not a pooled percentile. Decoded-RGB end-to-end timings exclude file decode. Missing early RSS remains missing; metric-sort peaks are not serving minima.', "display(pd.read_csv(D/'compute_summary.csv').round(4))\ndisplay(Image(filename=str(D/'figures/quality_latency.png')))"),
      ('Failure evidence and existing examples','Existing failure tables and original notebooks are the evidence. Inspect all 15 saved EfficientAD panels locally; no model rerun or new predictions are produced. Public outputs clear licensed dataset pixels.', "display(pd.read_csv(D/'failure_source_inventory.csv'))\npanels=pd.read_csv(ROOT/'reports/efficientad/failure_examples.csv').groupby('category',sort=True).first()\nfor category,row in panels.iterrows():\n    print(category,row['outcome'],row['path'])\n    panel=ROOT/row['panel']\n    if panel.exists():\n        display(Image(filename=str(panel),width=1050))\n    else:\n        print('Licensed local panel unavailable; source identity retained')"),
      ('Fine defects, halos and aggregation','The original size summaries have different seed scopes. Preserve each source, and distinguish observed misses/spatial spread from hypothesized bank-coverage, Gaussian-alignment or synthetic-transfer mechanisms.', "for model in MODELS:\n    p=ROOT/'reports'/model/'small_defect_summary.csv'\n    if p.exists():\n        print(model,'original saved size-summary scope')\n        display(pd.read_csv(p))"),
      ('Multicriteria research selection','Shortlist PatchCore and SSN Lite by macro image AUROC, then consider localization, frozen FPR/recall, latency, storage and provenance transparently. No arbitrary weighted total is constructed.', "shortlist=category_macro(runs).sort_values('image_auroc',ascending=False).head(2)\ndisplay(shortlist)\nprint('Quality-first research prototype: Simplified PatchCore; faster balanced alternative: SSN Lite')"),
      ('Engineering recommendation and provenance','The selected research candidate is not a production-cleared model. Resolve weight/data rights, validate new acquisition conditions and calibrate a justified deployment policy on independent normal data in a later phase.', "display(Markdown('PatchCore for quality-first prototyping; SSN Lite for latency/quality balance; PaDiM for low measured fitting cost. No engineering application is started.'))"),
      ('Research limitations','One dataset, three dependent seeds, unequal architectures/budgets, missing early RSS, cached fitting and historical GPU power variation limit comparisons. Pretraining associations are observational, not causal ablations.', "display(Markdown('Full limitations and observed-versus-hypothesized failure interpretations: [written report](../reports/final_comparison/report.md).'))"),
      ('Final conclusions','Pretrained feature methods are strongest descriptively here. Reduced EfficientAD has substantial threshold-transfer problems despite useful pixel responses. None of these results establishes universal industrial performance.', "display(category_macro(runs)[['image_auroc','pixel_ap','recall','fpr','f1']].round(4))"),
      ('References and reproducibility','All primary experimental claims come from pinned local records, not paper tables. Rebuild canonical analysis with the project interpreter; deep auditing also requires excluded local weights/maps. Read the report for exact commands and primary references.', "display(Markdown('See [research report and references](../reports/final_comparison/report.md), [analysis plan](../reports/final_comparison/analysis_plan.md), and [source provenance](../reports/final_comparison/source_provenance.json).'))")]
    for i,(title,before,code) in enumerate(sections,1):
        cells.append(nbf.v4.new_markdown_cell(f'## {i}. {title}\n\n{before}'))
        cell=nbf.v4.new_code_cell(code)
        if i==14:cell.metadata.contains_dataset_images=True
        if i==14:
            cell.source += "\nfor evidence in json.loads((D/'visual_evidence_provenance.json').read_text()):\n    panel=ROOT/evidence['path']\n    print(evidence['path'])\n    if panel.exists():\n        display(Image(filename=str(panel),width=1000))"
        cells.append(cell)
        # Include section-specific report prose as interpretation, not only generic narration.
        interpretations={1:'The analysis covers exactly 270 final evaluations without changing predictions. Research selection is conditional on this public benchmark and cannot substitute for a separate engineering validation dataset.',
         2:'Pretrained methods share representations but use different adaptation, aggregation and smoothing. Reduced EfficientAD omits the external-image penalty and selects only the local map; its result cannot be attributed to the complete published architecture.',
         3:'The fitting/development/calibration/test counts are 2551/539/539/1725. Calibration contains only normal train images and has as few as nine samples per category, a central limitation for a nominal five-percent operating point.',
         4:'All 270 primary image-metric comparisons have zero numerical error after the documented historical float32 restoration when needed. Large map/checkpoint identities are checked without recomputing predictions or native pixel metrics.',
         6:'PatchCore has the highest ranking/localization macro point estimates; SSN Lite has slightly stronger frozen F1. The scratch methods achieve low FPR alongside low recall, demonstrating why false alarms must be read with missed defects.',
         7:'Screw changes rankings substantially: AE image AUROC is 0.9573, PatchCore 0.8690, SSN Lite 0.8226, PaDiM 0.7662 and reduced EfficientAD 0.0572. AE nevertheless has screw recall only 0.0644 at its frozen cutoff.',
         8:'PatchCore macro pixel AP is 0.5402 versus SSN Lite 0.4361 and PaDiM 0.4538. Reduced EfficientAD reaches 0.4985 despite weaker image ranking; AE screw pixel AP is only 0.0263 despite strong image ranking.',
         9:'These summaries retain each model/category seed SD rather than treating seeds as independent datasets. Texture failures are strongly category dependent: leather can rank nearly perfectly while transferring its cutoff poorly.',
         10:'The complete paired table retains all effects and uncertainty, including null results. Small PatchCore/SSN image-ranking differences do not establish superiority. Category-bootstrap intervals and multiplicity-adjusted tests address different uncertainty questions.',
         11:'Calibration exceedance is measured on the same normals that set the cutoff and is not a future-risk guarantee. EfficientAD exhibits a large calibration-to-test FPR gap; ties and genuine texture hot spots are recorded evidence, while distribution shift remains a possible explanation.',
         12:'PatchCore leather/carpet FPR is 0.6979/0.5595; SSN Lite is 0.5625/0.4881. Reduced EfficientAD is 0.8021/0.5833. These actual rates prevent a claim that the frozen threshold transfers reliably even for the strongest ranking models.',
         13:'Using medians of per-run decoded-RGB end-to-end p50, PatchCore takes 33.93 ms, SSN Lite 28.66 ms, PaDiM 20.89 ms and EfficientAD 30.54 ms. These exclude file decoding; training times omit shared pretraining and differ in feature-cache reuse.',
         14:'These are genuine previously saved examples, not newly inferred maps. Screw scratch responses can coexist with image misses; normal leather shows actual hot spots, and grid responses extend beyond the annotated thread. See the original visual-review document for all 15 recorded observations.'}
        cells.append(nbf.v4.new_markdown_cell(interpretations.get(i,before+' Full quantitative values, scientific interpretation and practical limitations are retained in the accompanying written report and source-linked CSV tables.')))
    nb.cells=cells
    nbf.write(nb,ROOT/'notebooks/07_model_comparison.ipynb')


if __name__=='__main__':
    publish()
