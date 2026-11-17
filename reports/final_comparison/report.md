# Final six-model research comparison

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

| Method | image_auroc | image_ap | pixel_auroc | pixel_ap | au_pro | recall | fpr | precision | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Basic CNN | 0.7126 | 0.8590 | 0.7858 | 0.2354 | 0.5498 | 0.4018 | 0.0983 | 0.8733 | 0.5106 |
| Convolutional AE | 0.6346 | 0.8279 | 0.6900 | 0.1109 | 0.4494 | 0.2350 | 0.0939 | 0.7896 | 0.3146 |
| Simplified PatchCore | 0.9676 | 0.9885 | 0.9729 | 0.5402 | 0.9204 | 0.8904 | 0.1718 | 0.9400 | 0.9043 |
| SuperSimpleNet Lite | 0.9648 | 0.9862 | 0.9528 | 0.4361 | 0.8565 | 0.8920 | 0.1662 | 0.9412 | 0.9099 |
| PaDiM | 0.8979 | 0.9464 | 0.9659 | 0.4538 | 0.9119 | 0.6965 | 0.1445 | 0.9255 | 0.7554 |
| EfficientAD-S reduced-budget | 0.8163 | 0.9186 | 0.9199 | 0.4985 | 0.8184 | 0.6656 | 0.2201 | 0.8859 | 0.7401 |

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

| category | image_leader | image_auroc | pixel_ap_leader | pixel_ap | image_auroc_range | efficientad_fpr |
| --- | --- | --- | --- | --- | --- | --- |
| bottle | Simplified PatchCore | 1.0000 | Simplified PatchCore | 0.7290 | 0.3767 | 0.0000 |
| cable | Simplified PatchCore | 0.9824 | Simplified PatchCore | 0.6434 | 0.4763 | 0.0517 |
| capsule | Simplified PatchCore | 0.9581 | Simplified PatchCore | 0.4490 | 0.4975 | 0.1739 |
| carpet | Simplified PatchCore | 0.9778 | EfficientAD-S reduced-budget | 0.6534 | 0.6383 | 0.5833 |
| grid | SuperSimpleNet Lite | 0.9616 | EfficientAD-S reduced-budget | 0.3998 | 0.1818 | 0.0000 |
| hazelnut | Simplified PatchCore | 0.9999 | Simplified PatchCore | 0.5813 | 0.1650 | 0.0167 |
| leather | Simplified PatchCore | 1.0000 | EfficientAD-S reduced-budget | 0.5857 | 0.2543 | 0.8021 |
| metal_nut | SuperSimpleNet Lite | 0.9938 | Simplified PatchCore | 0.8506 | 0.7550 | 0.4242 |
| pill | SuperSimpleNet Lite | 0.9433 | Simplified PatchCore | 0.7114 | 0.2204 | 0.0385 |
| screw | Convolutional AE | 0.9573 | Simplified PatchCore | 0.2831 | 0.9001 | 0.9106 |
| tile | EfficientAD-S reduced-budget | 0.9977 | EfficientAD-S reduced-budget | 0.8189 | 0.3253 | 0.0000 |
| toothbrush | Simplified PatchCore | 0.9648 | Simplified PatchCore | 0.5779 | 0.3833 | 0.1111 |
| transistor | Simplified PatchCore | 0.9781 | PaDiM | 0.6295 | 0.5883 | 0.0389 |
| wood | PaDiM | 0.9912 | EfficientAD-S reduced-budget | 0.6150 | 0.0655 | 0.0877 |
| zipper | SuperSimpleNet Lite | 0.9902 | Simplified PatchCore | 0.5469 | 0.4005 | 0.0625 |

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

| model | kind | image_auroc | pixel_ap | recall | fpr |
| --- | --- | --- | --- | --- | --- |
| autoencoder | object | 0.6056 | 0.1256 | 0.2092 | 0.0498 |
| autoencoder | texture | 0.6926 | 0.0815 | 0.2865 | 0.1821 |
| basic_cnn | object | 0.6388 | 0.1779 | 0.2676 | 0.0823 |
| basic_cnn | texture | 0.8602 | 0.3504 | 0.6701 | 0.1304 |
| efficientad | object | 0.7377 | 0.4404 | 0.5744 | 0.1828 |
| efficientad | texture | 0.9735 | 0.6146 | 0.8481 | 0.2946 |
| padim | object | 0.8732 | 0.4904 | 0.6304 | 0.1739 |
| padim | texture | 0.9474 | 0.3806 | 0.8287 | 0.0858 |
| patchcore | object | 0.9629 | 0.5957 | 0.8668 | 0.0956 |
| patchcore | texture | 0.9769 | 0.4290 | 0.9377 | 0.3242 |
| supersimplenet | object | 0.9551 | 0.4903 | 0.8535 | 0.1040 |
| supersimplenet | texture | 0.9842 | 0.3276 | 0.9689 | 0.2907 |

Largest within-category image-AUROC seed SD for each architecture:

| model | category | image_auroc_seed_sd |
| --- | --- | --- |
| autoencoder | capsule | 0.2420 |
| basic_cnn | screw | 0.2166 |
| efficientad | metal_nut | 0.0773 |
| padim | zipper | 0.0738 |
| patchcore | screw | 0.0101 |
| supersimplenet | screw | 0.0189 |

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

| metric | difference | ci_low | ci_high | p_holm_135 |
| --- | --- | --- | --- | --- |
| image_auroc | 0.0028 | -0.0087 | 0.0148 | 1.0000 |
| image_ap | 0.0023 | -0.0023 | 0.0071 | 1.0000 |
| pixel_auroc | 0.0201 | 0.0112 | 0.0290 | 0.0649 |
| pixel_ap | 0.1041 | 0.0795 | 0.1277 | 0.0082 |
| au_pro | 0.0639 | 0.0381 | 0.0957 | 0.0217 |
| recall | -0.0016 | -0.0419 | 0.0385 | 1.0000 |
| fpr | 0.0056 | -0.0350 | 0.0456 | 1.0000 |
| precision | -0.0012 | -0.0162 | 0.0142 | 1.0000 |
| f1 | -0.0055 | -0.0281 | 0.0145 | 1.0000 |

There are 56 Holm-adjusted p<.05 comparisons in the full 135-test table.
Inspect `paired_comparisons.csv` for every pair, including effects and intervals.
A positive point estimate with an interval crossing zero is uncertain under this
category resampling scheme. An interval excluding zero and an adjusted test
not rejecting are different evidence statements, not a contradiction. No
equivalence or universal superiority claim follows from close macro means.

## Calibration and threshold transfer

| model | calibration_n | calibration_fpr | fpr | recall | precision | f1 | runs_with_threshold_ties |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Basic CNN | 35.9333 | 0.0295 | 0.0983 | 0.4018 | 0.8733 | 0.5106 | 0 |
| Convolutional AE | 35.9333 | 0.0295 | 0.0939 | 0.2350 | 0.7896 | 0.3146 | 0 |
| Simplified PatchCore | 35.9333 | 0.0295 | 0.1718 | 0.8904 | 0.9400 | 0.9043 | 0 |
| SuperSimpleNet Lite | 35.9333 | 0.0295 | 0.1662 | 0.8920 | 0.9412 | 0.9099 | 0 |
| PaDiM | 35.9333 | 0.0295 | 0.1445 | 0.6965 | 0.9255 | 0.7554 | 0 |
| EfficientAD-S reduced-budget | 35.9333 | 0.0274 | 0.2201 | 0.6656 | 0.8859 | 0.7401 | 3 |

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

| model | training_hours | evaluation_hours | model_p50_ms | model_p95_ms | end_to_end_p50_ms | end_to_end_p95_ms | deployment_mib_one_category | train_peak_vram_gib | inference_peak_vram_gib | cpu_peak_rss_gib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Basic CNN | 0.1352 | 0.5898 | 2.4298 | 3.8787 | 10.3426 | 13.3640 | 0.3272 | 0.4805 | 0.0441 | not recorded |
| Convolutional AE | 0.3777 | 0.9784 | 3.3273 | 5.1242 | 14.0110 | 16.4359 | 0.1251 | 0.6072 | 0.0370 | not recorded |
| Simplified PatchCore | 0.0330 | 1.1368 | 15.2991 | 17.1336 | 33.9325 | 37.2452 | 13.7126 | 1.1494 | 0.0784 | 5.7659 |
| SuperSimpleNet Lite | 1.4858 | 0.7493 | 9.6683 | 17.0836 | 28.6551 | 39.2048 | 12.7889 | 0.7092 | 0.0819 | 5.5177 |
| PaDiM | 0.0099 | 0.4942 | 12.8722 | 16.1123 | 20.8881 | 24.7457 | 168.4766 | 0.3839 | 0.2315 | 7.0004 |
| EfficientAD-S reduced-budget | 2.3821 | 0.7817 | 21.9824 | 26.4717 | 30.5405 | 35.0719 | 30.7558 | 0.3089 | 0.1072 | 5.8039 |

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
.\.venv\Scripts\python.exe scripts/build_model_comparison.py
.\.venv\Scripts\python.exe scripts/publish_model_comparison.py
.\.venv\Scripts\python.exe scripts/execute_model_comparison.py
.\.venv\Scripts\python.exe scripts/validate_model_comparison.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp
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
