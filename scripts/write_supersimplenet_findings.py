"""Write measured post-freeze findings without changing models or thresholds."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/supersimplenet'
summary=json.loads((out/'summary.json').read_text());frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())
comparison=pd.read_csv(out/'model_macro_comparison.csv',index_col=0)
categories=pd.read_csv(out/'all_runs.csv').groupby('category').mean(numeric_only=True)
stability=pd.read_csv(out/'calibration_stability.csv');examples=pd.read_csv(out/'failure_examples.csv')
small=pd.read_csv(out/'small_defect_summary.csv');types=pd.read_csv(out/'defect_type_size_summary.csv')
dev=pd.read_csv(out/'development_macro.csv');paired=pd.read_csv(out/'paired_image_difference_intervals.csv')
brightness=pd.read_csv(out/'normal_brightness_sensitivity.csv')
m=summary['macro'];lines=['# SuperSimpleNet Lite: measured findings','',
    f"All15 categories completed seeds {frozen['seeds']} with frozen `{frozen['selected_variant']}` / `{frozen['configuration']['aggregation']}` policy.",
    'This is an intentionally reduced-budget ICPR-style variant: ResNet18 ImageNet V1,',
    'aligned layer2/3 features, widths384/256/32,40 epochs, batch8 and gradient clipping10.',
    'It is not a reproduction of the published WideResNet50-2/300-epoch benchmark.',
    'The original adapted-classifier and unsupervised detached-segmentation routes are explicit.',
    'No official test image or mask entered development, fitting or threshold selection.','',
    '## Development evidence','',
    'The bounded study retained16 completed trials over four predefined object/texture categories.',
    'An externally interrupted focal-only cable trial retained its first two epochs and restarted',
    'once as v2 because no checkpoint existed. The other completed trials were not repeated.',
    'Reference width fit memory but projected to3.4 hours for45 fits at40 epochs and was excluded',
    'by the prospective90-minute training budget. Lite magnitude, adaptation and loss ablations',
    'used the same fixed .05 held-out synthetic proxies for final selection, avoiding an easier',
    'evaluation task for the stronger-noise candidate. Within-trial curves retained training noise.',
    f"Selected mean common-proxy image AUROC / pixel AP / objective: {summary['seed_budget_decision']['learning_evidence']['common_proxy_image_auroc']:.6f} / {summary['seed_budget_decision']['learning_evidence']['common_proxy_pixel_ap']:.6f} / {summary['seed_budget_decision']['learning_evidence']['objective']:.6f}.",
    'These proxies establish synthetic learnability, not industrial defect accuracy. The small',
    'development set, single development seed and coupled backbone/width reduction limit inference.',
    'There was no full-budget reference-quality comparison.','',
    'PatchCore and this model use ImageNet priors, whereas the earlier CNN and',
    'autoencoder trained from scratch. Matching partitions and metrics does not',
    'isolate the causal effect of learned synthetic discrimination: feature geometry,',
    'scoring and training differ as well. No additional unregistered ablation was run.','',
    '## Exact-record macro comparison','',
    '| Metric | Basic CNN | Autoencoder | PatchCore variant | SuperSimpleNet Lite |',
    '|---|---:|---:|---:|---:|']
for key,row in comparison.iterrows():lines.append(f"| {key} | {row['Basic CNN']:.6f} | {row['Autoencoder']:.6f} | {row['PatchCore variant']:.6f} | {row['SuperSimpleNet Lite']:.6f} |")
lines+=['',f"Image AUROC / pixel AP / recall / test FPR are {m['image_auroc']['mean']:.6f} / {m['pixel_ap']['mean']:.6f} / {m['recall']['mean']:.6f} / {m['fpr']['mean']:.6f}.",
    'The comparison must consider localization, recall and false alarms together with latency.',
    'AP and precision reflect the benchmark defect prevalence. Small macro differences do not',
    'establish statistical superiority. Paired intervals condition on the fitted models and',
    'calibrated thresholds, preserving category/image dependence and matching training seeds.',
    'They exclude fitting and calibration sampling uncertainty; seed standard deviations are separate.',
    f"Image FPR whole-image bootstrap95% interval: {m['fpr']['image_bootstrap_95ci']}.",
    f"Pixel operating macro: {summary['pixel_operating_macro']}.",
    'Pixel decisions use their own threshold and background pool, distinct from image decisions.','',
    '## Focal categories and normal calibration','',
    f"Leather FPR {categories.loc['leather','fpr']:.6f}; carpet FPR {categories.loc['carpet','fpr']:.6f}; screw image AUROC {categories.loc['screw','image_auroc']:.6f} and recall {categories.loc['screw','recall']:.6f}.",
    f"Lowest category image AUROC: {categories.image_auroc.idxmin()} ({categories.image_auroc.min():.6f}); highest FPR: {categories.fpr.idxmax()} ({categories.fpr.max():.6f}).",
    f"Reserved normal calibration counts range from {stability.calibration_images.min()} to {stability.calibration_images.max()} images per category/seed.",
    'Nominal image FPR is5%, using higher q95 with strict greater-than decisions. Under continuous',
    'IID normals, finite order-statistic sampling alone gives expected unseen-normal FPR',
    f"between {stability.iid_continuous_order_stat_expected_fpr.min():.2%} and {stability.iid_continuous_order_stat_expected_fpr.max():.2%} for these sample sizes; this is not a target guarantee.",
    'The stability table includes empirical calibration FPR, score concentration, normal test shift,',
    'threshold bootstrap intervals, binomial Wilson test-FPR intervals and IID order-statistic intervals.',
    'Zero empirical false alarms do not imply a zero population FPR. Bootstrap intervals alone',
    'can miss unseen tail values. Correlated pixels are not independent calibration observations.',
    'Calibration distributions contain only original clean normals. No test-tuned threshold was introduced.','',
    'Small normal-development brightness changes were evaluated only as a post-freeze diagnostic.',
    f"Across altered views, mean absolute image-score change was {brightness[brightness.brightness_factor.ne(1)].score_change.abs().mean():.6f}; {int((brightness[brightness.brightness_factor.ne(1)].alarm!=brightness[brightness.brightness_factor.ne(1)].base_alarm).sum())} alarm decisions changed.",
    'Those augmented views never entered fitting or calibration. This sensitivity does not prove',
    'that brightness caused test false positives, and no policy was changed in response.','',
    '## Observed failures and fine localization','',
    'Each category has deterministic original/mask/map/overlay examples in the full local notebook.',
    'Compact numerical diagnostics and paths remain public; original dataset images are not in Git.']
for category in ['leather','carpet','screw','transistor']:
    cases=examples[examples.category.eq(category)&examples.outcome.isin(['FP','FN'])]
    for row in cases.itertuples():
        lines.append(f"- `{row.path}` is {row.outcome}: image score {row.score:.6f} versus frozen threshold {row.image_threshold:.6f}; top1% native-pixel overlap with ground truth is {row.top1pct_pixel_defect_overlap:.4f}, and predicted pixel area is {row.predicted_native_pixel_fraction:.2%}. Image and pixel decisions are separate.")
lines+=['', 'Native defect-size strata (seed2026; retrospective, no selection):','',
    '| Size group | Images | Recall | Median native mask fraction |','|---|---:|---:|---:|',
    *[f'| {r.size_group} | {r.images} | {r.recall:.6f} | {r.median_mask_fraction:.6f} |' for r in small.itertuples()],
    '', 'A dense head can respond outside the actual defect or miss a defect even if synthetic',
    'discrimination is strong. The displayed examples and defect/background logit differences',
    'quantify that behavior; they do not isolate a single cause. The64-grid head interpolates',
    '32/16-grid backbone activations and supplies no new measured fine detail. Smoothing and',
    'coarse pretrained features are plausible limitations, but no higher-resolution counterfactual',
    'was run. Structural and scratch-type results remain in the per-defect tables, including misses.',
    'Synthetic-to-real transfer is tested by the final benchmark, not assumed from proxy AUROC.','',
    '## Measured compute and storage','',
    f"Final active fit wall seconds: {summary['total_fit_wall_seconds']:.2f}; training-loop seconds: {summary['total_training_loop_seconds']:.2f}; native evaluation seconds: {summary['total_evaluation_seconds']:.2f}.",
    f"Mean category/seed model p50/p95: {summary['mean_model_p50_ms']:.3f}/{summary['mean_model_p95_ms']:.3f}ms; decoded-RGB-to-native-CPU-map p50/p95: {summary['mean_end_to_end_p50_ms']:.3f}/{summary['mean_end_to_end_p95_ms']:.3f}ms.",
    f"Feature extraction / learned-head p50: {summary['mean_extraction_p50_ms']:.3f}/{summary['mean_heads_p50_ms']:.3f}ms.",
    f"Peak allocated training / inference VRAM: {summary['peak_training_gpu_allocated_gib']:.3f}/{summary['peak_inference_allocated_gib']:.3f}GiB; peak sampled evaluation RSS: {summary['peak_evaluation_cpu_rss_gib']:.3f}GiB.",
    f"Largest category-head checkpoint: {summary['max_head_checkpoint_bytes']}bytes; shared frozen extractor: {summary['shared_backbone_storage']['bytes']}bytes once.",
    f"Compatible per-image feature cache: {summary['deterministic_feature_images']} images, {summary['deterministic_feature_cache_bytes']/2**30:.3f}GiB; reused across seeds and not needed for inference.",
    'Timing uses float32, batch1,50 warmups and200 measurements; decoded input is memory-resident,',
    'file decode excluded. Gaussian/native interpolation and CPU map transfer appear in end-to-end',
    'cost rather than model-only time. GPU peaks use allocator counters; process RSS sampling is20Hz.',
    'Power context is recorded. Different job times and thermal conditions limit tiny latency comparisons.',
    'Final checkpoints support exact epoch recovery; interrupted downtime is excluded from active training.',
    'Pixel bootstrap quantization bias is reported explicitly, while primary native metrics remain exact.','',
    '## Conclusions and limits','',
    'This study evaluates a fixed, affordable learned feature-discrimination baseline and preserves',
    'its weak settings and failure cases. Results apply to these splits, widths, epochs, preprocessing',
    'and calibration policy. They do not establish paper equivalence or universal production quality.',
    'Future changes need a new preregistered study rather than post-test tuning. No Notebook05,',
    'PaDiM or EfficientAD work was started.']
(out/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Measured findings written; no training or calibration changes')
