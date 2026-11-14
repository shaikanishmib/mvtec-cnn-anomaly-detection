"""Write measured findings and update only the PaDiM roadmap entry."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=ROOT/'reports/padim'
    summary=json.loads((report/'summary.json').read_text());macro=summary['macro']
    categories=pd.read_csv(report/'all_category_means.csv').set_index('category')
    comparison=pd.read_csv(report/'model_macro_comparison.csv',index_col=0)
    worst=categories.sort_values('recall_mean').head(3)
    false=categories.sort_values('fpr_mean',ascending=False).head(3)
    lines=['# PaDiM findings','',
        'All 15 categories completed across seeds 2026, 2027 and 2028 (45 category-specific models). '
        'The real GPU pilot and 16 bounded development runs preceded the frozen confirmation. '
        'No hyperparameter or threshold changed after official test access.','',
        '## Selected method','',
        'Frozen ImageNet V1 ResNet18 layer1/2/3 features, nearest alignment to 64 square, '
        '100 seeded channels from 448, position-specific unbiased full covariances + 0.01 I, '
        'Cholesky Mahalanobis distances, bilinear 256-square maps and Gaussian sigma 4. '
        'Image score is maximum smoothed map. The selected four-category synthetic objective '
        'was 0.716183. Synthetic proxies are not real defects.','',
        'The shared whole-image 256-square bilinear geometry differs from the paper\'s '
        '224 center crop and bicubic interpolation; we use three separate seeds rather '
        'than a ten-model score ensemble. Report these as shared-protocol PaDiM results, '
        'not a direct reproduction of the publication.','',
        '## Measured macro results','',
        '| Metric | Mean | Seed SD | Conditional image-bootstrap 95% interval |',
        '|---|---:|---:|---|']
    for name in ['image_auroc','image_ap','pixel_auroc','pixel_ap','au_pro','recall','fpr','precision','f1']:
        r=macro[name];lo,hi=r['image_bootstrap_95ci']
        lines.append(f"| {name} | {r['mean']:.6f} | {r['seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines+=['',
        'Primary pixel metrics are exact at native resolution. Pixel intervals use '
        'whole-image resampling of 4096-bin calibration-scaled histograms; image metric '
        'intervals are exact conditional on fitted models and fixed thresholds. '
        'Seed SD and calibration-sampling uncertainty are distinct. The summary retains '
        'binned-minus-exact point differences as an approximation diagnostic.','',
        '## Comparison with all four earlier models','',
        '| Model | Image AUROC | Pixel AUROC | Pixel AP | AU-PRO | Recall | FPR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in comparison:
        lines.append('| '+name+' | '+' | '.join(f'{comparison.loc[k,name]:.6f}' for k in ['image_auroc','pixel_auroc','pixel_ap','au_pro','recall','fpr'])+' |')
    lines+=['','Paired category/seed/image bootstrap differences for every previous model '
        'are retained in paired_image_difference_intervals.csv. These are conditional '
        'benchmark comparisons, not a deployment generalization claim.','',
        '## Threshold errors and localization limits','',
        'The weakest category recalls were '+', '.join(f'{c} ({r.recall_mean:.3f})' for c,r in worst.iterrows())+'. '
        'The largest normal test FPRs were '+', '.join(f'{c} ({r.fpr_mean:.3f})' for c,r in false.iterrows())+'. '
        'The higher empirical normal q95 calibration target does not guarantee a 5% '
        'test or population false-alarm rate. Small calibration sets and changed normal '
        'variation can produce substantial threshold errors even when ranking is good.','',
        'All per-defect results, pixel operating points, category/seed predictions and '
        'confusion counts remain available. Mask-area strata and low-recall defect '
        'types are retrospective descriptions; repeated images across three seeds '
        'are identified as such. Licensed strongest-FP/weakest-FN panels stay local. '
        'Spatial Gaussian mismatch, normal multimodality and smoothing of small '
        'defects are plausible limitations; no counterfactual isolates their causes.','',
        '## Measured resources','',
        f"Total extraction + Gaussian fit time: {summary['total_fit_wall_seconds']:.2f}s across all 45 models. "
        f"Mean category/seed batch-one model p50/p95: {summary['mean_model_p50_ms']:.3f}/"
        f"{summary['mean_model_p95_ms']:.3f} ms; end-to-end p50/p95: "
        f"{summary['mean_end_to_end_p50_ms']:.3f}/{summary['mean_end_to_end_p95_ms']:.3f} ms. "
        'These are means of per-run quantiles, not pooled quantiles. Timing uses 50 '
        'warmups and 200 synchronized iterations; file decode is excluded. Model time '
        'contains feature extraction + Gaussian scoring; smoothing/native restoration '
        'are postprocessing.','',
        f"Peak fitting allocated VRAM: {summary['peak_training_gpu_allocated_gib']:.3f} GiB; "
        f"peak inference allocated VRAM: {summary['peak_inference_allocated_gib']:.3f} GiB; "
        f"peak evaluation process RSS: {summary['peak_evaluation_cpu_rss_gib']:.3f} GiB. "
        f"Maximum Gaussian checkpoint: {summary['max_checkpoint_bytes']:,} bytes. "
        'Shared frozen ImageNet weights are stored separately. No fit-time feature '
        'cache hides the repeated per-seed extraction cost; existing resized RGB '
        'caches may be reused. Native metric sweeps dominate evaluation cost.','',
        '## Scope and next work','',
        'Milestones 1–6 are preserved. This milestone completes PaDiM only. '
        'EfficientAD has not been started. A future family comparison or deployment '
        'choice needs its own authorized stage and independent confirmation for '
        'claims beyond this benchmark.']
    (report/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('| 05 | PaDiM | Planned |','| [05](notebooks/05_padim.ipynb) | PaDiM | Complete: verified GPU pilot, bounded development and all 15 categories across three seeds |')
    old='both successful and unsuccessful experiments. The Basic CNN, convolutional autoencoder, PatchCore variant and SuperSimpleNet Lite baselines are complete;'
    new='both successful and unsuccessful experiments. The Basic CNN, convolutional autoencoder, PatchCore variant, SuperSimpleNet Lite and PaDiM baselines are complete;'
    text=text.replace(old,new)
    marker='\n## Milestone 7 — PaDiM\n'
    assert marker not in text,'Preserve existing PaDiM section; investigate instead of appending twice'
    text+=marker+'\n'+(
        'Notebook 05 completes the frozen ResNet18 spatial-Gaussian model on all 15 '
        'categories across seeds 2026/2027/2028. The real normal-image GPU pilot '
        'passed before the 16-run bounded development study. Configuration: 100 '
        'seeded channels, covariance +0.01 I, sigma-4 smoothed Mahalanobis maps.\n\n'
        f"Macro image AUROC **{macro['image_auroc']['mean']:.6f}**, pixel AUROC "
        f"**{macro['pixel_auroc']['mean']:.6f}**, pixel AP **{macro['pixel_ap']['mean']:.6f}**, "
        f"AU-PRO **{macro['au_pro']['mean']:.6f}**; normal-q95 operating recall "
        f"**{macro['recall']['mean']:.6f}**, FPR **{macro['fpr']['mean']:.6f}**. "
        'Threshold transfer remains a material limitation; inspect category results '
        'and confusion counts alongside ranking.\n\n'
        'See [findings](reports/padim/findings.md), [full summary](reports/padim/summary.json), '
        '[all runs](reports/padim/all_runs.csv), '
        '[four-model comparison](reports/padim/model_macro_comparison.csv), '
        '[protocol](docs/padim_protocol.md) and [reproduction](docs/padim_reproduction.md). '
        'Original masks are evaluated at native resolution; only pixel bootstrap '
        'intervals use a disclosed histogram approximation. Checkpoints/maps/panels '
        'remain ignored locally. EfficientAD remains planned.\n')
    readme.write_text(text,encoding='utf-8')
    print('Wrote measured findings and PaDiM roadmap entry')


if __name__=='__main__':main()
