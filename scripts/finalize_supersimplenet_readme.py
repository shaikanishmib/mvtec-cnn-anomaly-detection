"""Publish only reconciled completed research; preserve earlier result sections."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/supersimplenet'
summary=json.loads((out/'summary.json').read_text());frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())
assert summary['models']==45 and summary['categories']==15 and summary['seeds']==[2026,2027,2028]
assert len(list((out/'evaluation').glob('04-test-*-result.json')))==45
path=ROOT/'README.md';source=path.read_text(encoding='utf-8')
assert '<!-- SUPERSIMPLENET_FINDINGS_START -->' not in source
source=source.replace('The Basic CNN, convolutional autoencoder and PatchCore variant baselines are complete;',
    'The Basic CNN, convolutional autoencoder, PatchCore variant and SuperSimpleNet Lite baselines are complete;')
source=source.replace('policy frozen for 15 categories × three seeds; final benchmark pending',
    'complete benchmark on all 15 categories × three seeds; reduced-budget ICPR-style variant')
source=source.replace('No PatchCore work was started.','At the autoencoder milestone, no PatchCore work had started.')
source=source.replace('No SuperSimpleNet work was started.','At the PatchCore milestone, no SuperSimpleNet work had started.')
comparison=pd.read_csv(out/'model_macro_comparison.csv',index_col=0)
lines=['','<!-- SUPERSIMPLENET_FINDINGS_START -->','## SuperSimpleNet Lite: completed benchmark','',
    'Notebook04 evaluates an explicitly reduced-budget ICPR-style SuperSimpleNet.',
    'Frozen ResNet18 ImageNet V1 layer2/3 features feed a learned adapter,',
    'segmentation discriminator and auxiliary image detector. Normal-only fitting',
    'uses Perlin-masked feature-space Gaussian noise, not RGB defect corruption.',
    'The bounded four-category study selected noise standard deviation0.05,',
    'margin-plus-focal loss,40 epochs and top1% smoothed-map image scoring.',
    'All15 categories completed seeds2026/2027/2028 before official evaluation.',
    'No post-test retraining or threshold changes were made.','',
    '| Metric | Basic CNN | Autoencoder | PatchCore variant | SuperSimpleNet Lite |',
    '|---|---:|---:|---:|---:|']
for key,row in comparison.iterrows():lines.append(f"| {key} | {row['Basic CNN']:.6f} | {row['Autoencoder']:.6f} | {row['PatchCore variant']:.6f} | {row['SuperSimpleNet Lite']:.6f} |")
lines+=['',
    f"Mean model p50/p95: **{summary['mean_model_p50_ms']:.3f}/{summary['mean_model_p95_ms']:.3f}ms**; decoded-RGB-to-native-CPU-map p50/p95: **{summary['mean_end_to_end_p50_ms']:.3f}/{summary['mean_end_to_end_p95_ms']:.3f}ms** (file decode excluded).",
    f"Peak allocated training VRAM: **{summary['peak_training_gpu_allocated_gib']:.3f}GiB**. Final training-loop time: **{summary['total_training_loop_seconds']/60:.2f}minutes**; native evaluation: **{summary['total_evaluation_seconds']/60:.2f}minutes**.",
    f"Shared frozen extractor: **{summary['shared_backbone_storage']['bytes']:,}bytes** once; largest category-head checkpoint: **{summary['max_head_checkpoint_bytes']:,}bytes**.",
    '', 'These are means of run percentiles rather than pooled serving percentiles.',
    'Ranking, recall, false-positive behavior, localization and latency must be',
    'considered together. Paired image intervals preserve matching categories,',
    'seeds and image draws; they condition on fitted models and thresholds.',
    'Small macro differences do not establish statistical superiority.',
    '', 'Image/pixel thresholds use reserved unmodified normal q95/q99.5 distributions.',
    'Finite calibration sample sizes do not guarantee the nominal5% unseen-normal FPR.',
    'The reports include category/seed threshold distributions, Wilson intervals,',
    'order-statistic uncertainty, real failure panels, fine-defect results and',
    'post-freeze normal brightness sensitivity. Pixel bootstrap intervals use a',
    'disclosed fixed-bin approximation; primary native point metrics remain exact.',
    '', 'The publication uses a larger backbone, longer training and a learned primary',
    'image head. Our reduced widths,40-epoch budget and selected map aggregation are',
    'explicit differences. Strong synthetic-proxy performance does not guarantee',
    'synthetic-to-real transfer. All rejected settings and the interrupted cable',
    'attempt remain recorded. Final fitting has exact epoch/RNG recovery.',
    'Public outputs are genuine and compact; complete original/mask/overlay panels,',
    'weights and caches remain local and ignored. No Notebook05 work was started.',
    '', 'Read [Notebook04](notebooks/04_supersimplenet.ipynb), [all-category results](reports/supersimplenet/all_category_means.csv), [measured findings](reports/supersimplenet/findings.md), [frozen policy](configs/supersimplenet_frozen.json), [experiment registry](experiments/04-development-selection-v1.json) and [reproduction guide](docs/supersimplenet_reproduction.md).',
    '<!-- SUPERSIMPLENET_FINDINGS_END -->','']
path.write_text(source+'\n'.join(lines),encoding='utf-8')
print('README updated with reconciled completed benchmark only')
