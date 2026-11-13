"""Write only measured post-confirmation interpretations; never tune policy."""
from pathlib import Path
import json
import pandas as pd
import nbformat
root=Path(__file__).resolve().parents[1];output=root/'reports/patchcore'
summary=json.loads((output/'summary.json').read_text());m=summary['macro']
comparison=pd.read_csv(output/'model_macro_comparison.csv',index_col=0)
categories=pd.read_csv(output/'all_runs.csv').groupby('category').mean(numeric_only=True)
kinds=pd.read_csv(output/'texture_object.csv').set_index('kind')
examples=pd.read_csv(output/'failure_examples.csv');fn=examples[examples.outcome.eq('FN')]
coverage=pd.read_csv(output/'memory_coverage.csv');small=pd.read_csv(output/'small_defect_summary.csv')
types=pd.read_csv(output/'defect_type_size_summary.csv');scratch=types[types.defect.str.contains('scratch')]
dev=pd.read_csv(output/'development_macro.csv')
paired=pd.read_csv(output/'paired_image_difference_intervals.csv')
lines=['# PatchCore variant: measured findings','',
    'Frozen ResNet18 ImageNet V1 layer2/layer3 features, local 3x3 pooling,',
    '384-dimensional concatenated embeddings and 1,024 patches per image.',
    'Normal banks contain 2,048 patches selected from at most 20,000 candidates',
    'using 32-dimensional projected farthest-first sampling. Scoring uses full',
    '384-dimensional blocked L2 and top-1%-patch mean image aggregation. All 15',
    'categories completed seeds 2026/2027/2028 without post-test tuning.','',
    f"The selected development objective was {dev.iloc[0].objective:.6f}; the next candidate was {dev.iloc[1].objective:.6f}.",
    'This small, single-development-seed proxy advantage is not proof of an',
    'optimal architecture. All budgets and both aggregations remain recorded.','',
    '| Metric | PatchCore variant | Basic CNN | Autoencoder |','|---|---:|---:|---:|']
for key,row in comparison.iterrows():lines.append(f"| {key} | {row['PatchCore variant']:.6f} | {row['Basic CNN']:.6f} | {row['Autoencoder']:.6f} |")
lines+=['',f"Recall was {m['recall']['mean']:.2%}; test FPR was {m['fpr']['mean']:.2%} versus the reserved-normal 5% target.",
    f"Whole-image bootstrap FPR interval: {m['fpr']['image_bootstrap_95ci'][0]:.2%}–{m['fpr']['image_bootstrap_95ci'][1]:.2%}.",
    'Ranking, localization AP/AU-PRO and recall improved over both earlier models,',
    'but false alarms increased and inference became slower. The results describe',
    'a quality/cost tradeoff, not unconditional production superiority. Image AP',
    'and precision reflect the benchmark defect prevalence, not a production rate.','',
    f"Lowest category image AUROC: {categories.image_auroc.idxmin()} ({categories.image_auroc.min():.4f}).",
    f"Lowest category recall: {categories.recall.idxmin()} ({categories.recall.min():.2%}).",
    f"Highest category test FPR: {categories.fpr.idxmax()} ({categories.fpr.max():.2%}).",
    f"Carpet test FPR was {categories.loc['carpet','fpr']:.2%}; leather was {categories.loc['leather','fpr']:.2%}.",
    f"Texture/object mean FPR: {kinds.loc['texture','fpr']:.2%} / {kinds.loc['object','fpr']:.2%}.",
    'Normal-score histograms and per-category threshold bootstrap intervals expose',
    'this shift relative to reserved calibration normals. High image AUROC can',
    'coexist with an unsuitable threshold. No threshold was changed after testing.','',
    f"At separate pixel thresholds, macro recall/precision/F1 were {summary['pixel_operating_macro']['pixel_recall']:.4f} / {summary['pixel_operating_macro']['pixel_precision']:.4f} / {summary['pixel_operating_macro']['pixel_f1']:.4f};",
    f"pooled-within-category background pixel FPR averaged {summary['pixel_operating_macro']['pixel_fpr']:.2%}.",
    'Pixel background includes normal and defective test images. These decisions',
    'use independent pixel calibration and must not be confused with image FPR.',
    'Broad responses beyond defects limit operating-point precision despite high',
    'pixel AUROC. The separate pixel tables include exact CNN/AE operating points.','',
    '## Observed failures and memory evidence','',
    '- `leather/test/good/001.png` and `carpet/test/good/003.png` are genuine normals with predicted areas over natural texture. Their nearest references are genuine fitting normals. This demonstrates false alarms despite visually similar normal references; it does not prove one cause such as lighting or insufficient bank size.',
    '- `carpet/test/metal_contamination/000.png` is missed at image score 1.146842 against threshold 1.256326; none of its top 11 patch cells intersect the displayed mask. The larger color defect `carpet/test/color/000.png` is detected, with a broad response beyond the native defect boundary.',
    '- `screw/test/manipulated_front/000.png` is missed at 1.788331 against threshold 1.878041. The normal `screw/test/good/014.png` scores 1.878239, just above that threshold; its marked highest-distance cell is in the background. These examples expose both missed fine structure and sensitivity to normal non-defect regions.',
    '- `transistor/test/misplaced/002.png` visibly lacks the component but scores 1.928370 below threshold 2.018568. Its selected query cell matches background in a fitting image. This is consistent with position-agnostic normal-patch matching accepting background at an abnormal location; no positional ablation was run to isolate that mechanism.','',
    f"The selected panels include {len(fn)} false negatives. Their median top-11 patch/mask-cell overlap was {fn.top_patch_defect_overlap.median():.4f}.",
    'Patch-cell overlap is not native pixel precision and receptive fields exceed',
    'the 8x8 grid cell. Smooth/broad maps also reflect the 32-square feature grid',
    'and interpolation; no counterfactual higher-resolution study was performed.',
    f"Minimum fraction of fitting source images represented in a bank: {coverage.fraction_fitting_images_represented.min():.2%}.",
    'Image-source coverage and projected candidate radius do not guarantee coverage',
    'of all rare local patterns. The random candidate cap remains an approximation.',
    f"Largest selected cached-feature NN replay difference: {examples.cached_replay_max_abs_difference.max():.8f}.",
    'No completed feature extraction was repeated for these reference panels.','',
    'Seed-2026 native mask-area analysis (descriptive and category-confounded):','',
    '| Defect area | Images | Recall |','|---|---:|---:|']
for row in small.itertuples(index=False):lines.append(f'| {row[0]} | {row.images} | {row.recall:.4f} |')
if len(scratch):lines+=['',f"Scratch-named defect types: {int(scratch.images.sum())} images, weighted seed-2026 recall {(scratch.images*scratch.recall).sum()/scratch.images.sum():.4f}."]
lines+=['','The smaller groups have lower recall here; category/defect composition and',
    'feature detail are confounded. These observations do not isolate a resolution',
    'effect. Exact native masks were retained for primary localization metrics.','',
    '## Measured cost and uncertainty','',
    f"All 45 bank fits took {summary['total_bank_fit_wall_seconds']:.2f} s, including {summary['total_coreset_seconds']:.2f} s of coreset construction.",
    f"Native evaluation took {summary['total_evaluation_seconds']/60:.2f} minutes.",
    f"The {summary['deterministic_feature_caches']} unique final feature caches contain {summary['deterministic_feature_cache_bytes']/2**30:.3f} GiB on disk; extraction stages totaled {summary['unique_extraction_stage_seconds']:.2f} s once.",
    'Extraction-stage time includes image loading, transfer and cache writes and is',
    'already within fitting/evaluation wall time; do not add it again. Later seeds',
    'reuse compatible features. External ImageNet pretraining cost is excluded.',
    f"Mean per-run feature/NN p50: {summary['mean_extraction_p50_ms']:.3f}/{summary['mean_nn_p50_ms']:.3f} ms.",
    f"Combined model p50/p95: {summary['mean_model_p50_ms']:.3f}/{summary['mean_model_p95_ms']:.3f} ms; end-to-end p50/p95: {summary['mean_end_to_end_p50_ms']:.3f}/{summary['mean_end_to_end_p95_ms']:.3f} ms.",
    'These are means of run percentiles, not pooled service percentiles. End-to-end',
    'starts with decoded RGB in memory and excludes disk read/decode.',
    f"Peak bank-fitting GPU allocation: {summary['peak_bank_gpu_allocated_gib']:.3f} GiB; batch-one inference allocation: {summary['peak_inference_allocated_gib']:.3f} GiB.",
    f"Peak sampled evaluation process RSS: {summary['peak_evaluation_cpu_rss_gib']:.3f} GiB, including native metric scratch work, not isolated serving RAM.",
    'Inference memory scope differs from earlier evaluators, which retained GPU',
    'test-image caches; avoid claiming an isolated memory advantage from those values.',
    f"Each bank has {summary['bank_vector_bytes']:,} vector bytes; largest serialized bank is {summary['max_bank_checkpoint_bytes']:,} bytes.",
    f"One shared frozen feature extractor is {summary['shared_backbone_storage']['bytes']:,} bytes with {summary['parameters']:,} parameters and zero trainable parameters.",
    'Bank-only storage must not be compared to complete CNN/AE model storage.',
    'Paired image difference intervals reuse identical whole-image draws across',
    'models/seeds; they are conditional on these fitted models. Seed variation is',
    'reported separately. Pixel intervals target the disclosed 4096-bin transform;',
    'primary point metrics use exact raw native distances.',
    f"Maximum absolute run pixel-AP binning bias was {summary['pixel_bootstrap_quantization_bias']['pixel_ap']['maximum_absolute_run_difference']:.6f}.",'',
    '## Limits and retention','',
    'This is a simplified PatchCore variant, not a paper-equivalent reproduction:',
    'ResNet18, capped candidate pool/projected selection, direct concatenation,',
    'no paper image-score neighborhood reweighting, no Gaussian smoothing and a',
    'documented whole-image 256 geometry override of classification transforms.',
    'ImageNet transfer differs from the scratch CNN/AE priors. Finite normal',
    'calibration does not guarantee population FPR; texture false alarms are a',
    'major limitation. No independent environment rebuild or deployment claim is',
    'made. All rejected trials, exact comparisons and measured resource costs remain.',
    'The pilot executed live; continuation cells analyze separately completed jobs',
    'without claiming live fitting replay. Licensed panels, features, maps and banks',
    'remain local and ignored. No SuperSimpleNet work was started.','']
text='\n'.join(lines)
(output/'findings.md').write_text(text,encoding='utf-8')
for path in [root/'notebooks/03_patchcore.ipynb',root/'outputs/patchcore/03_patchcore.complete.executed.ipynb']:
    notebook=nbformat.read(path,as_version=4)
    cell=nbformat.v4.new_markdown_cell(text,metadata={'generated_patchcore_findings':True})
    existing=next((i for i,c in enumerate(notebook.cells) if c.metadata.get('generated_patchcore_findings')),None)
    if existing is None:notebook.cells.insert(-1,cell)
    else:notebook.cells[existing]=cell
    path.write_bytes(nbformat.writes(notebook).encode('utf-8'))
readme=root/'README.md';content=readme.read_text(encoding='utf-8')
content=content.replace('| 03 | PatchCore | Planned |','| [03](notebooks/03_patchcore.ipynb) | PatchCore variant | Complete: bounded development and all 15 categories across three coreset seeds |')
content=content.replace('The Basic CNN and convolutional autoencoder baselines are complete;',
    'The Basic CNN, convolutional autoencoder and PatchCore variant baselines are complete;')
content=content.replace('Notebooks 01 and 02 are complete. PatchCore requires a separate request.',
    'Notebooks 01, 02 and 03 are complete. SuperSimpleNet requires a separate request.')
start='<!-- PATCHCORE_FINDINGS_START -->';end='<!-- PATCHCORE_FINDINGS_END -->'
section=start+'\n## PatchCore variant measured baseline\n\n'+text.split('\n',2)[2]
section+='\nRead [Notebook 03](notebooks/03_patchcore.ipynb), [results](reports/patchcore/all_category_means.csv), [findings](reports/patchcore/findings.md), [protocol](docs/patchcore_protocol.md) and [reproduction](docs/patchcore_reproduction.md).\n'+end
if start in content:
    before,remaining=content.split(start,1);_,after=remaining.split(end,1);content=before+section+after
else:content+='\n'+section+'\n'
readme.write_text(content,encoding='utf-8')
print(text)
