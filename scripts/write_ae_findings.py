"""Write measured interpretations after frozen confirmation; never tune models."""
from pathlib import Path
import json
import pandas as pd
import nbformat

root=Path(__file__).resolve().parents[1]
output=root/'reports/autoencoder'
summary=json.loads((output/'summary.json').read_text())
cnn=json.loads((root/'reports/basic_cnn/summary.json').read_text())
macro=summary['macro']
comparison=pd.read_csv(output/'cnn_macro_comparison.csv',index_col=0)
categories=pd.read_csv(output/'all_category_means.csv').set_index('category')
kinds=pd.read_csv(output/'texture_object.csv').set_index('kind')
examples=pd.read_csv(output/'failure_examples.csv')
small=pd.read_csv(output/'small_defect_summary.csv')
normal=examples[examples.path.str.contains('/good/')]
fn=examples[examples.outcome.eq('FN')]
low_residual_fn=fn[fn.residual_inside_mask.le(fn.residual_outside_mask)]
detail_ratio=examples.reconstruction_detail_ratio.median()
detail_interpretation=(
    'The lower neighbor-detail measure is consistent with blur in these selected reconstructions; it does not establish that blur alone causes missed defects.'
    if detail_ratio<1 else
    'This detail measure does not show an aggregate reduction in these selected reconstructions; inspect individual panels before attributing failures to blur.')
size_index=small.set_index('size_group')
size_interpretation='Size groups mix categories and defect types; the table is descriptive rather than a matched resolution ablation.'
if {'at most 0.1%','above 1%'}.issubset(size_index.index):
    delta=size_index.loc['at most 0.1%','recall']-size_index.loc['above 1%','recall']
    size_interpretation=f'The smallest-group recall minus above-1%-group recall was {delta:+.4f}. '+size_interpretation
best=categories.image_auroc_mean.idxmax()
worst=categories.image_auroc_mean.idxmin()
lines=['# Convolutional autoencoder: measured findings','',
    'The frozen shallow depth-3, width-16, latent-8 autoencoder used MSE, 256-square RGB,',
    '20 fixed epochs, squared channel-mean residual maps and top-1% mean image scores.',
    'All 15 categories completed seeds 2026/2027/2028. Calibration used only the',
    'reserved normals. There was no post-test tuning or retraining.','',
    '| Metric | Autoencoder | Basic CNN | AE minus CNN |','|---|---:|---:|---:|']
for key,row in comparison.iterrows():
    lines.append(f"| {key} | {row['Autoencoder']:.6f} | {row['Basic CNN']:.6f} | {row['AE minus CNN']:+.6f} |")
lines+=['',f"Image recall was {macro['recall']['mean']:.2%} with test FPR {macro['fpr']['mean']:.2%},",
    'against a calibration image-FPR target of 5%. This is an empirical normal',
    'quantile, not a guarantee under test distribution shift. Category/seed calibration',
    'counts and threshold bootstrap intervals are retained in each evaluation record.',
    f"Two runs emitted no image alarms; primary macro precision follows the shared zero-division convention. Defined-run-only precision was {summary['metric_edge_cases']['image_precision_defined_run_mean']:.6f}, reported separately.",
    f"The highest category image AUROC was {best} ({categories.loc[best,'image_auroc_mean']:.4f});",
    f"the lowest was {worst} ({categories.loc[worst,'image_auroc_mean']:.4f}).",'',
    f"Texture image AUROC/recall: {kinds.loc['texture','image_auroc']:.4f} / {kinds.loc['texture','recall']:.4f};",
    f"object image AUROC/recall: {kinds.loc['object','image_auroc']:.4f} / {kinds.loc['object','recall']:.4f}.",
    'These are measured category summaries, not evidence that one architecture',
    'mechanism alone causes the differences.','',
    f"Mean per-run model-stage p50 was {summary['mean_model_p50_ms']:.3f} ms versus CNN {cnn['mean_model_p50_ms']:.3f} ms;",
    f"end-to-end p50 was {summary['mean_end_to_end_p50_ms']:.3f} ms versus {cnn['mean_end_to_end_p50_ms']:.3f} ms.",
    'Both exclude file read/decode; AE model timing includes residual construction.',
    'These are means of run percentiles, not pooled service latency.',
    f"Final training took {summary['total_training_seconds']/60:.2f} minutes including loading/validation.",
    f"Exact native evaluation took {summary['total_evaluation_seconds']/60:.2f} minutes.",
    f"Peak allocated training VRAM was {summary['peak_training_vram_gib']:.3f} GiB;",
    f"the model has {summary['parameters']:,} parameters and largest weight checkpoint {summary['max_checkpoint_bytes']:,} bytes.",'',
    '## Failure evidence','',
    f"Deterministically selected panels contain {len(fn)} false-negative examples; in {len(low_residual_fn)},",
    'mean residual inside the displayed defect mask was no larger than outside it.',
    'The table and panel identify each image. Low inside/outside contrast, when',
    'present, documents weak local separation. Other false negatives can have',
    'higher local error but still fail image top-k aggregation or the normal threshold.',
    f"Selected false negatives had median top-1%-error/mask overlap {fn.top1pct_defect_overlap.median():.4f}",
    f"and median overlap enrichment over mask-area fraction {fn.top1pct_defect_enrichment.median():.3f}.",
    'These display-resolution diagnostics distinguish dominant non-defect residuals',
    'from errors at the actual defect; tiny mask area also limits achievable overlap.',
    f"{int(fn.mask_pixels.eq(0).sum())} selected false-negative masks disappeared under nearest-neighbor 256-square display resizing;",
    'all primary localization metrics still used their original native masks.',
    f"Across the selected panels, median reconstructed/original neighbor-detail ratio was {examples.reconstruction_detail_ratio.median():.3f}.",
    detail_interpretation,
    f"The displayed normals had median residual on high-gradient pixels {normal.residual_on_edges.median():.6f}",
    f"versus {normal.residual_off_edges.median():.6f} off those pixels.",
    'These deliberately high-scoring normals expose normal-structure residuals;',
    'they are not a random sample of normal reconstruction quality.','',
    'Directly inspected seed-2026 panels show the following concrete cases:','',
    '- `carpet/test/color/000.png`: the dark color defect is visibly retained in the reconstruction. Its mean in-mask residual is 0.002432 versus 0.010088 outside; normal weave errors dominate this false negative.',
    '- `capsule/test/crack/000.png`: the small crack has weak residuals relative to normal printing and silhouette errors. Only 0.003049 of top-1%-error pixels overlap the displayed mask. Normal `good/002.png` is a false positive with visible printing/edge residuals.',
    '- `metal_nut/test/good/009.png`: normal rims and surface detail produce a false-positive image score of 0.070304. The bent false negative `bent/000.png` retains its coarse altered contour and scores only 0.031789.',
    '- `screw/test/scratch_head/003.png`: the image alarm is correct, but the overlay also marks normal threads; only 0.001524 of top-error pixels overlap the tiny displayed defect mask. Image detection success does not establish accurate localization.','',
    'Seed-2026 defect-size analysis (native mask area; descriptive, after freeze):','',
    '| Defect area | Images | Recall |','|---|---:|---:|']
for row in small.itertuples(index=False):lines.append(f'| {row[0]} | {row.images} | {row.recall:.4f} |')
lines+=['',size_interpretation,
    '256-square resizing and top-k dilution can suppress tiny defects in principle;',
    'no higher-resolution counterfactual was trained after test to measure that effect.',
    'See defect_type_size_summary.csv for scratch-specific counts and outcomes.','',
    '## Limits and retention','',
    'No pretrained features, skip connections, smoothing or combined loss were used.',
    'The bounded development comparison selected MSE/squared residuals over the',
    'four fixed object/texture categories. Proxy scores were modest and do not',
    'guarantee transfer to real defects. Twenty epochs limit compute and do not',
    'prove convergence for every category. Three seeds do not remove sampling',
    'uncertainty. Small normal calibration partitions can yield unstable thresholds.',
    'Pixel confidence intervals use fixed-bin whole-image bootstrap; point metrics',
    'are exact at native resolution and quantization bias is separately reported.',
    f"Mean binned-minus-exact pixel AUROC bias was {summary['pixel_bootstrap_quantization_bias']['pixel_auroc']['mean_binned_minus_exact']:+.6f},",
    f"with maximum absolute run difference {summary['pixel_bootstrap_quantization_bias']['pixel_auroc']['maximum_absolute_run_difference']:.6f}.",
    'This is material: those intervals describe the binned approximation and must',
    'not be treated as coverage intervals for the exact pixel AUROC point estimate.',
    'Compare recall, FPR, pixel AP/AU-PRO and latency together before claiming an',
    'improvement. This research benchmark does not establish production suitability.','',
    'The complete local notebook keeps original/reconstruction/error/overlay/mask',
    'panels. Public outputs contain compact measured tables and aggregate plots.',
    'Registry records are append-only, including all rejected variants. The pilot',
    'executed live; later notebook cells analyze separately recorded jobs and',
    'explicitly do not claim live training replay. No PatchCore work was started.','']
text='\n'.join(lines)
(output/'findings.md').write_text(text,encoding='utf-8')
for path in [root/'notebooks/02_convolutional_autoencoder.ipynb',root/'outputs/autoencoder/02_convolutional_autoencoder.complete.executed.ipynb']:
    notebook=nbformat.read(path,as_version=4)
    cell=nbformat.v4.new_markdown_cell(text,metadata={'generated_ae_findings':True})
    existing=next((i for i,c in enumerate(notebook.cells) if c.metadata.get('generated_ae_findings')),None)
    if existing is None:notebook.cells.insert(-1,cell)
    else:notebook.cells[existing]=cell
    path.write_bytes(nbformat.writes(notebook).encode('utf-8'))
readme=root/'README.md'
content=readme.read_text(encoding='utf-8')
content=content.replace('| 02 | Convolutional autoencoder | Planned |','| [02](notebooks/02_convolutional_autoencoder.ipynb) | Convolutional autoencoder | Complete: bounded development and all 15 categories across three seeds |')
content=content.replace('The Basic CNN baseline is complete;\nthe remaining model families and engineering application are planned.',
    'The Basic CNN and convolutional autoencoder baselines are complete;\nthe remaining model families and engineering application are planned.')
content=content.replace('Notebook 01 is complete. Notebook 02 requires a separate request.',
    'Notebooks 01 and 02 are complete. PatchCore requires a separate request.')
start='<!-- AE_FINDINGS_START -->'
end='<!-- AE_FINDINGS_END -->'
section=start+'\n## Convolutional autoencoder measured baseline\n\n'+text.split('\n',2)[2]
section+='\nRead [Notebook 02](notebooks/02_convolutional_autoencoder.ipynb), [results](reports/autoencoder/all_category_means.csv), [findings](reports/autoencoder/findings.md), [experiment registry](experiments/), [protocol](docs/autoencoder_protocol.md) and [reproduction](docs/autoencoder_reproduction.md).\n'+end
if start in content:
    before,remaining=content.split(start,1)
    _,after=remaining.split(end,1)
    content=before+section+after
else:content+='\n'+section+'\n'
readme.write_text(content,encoding='utf-8')
print(text)
