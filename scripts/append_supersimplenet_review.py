"""Append visually checked observations and the genuinely executed matched session."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import pandas as pd,nbformat as nbf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.data import sha256
from anomaly_detection.tracking import record
out=ROOT/'reports/supersimplenet';local=ROOT/'outputs/supersimplenet'
review=out/'observed_panel_analysis.md';assert not review.exists()
summary=json.loads((out/'summary.json').read_text());macro=summary['macro']
examples=pd.read_csv(out/'failure_examples.csv').set_index('path')
matched=pd.read_csv(out/'matched_frozen_cases.csv').set_index(['path','model'])
stability=pd.read_csv(out/'calibration_stability.csv')
paired=pd.read_csv(out/'paired_image_difference_intervals.csv')
pc=paired[paired.comparison.eq('PatchCore variant')].set_index('metric')
def case(path):return examples.loc[path]
carpet=case('carpet/test/metal_contamination/000.png');screw=case('screw/test/manipulated_front/000.png')
lead=case('transistor/test/bent_lead/003.png');wood=case('wood/test/scratch/001.png')
missing=matched.loc[('transistor/test/misplaced/002.png','SuperSimpleNet Lite')]
leather=stability[stability.category.eq('leather')];carpet_stability=stability[stability.category.eq('carpet')]
text=f'''# Observed panel review and comparative conclusion

All 15 deterministic category panels and the four matched-case panels were visually inspected after the 45 frozen evaluations completed. These observations describe retained outputs, not new model fitting or score selection. Original images, ground truth and large overlays remain local; public numeric evidence is in `failure_examples.csv`, `matched_frozen_cases.csv`, `calibration_stability.csv` and the all-category tables. Heatmap scales differ across panels, so color brightness is not a cross-model score comparison.

## What the visible maps support

- **Texture false positives persist.** The leather and carpet normal panels have empty ground truth but thresholded responses on ordinary texture and image borders. Both pretrained-feature methods classify the earlier `leather/test/good/001.png` and `carpet/test/good/003.png` cases as false positives. Basic CNN and Autoencoder also false-alarm on that leather image, but classify that carpet image as normal. This is shared and category-dependent behavior, not evidence that one model fixes normal-texture calibration.
- **Large visible defects often produce strong, broad responses.** Bottle broken rims, carpet color patches, large capsule cracks and tile cracks produce detections overlapping the actual defect. Their overlays also extend beyond the ground-truth boundaries. This agrees with the separate pixel operating point: macro pixel recall {summary['pixel_operating_macro']['pixel_recall']:.4f}, precision {summary['pixel_operating_macro']['pixel_precision']:.4f}, F1 {summary['pixel_operating_macro']['pixel_f1']:.4f} and background FPR {summary['pixel_operating_macro']['pixel_fpr']:.2%}. High recall does not imply precise boundaries.
- **Fine defects can be missed.** `carpet/test/metal_contamination/000.png` has image margin {carpet.score_minus_threshold:.6f} and defect-pixel recall {carpet.defect_pixel_recall:.2%}. The fine `wood/test/scratch/001.png` has image margin {wood.score_minus_threshold:.6f} and defect-pixel recall {wood.defect_pixel_recall:.2%}; its thin ground-truth line lacks a corresponding continuous response. Capsule fine-crack/scratch examples also have responses on printed markings or background rather than only the target. The 64-grid head interpolates 32/16-grid backbone features and is smoothed afterward; those operations can limit detail, but no resolution counterfactual isolated the cause.
- **Normal structures and background can dominate.** The screw false-positive panel responds on the ordinary shaft and a corner; capsule normals respond around printed text and pill normals can have positive pixel overlays despite a negative image decision. Hazelnut and toothbrush normal panels include background hotspots. These are observed locations, not proof that illumination or any one augmentation caused the errors.
- **Image and pixel failures differ.** `transistor/test/bent_lead/003.png` is an image false negative at margin {lead.score_minus_threshold:.6f}, despite defect-pixel recall {lead.defect_pixel_recall:.2%}. The image aggregation/cutoff can reject a case with a localized pixel response. Conversely, an image true negative can still contain pixel false alarms. Neither threshold was changed.
- **Structural presence is not reliably captured.** All four methods miss the earlier `transistor/test/misplaced/002.png` image, which visibly lacks the expected component. SuperSimpleNet Lite's image margin is {missing.image_margin:.6f} and only {missing.defect_pixel_recall:.2%} of the large ground-truth region is alarmed. Hotspots appear on board patterns rather than covering the missing component. This demonstrates a failure on this structural example; no positional or learned-head counterfactual established one cause.
- **Screw remains difficult.** All four methods miss the matched manipulated-tip case. SuperSimpleNet Lite has image margin {screw.score_minus_threshold:.6f} and zero predicted defect-pixel recall there. Other screw examples are detected, including a scratch-head case close to the image threshold. The complete screw category metrics and per-defect table, rather than these selected examples alone, establish its weakness.

## Calibration evidence

Leather has 36 calibration normals; carpet has 42. Across seeds, leather test FPR ranges from {leather.test_fpr.min():.2%} to {leather.test_fpr.max():.2%}, and carpet from {carpet_stability.test_fpr.min():.2%} to {carpet_stability.test_fpr.max():.2%}. Both have higher mean test-normal scores than calibration-normal scores in every seed. Their category histograms and threshold intervals expose an empirical distribution shift. Finite calibration size alone is not an established explanation for these large false-positive rates. The IID continuous order-statistic intervals explicitly rely on assumptions that this shift can violate.

Image AUROC is 1.0 for leather in all seeds, yet normal false positives remain high at the frozen operating threshold. This directly separates ranking quality from calibration. Post-freeze brightness diagnostics on the predefined development subset show sensitivity without contaminating the original calibration distribution; they do not prove the cause of the observed test shift.

## What the complete comparison establishes

The image AUROC difference versus PatchCore is {pc.loc['image_auroc','point_difference']:.6f}, with paired conditional 95% interval [{pc.loc['image_auroc','low']:.6f}, {pc.loc['image_auroc','high']:.6f}]. The F1 difference is {pc.loc['f1','point_difference']:.6f}, interval [{pc.loc['f1','low']:.6f}, {pc.loc['f1','high']:.6f}]. All six paired image-metric intervals versus PatchCore span zero. This does not establish equivalence or superiority; it means these small point differences do not resolve an image-quality advantage under the stated conditional uncertainty analysis.

Localization point estimates are lower than PatchCore: pixel AP {macro['pixel_ap']['mean']:.6f} versus 0.540154023315294, and AU-PRO {macro['au_pro']['mean']:.6f} versus 0.9203990769959395. The baseline values are read from its exact stored summary in the comparison tables. Median measured model and end-to-end latency are lower, but mean end-to-end p95 is {summary['mean_end_to_end_p95_ms']:.3f}ms versus PatchCore's 37.72269044455329ms, so there is no uniform tail-latency improvement. Native evaluation took {summary['total_evaluation_seconds']/60:.2f} minutes. Fitting required {summary['total_fit_wall_seconds']/60:.2f} minutes and the feature cache is larger than PatchCore's; inference avoids its memory-bank search.

Relative to the scratch CNN and Autoencoder, ranking and recall improve substantially, but image FPR is higher. ImageNet priors differ, so the comparison does not isolate learned synthetic discrimination alone. The original paper also uses a larger backbone, longer training and a learned primary image head. This selected Lite/top1% variant is not paper-equivalent.

The research outcome is a useful tradeoff measurement: strong image discrimination and lower median inference cost, weaker localization than PatchCore, and an unresolved normal false-positive problem. The calibration objective was not demonstrably solved. No thresholds, weights or model choices were changed after seeing official test results, and no subsequent model family was started.
'''
# Keep all comparison constants sourced from the exact saved baseline records.
baseline=json.loads((ROOT/'reports/patchcore/summary.json').read_text())
text=text.replace('0.540154023315294',repr(baseline['macro']['pixel_ap']['mean'])).replace('0.9203990769959395',repr(baseline['macro']['au_pro']['mean']))
text=text.replace('37.72269044455329',repr(baseline['mean_end_to_end_p95_ms']))
review.write_text(text,encoding='utf-8')
full_path=local/'04_supersimplenet.full.executed.ipynb';before=local/'04_supersimplenet.full.before-matched-cases.executed.ipynb'
with before.open('xb') as stream:stream.write(full_path.read_bytes())
full=nbf.read(full_path,as_version=4);extra=nbf.read(local/'04_supersimplenet.matched_cases.executed.ipynb',as_version=4)
full.cells.extend(extra.cells)
interpretation=nbf.v4.new_markdown_cell(text);interpretation.metadata.update(execution_session='post-execution-visual-review',reviewed_at_utc=datetime.now(timezone.utc).isoformat())
full.cells.append(interpretation);nbf.validate(full);full_path.write_bytes(nbf.writes(full).encode())
for cell in full.cells:
    if cell.cell_type=='code' and cell.metadata.get('local_dataset_pixels'):cell.outputs=[]
public=ROOT/'notebooks/04_supersimplenet.ipynb';public.write_bytes(nbf.writes(full).encode())
readme=ROOT/'README.md';body=readme.read_text(encoding='utf-8')
marker='<!-- SUPERSIMPLENET_FINDINGS_END -->'
assert marker in body
body=body.replace(marker,'Image-quality differences versus PatchCore remain unresolved by the paired conditional intervals; localization is weaker. Median inference is faster, but end-to-end p95 is higher and texture false positives persist. Read the [visually checked panel analysis](reports/supersimplenet/observed_panel_analysis.md) and [matched frozen cases](reports/supersimplenet/matched_frozen_cases.csv).\n'+marker)
readme.write_text(body,encoding='utf-8')
manifest=out/'artifact_manifest.json';artifacts=json.loads(manifest.read_text())
for path in [review,out/'matched_frozen_cases.csv',local/'04_supersimplenet.matched_cases.executed.ipynb',full_path,public]:
    artifacts.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),bytes=path.stat().st_size))
manifest.write_text(json.dumps(artifacts,indent=2)+'\n',encoding='utf-8')
record(ROOT/'experiments',dict(experiment_id='04-observed-panel-review-v1',model_family='FrozenCrossModelReview',categories=list(json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())['categories']),
    hypothesis='Check observed failure locations and the quality/calibration/cost tradeoff against retained metrics and genuine panels',
    preprocessing='Visual inspection of all15 category panels and four matched prior-failure cases; no model inference',hyperparameters=dict(official_test_policy='unchanged'),seed=2026,
    environment=dict(source_sha256=sha256(Path(__file__))),training=dict(phase='post-evaluation interpretation only'),
    metrics=dict(review_sha256=sha256(review),matched_cases=16,category_panels_inspected=15,public_notebook_sha256=sha256(public)),calibration=dict(policy='All recorded thresholds unchanged'),
    paths=dict(review=review.relative_to(ROOT).as_posix()),interpretation='Observed locations and quantitative evidence support descriptive conclusions, not causal proof or new selection',decision='Publish balanced findings; no next model',status='completed'))
print('Appended actual third execution session and visually checked interpretation; preserved original full notebook and all benchmark records')
