"""Record inspected genuine panels; no inference, tuning or altered metrics."""
from pathlib import Path
import pandas as pd
import nbformat as n

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/efficientad'
OBSERVATIONS={
    'bottle':'Central contamination responds locally, with bright corner/rim responses; the image is missed at a score tied with its cutoff.',
    'cable':'The missing-cable interior produces a clear response and overlapping pixel alarms, yet its image maximum is below the cutoff.',
    'capsule':'A compact response aligns with the faulty imprint on the red half; it remains below the image cutoff.',
    'carpet':'The color defect receives a local response alongside off-target texture spots. The unselected AE map has strong border responses.',
    'grid':'A diagonal thread is visible in the local response; the alarm band is broader/more connected than the fragmented ground-truth mask.',
    'hazelnut':'Two cut regions respond, but the image maximum is only slightly below the calibrated cutoff.',
    'leather':'A genuine normal texture has a bright local hot spot and diffuse pixel alarms despite an empty mask: an actual false positive.',
    'metal_nut':'The flipped object has broad ring/boundary responses and partial object alarms, but still fails the image cutoff.',
    'pill':'The faulty-imprint area responds together with rim/off-target spots; the image remains below its cutoff.',
    'screw':'Neck scratches receive local pixel alarms while the image is missed; the unselected AE map also emphasizes the head/tip.',
    'tile':'The large glue-strip region responds broadly and overlaps the mask, but its maximum is below the image cutoff.',
    'toothbrush':'The bristle-top region responds, with partial coverage and separate off-target spots; the image is still missed.',
    'transistor':'Upper-case damage receives a local response with broader surrounding responses; the image maximum remains below its cutoff.',
    'wood':'Three liquid regions are visible in the local map and alarm overlay, yet the image maximum falls below its cutoff.',
    'zipper':'Squeezed upper teeth receive a local response alongside border responses; the image remains below its cutoff.'}


def main():
    examples=pd.read_csv(OUT/'failure_examples.csv',float_precision='round_trip')
    selected=examples.groupby('category',sort=False).first()
    lines=['### Post-evaluation visual review: actual observations','',
        'The all-category contact sheet and individual screw/leather panels were visually inspected after all 45 evaluations. The following observations refer to the first retained genuine case in each category; full-resolution panels and additional fine-defect zooms remain local. No model, score, border handling or threshold was changed. Heatmaps have separate signed-value colorbars, so brightness alone is not a cross-branch comparison.','',
        '| Category / seed | Genuine source | Outcome | Score / cutoff | Visual observation |',
        '|---|---|---|---|---|']
    for category,row in selected.iterrows():
        lines.append(f"| {category} / {row.seed} | `{row.path}` | {row.outcome} | {row.score:.6g} / {row.threshold:.6g} | {OBSERVATIONS[category]} |")
    lines.extend(['','Observed versus proposed explanation: localized responses below an image cutoff, normal-texture alarms, corner/border responses and spatial spread beyond masks are directly visible. Padding/receptive fields, insufficient optimization, missing external regularization or acquisition-domain differences are possible explanations; these panels do not causally separate them. A combined/global map looking different does not establish that an alternative score would improve the frozen benchmark. No alternative test-driven score was selected.','',
        'The frozen reduced variant met the compute target but is not a reliable universal operating-point detector here. The severe screw failure and carpet/leather false alarms remain central limitations even where pixel ranking or image AP is high. There is no claim of full-paper equivalence or deployment readiness.','',
        '### Paired uncertainty: selected exact-image contrasts','',
        'Differences below are EfficientAD minus the recorded baseline, using the same 2,000 whole-image draws and fixed models/thresholds. Lower FPR is favorable. These conditional intervals differ from the separately retained 15-category resampling intervals and do not imply unrestricted statistical superiority.','',
        '| Baseline | Metric | Difference | Conditional paired 95% interval |','|---|---|---:|---|'])
    pairs=pd.read_csv(OUT/'paired_image_difference_intervals.csv')
    for row in pairs[pairs.metric.isin(['image_auroc','fpr','f1'])].itertuples():
        lines.append(f'| {row.comparison} | {row.metric} | {row.point_difference:+.4f} | [{row.low:+.4f}, {row.high:+.4f}] |')
    lines.append('')
    text='\n'.join(lines)
    review=ROOT/'docs/efficientad_visual_review.md'
    assert not review.exists(),'Preserve existing reviewed observations'
    review.write_text(text,encoding='utf-8')
    report=ROOT/'docs/efficientad_final_results.md'
    with report.open('a',encoding='utf-8') as stream:stream.write('\n'+text)
    for path in (ROOT/'notebooks/06_efficientad.ipynb',ROOT/'outputs/efficientad/final/06_efficientad.full.executed.ipynb'):
        nb=n.read(path,4);assert nb.metadata.get('efficientad_final_replay',{}).get('models')==45
        assert not nb.metadata.get('efficientad_visual_review')
        nb.cells.append(n.v4.new_markdown_cell(text));nb.metadata['efficientad_visual_review']=True;n.write(nb,path)
    print('Appended genuine visual observations for 15 categories and paired contrasts; prior outputs unchanged')


if __name__=='__main__':main()
