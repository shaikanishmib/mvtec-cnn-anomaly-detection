### Post-evaluation visual review: actual observations

The all-category contact sheet and individual screw/leather panels were visually inspected after all 45 evaluations. The following observations refer to the first retained genuine case in each category; full-resolution panels and additional fine-defect zooms remain local. No model, score, border handling or threshold was changed. Heatmaps have separate signed-value colorbars, so brightness alone is not a cross-branch comparison.

| Category / seed | Genuine source | Outcome | Score / cutoff | Visual observation |
|---|---|---|---|---|
| bottle / 2027 | `bottle/test/contamination/000.png` | FN | 0.47813 / 0.47813 | Central contamination responds locally, with bright corner/rim responses; the image is missed at a score tied with its cutoff. |
| cable / 2026 | `cable/test/missing_cable/006.png` | FN | 0.520378 / 0.535737 | The missing-cable interior produces a clear response and overlapping pixel alarms, yet its image maximum is below the cutoff. |
| capsule / 2026 | `capsule/test/faulty_imprint/005.png` | FN | 0.305825 / 0.366305 | A compact response aligns with the faulty imprint on the red half; it remains below the image cutoff. |
| carpet / 2026 | `carpet/test/color/002.png` | FN | 0.315676 / 0.329155 | The color defect receives a local response alongside off-target texture spots. The unselected AE map has strong border responses. |
| grid / 2026 | `grid/test/thread/009.png` | FN | 0.660734 / 0.769681 | A diagonal thread is visible in the local response; the alarm band is broader/more connected than the fragmented ground-truth mask. |
| hazelnut / 2027 | `hazelnut/test/cut/001.png` | FN | 0.389068 / 0.389507 | Two cut regions respond, but the image maximum is only slightly below the calibrated cutoff. |
| leather / 2026 | `leather/test/good/030.png` | FP | 0.797814 / 0.303777 | A genuine normal texture has a bright local hot spot and diffuse pixel alarms despite an empty mask: an actual false positive. |
| metal_nut / 2027 | `metal_nut/test/flip/007.png` | FN | 0.303931 / 0.310598 | The flipped object has broad ring/boundary responses and partial object alarms, but still fails the image cutoff. |
| pill / 2028 | `pill/test/faulty_imprint/011.png` | FN | 0.410777 / 0.510062 | The faulty-imprint area responds together with rim/off-target spots; the image remains below its cutoff. |
| screw / 2028 | `screw/test/scratch_neck/020.png` | FN | 0.362458 / 0.399855 | Neck scratches receive local pixel alarms while the image is missed; the unselected AE map also emphasizes the head/tip. |
| tile / 2026 | `tile/test/glue_strip/003.png` | FN | 1.12063 / 1.27803 | The large glue-strip region responds broadly and overlaps the mask, but its maximum is below the image cutoff. |
| toothbrush / 2026 | `toothbrush/test/defective/014.png` | FN | 0.245262 / 0.261065 | The bristle-top region responds, with partial coverage and separate off-target spots; the image is still missed. |
| transistor / 2028 | `transistor/test/damaged_case/008.png` | FN | 0.276791 / 0.322679 | Upper-case damage receives a local response with broader surrounding responses; the image maximum remains below its cutoff. |
| wood / 2026 | `wood/test/liquid/002.png` | FN | 0.360663 / 0.373221 | Three liquid regions are visible in the local map and alarm overlay, yet the image maximum falls below its cutoff. |
| zipper / 2028 | `zipper/test/squeezed_teeth/001.png` | FN | 0.449607 / 0.463272 | Squeezed upper teeth receive a local response alongside border responses; the image remains below its cutoff. |

Observed versus proposed explanation: localized responses below an image cutoff, normal-texture alarms, corner/border responses and spatial spread beyond masks are directly visible. Padding/receptive fields, insufficient optimization, missing external regularization or acquisition-domain differences are possible explanations; these panels do not causally separate them. A combined/global map looking different does not establish that an alternative score would improve the frozen benchmark. No alternative test-driven score was selected.

The frozen reduced variant met the compute target but is not a reliable universal operating-point detector here. The severe screw failure and carpet/leather false alarms remain central limitations even where pixel ranking or image AP is high. There is no claim of full-paper equivalence or deployment readiness.

### Paired uncertainty: selected exact-image contrasts

Differences below are EfficientAD minus the recorded baseline, using the same 2,000 whole-image draws and fixed models/thresholds. Lower FPR is favorable. These conditional intervals differ from the separately retained 15-category resampling intervals and do not imply unrestricted statistical superiority.

| Baseline | Metric | Difference | Conditional paired 95% interval |
|---|---|---:|---|
| Basic CNN | image_auroc | +0.1037 | [+0.0778, +0.1277] |
| Basic CNN | fpr | +0.1217 | [+0.0942, +0.1522] |
| Basic CNN | f1 | +0.2295 | [+0.2065, +0.2522] |
| Convolutional Autoencoder | image_auroc | +0.1817 | [+0.1482, +0.2148] |
| Convolutional Autoencoder | fpr | +0.1261 | [+0.0965, +0.1567] |
| Convolutional Autoencoder | f1 | +0.4254 | [+0.3972, +0.4540] |
| PatchCore variant | image_auroc | -0.1513 | [-0.1705, -0.1341] |
| PatchCore variant | fpr | +0.0483 | [+0.0135, +0.0852] |
| PatchCore variant | f1 | -0.1643 | [-0.1853, -0.1457] |
| SuperSimpleNet Lite | image_auroc | -0.1485 | [-0.1673, -0.1300] |
| SuperSimpleNet Lite | fpr | +0.0538 | [+0.0207, +0.0867] |
| SuperSimpleNet Lite | f1 | -0.1698 | [-0.1903, -0.1528] |
| PaDiM | image_auroc | -0.0816 | [-0.1046, -0.0598] |
| PaDiM | fpr | +0.0756 | [+0.0374, +0.1145] |
| PaDiM | f1 | -0.0153 | [-0.0376, +0.0067] |
