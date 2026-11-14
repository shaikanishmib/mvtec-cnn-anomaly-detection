# EfficientAD fidelity audit for bounded development

Reference: Batzner et al., WACV 2024, author's arXiv v3 reviewed in the pilot
phase; Algorithms 1/2 and architecture Tables 6/8. This audit inspects existing
`src/anomaly_detection/efficientad.py`; the architecture is not replaced.

| Component | Current implementation | Assessment |
|---|---|---|
| Teacher PDN-S | Four Conv2d layers, 128/256/256/384, 4/4/3/4 kernels, average pools, paper padding | Matches padded Table 6; 384x64x64 on 256 RGB |
| Teacher training status | `requires_grad_(False)`, `train()` override calls teacher.eval(), feature extraction under no_grad | Frozen; focused tests assert no teacher gradients |
| Teacher normalization | Channel population moments across fitting images and spatial positions, float64 accumulation | Full fitting partition for development, unlike eight-image pilot; variance floor 1e-12 is a numerical safeguard |
| Student | Same padded PDN, 768 output channels; first/last 384 heads | Matches dual-head design |
| Hard loss | Mean of squared discrepancies at/above detached .999 quantile | Matches hard-feature mining; ties included |
| Global AE | Five stride-two convs, 8x8 collapse to 64x1x1; bilinear decoder, dropout .2, 384x64x64 output | Matches padded Table 8 |
| AE objectives | Teacher-to-AE MSE and AE-to-student-head MSE on same augmented input | Both trainable branches receive STAE gradients; no teacher gradient |
| Photometric augmentation | Random brightness, contrast or saturation U(.8,1.2) for development | Paper-style recipe, tensor RGB rather than quantized PIL. Initial pilot fixed brightness remains preserved |
| Anomaly maps | Local teacher/student MSE and global AE/student MSE, channel means, bilinear resize | Matches Algorithm 2; no arbitrary backbone substitution |
| Map scaling | Independent development-normal q90/q99.5; .1 scale; equal combination | Matches principle and padded-paper ordering; strict degenerate-quantile guard |
| Image scoring | Maximum combined map; local-only candidate explicitly studied | Combined-max matches reference; selecting local-only would be an explicitly reduced variant |
| Training penalty | Optional external student-output squared norm implemented, external=None in all executed fits | Missing ImageNet/unrelated-image regularization; no external dataset acquired |
| Budget/optimization | 3/9 shuffled full epochs, Adam 1e-4, wd 1e-5, no late LR decay | Material reduction from 70k steps and 66,500-step decay; not paper-equivalent training |
| Calibration | Separate frozen normal split, 95th percentile higher, strict greater | Project operating-point policy; independent of development map scaling |

Without the external-image penalty, hard mining still directs learning toward
poorly imitated normal features, but the student is not explicitly penalized on
unrelated images. It may generalize its teacher predictions to anomalies and
reduce anomaly contrast. The direction/magnitude on real defects is unknown;
no claim is made that hard mining substitutes for that regularization. Fixed
synthetic proxy edits can detect some lost sensitivity but cannot cover all
defects or logical anomalies. No causal penalty ablation is possible without
legitimate external inputs; none is fabricated.

Teacher asset: `nelson1425/EfficientAD`, an **unofficial** implementation,
revision `fcab5146f84ae17597044ad5ddf1656ccf805401`, `models/teacher_small.pth`,
10,779,715 bytes, SHA256
`b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337`.
Its repository is Apache-2.0, with the license retained under
docs/third_party/efficientad/LICENSE. A separate weight license/model card and
original distillation log are unavailable. Accompanying pretraining source states
ImageNet and torchvision WideResNet101_2 IMAGENET1K_V1, layer2/layer3 descriptors.
That states the intended distillation recipe; tensor contents alone cannot prove
the checkpoint's entire training history. No weights are committed or redistributed.

The correct name for ensuing results is **EfficientAD-S reduced-budget variant**.
This phase tests whether it is a credible bounded baseline, not whether it
reproduces published EfficientAD metrics or millisecond figures on different hardware.
