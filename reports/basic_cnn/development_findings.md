# Basic CNN: completed controlled development

No official calibration or test inference was used to select this configuration.
Synthetic edits/proxies used only outer fitting normals, with disjoint inner
gradient/probe sources. Official development images remained normal and unmodified.

The bottle pilot completed eight epochs with 83,857 trainable parameters. Training
loss decreased 0.427 to 0.204; fixed-probe loss decreased 0.508 to 0.194. The
selected epoch 7 achieved synthetic-proxy image AUROC 0.932, pixel AP 0.701 and
recall 0.897 at the official-normal development q95 threshold. These values do
not estimate real-defect benchmark performance. Peak VRAM was 0.51 GiB.

All 20 predefined category/configuration fits completed, including the reused
pilot. Macro proxy recall/pixel AP: mixed BCE+Dice 0.849/0.716, mixed BCE
0.811/0.689, smaller mixed CNN 0.766/0.637, perturbation-only 0.582/0.498,
patch-only 0.314/0.210. Preserve rejected variants and per-epoch curves; patch-only
failure supports diversity within this proxy task, not a claim about real defects.

The frozen choice uses width 16, mixed synthetic edits, positive pixel weight 8,
Dice weight 1, resolution 256, batch 16 and AdamW lr .001 / weight decay .0001.
The selected configuration's median best epoch across the four development
categories is eight; final models use eight fixed epochs with all fitting normals.
There is no final category-specific early stopping or test-driven adaptation.

The complete evaluation pipeline was checked on tiny synthetic files before
freezing, including calibration, native maps, exact ranking metrics, bootstrap,
timing and immutable result reservation. Six exact metric tests match sklearn
and the foundation AU-PRO reference, including score ties across chunk boundaries.
The first integration fixture omitted source SHA256 metadata; the fixture was
corrected before any official calibration/test inference. No model-training run
failed, and none of the unsuccessful proxy configurations was deleted.

Final confirmation remains pending at this development milestone. The planned
45 fresh models cover all 15 categories and three seeds. Synthetic-to-real gap,
small-defect localization and normal-calibrated recall/FPR remain open hypotheses.

Subsequent frozen confirmation is now recorded in Notebook 01 and summary.json.
The development statements above describe the earlier, independently committed
milestone; they are not rewritten to match the later real-test outcomes.
