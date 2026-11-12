# Notebook 02 preregistration — 2026-11-12

Preserve all Notebook 01 evidence and frozen partitions. No official autoencoder
test access before selection/freeze. Development categories: bottle, cable,
carpet, tile, for object/texture coverage without test-driven category selection.

Initial pilot: bottle, seed 2026, RGB 256-square PIL bilinear preprocessing,
float32, batch 16, Adam learning rate 0.001, three epochs, width 16, depth 3,
latent channels 8, L1. No augmentation, pretrained weights or skip connections.
Fit gradients use fitting normals only; development reconstruction is monitored.
Retain every epoch and the last checkpoint. Pilot establishes resources, not
final detection quality or convergence. Maximum pilot wall budget: 15 minutes.

Bounded development: initial model, depth 4, latent 2, latent 32, MSE loss,
one factor at a time on the four fixed categories. Set epoch budget using pilot
throughput before development. Combined loss, smoothing or increased resolution
require evidence and a new append-only hypothesis before fitting.

Scoring amendment to foundation protocol: compare channel-mean absolute residual
and channel-mean squared residual. Aggregate the highest ceil(0.01*H*W) map values
by their mean for image scores. A maximum can overreact to a single noisy pixel.
Select map on permitted development evidence before calibration/test. No test
normalization or tuning. Restore final continuous maps to native resolution.
Image calibration uses the existing higher empirical 95th normal quantile and
strict > alarm. Pixel calibration must be independently specified before final
evaluation. No all-category training until measured pilot workload is reported.

Final benchmark and exact stored CNN comparison remain pending. Three seeds
2026/2027/2028 are preferred subject to measured workload. Freeze full policy,
epoch count, source identities and calibration rules before test access. Retain
all starts, results and failures; never overwrite existing experiment IDs.

## Measured pilot and development budget

The completed bottle pilot used 147 fitting and 31 development normals (see
immutable result for authoritative counts), three epochs, 609,729,536 allocated
bytes and 901,775,360 reserved bytes. Training epochs took 2.624, 1.217 and 1.208
seconds. Development uses 20 fixed epochs for each of five variants across four
categories (20 fits); fixed last-epoch selection avoids a checkpoint search.
The inner 20% of fitting normals is excluded from gradient updates and supplies
fixed synthetic probes (same generator as Notebook 01). Official development
normals set a temporary q95 threshold for proxy comparisons only.

Selection: maximize the unweighted four-category mean of
(proxy image AUROC + proxy pixel AP)/2 across variant/map pairs. Break exact ties
using proxy recall, then fewer parameters. Report both map rules for every run.
Do not compare raw L1 and MSE loss magnitudes as an architecture ranking.
These proxies are not official defect evidence. Combined loss is not automatic;
retain a simple loss unless development shows a specific reason to add complexity.
Final epoch budget is 20 unless recorded convergence/resource evidence justifies
a new preregistered development phase before test access. Final confirmation
trains on the complete fitting partition, including former inner probe sources.
Pixel threshold: upper edge of the bin containing the pooled calibration
residual q99.5, 65,536 fixed [0,1] bins. This is separate from image q95.
Threshold uncertainty: 2,000 whole-calibration-image bootstrap draws. Test metric
uncertainty reuses shared whole-image bootstrap and native exact point metrics.
