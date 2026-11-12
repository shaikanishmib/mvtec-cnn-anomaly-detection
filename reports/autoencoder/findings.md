# Convolutional autoencoder: measured findings

The frozen shallow depth-3, width-16, latent-8 autoencoder used MSE, 256-square RGB,
20 fixed epochs, squared channel-mean residual maps and top-1% mean image scores.
All 15 categories completed seeds 2026/2027/2028. Calibration used only the
reserved normals. There was no post-test tuning or retraining.

| Metric | Autoencoder | Basic CNN | AE minus CNN |
|---|---:|---:|---:|
| image_auroc | 0.634623 | 0.712589 | -0.077966 |
| image_ap | 0.827916 | 0.858954 | -0.031038 |
| pixel_auroc | 0.690002 | 0.785832 | -0.095830 |
| pixel_ap | 0.110897 | 0.235382 | -0.124485 |
| au_pro | 0.449350 | 0.549807 | -0.100457 |
| recall | 0.235000 | 0.401760 | -0.166760 |
| fpr | 0.093944 | 0.098341 | -0.004397 |
| precision | 0.789620 | 0.873299 | -0.083679 |
| f1 | 0.314635 | 0.510593 | -0.195958 |

Image recall was 23.50% with test FPR 9.39%,
against a calibration image-FPR target of 5%. This is an empirical normal
quantile, not a guarantee under test distribution shift. Category/seed calibration
counts and threshold bootstrap intervals are retained in each evaluation record.
Two runs emitted no image alarms; primary macro precision follows the shared zero-division convention. Defined-run-only precision was 0.826347, reported separately.
The highest category image AUROC was hazelnut (0.9588);
the lowest was metal_nut (0.2388).

Texture image AUROC/recall: 0.6926 / 0.2865;
object image AUROC/recall: 0.6056 / 0.2092.
These are measured category summaries, not evidence that one architecture
mechanism alone causes the differences.

Mean per-run model-stage p50 was 2.871 ms versus CNN 3.139 ms;
end-to-end p50 was 13.272 ms versus 12.538 ms.
Both exclude file read/decode; AE model timing includes residual construction.
These are means of run percentiles, not pooled service latency.
Final training took 22.66 minutes including loading/validation.
Exact native evaluation took 58.70 minutes.
Peak allocated training VRAM was 0.607 GiB;
the model has 31,499 parameters and largest weight checkpoint 131,133 bytes.

## Failure evidence

Deterministically selected panels contain 15 false-negative examples; in 1,
mean residual inside the displayed defect mask was no larger than outside it.
The table and panel identify each image. Low inside/outside contrast, when
present, documents weak local separation. Other false negatives can have
higher local error but still fail image top-k aggregation or the normal threshold.
Selected false negatives had median top-1%-error/mask overlap 0.0518
and median overlap enrichment over mask-area fraction 2.222.
These display-resolution diagnostics distinguish dominant non-defect residuals
from errors at the actual defect; tiny mask area also limits achievable overlap.
0 selected false-negative masks disappeared under nearest-neighbor 256-square display resizing;
all primary localization metrics still used their original native masks.
Across the selected panels, median reconstructed/original neighbor-detail ratio was 0.434.
The lower neighbor-detail measure is consistent with blur in these selected reconstructions; it does not establish that blur alone causes missed defects.
The displayed normals had median residual on high-gradient pixels 0.006548
versus 0.001377 off those pixels.
These deliberately high-scoring normals expose normal-structure residuals;
they are not a random sample of normal reconstruction quality.

Directly inspected seed-2026 panels show the following concrete cases:

- `carpet/test/color/000.png`: the dark color defect is visibly retained in the reconstruction. Its mean in-mask residual is 0.002432 versus 0.010088 outside; normal weave errors dominate this false negative.
- `capsule/test/crack/000.png`: the small crack has weak residuals relative to normal printing and silhouette errors. Only 0.003049 of top-1%-error pixels overlap the displayed mask. Normal `good/002.png` is a false positive with visible printing/edge residuals.
- `metal_nut/test/good/009.png`: normal rims and surface detail produce a false-positive image score of 0.070304. The bent false negative `bent/000.png` retains its coarse altered contour and scores only 0.031789.
- `screw/test/scratch_head/003.png`: the image alarm is correct, but the overlay also marks normal threads; only 0.001524 of top-error pixels overlap the tiny displayed defect mask. Image detection success does not establish accurate localization.

Seed-2026 defect-size analysis (native mask area; descriptive, after freeze):

| Defect area | Images | Recall |
|---|---:|---:|
| at most 0.1% | 17 | 0.2353 |
| 0.1% to 1% | 482 | 0.2573 |
| above 1% | 759 | 0.2411 |

The smallest-group recall minus above-1%-group recall was -0.0058. Size groups mix categories and defect types; the table is descriptive rather than a matched resolution ablation.
256-square resizing and top-k dilution can suppress tiny defects in principle;
no higher-resolution counterfactual was trained after test to measure that effect.
See defect_type_size_summary.csv for scratch-specific counts and outcomes.

## Limits and retention

No pretrained features, skip connections, smoothing or combined loss were used.
The bounded development comparison selected MSE/squared residuals over the
four fixed object/texture categories. Proxy scores were modest and do not
guarantee transfer to real defects. Twenty epochs limit compute and do not
prove convergence for every category. Three seeds do not remove sampling
uncertainty. Small normal calibration partitions can yield unstable thresholds.
Pixel confidence intervals use fixed-bin whole-image bootstrap; point metrics
are exact at native resolution and quantization bias is separately reported.
Mean binned-minus-exact pixel AUROC bias was -0.008928,
with maximum absolute run difference 0.047090.
This is material: those intervals describe the binned approximation and must
not be treated as coverage intervals for the exact pixel AUROC point estimate.
Compare recall, FPR, pixel AP/AU-PRO and latency together before claiming an
improvement. This research benchmark does not establish production suitability.

The complete local notebook keeps original/reconstruction/error/overlay/mask
panels. Public outputs contain compact measured tables and aggregate plots.
Registry records are append-only, including all rejected variants. The pilot
executed live; later notebook cells analyze separately recorded jobs and
explicitly do not claim live training replay. No PatchCore work was started.
