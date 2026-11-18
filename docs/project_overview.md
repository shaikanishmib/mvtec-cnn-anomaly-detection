# Project overview

Industrial inspection needs both an image-level warning and an explanation of
where a model responded. Defects are diverse, while fitting data often contains
only normal examples. This project compares six ways to learn or reuse visual
representations under that constraint, then carries one frozen method into an
inspection application.

## Research question and method

Eight notebooks study Basic CNN, a convolutional autoencoder, simplified
PatchCore, SuperSimpleNet Lite, PaDiM and an EfficientAD-S reduced-budget variant.
The dataset is the original 15-category MVTec AD, not MVTec AD 2. The completed
benchmark contains 270 category-seed evaluations: six methods, 15 categories and
three seeds. Fitting, development and reserved normal calibration partitions
remain separate from final official test evaluation. Methods are not assumed
to be paper-equivalent or identical in their training and scoring protocols.

The [final report](../reports/final_comparison/report.md) compares image ranking,
pixel localization and calibrated decisions separately. Macro results weight
categories equally. Category-paired uncertainty avoids treating pixels or repeated
seeds on the same test images as independent observations.

## What the experiments showed

Pretrained representation methods performed more strongly than the scratch
baselines in this benchmark. Simplified PatchCore reached image AUROC 0.9676,
pixel AP 0.5402 and AU-PRO 0.9204. SuperSimpleNet Lite was close in image ranking,
with a different latency/localization tradeoff. PaDiM localized well despite
weaker image detection. Reduced-budget EfficientAD had poorer image ranking and
threshold transfer; that finding does not characterize full-budget EfficientAD.
These are observations on MVTec, not proof of causal mechanisms or generalization
to other factories and cameras.

PatchCore was selected as the engineering research candidate for its balance of
detection, localization, fitting cost and traceability. The deployed seed is
always 2026, the lowest preregistered seed, independently of test performance.
The original 2,048-vector banks and normal-calibration thresholds are reused.
There is no deployment-time fitting or retrospective threshold optimization.

## From research to application

The frozen Python adapter feeds one guarded FastAPI worker. A React/TypeScript
dashboard discovers available models and categories, validates uploads, displays
raw score and cutoff, and compares original, heatmap and overlay. nginx serves
the compiled frontend and proxies the private backend. The CPU deployment is the
default; a separate verified CUDA variant uses the RTX 4060. Artifact checksums
are verified before serving. Only PatchCore is actually available in the app.

Milestone 13 verified all 15 category paths on CPU and CUDA, real object/texture
dashboard flows, cache bounds, concurrent rejection, invalid inputs, corruption,
visualization expiry, restart and crash recovery. Its
[measurements](../reports/deployment/results.md) separate tensor computation,
backend, HTTP and browser timing. They are retained evidence, not rerun or
replaced by this portfolio review.

## Limits that matter

PatchCore's category-macro test FPR was about 17.2% across the recorded research
seeds despite a nominal 5% normal-calibration target. Small calibration sets,
distribution changes, textures and score behavior limit threshold transfer.
An anomaly is not a verified manufacturing defect; misses and false alarms
remain possible. Heatmap colors are display-scaled, not probabilities.

The application is an unauthenticated local research prototype. Known dependency
findings, incomplete image-CVE scanning, a large CUDA image, licensing questions
and lack of prospective industrial validation prevent production-readiness
claims. The [fresh-clone guide](reproducibility.md) distinguishes source checks
from unavailable retained artifacts, and the [rights review](licensing_and_attribution.md)
distinguishes public source availability from reuse permission.
