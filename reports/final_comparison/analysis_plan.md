# Milestone 9 analysis plan

Baseline: 815a14a81f30ccac4eb1b0678b3d1d2242da1ceb. Use the existing project
interpreter and saved results only. No training, inference, threshold refitting,
frozen-policy changes, raw-image publication or engineering implementation.

Audit all 6 x 15 x 3 final evaluations, not just summary means. Link original
result, prediction, calibration, checkpoint/map identities, frozen configurations,
registries and global partition hashes. Hash referenced local large artifacts
once where their original identities are available; missing measurements remain
missing. Reconcile primary image metrics from saved predictions using the shared
binary metric function, without silently repairing old recorded results.

Build 270 source-linked rows. Use equal category weighting after seed means.
Keep ranking metrics separate from the frozen calibrated operating point and
keep all models' original scoring/calibration conventions visible. Threshold
transfer uses saved normal-calibration scores where available, never new inference.

Statistics: pair 15 category means (three seeds within each category) for each
of 15 model pairs and nine metrics. Use 10,000 paired category-cluster bootstrap
draws, fixed seed 9009. Preserve within-category seed correspondence; model seeds
are different experimental factors (e.g. coreset/channel selection), not universal
independent replications. Use exact paired sign-randomization of the category
differences (2^15 signs) under its exchangeability assumption and Holm correction
across the entire 135-comparison family. Report effect differences, paired dz,
median and win/tie/loss counts with FPR direction reversed only for interpretation.
CI overlap is not a formal equivalence test. No independent-pixel significance
test and no claim of inference to all industrial categories.

Model choice: no weighted composite score. Shortlist the two strongest recorded
macro image-ranking models, inspect localization Pareto dominance among them,
then explicitly weigh recall/FPR, latency, storage, complexity, stability and
provenance as documented tradeoffs. Choose a research engineering prototype,
not a production-ready detector. Keep alternative localization/latency/minimal-fit
recommendations and commercial weight-license uncertainty explicit. Official test
comparison is post-benchmark research selection, not an unbiased new validation
of the selected engineering candidate.

Use existing genuine failure panels, prediction disagreements and defect-size
tables. Original licensed panels stay local; the Git notebook includes executed
panel replay code and source inventory, while aggregate figures are public.
No oracle threshold is needed or introduced. Freeze the analysis source identities
in the final provenance package; retain the original research files unchanged.
