# Bounded development protocol, declared before training

This phase continues verified pilot commit ee19ec85e914bd0b8cda8c6fb1c0c8768c1557a1.
It does not rerun the pilot, change its results, open official test images or start
the final benchmark. The machine-readable plan is configs/efficientad_development.json.

The previous proposed 12-fit plan is superseded by **four fresh fits**: bottle
(147 fitting normals) and carpet (196), each for **3 and 9 complete epochs**,
seed 2026, batch one. This is 441+1323+588+1764=4116 optimizer steps, with a
two-hour development wall-time cap. Every image occurs once per shuffled epoch.
Fresh initialization is reset to the same seed for a controlled budget comparison.
All epoch losses and timings are saved. No open-ended extension is permitted.

Teacher/student/AE capacity, FP32 and hard mining remain unchanged. Use full fitting
normalization and randomized paper-style photometric augmentation (tensor RGB
rather than quantized PIL). Constant Adam learning rate 1e-4 and weight decay
1e-5; no paper 66,500-step schedule in these short budgets. This is explicitly
**EfficientAD-S reduced-budget variant**, without the external-image penalty.

Budget selection must consider actual discrimination, not just decreasing loss.
Use 16 deterministically chosen fitting normals and three fixed proxy edits per
image: thin scratch, rectangular patch and color patch. These backgrounds were
trained on: proxy discrimination is optimistic and cannot estimate real-defect
performance. All clean development normals measure held-out score distributions,
normal reconstruction, quantile stability and branch behavior. Score candidates
are normalized combined-map maximum and normalized local-map maximum. Global-only,
raw-map and top-1%-mean scores are diagnostics, not additional tuning candidates.
AE contribution is an **inference ablation**, not a separately trained no-AE model.

Predeclared eligibility: finite values and gradients; lower fixed fitting-probe
hard and AE losses; proxy AUROC at least .65 for both categories. Rank by macro
proxy AUROC, selecting three epochs if within .02 of best eligible nine-epoch
candidate; prefer combined score if within .02 of local score at selected budget.
If none is eligible, report NO-GO and retain the best diagnostic setting. No
selection decision reads calibration or test scores. These rules establish only
minimum engineering/proxy credibility, never a promised real anomaly AUROC.

After selection, lock the score and threshold policy, then read clean reserved
calibration images only. Use project `calibrate_normal` at nominal 5% FPR,
`method="higher"`, strict score > threshold. Report empirical FPR, distribution,
order-statistic/bootstrap sensitivity, leave-one-out sensitivity, uncertainty and
cross-check on development normals. No threshold rule changes after looking at
calibration. Calibration findings may expose a limitation but cannot select a
different configuration.

Latency audit keeps the same model semantics and deterministic FP32. Batch-one
resident GPU input isolates forward execution from transfer and disk decode.
Three repeats of 100, 50 warmups each, synchronized wall and CUDA event timing,
plus contiguous bursts, separate transfer and end-to-end timings. CUDA events
measure elapsed GPU-stream time including launch gaps, not pure summed kernel
duration. Sample nvidia-smi utilization, clocks, power and P-state; Windows WDDM
and background scheduling are possible explanations, not asserted causes.
No power policy or clock settings are changed.

For final feasibility use all-category manifest counts, measured complete-epoch
seconds per image from both categories, full teacher-stat preparation, map scaling,
clean calibration, measured inference+decode, serialization and validation overhead.
Use the slower measured category rate and a factor-two safety margin plus at
least 60 s/model CPU metric/artifact overhead. Probe exact native-pixel metric
cost on normal-derived synthetic masks/scores to make evaluation overhead explicit.
Official test counts and native dimensions may be read from existing metadata,
but no test pixels, labels, masks or scores enter this phase. Preserve all 15
categories and three seeds. GO requires both technical eligibility and conservative
total <=12 hours. Final benchmark remains unlaunched even on GO in this phase.

Recovery: inspect reports/efficientad/development, experiments/06-dev-*.json and
docs/milestone_status.md. Completed fit results/checkpoints are reused after hash
verification. Start events without results require inspection; never overwrite
them. Checkpoints and result events are saved immediately for each completed fit.

Final evaluator compatibility: normalized EfficientAD maps may be negative.
The project's exact native-pixel ranking metrics accept signed scores; do not
reuse positive-distance-only histogram transforms without adapting and testing
their monotonicity. No signed score is clipped merely to satisfy an older model's
evaluation helper. Full native final evaluation remains a separate gated phase.
