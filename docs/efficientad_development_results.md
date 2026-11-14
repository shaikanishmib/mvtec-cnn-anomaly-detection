# EfficientAD bounded development: measured results

Four predefined fresh fits; no official test pixels/masks or final model evaluations. All results refer to the EfficientAD-S reduced-budget variant. The pilot is unchanged.

| Category | Epochs | Steps | Train s | Hard probe before / after | AE probe before / after | Combined proxy AUROC | Local proxy AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 3 | 441 | 65.62 | 18.000 / 4.274 | 1.0038 / 0.4265 | 0.6745 | 0.8783 |
| bottle | 9 | 1323 | 185.60 | 18.000 / 2.399 | 1.0038 / 0.2300 | 0.9987 | 0.9987 |
| carpet | 3 | 588 | 65.87 | 21.077 / 8.211 | 1.0889 / 0.6127 | 0.9128 | 0.9792 |
| carpet | 9 | 1764 | 226.66 | 21.077 / 6.854 | 1.0889 / 0.6133 | 0.9466 | 0.9909 |

Selected **9 epochs, local_max**, macro fit-proxy AUROC 0.9948, eligible=True. Selection followed the declared .65 category floor and .02 shorter-budget/combined-map preference. Calibration was opened only after the selection file was locked. The autoencoder is still trained, but its map is excluded from the selected image score. Its carpet fitting-probe loss plateaus (.6127 at three epochs versus .6133 at nine); decreasing teacher/student loss alone cannot demonstrate a useful global branch.

The three-epoch bottle combined score is weak on block edits (AUROC .3359), despite decreasing losses. Nine-epoch bottle combined proxy AUROC rises to .9987. These are actual synthetic-edit observations, not real defect findings. Proxy AP uses 48 edited and 16 clean images (random ranking reference .75); all edits share fitting backgrounds. AE contribution is inference-only ablation, not a separate no-AE training experiment.

## Latency diagnosis

Unchanged pilot model, deterministic FP32, 50 warmups per repeat, 3 x 100 resident batch-one measurements: wall **41.99/54.11 ms p50/p95**, CUDA event **41.92/53.26 ms**. H2D **0.274/0.402 ms**. Decode-through-CPU-map/score **47.14/80.44 ms** (100). Both synchronization and event streams are retained in latency.json.

Sampled GPU P0/clocks were stable and CPU process utilization was near one core. Burst execution did not materially lower latency. CUDA events include gaps between launches and do not isolate pure kernel computation; OS scheduling is a possible explanation, not established causation. No clock/power settings changed.

Exploratory paired selected-checkpoint control (300 measurements each): eager wall **25.98/32.23 ms**; fixed-shape CUDA graph **19.26/19.63 ms**. Both time the full two-map forward, not an optimized local-only evaluator. Capture/transfer excluded; all output equality checks passed. This supports a dispatch/scheduling contribution; graph execution is not adopted in the frozen configuration or conservative estimate. Profiler CPU launch overhead supports this interpretation but profiling perturbs timings and does not establish a specific WDDM cause.

## Normal-only threshold feasibility

| Category | Calibration n | Higher 95% threshold | Nominal FPR | Observed calibration FPR | IID future FPR 95% interval | Bootstrap threshold 95% range |
|---|---:|---:|---:|---:|---|---|
| bottle | 31 | 0.506691 | .05 | 0.0000 | not applicable: tied scores | [0.506691, 0.506691] |
| carpet | 42 | 0.329155 | .05 | 0.0476 | [0.0150, 0.1616] | [0.210839, 0.522247] |

A serialization audit found decimal float32 CSV values compared as float64 could inflate tied-score exceedances. Original reports are retained; corrected reports promote exact float32 values before 17-digit export and use a round-trip reader. No weights, thresholds or selection changed. Bottle calibration has one unique score across 31 images; the max response floor is not evidence that all feature maps collapsed. Its zero-width bootstrap range does not imply certainty, and the continuous Beta interval is marked inapplicable. Exact-score normal maps and peak positions are diagnosed in the corrected report.


Policy stays higher quantile with strict score > threshold, per category and freshly fitted model. Calibration exceedance intervals are descriptive because the same normals fit the threshold. Order-statistic intervals assume IID continuous scores; image dependence or discrete ties weaken that model. Small tails, especially toothbrush n=9 in a later benchmark, cannot guarantee nominal 5% FPR. No configuration or policy was changed from these calibration observations.

## Compute accounting and final feasibility

Synchronized training total: **543.75 s**. Measured compute pipeline total including full fit diagnostics, latency controls, calibration, decode and synthetic native-metric probe: **871.20 s**. Training is CUDA-active elapsed wall time, not an exclusive GPU utilization/energy integral. Notebook/tests/documentation add CPU wall time outside this counter.

Peak process working set: **1.966 GiB** (OS cumulative process peak); peak CUDA allocated **314.30 MiB**, reserved **434.00 MiB** (training through diagnostics and reload). Checkpoint bytes: 32,253,487.

Requested coverage stays **15 categories x 3 seeds = 45 fresh final models**. Projected raw training **2.68 h**; conservative complete sequential workload **10.16 h**, against 12 h. The entire estimate receives a factor-two margin and includes CPU work, so it is deliberately stricter than GPU-only training time. It is not a measured final runtime.

| Category | Train/model s | Prepare/model s | Map scale/model s | Calibrate/model s | Infer/model s | Native metrics/model s | 3-seed total with margin h |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 185.6 | 11.7 | 62.7 | 2.5 | 6.7 | 20.8 | 0.584 |
| cable | 199.5 | 11.8 | 66.8 | 2.7 | 12.1 | 50.9 | 0.674 |
| capsule | 195.7 | 11.8 | 64.8 | 2.6 | 10.6 | 42.3 | 0.647 |
| carpet | 247.5 | 12.2 | 85.0 | 3.4 | 9.4 | 39.2 | 0.762 |
| grid | 234.8 | 12.1 | 78.9 | 3.2 | 6.3 | 25.6 | 0.703 |
| hazelnut | 347.2 | 13.0 | 117.4 | 4.8 | 8.8 | 36.7 | 0.981 |
| leather | 218.4 | 12.0 | 72.9 | 2.9 | 10.0 | 41.7 | 0.697 |
| metal_nut | 194.4 | 11.8 | 66.8 | 2.7 | 9.3 | 17.2 | 0.605 |
| pill | 236.1 | 12.1 | 80.9 | 3.3 | 13.4 | 33.9 | 0.734 |
| screw | 282.8 | 12.5 | 97.1 | 3.9 | 12.9 | 54.5 | 0.874 |
| tile | 204.5 | 11.9 | 68.8 | 2.8 | 9.4 | 25.8 | 0.640 |
| toothbrush | 53.0 | 10.7 | 18.2 | 0.7 | 3.4 | 13.3 | 0.267 |
| transistor | 190.6 | 11.8 | 62.7 | 2.5 | 8.0 | 33.2 | 0.616 |
| wood | 218.4 | 12.0 | 74.9 | 3.0 | 6.4 | 25.9 | 0.669 |
| zipper | 212.1 | 11.9 | 72.9 | 2.9 | 12.1 | 51.3 | 0.706 |

Add measured checkpoint serialization/reload and 60 s/model fixed overhead before applying the factor-two margin. The fixed allowance covers further latency sweeps, bootstrap/artifact handling and validation; actual final native-metric and sustained thermal cost remain uncertain. Native-metric cost was measured on synthetic truth/scores, then scaled by native pixel count and N log N; no official masks were loaded.

Recommendation: **GO** for the locked reduced protocol only. Frozen identity: `d728cf544ebc6f0c3d012adbca044a7aa5ddf4c0ced2e3210bef19b77a71d321`. This phase launches **zero** final models; no seed/category reductions are hidden. If actual resumed final costs exceed the target, stop and report before changing protocol.

## Limitations and remaining work

Teacher is the unofficial nelson1425/EfficientAD PDN-S at revision fcab5146f84ae17597044ad5ddf1656ccf805401, SHA256 b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337. Repository Apache-2.0; separate weight licensing and original training history remain unconfirmed. Full audit: docs/efficientad_fidelity_audit.md. No external-image penalty or paper-length optimization; one development seed and two categories. All final nine metrics, real-defect failure analysis and six-model comparison are unavailable. Notebook 07 is not started. Tests and validator evidence are recorded separately in the milestone status and development validation report.
