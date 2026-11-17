# Research milestone progress

## Milestone 11 - FastAPI backend complete

Final verified state: eight HTTP endpoints, unchanged M10 PatchCore adapter,
all 15 categories, CPU/CUDA and fixed seed 2026. Lazy two-model LRU; one guarded
inspection; memory-only bounded uploads; PNG cache 8 items/32 MiB/300 s.
Thirty normal-image category/device API/direct checks and four object/texture
defect checks passed with zero score error and matching thresholds/decisions;
native heatmap/overlay arrays matched for all 30 normal-image checks.

Complete suite: 203 passed, zero failures/errors/skips. A final OpenAPI-only error
schema clarification was followed by 25 passing contract/security tests. The
pinned Starlette/httpx test-client deprecation warning is documented. Backend
completion validation passed. All 3,086 protected files, 15 selected bank and
calibration identities, and eight notebooks/205 saved executed code cells remain
unchanged and validated. No models were trained or historical benchmarks rerun.

Isolated real loopback HTTP p50/p95 (ms): CPU bottle 288.00/299.18, carpet
360.61/375.84; CUDA bottle 144.30/147.16, carpet 228.57/247.17. Cold category
requests span 1.01-3.46 s; process startup-to-live 21.86/22.72 s. Stage timings,
model loading, optional PNG cost and actual server memory are retained separately
in reports/backend/performance.json and runtime_summary.csv. Loopback workers
were stopped; no frontend was started. Earlier validation/cleanup failures are
retained in recovery_history.json, and completed measurement reports are reused.

Publication evidence: validation.json, completion_validation.json, test_summary.json,
OpenAPI snapshot, genuine example response, performance and parity tables, and
the publication seal. Finish -> verify -> commit -> push; verify current SHA with
`git log -1 --format=%H` and `git ls-remote origin refs/heads/main`. The seal avoids
a self-referential commit hash. Recovery must inspect the seal/Git state before
repeating tests or measurement scripts. Next milestone: interactive inspection
frontend, awaiting a separate instruction. The active-phase notes below are history.

Baseline 3a4eac2237ca3e1a4c35b9c18974ba81d663c21e; initially clean. The M10
adapter, manifest, banks, thresholds, seed and research files are unchanged.
Backend-only dependencies are pinned in requirements-backend.txt; pip check
passes and CUDA PyTorch remains 2.11.0+cu128. No frontend or model fitting.

Implemented health/discovery/model-info/predict and TTL heatmap/overlay retrieval.
Single admission gate, guarded model worker, bounded LRU model cache and bounded
in-memory visualizations. Upload byte/pixel/dimension, MIME/extension/decoded
format, multipart field and timeout checks are in place. No raw image persistence.
Focused HTTP tests plus real CPU/CUDA integration tests: 41 passed. Initial six
error-schema checks exposed duplicate X-Request-ID headers; fixed and rerun.
Real uvicorn loopback launch on port 8000 passed and smoke process was stopped.

Current commands/evidence: scripts/validate_backend_real.py is completing the
bounded 15-category CPU/CUDA parity and timing report. Inspect
reports/backend/parity.csv, integration.json, server_smoke.json and existing
processes before rerunning. The completed integration report is exclusive-write.
Remaining: final timing analysis/docs, full suite, artifact/notebook/protection
validators, staged-content audit and finish -> verify -> commit -> push.
Recovery uses reports/backend/research_protection.json to protect all prior
tracked files except rolling README/progress/checkout metadata. Do not rerun
earlier training/evaluation scripts or modify the inference package.

## Milestone 10 - PatchCore engineering foundation complete

Completion evidence: 161 tests passed, zero failures/errors/skips; all 28 new
engineering tests plus all 133 research tests. Real-image validation passed all
15 categories on CPU/CUDA (30 images, 60 predictions, 60 identical repeats).
Every sampled reference image decision matched. Max score/reference-map errors
were .000194788/.00220728; max CPU/CUDA native-map error .00000464917.
The earlier overly strict TF32-reference check and invalid-device error were
corrected/disclosed in retained engineering history; frozen models were untouched.

Validator passed: 15 selected banks, all 45 original banks available, pinned
extractor/calibration/config identities, 3,065 prior research files unchanged.
CLI smoke passed with JSON/raw-map/heatmap/overlay outputs kept local. Runtime:
bottle CUDA model wall p50/p95 20.63/23.55 ms, decoded E2E 34.67/40.41 ms;
20 warmups and 100 measurements, FP32, four CPU threads. Full measurements and
memory are in `reports/engineering/patchcore/integration.json` and runtime CSV.

Package/documentation and deployment v1 are complete. Publication follows seal,
staged audit, commit, push, and `git ls-remote origin refs/heads/main` verification.
Current completion SHA is obtained from Git, avoiding self-reference inside its
own seal. Check the publication manifest before repeating any validation.
Next milestone: FastAPI backend; no backend or frontend has been started.
The active-phase notes below are historical and superseded by this completion.

Core package, local CLI and versioned deployment manifest implemented. All 45
original banks verified; seed 2026 selected uniformly before test inspection.
Manifest SHA256: 58be2eafb16ac012ac18e0feb69f11b2f510e7417464eaba5bd9797b423272fa.
Real-image validation complete: 15 categories, 30 images, 60 CPU/CUDA predictions
and 60 repeats. All sampled original image decisions match; max score error
.000194788, max historical model-map error .00220728, max CPU/CUDA native-map
error .00000464917. Timing: 20 warmups/100 measurements, bottle/carpet, CPU/CUDA.
Historical TF32/cache rounding discrepancy and corrected invalid CUDA-index error
are retained in `reports/engineering/patchcore/recovery_history.json`.

Completed: explicit project interpreter, CUDA/RTX4060 verification, exclusive
deployment registration, core implementation and real-image measurement command.
Remaining: full suite, saved-evidence/historical validator, CLI smoke, final docs
and publication audit, commit/push and remote verification.
Recovery: read engineering integration/parity/validation records and manifest
before any command; do not rerun completed real-image timing or research fitting.
`scripts/validate_patchcore_engineering.py` validates saved outputs and old file
identities; it does not rerun inference. FastAPI and frontend remain unstarted.

Baseline c8fd9fb6bf6ef9e0bfa5b63c9060350412df70ea; initially clean. Audited
frozen PatchCore source, preprocessing, 45 original banks and calibration records.
Deployment policy: use seed 2026 for every category (lowest preregistered seed;
no selection from official test metrics). No fitting/calibration policy changes.
Core package is under `src/anomaly_detection/inference/`; versioned deployment
manifest `configs/patchcore_deployment_v1.json` registers existing artifacts.
Progress/evidence: `reports/engineering/patchcore/`. Recovery: inspect manifest,
artifact availability, integration outputs and Git status before repeat execution.
Do not rerun completed research fits or start FastAPI/frontend.

## Milestone 9 — final research comparison complete

Final verified research state (2026-11-17): 270 source-linked evaluations,
135 category-paired contrasts, ten figures, professional report and 20/20 executed
Notebook 07 code cells. Fresh complete suite: 133 passed, 0 failed, 0 errors,
0 skipped (152.08 s). Comparison validator passed: all original metrics and
derived tables/report agree; 3,091 original source identities and 3,017 protected
historical files preserved. Three additional teacher/shared-trunk assets verified.
Public notebook excludes licensed panel outputs; complete replay remains local.

Publication evidence: `reports/final_comparison/test_summary.json`, `validation.json`
and `publication_manifest.json`. Use `scripts/seal_model_comparison.py --check`
to verify compact publication bytes without rerunning expensive source checks.
Git publication follows finish -> verify -> commit -> push. Obtain the completion
commit from `git log -1 --format=%H` and verify it against
`git ls-remote origin refs/heads/main`; a self-referential commit SHA is not stored
inside its own manifest. Stage only the compact comparison, useful aggregate
plots, notebook, source/tests, README, this progress log and checkout attributes.

No scientific tasks remain incomplete. No training, inference, threshold tuning,
new model registry events or engineering work were performed in this milestone.
Next phase is Engineering and requires a separate user instruction.

Recovery audit (2026-11-17): existing analyses were reused, not restarted. HEAD
was still 815a14a81f30ccac4eb1b0678b3d1d2242da1ceb; no Milestone 9 commit or
running benchmark remained. The saved full-suite log has 133 passed, zero failed
and zero skipped. Notebook 07 replay passed again (20/20 cells). Additional
pinned teacher/shared-backbone identities passed. See
`reports/final_comparison/recovery_audit.md` for the task-by-task evidence.

The original recovery/progress entries below are history, superseded by the
verified completion record above. On interruption, inspect Git status and remote
identity before any repeat command; preserve the sealed comparison and model files.

Progress: all 270 final records reconciled exactly against saved image predictions
and original source metrics. 3,091 source identities (~20.37 GB including local
artifacts) audited; recovery reuses SHA256 entries only for unchanged size/mtime.
Canonical tables, 135 category-paired contrasts, five category heatmaps, calibration,
compute and seed figures, written report and Notebook 07 are built. Notebook replay
executed 20/20 cells; local licensed panels are cleared in the public copy.
No training, inference, threshold fitting or engineering performed.

Completed commands:
- `.\.venv\Scripts\python.exe scripts/build_model_comparison.py`
- `.\.venv\Scripts\python.exe scripts/publish_model_comparison.py`
- `.\.venv\Scripts\python.exe scripts/execute_model_comparison.py`

Remaining: complete full pytest, comparison/historical validators, publication
manifest and staged-content audit, commit/push and remote SHA verification.
Recovery: inspect `reports/final_comparison/audit_summary.json`, source provenance,
`outputs/final_comparison/hash_cache.json`, notebook execution and Git status before
rerunning anything. Do not rerun prior model scripts. Rebuild compact analysis only
if its inputs/code change; the builder preserves historical records.

Baseline 815a14a81f30ccac4eb1b0678b3d1d2242da1ceb verified; initially clean.
Current phase: source audit and canonical 270-row result construction. Read
reports/final_comparison/analysis_plan.md for statistical/selection rules.
No training, inference, threshold fitting or engineering work authorized here.
All existing experimental records, datasets, splits, policies and checkpoints
are read-only. Notebook 07 will execute saved-artifact comparison replay only.
Recovery: inspect reports/final_comparison, its provenance/validation records,
and this file before resuming. Existing Milestone 8 completion below is preserved.

## Frozen final benchmark — active

Baseline 664e106dbdd25104ea4e3c3be192fc8be7a0b772 verified, initially clean.
Current phase: Milestone 8 research execution and validation complete. Publication
package is being sealed and audited for the completion commit/push. Do not rerun
completed fits. The completion record below supersedes historical pending actions.
Readiness checks passed: exact project .venv, RTX 4060 / torch 2.11.0+cu128,
15 categories, all frozen source/teacher/partition hashes, ~502 GiB free disk,
no competing runner and no preexisting final run artifacts. Ten focused tests passed.
Frozen identity d728cf544ebc6f0c3d012adbca044a7aa5ddf4c0ced2e3210bef19b77a71d321.
See docs/efficientad_final_protocol.md for pre-test execution/evaluation rules.
No pilot/development reruns. No Notebook 07 or personal-project access.

Recovery: inspect reports/efficientad/final/benchmark_manifest.json and
experiments/06-final-*.json, then use project Python scripts/run_efficientad_final.py
--all. Per-epoch snapshots permit interrupted fitting recovery. Verified complete
category-seed attempts are skipped; a result is complete only after evaluation
and artifact verification. Final progress will be updated by the manager below.

Safe restart/status wrapper: project Python scripts/resume_efficientad_final.py
--status, then scripts/resume_efficientad_final.py only if no manager is live.
See docs/efficientad_final_recovery.md. On the user's resume message the original
manager remained active; no second manager or worker was launched and no completed
run was repeated. Continue monitoring execution session and saved manifest.

## Bounded development continuation

Baseline verified: ee19ec85e914bd0b8cda8c6fb1c0c8768c1557a1, initially clean.
Current phase: bounded development complete, GO for the locked reduced-budget
research protocol; no final benchmark started. The following declaration was saved
before training. See configs/efficientad_development.json
and docs/efficientad_development.md. Authorized phase: latency diagnosis, four
fresh fits (bottle/carpet x 3/9 epochs, seed 2026; 4116 total steps), selection,
then reserved-normal calibration feasibility and conservative 45-model cost gate.
No official test image/mask access and no final benchmark. Two-hour development
wall cap. Preserve successful pilot and prior milestone artifacts.

Recovery command (after inspecting saved events/results):
`.\.venv\Scripts\python.exe scripts/develop_efficientad.py`.
This command will skip verified completed fits; incomplete starts stop for review.
Initial remaining actions (completed below): phase-specific tests and validators,
notebook update and publication commit/push.

Latency audit saved: reports/efficientad/development/latency.json, registry
06-latency-audit-v1 start/result. 300 resident-input measurements: synchronized
wall p50/p95 41.986/54.112 ms, CUDA events 41.916/53.264 ms. Transfer .274/.402 ms;
decode-through-CPU-map 47.137/80.442 ms (100). P0 and ~2565 MHz observed; CPU
~95% of one core, sampled GPU ~37-46%. Causes remain hypotheses, not proven.

First completed development fit: 06-dev-bottle-e3-seed2026-v1, 441 steps,
65.621 s training (three complete 147-image epochs). Checkpoint and registry
result saved immediately after diagnostics. Four-image fitting-probe hard loss
17.9997 to 4.2737; AE 1.00377 to .42652. Nine-epoch bottle fit is running.
No selected configuration yet and no calibration pixels read at this point.

Second completed fit: 06-dev-bottle-e9-seed2026-v1. All nine full epochs saved,
1,323 steps; teacher statistics reused after identity verification. Carpet
three-epoch fit is running. No fits are rerun. Deferred all testing and further
GPU timing until training completes to avoid contaminating compute measurements.

Third completed fit: 06-dev-carpet-e3-seed2026-v1 (588 steps). Nine-epoch carpet
is the final running fit; no budget extension planned. After selection, one
exploratory execution-only CUDA graph control will compare identical fixed-shape
outputs to eager inference to investigate host dispatch. It is not used for
selection or to reduce the conservative eager-based final workload estimate.

All four fits complete: 4,116 steps, 543.749 s synchronized training, 788.212 s
fit pipelines. Nine-epoch carpet training 226.664 s. Predeclared choice: nine
epochs, normalized local_max (both AE branches still trained; combined score
rejected by .02 preference rule). Macro fit-proxy AUROC .99479, not real-defect
performance. Compute gate GO: 10.159 h conservative sequential 45-model estimate,
including CPU work and factor-two margin; raw projected training 2.684 h.
Official benchmark models remain 0/45.

Calibration audit found decimal float32 CSV values read as float64 inflated bottle
development exceedances. Corrected without training or policy changes using exact
float32-to-float64 promotion before 17-digit export and round_trip reading.
Original reports retained; authoritative new reports are calibration-*-corrected.json.
Bottle development exceedance corrected from 1.0 to 1/31; calibration stays 0/31.
All bottle calibration scores tie: bootstrap degeneracy does not establish
certainty and continuous-IID Beta FPR interval is marked inapplicable.
Carpet calibration 2/42 (.04762), threshold .329154849; bottle .506690860.
No configuration/threshold selection was revisited after calibration.

Exploratory paired launch control completed on selected bottle checkpoint:
300 eager wall p50/p95 25.976/32.233 ms; 300 fixed-shape graph 19.256/19.626 ms.
Output equality was bitwise on capture input and zero error on three further
normals. Profiler shows substantial cudaLaunchKernel CPU cost; no OS-specific
causation asserted. Eager cost remains the frozen execution policy.

Development phase complete. Frozen reduced protocol:
configs/efficientad_reduced_frozen.json SHA256
d728cf544ebc6f0c3d012adbca044a7aa5ddf4c0ced2e3210bef19b77a71d321.
Both trainable branches remain trained, but local-only image scoring was selected;
this is not full EfficientAD paper equivalence. Final GO is for a research
benchmark only; score-floor/proxy/penalty limitations are explicit.

Actual measured compute pipelines: 871.197 s total, including 543.749 s training,
all full fit diagnostics, 32.292 s latency audit, 17.705 s launch control, 9.825 s
precision audit, calibration/data loading and synthetic metric-cost probe.
Not an exclusive GPU-busy integral; documentation/tests add CPU wall time.
Peak process working set 1.966 GiB; CUDA allocated 314.30 MiB, reserved 434 MiB.
Four checkpoints retained locally (~32.25 MB each), raw images/weights excluded.

Validation: full suite 114 passed in 50.55 s. Deep development validator passed:
4 fits, 4116 steps, 16 fit artifact checks, 37 manifest artifacts, 21 registry
events, 60 notebook cells / 14 executed replay code cells. Original pilot cell
sources, execution counts and outputs verified identical to pilot commit.
The validator initially loaded teacher reference tensors on their original CUDA
device while checking CPU checkpoints; fixed map_location=cpu and reran validation
without any feature extraction or training. Notebook extension executes only new
replay cells; no successful pilot/fit rerun. Calibration precision regression test
and signed-score metric compatibility passed.

Recovery: reuse completed JSON/CSV/checkpoints, do not run pilot or pilot-only
notebook builder. During an interrupted pre-freeze study, the main development
driver verifies and skips completed fits. Once frozen, use the validator rather
than rerunning the driver or overwriting its timing/feasibility reports.
Authoritative normal calibration: *-corrected.json and *-exact-scores.csv;
original diagnosis reports remain as superseded history. Use
`.\.venv\Scripts\python.exe scripts/validate_efficientad_development.py` for deep
validation. No final model or Notebook 07 started. Git staged audit/commit/push
is the remaining publication action for this phase.

Pilot-stage validator also passed read-only after notebook extension, without
rewriting reports/efficientad/validation.json. Notebook is 433,828 bytes and
contains only aggregate plots/tables; all original pilot outputs are preserved.
New frozen configuration has an explicit CRLF Git attribute so its byte identity
survives Windows/Linux checkouts, matching existing frozen-record conventions.

Staged publication audit passed: 74 text/notebook files, largest 433,828 bytes;
no weights, raw pixels, caches, temporary files or earlier milestone result changes.
`git diff --cached --check` passed. Git checkout-filter verification reproduced
38 recorded SHA256 identities (reports, source and frozen configuration), guarding
against CRLF/LF identity changes on clone. Publication identity is available with
`git log -1 --format=%H`; verify origin/main with `git ls-remote origin refs/heads/main`.
The final phase report records the published SHA and remote verification result.

## Preserved pilot-stage record

Scope of this execution: reference/asset verification, core implementation and one real-data CUDA pilot. Full development and benchmark remain gated; Notebook 07 is out of scope.

Baseline HEAD: 632f26eb83a3c7092fe25c54eba51853c98e310c; initially clean.
Environment: project .venv only; torch 2.11.0+cu128, CUDA 12.8, RTX 4060 Laptop confirmed.
Existing partitions, append-only tracking and metric utilities inspected. No previous assets modified.
CVF HTML/PDF returned HTTP 403; read author's arXiv v3 full text, including Algorithms 1–2 and architecture Tables 6–8.
Teacher candidate: unofficial nelson1425/EfficientAD, revision fcab5146f84ae17597044ad5ddf1656ccf805401, models/teacher_small.pth (10,779,715 bytes). Repository Apache-2.0; checkpoint has no separate model card or independently verified distillation log. Do not claim author-provided weights.
Initial GitHub socket access denied by sandbox; escalated metadata query succeeded.

Current phase: pilot passed; focused tests and notebook/artifact validation. External ImageNet training images unavailable; penalty implementation supported but pilot explicitly omits penalty. No dataset downloads.

Predeclared pilot: bottle, seed 2026, first eight sorted fitting normals, 32 optimizer steps, batch one, 256 pixels. Teacher statistics use these eight only; this is a learning/engineering gate, not a performance experiment. Check finite losses, gradients, branch changes and fixed-image loss reduction; save/reload; 20 warmups/100 timed model forwards.

Proposed development budget (not authorized to run until pilot reviewed): bottle and carpet; at most 12 category/configuration fits, 2,000 steps each, seed 2026. Baseline, mean loss, no AE, raw maps, alternative image aggregation, and one capacity/budget control. Penalty contrast only with documented legitimate external inputs, replacing a trial rather than extending budget. Select using normal development and fit-derived synthetic probes. Freeze before calibration/test. Final 45-run cost must be projected from measured pilot; no final runs started.

Recovery: inspect this file, experiments/06-*.json and reports/efficientad before running `.\.venv\Scripts\python.exe scripts/pilot_efficientad.py`. Never overwrite existing events or retrain completed runs.

Remaining: reviewable commit/push, then pilot/budget review before bounded development. Milestone remains incomplete until later bounded development and frozen benchmark.

Pilot completed: `06-pilot-bottle-seed2026-v1`, 32 steps, 5.017 s training; hard loss 17.1074→9.3763, AE .989384→.956559. Finite gradients and branch updates passed; 384×64×64 features; 256×256 maps; exact save/load inference. Teacher SHA256 b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337. Checkpoint SHA256 4584f177c3f5ed5a1dc648e5cdaa05c455e4f6216420850fbfa94069ccdb375f (32,253,242 bytes). Evidence: reports/efficientad/pilot.json and two append-only experiment events.

Measured model latency p50/p95 33.472/145.272 ms (20 warmups/100 measurements, deterministic FP32); tail needs investigation. Peak CUDA allocated 318,654,464 bytes; reserved 387,973,120; process peak working set 1,596,932,096 bytes. Linear training projection: 137.19 hours for 45×70k steps, excluding loading/evaluation/thermal effects. This does not validate sustained cost or authorize a final benchmark.

Focused core tests: 3 passed in 17.50 s. Notebook replay executed successfully, but IPython attempted its default read-only history path and ZMQ printed a shutdown socket warning; builder corrected to use project-local ignored IPython/Jupyter directories. No training rerun. Final benchmark count 0/45, development fits 0; nine final macro metrics, comparisons, calibrated thresholds and category failure observations remain unavailable.

Final verification: full suite ` .\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp` passed 107 tests in 62.32 s. `scripts/validate_efficientad.py` passed deep teacher/checkpoint/registry/input identities and Notebook 06 validation (20 sections, five executed replay cells). Staged audit: 15 text/notebook files, no dataset pixels, weights, caches or temporary files. `git diff --cached --check` passed. Previous milestone files/results preserved except README roadmap status. Git staging required escalation because .git is read-only in sandbox. Recovery: do not rerun pilot; replay saved report and run validator. Model-only latency measured; end-to-end latency, final evaluation time, calibration and failure examples await later phases.

<!-- efficientad-final-live -->

## Live final benchmark progress

UTC 2026-11-14T06:35:24.662066+00:00: 45/45 verified complete. All 45 run artifacts verified. Aggregation, failure analysis, notebook, final tests and publication remain.

Manifest: reports/efficientad/final/benchmark_manifest.json. Recovery: project Python scripts/run_efficientad_final.py --all. Complete outputs are hashed before skipping; each epoch saves model/optimizer/RNG state.

## Milestone 8 final completion record

Execution complete: 1 pilot, 4 bounded development fits, 45 fresh final models
(15 categories x seeds 2026/2027/2028), 68,877 optimizer steps, nine full epochs
per final model. All final attempts a01 completed without execution failure or
training recovery. The user's resume reused the existing live manager; no
completed run was repeated. No post-test model/threshold changes and no seed or
category omissions. Notebook 07 and the engineering application remain unstarted.

Final macro metrics: image AUROC .816297536, image AP .918592537, pixel AUROC
.919887356, pixel AP .498475624, AU-PRO .818373389, recall .665639183, FPR
.220080859, precision .885938659, F1 .740084396. Exact native primary pixel
metrics; 2,000 shared whole-image bootstrap draws, approximate pixel intervals
with reported binning bias; paired category and exact-image comparisons retained.
All five previous models use their exact stored results, unmodified.

Actual training 8,575.551 s (2.382 h), calibration 110.507 s, evaluation 2,814.064 s
(46.901 min), final checkpoint serialization/replay 9.244 s. Complete run
pipelines 12,045.302 s (3.346 h), within 12 h; historical projections 2.68/10.16 h
were not substituted for measurements. No resource stop. Mean per-run model
p50/p95 23.066/27.601 ms, decoded-image E2E 31.663/36.494 ms, separate warm-cache
file-decode-inclusive 61.160/66.308 ms. Full two-map eager forward timed; local
map alone is selected for primary scoring. Training CUDA allocated/reserved peaks
.308905/.458984 GiB; inference .107182/.189453 GiB; sampled worker RSS peak
5.803856 GiB (native evaluation included). Maximum checkpoint 32,249,772 bytes.

Calibration: frozen higher .95, strict greater, reserved normal sample sizes
9–58; mean empirical calibration FPR .027368 versus test macro .220081. Six
calibration sets have ties. Bottle response floor is retained, not cropped;
continuous-score Beta intervals are inapplicable for tied samples. Screw mean
image AUROC .057184/FPR .910569; carpet FPR .583333; leather FPR .802083 despite
image AUROC .998755. Tile/wood rank well. There are 1,035 image FN seed-evaluations
with at least one overlapping pixel alarm, not 1,035 independent new images.

Genuine failure analysis: 67 case panels across all 15 categories plus small-defect
GT-bounding-box zooms, original images, local/global/combined maps and masks.
All-category contact sheet and screw/leather detail were visually inspected;
actual observations and hypotheses are separated in docs/efficientad_visual_review.md.
Local full executed image-output notebook is under outputs/efficientad/final/.
Git notebook retains all aggregate plots/numerics and executed panel code, clearing
only licensed dataset-derived image outputs. Pilot/development cells and outputs
were verified unchanged.

Validation complete: full suite 121 passed, 0 failed/errors/skipped (44.36 s;
JUnit reports/efficientad/final/pytest.xml). Deep validator passed in 22.81 s:
45 checkpoints, 495 run artifacts, 15 teacher-normalization identities, 111
registry records, 2,655 protected prior files byte-identical (zero line-ending
fallbacks required), Notebook 06 95 cells / 25 executed code cells. No duplicate
long-running native metric sweep or feature extraction was used for validation.
Published notebook is 1,178,160 bytes after Git-compatible LF normalization.

Sealed publication manifest: reports/efficientad/final/publication_manifest.json
contains 587 artifact identities. Verification summary, resource guard, all run
histories/scores and final scientific reports are retained. New frozen-evaluator
sources and threshold policy remain pinned. The public teacher is the same
unofficial PDN-S, SHA256 b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337;
repository Apache-2.0 does not resolve separate weight licensing/training history.
Missing penalty, nine-epoch budget, optimistic two-category proxy selection,
local-only score and calibration failures prevent paper-equivalence/deployment
claims. The compute goal was met; universal anomaly detection quality was not.

Publication staging audit passed: 365 text/notebook files, largest 1,178,160 bytes;
271 Git-checkout-filtered identities matched the sealed publication manifest.
`git diff --cached --check` passed. No weights, raw images/maps, caches, temporary
data or previous milestone result changes are staged. Finish/verification complete;
the completion commit and push follow.
The completion commit includes this record. Obtain its identity with
`git log -1 --format=%H` and verify origin/main using `git ls-remote origin refs/heads/main`.
The final response records the actual published SHA. Do not rerun the successful
benchmark; use the deep validator and saved artifacts for later review.
