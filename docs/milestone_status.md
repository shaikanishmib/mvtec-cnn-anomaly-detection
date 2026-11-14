# Milestone 8 progress

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
