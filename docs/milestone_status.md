# Milestone 8 progress

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
