# Milestone 13: deployment and final-release preparation

Implementation and local CPU deployment verification are complete. GPU application status is reported below independently from hardware/container pass-through. No public tagged release or image-registry publication was performed. This is a local research release, not an authenticated or security-approved public service.

## Verification summary

- Python: 220 passed; 0 failed, 0 errors, 0 skipped. Includes prior research/inference/backend and new deployment contracts.
- Frontend: 39 unit tests and 3 real compiled-asset browser tests passed; no failures/skips. Typecheck, lint and build passed.
- Protected baseline: 3,155 unchanged files. Eight notebooks / 205 saved executed code cells and all 270 canonical category-seed research records remain intact. The existing comparison validator reconciled original sources without retraining or rerunning benchmark evaluations.
- CPU and CUDA Compose schemas validated. Backend is non-root/read-only, private-network-only and single-worker; nginx is non-root/read-only with a loopback-only published port. All 33 external artifact files and original thresholds were checked.
- The CPU stack serves genuine object and texture inspections with correct score/threshold/decision, native PNGs, reset/repeat, refresh and connection recovery. Only existing authored-fixture screenshots remain public.

## Hardware and software

Windows-11-10.0.26300-SP0; Intel64 Family 6 Model 191 Stepping 2, GenuineIntel. Docker Desktop 4.97.0 (244584), Engine 29.9.2, Compose 5.6.1; Docker VM 24 CPUs, 15.47 GiB. GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 610.91, 8188 MiB. WSL2 and NVIDIA container nvidia-smi pass-through were actually checked. The Docker engine initially required startup; no host CUDA/Python recreation occurred.

Runtime Python/Pillow differences from the historical fitting environment are explicit in validation.json. Container-only Pillow 11.3.0 and setuptools 78.1.1 security updates leave the original project environment unchanged. All-category parity constrains their observed numerical impact; it is not a proof for every possible input.

## Builds and startup

| Image | Runtime user | Uncompressed local size bytes | Final build seconds |
|---|---|---:|---:|
| backend | 10001:10001 | 1,596,962,785 | 2.293 |
| frontend | 101:101 | 93,067,548 | 17.823 |
| backend-cuda | 10001:10001 | 12,524,680,490 | 62.330 |

Final build time may be cached: the CPU final identity check reused layers, and the successful GPU build reused its already downloaded official base. build_history.jsonl retains the earlier measured 93.065-second dependency-lock refresh; the initial frontend build took 17.823 seconds. GPU preparation also included two timed-out 657.9-MB cuDNN downloads and a later official-runtime layer download of 4.10 GB taking 263.2 seconds before extraction. Those failed/preparatory stages are not included in the last successful build duration. Uncompressed image sizes differ from registry transfer bytes. Detailed attempts are preserved only in ignored build logs. Images contain runtime code/dependencies, not pretrained weights, banks or MVTec images. The large optional GPU runtime is not presented as lightweight.

| Device | Stopped-container live ms | Backend ready ms | Frontend/proxy ready ms |
|---|---:|---:|---:|
| cpu | 10065.91 | 11121.68 | 5901.31 |
| cuda | 10971.08 | 13966.99 | 4310.97 |

Single controlled stopped-container starts use existing images/artifacts and potentially warm host page caches. docker exec polling overhead is included; Docker Desktop boot/build time and Compose health cadence are excluded. up.json separately records Compose wait-to-health elapsed time for its last successful CPU launch.

## Bounded inference and HTTP measurements

Values are p50 / p95 milliseconds. Every device/category uses five warmups and 30 measured serial no-visual requests, 30 tensor-path measurements after five warmups, five visual HTTP requests and ten PNG GETs. Builds and tests do not overlap these measured phases. Only bottle and carpet provide performance samples; all 15 categories provide parity coverage.

| Device/category | Tensor model wall | Adapter inference | Backend processing | HTTP round trip | Visual HTTP | PNG GET |
|---|---:|---:|---:|---:|---:|---:|
| cpu/bottle | 26.88 / 29.68 | 30.08 / 30.95 | 54.86 / 65.79 | 58.64 / 69.47 | 369.14 / 370.72 | 3.95 / 5.75 |
| cpu/carpet | 26.17 / 30.66 | 31.09 / 34.81 | 80.93 / 85.24 | 85.55 / 89.67 | 658.06 / 675.68 | 6.21 / 10.72 |
| cuda/bottle | 4.84 / 7.16 | 25.04 / 26.83 | 51.92 / 55.02 | 56.29 / 59.79 | 377.00 / 401.21 | 4.89 / 7.08 |
| cuda/carpet | 4.88 / 5.58 | 12.65 / 15.31 | 63.88 / 83.37 | 69.71 / 88.73 | 667.79 / 742.41 | 7.38 / 11.31 |

Tensor model wall includes the frozen feature extractor/alignment and nearest-neighbor distances on a preprocessed, device-resident batch-one tensor. It excludes decoding, transfers and native map assembly. Adapter inference includes preprocessing and native-map generation; backend adds decode/response work; HTTP adds nginx/transport. Visual HTTP includes PNG generation; PNG GET is separate. These boundaries must not be compared as interchangeable latency improvements over previous milestones.

### Container memory and reliability

- cpu: backend RSS snapshot 520.35 MiB; CUDA allocated/reserved/peak allocated 0/0/0 bytes. 28 docker stats samples retain CPU %, memory and PID observations in container-cpu.json; they are sampled bounds rather than exhaustive peaks.
- cpu: maximum sampled container memory 886.40 MiB and CPU 374.73% (Docker 100% represents one CPU core). The sampling window also includes the separate tensor-profiling process, category swaps and recovery; this is not a pure steady-worker memory estimate.
- cpu: four simultaneous requests returned [200, 503, 503, 503]; expected busy rejections are distinct from the zero unexpected errors among asserted successful serial operations. Cache remained at two categories and evicted as categories changed.
- cpu: empty/corrupt/oversized images rejected; missing artifacts fail fast; copied corrupted bank rejected. Short-TTL authored-image PNG expiry and API-context cleanup passed in an isolated CPU container. No live retention or frozen model policy was changed.
- cpu: graceful backend/frontend restarts and post-restart prediction passed. A real uvicorn crash produced HTTP 502 then automatically recovered in 11084.50 ms. Docker CLI kill was initially found to be a manual stop, so the corrected probe kills only the owned uvicorn child inside its namespace.
- cuda: backend RSS snapshot 1370.02 MiB; CUDA allocated/reserved/peak allocated 62,170,112/109,051,904/83,029,504 bytes. 29 docker stats samples retain CPU %, memory and PID observations in container-cuda.json; they are sampled bounds rather than exhaustive peaks.
- cuda: maximum sampled container memory 1920.00 MiB and CPU 202.19% (Docker 100% represents one CPU core). The sampling window also includes the separate tensor-profiling process, category swaps and recovery; this is not a pure steady-worker memory estimate.
- cuda: four simultaneous requests returned [200, 503, 503, 503]; expected busy rejections are distinct from the zero unexpected errors among asserted successful serial operations. Cache remained at two categories and evicted as categories changed.
- cuda: empty/corrupt/oversized images rejected; missing artifacts fail fast; copied corrupted bank rejected. Short-TTL authored-image PNG expiry and API-context cleanup passed in an isolated CPU container. No live retention or frozen model policy was changed.
- cuda: graceful backend/frontend restarts and post-restart prediction passed. A real uvicorn crash produced HTTP 502 then automatically recovered in 14169.98 ms. Docker CLI kill was initially found to be a manual stop, so the corrected probe kills only the owned uvicorn child inside its namespace.

RSS before/after each 30-call tensor probe is retained. Short bounded runs show allocation/cache behavior; they cannot rule out long-duration leaks or establish production availability. CUDA peak allocation is the PyTorch allocator peak, not total driver/context/display VRAM. Resource sampling is deliberately concurrent observation; direct parity checks occur outside the timed serial loops. benchmark_images.json preserves the CPU image measured before a diagnostics-only preflight message refinement; model source, runtime dependency pins and artifact identities did not change.

## Browser turnaround and production bundle

The compiled nginx frontend was exercised by Edge against the CPU stack. Two genuine normal-image categories provide descriptive browser timings, not a statistically stable UI p95. Inspection complete ends when both PNGs resolve/load, before guaranteed compositor presentation; driver timing includes automation waits/assertions.

| Category | Server processing ms | UI request-to-PNG completion ms | Driver turnaround ms |
|---|---:|---:|---:|
| bottle | 387.33 | 418.80 | 479.46 |
| carpet | 1348.30 | 1383.40 | 1430.27 |

| Static asset | Raw bytes | Gzip reference bytes |
|---|---:|---:|
| assets/index-barjicEm.css | 16,121 | 4,261 |
| assets/index-YjsUVyn6.js | 249,892 | 76,508 |
| index.html | 521 | 337 |

Frontend idle resource snapshot: 4.824MiB / 128MiB, CPU 0.00%, 3 PIDs. This is a single observation, not a peak. Final owned-stack shutdown completed successfully in 1.883 seconds; external artifacts and local images were retained.

These unchanged production assets are also built inside the frontend image. Reference gzip sizes are local compressed sizes, not a claim that nginx gzip is enabled. Browser navigation, initial-render, preview, image-load and submission measures are retained in validation.json.

## GPU status

The NVIDIA pass-through probe, actual CUDA-enabled PyTorch container, all-category model inference/parity, bounded CUDA benchmarks and recovery checks passed. CPU/CUDA were explicit configurations; no silent CPU fallback occurred. CUDA event timings and allocated/reserved/peak device memory are recorded separately from synchronized tensor-wall timing.

## Security audit and release limits

OSV audited 35 exact release Python package versions. After compatible container-only updates, 100 advisory records remain (records can share CVE aliases and are not a count of independent exploitable vulnerabilities). Findings and fixed-version ranges are retained in dependency_audit.json; baseline findings are retained separately. Larger Pillow fixes require a new major version; PyTorch's reported JIT issue is fixed in a later release. This adapter does not call torch.jit.script. Remaining setuptools records include platform/build-tool cases, with fixes incompatible with the frozen torch constraint. None of these scope observations is a claim of proven unreachability.
npm production dependency audit: 0 findings. Full frontend toolchain: 12 affected packages, from the known picomatch ReDoS advisory. Its published range remains unpatched in the pinned toolchain; forced major downgrades were not treated as a verified fix.
Docker Scout 1.27.0 is installed, but its local-image CVE scan requires Docker Hub authentication and was blocked. No account credentials were requested or embedded. OS/image CVE scanning therefore remains unverified; base-image digest pinning is not proof that the images are vulnerability-free. The official GPU base includes additional inherited packages outside the 35 selected Python pins; that broader image surface has not been cleared by this application dependency audit.
No public image push or GitHub tagged release occurred. The code is prepared and verified for bounded local research use; authenticated hardened public deployment, complete image CVE clearance, broader hardware/OS validation and prospective industrial calibration remain outside demonstrated readiness. No project-wide redistribution license is invented; review code, weight and dataset rights before reuse.

## Research preservation and interpretation

The original six-model comparison remains unchanged. Simplified PatchCore was selected through the documented quality/efficiency/provenance tradeoff, not a test-picked deployment seed. Seed 2026 is fixed for every category. The model omits parts of the published PatchCore pipeline; no paper-equivalence claim is made. Small normal-calibration sets, distribution changes and border/texture responses retain false-alarm and missed-defect risks. Nominal calibration FPR does not guarantee official-test or manufacturing FPR.

## Recovery and reproduction

Follow [the deployment guide](../../docs/deployment_guide.md). Restore legitimate byte-identical artifacts, validate/export them, build/up the CPU stack, run tests/benchmarks/browser checks and scripts/validate_release.py. Inspect existing final JSON, registry/history and local logs before repeating work. Benchmark category progress and build identities permit reuse after interruption. Stop only the owned Compose project; do not delete artifacts or change frozen thresholds.
