# Milestone 12 verification and measurements

The Fieldwork React/TypeScript dashboard is verified against the unchanged local PatchCore API. No fitting, recalibration, historical inference recreation or notebook modification occurred.

## Verification

- 39 frontend unit tests; six Edge browser tests; 70 backend/inference tests passed. Zero failures or skips in these final runs.
- TypeScript checking, ESLint and the Vite production build passed. Exact dependency pins and the npm lockfile were checked; a clean offline npm ci also passed.
- 3,118 protected baseline files remain byte-identical. All 15 selected bank/calibration identities and the pinned extractor/policy passed the existing deep backend validator.
- Eight historical notebooks / 205 saved executed code cells passed format, syntax, saved-output and byte-identity checks. Notebooks were not rerun.
- The saved OpenAPI prediction fields, literal model/version/seed and default upload limits agree with the frontend contracts.
- Desktop and 390 x 844 mobile axe checks found zero scoped WCAG A/AA violations; keyboard tabs, zoom and no mobile horizontal overflow passed. This is a scoped automated check, not full accessibility certification.
- Actual documented backend and npm development-server commands launched successfully. Owned service processes were stopped after checks.

## Genuine backend inspections

Four unmocked CPU inspections use existing M10 bottle/carpet normal/defect parity examples. They verify API-to-display fidelity, not new benchmark accuracy. The server remained authoritative for decisions. M11 already verified all 15 category paths on CPU/CUDA; those original results remain unchanged. No new GPU-browser timing claim is made.

| Category | Recorded case | Raw score | Original threshold | Research decision | PNG bytes |
|---|---|---:|---:|---|---|
| bottle | normal | 1.337237835 | 1.426850319 | Normal | Unchanged |
| bottle | defect | 3.250213385 | 1.426850319 | Anomaly | Unchanged |
| carpet | normal | 1.134422779 | 1.256325603 | Normal | Unchanged |
| carpet | defect | 2.255002022 | 1.256325603 | Anomaly | Unchanged |

Main UI numbers match the API rounded to six decimals; expanded details retain full numeric precision. Each displayed heatmap/overlay blob hash matched the corresponding cached backend PNG. The expiry test faults only image retrieval and preserves a genuine accepted numeric prediction.

## Performance boundaries

Edge headless, Windows, Vite development server, loopback FastAPI CPU/4 threads, warmed bottle readiness and two-category cache; four genuine bottle/carpet cases. Small descriptive sample; p95 interpolated, not a production guarantee. Initial-render excludes discovery; inspection:complete ends when both PNGs resolve, before guaranteed compositor presentation. Driver turnaround includes Playwright waits and assertions. Original data images and full local logs are excluded from publication.

Hardware/environment: Windows NT 10.0.26300.0, Intel64 Family 6 Model 191 Stepping 2, CPU backend with four PyTorch threads; installed Microsoft Edge, Node 26.13.0/npm 11.20.0. Browser tests ran serially with no backend regression tests competing. Timings are milliseconds; n=4 and linearly interpolated percentiles are descriptive only. Different image dimensions and visualization costs contribute variability. These are not equivalent to M10 model-only GPU timings.

| Boundary | p50 ms | p95 ms | n |
|---|---:|---:|---:|
| backend_latency_ms | 957.975 | 1489.363 | 4 |
| total_browser_turnaround_ms | 1332.110 | 1870.224 | 4 |
| app:initial-render | 335.100 | 359.545 | 4 |
| inspection:preview | 100.650 | 127.795 | 4 |
| inspection:submission | 0.100 | 0.100 | 4 |
| inspection:response | 975.500 | 1507.695 | 4 |
| inspection:heatmap | 42.550 | 58.770 | 4 |
| inspection:overlay | 50.900 | 75.320 | 4 |
| inspection:complete | 1070.100 | 1596.040 | 4 |
| first_contentful_paint | 752.000 | 789.400 | 4 |

`app:initial-render` measures module entry to first mounted effect, not backend readiness. First contentful paint comes from browser paint entries. Preview includes client validation/decode and the original-image load event. Submission is local synchronous preparation before fetch; response ends at parsed JSON. PNG measures include retrieval/decode until image load, not guaranteed screen presentation. Complete is the UI request-to-both-PNG-resolution boundary; driver turnaround also includes automation waits/assertions. Backend latency includes image decoding, inference, visualization and response preparation; it is included within the client response duration, not added to it.

### Production bundle

| File | Raw bytes | Gzip bytes |
|---|---:|---:|
| assets/index-barjicEm.css | 16,121 | 4,261 |
| assets/index-YjsUVyn6.js | 249,892 | 76,508 |
| index.html | 521 | 337 |

Gzip sizes are measured locally with Python gzip; hosting compression is not configured by this milestone. The real browser measurements use the development server, not these production assets.

## Safe public captures

The desktop/mobile screenshots are from the actual running UI and an actual backend response to an original procedural canvas fixture. No MVTec pixels are published. The out-of-distribution illustration is not a model-quality claim. Raw genuine-image screenshots, absolute input paths and service logs remain under ignored outputs/frontend. Screenshot and fixture SHA-256 values are in validation.json.

## Dependency and operational limitations

The production npm audit has zero vulnerabilities. The full audit reported 12 high-severity affected development-tool packages caused by the unpatched picomatch advisory [GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf), checked 2026-11-17. Latest 4.0.7 remained affected and no patched version was listed. A major forced downgrade was not applied as an unverified fix. Keep development tooling on loopback with trusted sources/glob patterns and revisit before deployment.

Client cancellation prevents stale UI results but does not terminate server inference. Readiness is a point-in-time default-category check. Backend PNGs expire or are evicted; downloaded browser blobs last only for the current page inspection. No inspection history is persisted. Original frozen cutoffs retain category-dependent false alarms and misses. The application is a local research prototype; hosting, authentication, production security review and broader assistive-technology testing remain future work.

## Reproduction

Follow [the frontend guide](../../docs/frontend_guide.md). Run frontend checks, scripts/verify_frontend_browser.py, the focused pytest command and scripts/validate_frontend.py using the existing project environment. Source identities, exact case values, PNG hashes and aggregates are in [validation.json](validation.json). No raw data or model weights are needed in Git.
