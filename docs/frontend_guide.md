# Fieldwork inspection dashboard

Milestone 12 adds a local React inspection workspace over the unchanged Milestone
11 FastAPI service. It performs no browser-side model inference. PatchCore is the
only verified adapter; model and category choices come from the service.

## Installation and startup

Use Node **26.13.0**, npm **11.20.0**, and the existing project Python environment.
The exact frontend dependencies and transitive resolution are pinned in
`frontend/package.json` and `frontend/package-lock.json`. No Python dependency or
CUDA installation changes are required. From the repository root in PowerShell:

```powershell
node --version
npm.cmd --version
npm.cmd --prefix frontend ci
```

In terminal one, start the existing backend:

```powershell
$env:ANOMALY_DEVICE = 'cpu'
$env:ANOMALY_CPU_THREADS = '4'
.\.venv\Scripts\python.exe -m uvicorn anomaly_detection.backend.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

In terminal two:

```powershell
npm.cmd --prefix frontend run dev
```

Open `http://127.0.0.1:5173`. These commands were exercised by the browser
integration runner. Keep one backend worker: its guarded inference execution,
two-category model cache and ephemeral PNG cache are process-local. To request
CUDA, use `$env:ANOMALY_DEVICE = 'cuda'` before starting that same backend command.
The frontend reads the returned device and never selects its own execution path.
This milestone's real browser measurements use CPU; M11's preserved CPU/CUDA
API/direct parity evidence remains the GPU validation source.

The pinned local extractor, seed-2026 banks and calibration records must already
exist. See [artifact reproduction](patchcore_inference.md) and the
[backend guide](backend_api.md). Weights, banks and dataset images are not shipped
with this UI. Readiness checks load and verify the default category; a different
category can still fail later if its own artifacts are unavailable.

## Configuration and network boundaries

Vite proxies `/health` and `/api` to `http://127.0.0.1:8000`. To change the proxy
destination, set `$env:BACKEND_URL` before starting Vite, or copy
`frontend/.env.example` to `frontend/.env.local`. `BACKEND_URL` belongs to the Vite
process. `VITE_API_BASE_URL` is optional, public browser configuration: never put
secrets there. An explicit cross-origin API URL requires matching restricted
backend `ANOMALY_CORS_ORIGINS`; the default proxy needs no CORS changes.

The dev server binds loopback, uses a strict port and is a local development tool.
Production hosting, authentication, TLS and external deployment are outside M12.
`npm.cmd --prefix frontend run build` creates `frontend/dist`; `run preview` serves
a local build preview at port 4173. Preview configuration/build are checked, while
the measured real-backend browser flow uses `run dev`, not a deployed build.

## Workflow and interpretation

1. Wait for **Backend ready**; retry connection if the server or artifacts fail.
2. Choose a discovered model and category. All 15 MVTec categories are available
   with the local artifacts: bottle, cable, capsule, carpet, grid, hazelnut,
   leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood and zipper.
3. Drop one PNG/JPEG or use the keyboard-accessible file picker. The UI checks
   MIME/extension agreement, non-empty content, signature, browser decoding and
   dimensions. Limits are **8 MiB, 4,000,000 decoded pixels and 4,096 per side**.
   These match default backend settings. Server validation remains authoritative,
   including stricter format and decompression safeguards. Browser decoding occurs
   before the client can check decoded dimensions; it is not a security boundary.
4. Run inspection. Repeated submission is blocked while running. The service
   receives multipart `image`, `category`, `model` and `visualize=true`.
5. Inspect raw score, original category cutoff and strict `score > threshold`
   research decision. The main summary rounds to six decimals; expandable details
   retain full JSON numeric precision. No probability or confidence is invented.
6. Compare original, heatmap and overlay together, or use individual tabs. Arrow
   keys move between tabs. Fit/150%/200% zoom changes the view and allows scrolling.
   The browser uses native backend PNG bytes without recoloring or rescaling data.
7. Reset or change the image/category. Old results are cleared; category changes
   retain the selected image but require a fresh request. Reset releases previews.

Heatmap colors use the backend's per-image visualization scaling, not threshold
calibration. Colors across different images are not absolute severity values.
The score comparison indicator changes only its display axis, never the score.
An anomaly is a research response, not a verified manufacturing defect. Normal
decisions can miss defects; small normal calibration sets and distribution
differences produce category-dependent false alarms. Original thresholds are
never recalibrated in the UI.

## Architecture and contracts

| Location | Responsibility |
|---|---|
| `src/api/types.ts` | Maintained backend prediction/model/provenance contracts |
| `src/api/client.ts` | Fetch, multipart, safe response validation and error translation |
| `src/upload.ts` | Local upload/signature/dimension validation and preview |
| `src/useInspection.ts` | Connection/inspection states, cancellation, stale-result isolation |
| `src/components/Controls.tsx` | Model/category selectors, drag/drop and reset |
| `src/components/PredictionSummary.tsx` | Authoritative score, cutoff, decision and timing |
| `src/components/ImageWorkspace.tsx` | Original/PNG comparison, zoom and expiry states |
| `src/components/Metadata.tsx` | Returned preprocessing, provenance and limitations |
| `src/styles.css` | Shared design tokens, responsive layout, focus and reduced motion |

Display components accept structured data; adding another actual adapter requires
extending and validating the API contracts. Discovery does not fabricate availability.
The client validates schema/version, finite numbers, strict decision orientation,
UUIDs, checksum formats and same-prediction visualization routes. The saved OpenAPI
schema's prediction field set is reconciled by `scripts/validate_frontend.py`.

| Endpoint | Use |
|---|---|
| `GET /health/live` | Service connectivity |
| `GET /health/ready` | Verified default artifact/device readiness |
| `GET /api/v1/models` | Actual available adapters |
| `GET /api/v1/categories` | Available categories |
| `GET /api/v1/model-info` | Selected category metadata and thresholds |
| `POST /api/v1/predict` | Multipart image inspection |
| `GET /api/v1/predictions/{prediction_id}/heatmap` | Native PNG |
| `GET /api/v1/predictions/{prediction_id}/overlay` | Native PNG |

The authoritative schema is `/openapi.json`; interactive backend docs are `/docs`.
The frontend does not copy nearest-neighbor, extractor or calibration logic.

## State, privacy and errors

Connecting, unavailable, artifact-unavailable, invalid upload, running, success,
failure and missing visualization states have explicit UI messages. Errors are
translated from known codes; internal exception text and paths are never rendered.
Malformed responses are rejected rather than treated as success. Reset, selection
changes and unmount abort pending client requests and ignore stale completions.
The 120-second fetch deadline bounds waiting for response headers; downloading a
response body and server/GPU work are not hard-cancelled by this client deadline.
Wait before retrying a timed-out operation; `server_busy` is recoverable.

Uploaded originals remain browser object URLs until reset/change/unmount. The UI
does not store inspection history or use localStorage. Backend visualizations
expire after 300 seconds, cache eviction or restart; fetched browser blobs can
remain visible for the current inspection until released. Missing PNGs show an
unavailable explanation while preserving the accepted numeric result. There is
no promise of persistence. Do not upload sensitive production images to an
unreviewed research service.

## Verification and recovery

```powershell
npm.cmd --prefix frontend run typecheck
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend test -- --reporter=json --outputFile=../outputs/frontend/unit-tests.json
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe scripts/verify_frontend_browser.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_backend.py tests/test_backend_real.py tests/test_patchcore_inference.py --basetemp=.pytest-tmp-frontend --junitxml=outputs/frontend/backend-regression.xml
.\.venv\Scripts\python.exe scripts/validate_frontend.py
```

The real browser runner requires installed Microsoft Edge and ports 8000/5173
free. It uses headless Edge via Playwright, starts only owned services, preserves
attempt logs and stops its owned workers in `finally`. It derives genuine
bottle/carpet samples from existing M10 parity records, performs no tuning and
keeps all raw images and detailed paths under ignored `outputs/frontend`.
Lightweight unit tests use clearly marked fixtures derived from the real schema;
the four genuine integration cases and public screenshot prediction use the
actual backend. The expiry test injects only the PNG failure, not a prediction.

The [validation report](../reports/frontend/validation.json) records 39 frontend
unit tests, six browser tests, 70 backend/inference regressions, schema consistency,
15 selected bank/calibration identities, historical notebook checks and 3,118
protected file hashes. Automated axe checks found zero scoped WCAG A/AA violations
on desktop and mobile; keyboard tabs and a 390-pixel layout passed. This is not
a comprehensive assistive-technology certification. The responsive breakpoints
are also supported by maintainable CSS rather than separate mobile logic.

For connection failure, check both terminals and `/health/live`. For readiness
failure, inspect backend-local logs and artifact instructions. For a busy worker,
wait and retry. For an expired image, repeat inspection. For a port conflict the
runner stops without killing unrelated processes. Do not rerun model training.

## Performance, screenshots and dependencies

See [measured results](../reports/frontend/results.md) for timing boundaries,
sample count, build sizes and limitations. Production build size is measured
separately from the Vite development-server browser load. Request, server and
visualization timings are deliberately reported as different boundaries.

Public screenshots in `docs/images` show an original procedural canvas fixture
and its genuine backend response. They contain no MVTec pixels. The fixture is
out of distribution and illustrates UI behavior, not detection accuracy. Genuine
MVTec screenshots remain ignored and unpublished.

`npm audit --omit=dev` reports zero vulnerabilities. The full audit reports 12
high-severity affected development packages from the unpatched picomatch advisory
[GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf).
As checked on 2026-11-17, latest picomatch 4.0.7 is affected and no patched version
is listed. A forced major tooling downgrade is not a verified fix. Use trusted
source/glob inputs, keep the dev server on loopback, and revisit the advisory
before external deployment. This is a documented unresolved development-tool
risk, not a claim of production security approval.

References: [Vite server/proxy configuration](https://vite.dev/config/server-options),
[Playwright browser channels](https://playwright.dev/docs/browsers).
