# FastAPI inspection backend

Milestone 11 supplies a local research API around the **unchanged Milestone 10
inference adapter**. Only `patchcore` has an implemented, verified adapter.
All 15 original MVTec AD categories use the pinned deployment-v1 seed-2026 banks,
ResNet18 features, preprocessing, top-1% patch score and original strict cutoffs.
There is no fitting, recalibration, automatic category recognition or frontend.

## Installation and launch

Use the existing Python 3.12 project environment. Backend additions are pinned in
`requirements-backend.txt`; CUDA/research lock files and PyTorch are unchanged.
The new modules are discovered by the existing editable `src` installation.

```powershell
# Run from the repository root. Activation is optional with explicit interpreter.
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
$env:ANOMALY_DEVICE = 'cpu'   # or 'cuda', 'cuda:0'
.\.venv\Scripts\python.exe -m uvicorn anomaly_detection.backend.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

This actual import/launch command was tested on loopback with health, readiness,
OpenAPI and genuine-image prediction. See `reports/backend/server_smoke.json`.
Use one worker: caches, admission control and results are process-local. Multiple
workers would multiply GPU allocations and are outside the validated protocol.
Stop with Ctrl+C. Do not use `--reload` for measurements. Bind to loopback for this
local research application; external deployment needs a separate security review.

OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`; Swagger UI: `/docs`;
ReDoc: `/redoc`. Interactive documentation assets may require CDN access; tests
use the local OpenAPI JSON and require no external network. The schema snapshot
is published in `reports/backend/openapi.json`.

## HTTP endpoints

| Method/path | Behavior |
| --- | --- |
| GET `/health/live` | Process/API is responding; does not load weights |
| GET `/health/ready` | Lazily verifies/loads the configured category, then tests a synthetic forward once per model instance; 503 on load/runtime failure or busy worker |
| GET `/api/v1/models` | Only verified adapter identifier `patchcore`, release `patchcore-inference-v1` |
| GET `/api/v1/categories` | All 15 supported categories |
| GET `/api/v1/model-info?category=bottle&model=patchcore` | Thresholds, calibration count, preprocessing/software identities, loaded-cache state, process RSS and GPU allocator counters; no internal paths |
| POST `/api/v1/predict` | Bounded multipart image inspection |
| GET `/api/v1/predictions/{prediction_id}/heatmap` | Native-size display-only PNG, if requested and retained |
| GET `/api/v1/predictions/{prediction_id}/overlay` | Native-size overlay PNG, if requested and retained |

Readiness guarantees the **configured readiness category**, not that every other
category's files exist. Other banks are validated on first use. The completion
validation separately checked every category on CPU/CUDA. A ready result does
not imply production calibration quality. Live health remains independent of
artifact/device availability. Readiness can warm the model, so omit it before a
cold-category timing experiment.

## Prediction request

Multipart fields: required `image` file and `category` text; optional `model`
(default `patchcore`) and `visualize` (`true`/`false`, default `false`). Unknown or
duplicate fields, extra files and invalid values are rejected. Device and artifact
paths are server configuration, never HTTP upload parameters.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/predict -F "image=@your-inspection.png;type=image/png" -F "category=bottle" -F "model=patchcore" -F "visualize=true"

# Capture the JSON in PowerShell while explicitly setting the image-part MIME:
$prediction = (curl.exe -sS -X POST http://127.0.0.1:8000/api/v1/predict -F "image=@your-inspection.png;type=image/png" -F "category=bottle" -F "visualize=true") | ConvertFrom-Json
Invoke-WebRequest -Uri ('http://127.0.0.1:8000' + $prediction.visualizations.overlay_url) -OutFile '.\local-overlay.png'
```

Uploads must have a matching extension, declared MIME type **and decoded format**:
PNG `.png` / `image/png` / PNG or JPEG `.jpg`/`.jpeg` / `image/jpeg` / JPEG. Case
of extensions is normalized. A valid PNG renamed `.exe`, JPEG bytes advertised as
PNG, SVG/WebP/GIF, animation, corrupt/truncated images and empty uploads fail.
Client filenames are used only to inspect the suffix; they are never opened,
persisted, returned or logged. No URL or filesystem-path input endpoint exists.

Defaults: **8 MiB image**, request body at most image limit + 64 KiB framing,
**4,000,000 pixels**, maximum width/height **4,096**, upload deadline **15 s**.
Actual received bytes are counted, including chunked requests; a lying or absent
Content-Length does not bypass the bound. PIL header dimensions, decompression
warnings, decoded format and animation are checked before full decoding. The
original adapter still performs its own input validation. RGB conversion, no EXIF
rotation, PIL bilinear resizing and research normalization are unchanged.

## Stable response v1.0

Pydantic/OpenAPI define required fields and types. The response contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1.0` |
| `prediction_id`, `request_id` | Server-generated UUIDs; request ID also in `X-Request-ID` |
| `model`, `model_version`, `category`, `device` | Verified model/release and execution selection |
| `anomaly_score`, `threshold`, `is_anomaly` | Raw L2 distance, original normal-calibration cutoff, strict `score > threshold` decision |
| `pixel_threshold`, `decision_rule`, `score_semantics` | Original pixel cutoff and explicit score interpretation |
| `input_width`, `input_height` | Original decoded geometry |
| `latency_ms` | Server middleware entry through response-model creation; excludes final JSON serialization and network return |
| `timings_ms` | Decode, category model-load, adapter inference, optional render/PNG encoding, total service time |
| `provenance` | Seed 2026; manifest, bank, extractor, ImageNet weight, calibration and preprocessing SHA256 identities |
| `visualizations` | Optional relative heatmap/overlay URLs, expiry and display-scaling metadata |
| `limitations` | Required research-use/calibration notice |

There are **no probabilities or confidence scores**. Raw scores and maps are not
clipped or normalized for decisions. Full-resolution floating maps are not sent
as JSON arrays. Use the direct inference package for raw numerical arrays.
The genuine example response is `reports/backend/example_response.json`; its UUID
is an example of a completed request, not a durable stored prediction resource.

Visualizations are opt-in PNGs made with the existing M10 `render` function.
Per-image min/max scaling affects display only and is reported explicitly;
colors cannot compare absolute severity across images. Metadata URLs can be
null. An absent, expired, evicted or restarted result returns 404. Retrieval UUIDs
are identifiers, not authentication or authorization credentials.

## Model management and resource bounds

`contracts.py` defines a model-adapter protocol and stable schemas.
`service.py` owns the registered PatchCore adapter, LRU cache, input validation and
visualization store. `app.py` owns HTTP validation, admission, lifespan and errors.
PatchCore feature extraction/scoring are **not reimplemented** inside FastAPI:
the adapter invokes `PatchCorePredictor.predict` directly.

No banks load on startup. A prediction or readiness request loads its category;
SHA256/size, category, seed, configuration and threshold identities are verified
before deserialization. The trusted manifest is pinned by default to
`58be2eafb16ac012ac18e0feb69f11b2f510e7417464eaba5bd9797b423272fa`.
No downloads or training occur. Restore legitimate local artifacts described in
`docs/patchcore_inference.md`. Missing or corrupt files cause safe 503 responses.

The default model LRU holds **two categories**. Eviction releases the old adapter
before allocating another; CUDA synchronization/cache cleanup is guarded. One
prediction is admitted from upload through completion; competing predictions get
503 instead of queuing unbounded GPU work. A separate manager lock serializes
all model use, readiness and eviction. Model loading/inference run in a worker
thread so live health remains responsive.

Multipart bodies are bounded before parsing; the pinned Starlette parser's spool
limit is set above that bound. Uploaded files therefore remain memory-backed and
are closed in `finally`, including error paths. A >1 MiB upload test confirms no
disk rollover. Raw uploads are never deliberately persisted. Optional generated
PNGs live in a thread-safe in-memory cache bounded by **8 results / 32 MiB / 300 s**.
Expiry prevents retrieval immediately; a periodic task removes expired entries
within at most five additional seconds. Item/byte pressure can evict sooner.
Shutdown clears models, PNGs and the cleanup task.

An upload deadline is enforced. Running GPU work is not forcibly killed on a
short request timeout: doing so would leave an active thread and unsafe ownership.
Inference is bounded by validated dimensions and serialized execution, but there
is no hard kernel-time SLA. A hung native driver needs operator/process recovery.
The current per-process admission bound is not a substitute for reverse-proxy
connection/rate limits. Cancellation cannot make the guarded model execute in
parallel; the underlying worker retains its model lock until completion.

## Configuration

All variables use the `ANOMALY_` prefix; settings are read at import/startup.

| Variable suffix | Default |
| --- | --- |
| `DEVICE` | `cpu`; set `cuda` or an available `cuda:index` explicitly |
| `CPU_THREADS` | 4 |
| `ARTIFACT_ROOT` | Repository/restored artifact root |
| `MANIFEST` | `configs/patchcore_deployment_v1.json` under the repository |
| `MANIFEST_SHA256` | Pinned hash above; change only for a separately approved release |
| `READY_CATEGORY` | bottle |
| `MODEL_CACHE_SIZE` | 2, allowed 1–15 |
| `MAX_UPLOAD_BYTES` | 8388608 |
| `MAX_PIXELS`, `MAX_DIMENSION` | 4000000 / 4096 |
| `UPLOAD_TIMEOUT_SECONDS` | 15 |
| `VISUALIZATION_BYTES`, `VISUALIZATION_ITEMS`, `VISUALIZATION_TTL_SECONDS` | 33554432 / 8 / 300 |
| `CORS_ORIGINS` | Empty; comma-separated explicit origins if needed |

CPU threads are applied during application lifespan and restored on teardown in
embedded tests. Cache settings are per process. CORS has no wildcard origins or
credential sharing. Do not enable broad CORS as a substitute for authentication.
No client option can select arbitrary model paths or enable an unverified model.

## Error contract and logging

Errors have `{ "error": { "code": "...", "message": "...", "request_id": "..." } }`.
API endpoint responses use `Cache-Control: no-store` and `X-Content-Type-Options:
nosniff`. CORS preflight is handled separately by the framework and may return a
plain protocol-level rejection; endpoint errors use the JSON schema above.

| HTTP | Typical codes |
| --- | --- |
| 400 | `invalid_multipart`, `invalid_content_length`, `upload_disconnected` |
| 408 | `upload_timeout` |
| 413 | `request_too_large`, `upload_too_large`, `image_dimensions`, `visualization_limit` |
| 415 | `multipart_required`, `unsupported_image_type`, `format_mismatch` |
| 422 | `invalid_parameters`, `unsupported_model`, `unsupported_category`, `empty_image`, `invalid_image`, `animated_image` |
| 404 | `visualization_not_found` or unknown HTTP route |
| 500 | `internal_error` |
| 503 | `server_busy`, `model_unavailable` |

Messages do not expose exceptions, absolute paths or supplied data. Failed
inspection is never converted into a normal decision. Clients can back off and
retry a busy response. Invalid uploads should be corrected, not retried blindly.
Structured JSON request logs contain generated ID, known route, status and elapsed
time; unexpected errors log their class only. Image bytes, filenames, arbitrary
query strings and raw predictions are not logged. Uvicorn access logging is
disabled in the documented launch command to avoid duplicating unfiltered URLs.

## Validation and measured performance

`reports/backend/integration.json` and `parity.csv` record all **15 categories on
CPU and CUDA**, using 15 genuine normal images and 30 category/device checks.
API/direct scores matched exactly in the measured environment; thresholds and
image decisions matched, and native heatmap/overlay arrays were identical.
These checks compare the unchanged M10 implementation, not a new accuracy
benchmark. No official labels selected a new score, cutoff, seed or model.

The primary timing tables below and `reports/backend/runtime_summary.csv` derive from
`performance.json`: real loopback HTTP measurements collected sequentially after
tests, with cold-category and warm-request boundaries separated. The broader
initial ASGI validation overlapped focused tests and is retained as diagnostic
evidence; its timings are not the primary latency recommendation.

Warm API timing includes multipart HTTP upload, validation, decoding, inference,
serialization and response return over loopback. Uploaded bytes are resident in
the client; filesystem decode is not included. Default warm requests omit PNG
generation; visualization generation/encoding is separately measured. Cold API
adds checksum/model loading; startup-to-live includes interpreter and imports but
no model. Stage profiling wraps unchanged M10 preprocessing/extractor/NN functions
with synchronization in an independent direct call; overhead means stage
percentiles cannot be added or equated to API latency. RSS and GPU allocator
memory come from the actual API worker. Hardware/power and sample-count limits
preclude universal service-level guarantees.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_backend.py --basetemp=.pytest-tmp-backend
.\.venv\Scripts\python.exe -m pytest -q tests/test_backend_real.py --basetemp=.pytest-tmp-backend-real
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp
.\.venv\Scripts\python.exe scripts/validate_backend.py
```

Mocked contract/security tests are separate from actual-model tests. No pytest
test needs external network access. Dedicated loopback smoke/performance scripts
start and stop their own local Uvicorn process; ensure port 8000 is free. Completed
real-model/performance reports are exclusive-write to avoid accidental reruns.
Eight original notebooks are validated from saved executed outputs; no research
notebook, model, threshold or prediction record is rewritten.

<!-- BACKEND_TIMINGS_START -->
## Measured performance and validation

Primary measurements are isolated real loopback HTTP requests, **after** the test suite.
Encoded PNG uploads use persistent HTTP/1.1 connections. Each category/device has five
warmups and 30 warm requests without visualization; ten separate requests measure
rendering/PNG encoding. Four CPU threads and unchanged full-FP32 inference are used.
The broader ASGI parity run overlapped focused tests, so its timings are retained
as secondary evidence; the following loopback measurements are the primary results.

| Category/device | Warm adapter p50/p95 ms | HTTP round-trip p50/p95 ms | Cold category request ms | Model loading ms |
| --- | ---: | ---: | ---: | ---: |
| bottle / cpu | 205.04 / 215.28 | 288.00 / 299.18 | 1139.56 | 674.89 |
| carpet / cpu | 213.89 / 220.33 | 360.61 / 375.84 | 1020.73 | 653.72 |
| bottle / cuda | 59.85 / 62.00 | 144.30 / 147.16 | 3464.91 | 2275.20 |
| carpet / cuda | 72.48 / 74.43 | 228.57 / 247.17 | 1009.99 | 765.80 |

| Category/device | Decode p50/p95 ms | Preprocess p50/p95 ms | Features p50/p95 ms | NN p50/p95 ms | Render+PNG p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottle / cpu | 70.83 / 73.87 | 11.60 / 12.22 | 122.32 / 129.09 | 52.72 / 55.73 | 708.90 / 736.91 |
| carpet / cpu | 117.54 / 120.12 | 16.34 / 17.12 | 133.25 / 157.79 | 47.06 / 63.58 | 828.23 / 871.34 |
| bottle / cuda | 71.77 / 73.25 | 11.80 / 12.03 | 23.01 / 24.94 | 7.49 / 8.39 | 767.90 / 791.54 |
| carpet / cuda | 124.45 / 127.99 | 16.70 / 17.42 | 21.03 / 22.35 | 6.27 / 7.06 | 865.04 / 919.67 |

Process startup-to-live was 21.86 s for CPU and 22.72 s for CUDA.
This includes interpreter/imports and API initialization, excludes model loading, and
has only one observation per device. Cold-category requests include load/verification
in a running Python process; they are not a cold-filesystem benchmark. Warm adapter
time includes preprocessing, transfers and native map/score construction; HTTP time
additionally includes multipart upload, decode, validation, JSON and loopback transport.
The independent stage profiles use 20 unchanged-adapter calls with explicit CUDA
synchronization around each stage. Extra synchronization and differing boundaries mean
these percentiles must not be added as though they were one request decomposition.
Visualization includes both heatmap/overlay rendering and PNG encoding, and is excluded
from the warm HTTP series. Its substantial cost is retained rather than hidden.

Maximum sampled API-worker RSS was 1,269,243,904 bytes (1.18 GiB). GPU peak allocated
was 82,898,432 bytes (79.06 MiB); maximum reserved was 109,051,904 bytes
(104.00 MiB). RSS includes libraries, CUDA context and bounded cached PNGs;
it is not minimum serving RAM. GPU values are PyTorch allocator counters, not driver totals.
No profiling-client memory is included in these server counters. Timings characterize
this Windows/RTX 4060 Laptop environment, not production tail-latency guarantees.
Thirty repetitions offer limited p95 precision; scheduling/power and input complexity
can change performance. These scopes differ from historical model-only research timing.

Validation: all 15 categories on CPU/CUDA reproduced direct M10 scores exactly (30
normal-image category/device checks), with exact thresholds, decisions and PNG arrays.
Four additional genuine bottle/carpet defect checks also matched exactly. The full
suite passed **203 tests, zero failures/errors/skips**. A pinned Starlette/httpx TestClient
deprecation warning is retained; actual Uvicorn HTTP behavior was separately verified.
All 3,086 historical files and all eight notebooks (205 executed code cells) were
preserved/validated without model training or rerunning historical benchmarks.

Evidence: `reports/backend/performance.json`, `runtime_summary.csv`, `integration.json`,
`parity.csv`, `defect_parity.csv`, `server_smoke.json`, `test_summary.json`,
`validation.json`, `environment.json` and the OpenAPI snapshot.

<!-- BACKEND_TIMINGS_END -->

## Security and research limitations

This local backend has no authentication, TLS termination, durable prediction
store, cross-worker result store, tenant isolation, production monitoring or
security certification. UUID retrieval and restricted CORS are not access control.
Use loopback; a deployment needs explicit authentication, trusted proxy/body/rate
limits, review and independent operational validation. Administrator-configured
manifests/paths must come from trusted releases. Generated overlays contain image
content and should be treated as private even though raw uploads are discarded.

The historical 5% nominal calibration policy did not guarantee 5% official-test
FPR. Leather/carpet false alarms and screw misses remain. No probabilities are
invented, and no post-test tuning conceals those limitations. Full FP32 versus
historical TF32/batch-cache numerical differences and near-threshold risks remain
as documented in Milestone 10. Weight/data licensing and external-domain
validation are unresolved deployment requirements. The interactive inspection
frontend is the next milestone and has not been started.

Implementation references: [FastAPI uploads](https://fastapi.tiangolo.com/tutorial/request-files/),
[Starlette requests/parser limits](https://starlette.dev/requests/).
These describe framework behavior; this repository's bounded-memory overrides,
tests and actual saved measurements are the evidence for the backend claims.
