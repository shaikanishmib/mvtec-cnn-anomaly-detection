"""Reconcile saved browser evidence, API contract and immutable research identities.

No fitting, inference, notebook execution or historical report rewriting occurs.
Run with the project .venv after verify_frontend_browser.py and frontend checks.
"""
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from validate_backend import validate as backend_validate
from anomaly_detection.backend.contracts import Settings


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def validate():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    protected = read('reports/frontend/research_protection.json')
    for name, expected in protected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    historical = backend_validate()  # Checks banks/calibration/OpenAPI/notebooks without writing them.
    schema = read('reports/backend/openapi.json')
    types = (ROOT / 'frontend/src/api/types.ts').read_text(encoding='utf-8')
    fields = re.search(r'PREDICTION_FIELDS = \[(.*?)\]', types).group(1)
    assert set(re.findall(r"'([^']+)'", fields)) == set(schema['components']['schemas']['PredictionResponse']['properties'])
    props = schema['components']['schemas']['PredictionResponse']['properties']
    for key in ['schema_version', 'model', 'model_version', 'decision_rule']:
        assert repr(props[key]['const']) in types
    assert schema['components']['schemas']['Provenance']['properties']['seed']['const'] == 2026
    settings = Settings()
    upload = (ROOT / 'frontend/src/upload.ts').read_text(encoding='utf-8')
    assert 'bytes: 8 * 1024 * 1024' in upload and settings.max_upload_bytes == 8*1024*1024
    assert 'pixels: 4_000_000' in upload and settings.max_pixels == 4_000_000
    assert 'dimension: 4096' in upload and settings.max_dimension == 4096
    package = read('frontend/package.json')
    lock = read('frontend/package-lock.json')
    for group in ['dependencies', 'devDependencies']:
        assert package[group] == lock['packages'][''][group]
        for name, version in package[group].items():
            assert re.fullmatch(r'\d+\.\d+\.\d+', version), name
            assert lock['packages']['node_modules/' + name]['version'] == version
    unit = read('outputs/frontend/unit-tests.json')
    assert unit['success'] and unit['numFailedTests'] == 0 and unit['numPendingTests'] == 0
    browser = read('outputs/frontend/browser-tests.json')['stats']
    assert browser['expected'] == 6 and not any(browser[k] for k in ['unexpected', 'flaky', 'skipped'])
    root = ET.parse(ROOT / 'outputs/frontend/backend-regression.xml').getroot()
    suites = list(root.iter('testsuite'))
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert counts['tests'] == 70 and not any(counts[k] for k in ['failures', 'errors', 'skipped'])
    rows = [read(f'outputs/frontend/real-{cat}-{case}.json') for cat in ['bottle', 'carpet'] for case in ['normal', 'defect']]
    for row in rows:
        assert row['model'] == 'patchcore' and row['device'] == 'cpu'
        assert row['displayed_score'] == f"{row['score']:.6f}"
        assert row['displayed_threshold'] == f"{row['threshold']:.6f}"
        assert row['is_anomaly'] == (row['score'] > row['threshold'])
        assert row['png_bytes_unchanged'] and set(row['hashes']) == {'heatmap', 'overlay'}
    capture = read('outputs/frontend/public-capture.json')
    assert capture['actual_backend_result'] and not capture['desktop_axe_violations'] and not capture['mobile_axe_violations'] and not capture['mobile_horizontal_overflow']
    fixture = read('reports/backend/example_response.json')['prediction_id']
    assets = []
    for path in sorted((ROOT / 'frontend/dist').rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            assert fixture.encode() not in data, 'Recorded prediction must not enter application bundle'
            assets.append(dict(path=path.relative_to(ROOT / 'frontend/dist').as_posix(), bytes=len(data), gzip_bytes=len(gzip.compress(data, mtime=0))))
    assert len(assets) == 3
    def summary(values):
        values = sorted(values)
        def percentile(q):
            pos = (len(values)-1)*q; low = int(pos); high = min(low+1,len(values)-1)
            return round(values[low] + (values[high]-values[low])*(pos-low), 3)
        return dict(n=len(values), p50_ms=percentile(.5), p95_ms=percentile(.95), min_ms=round(values[0],3), max_ms=round(values[-1],3))
    timing = {key: summary([r[key] for r in rows]) for key in ['backend_latency_ms', 'total_browser_turnaround_ms']}
    for name in ['app:initial-render', 'inspection:preview', 'inspection:submission', 'inspection:response', 'inspection:heatmap', 'inspection:overlay', 'inspection:complete']:
        timing[name] = summary([next(m['duration_ms'] for m in r['metrics'] if m['name']==name) for r in rows])
    timing['first_contentful_paint'] = summary([next(p['startTime'] for p in r['navigation']['paint'] if p['name']=='first-contentful-paint') for r in rows])
    screenshots = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'docs/images').glob('inspection-dashboard-*.png')}
    assert len(screenshots) == 2
    result = dict(status='passed', baseline_commit='8e08412f1179b6d92b041efad840f1f296d62369', protected_files=len(protected),
        backend_artifacts=historical, prediction_fields=len(props), unit_tests=unit['numPassedTests'], browser_tests=browser,
        backend_regression=counts, real_cases=rows, public_capture=capture, screenshot_sha256=screenshots,
        build_assets=assets, timings=timing,
        methodology='Edge headless, Windows, Vite development server, loopback FastAPI CPU/4 threads, warmed bottle readiness and two-category cache; four genuine bottle/carpet cases. Small descriptive sample; p95 interpolated, not a production guarantee. Initial-render excludes discovery; inspection:complete ends when both PNGs resolve, before guaranteed compositor presentation. Driver turnaround includes Playwright waits and assertions. Original data images and full local logs are excluded from publication.')
    # Strip verbose navigation URLs/timing arrays; retain the measured aggregates above.
    for row in result['real_cases']:
        row.pop('navigation'); row.pop('metrics')
    (ROOT/'reports/frontend/validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines = ['# Milestone 12 verification and measurements', '',
        'The Fieldwork React/TypeScript dashboard is verified against the unchanged local PatchCore API. No fitting, recalibration, historical inference recreation or notebook modification occurred.', '',
        '## Verification', '',
        f'- {unit["numPassedTests"]} frontend unit tests; six Edge browser tests; 70 backend/inference tests passed. Zero failures or skips in these final runs.',
        '- TypeScript checking, ESLint and the Vite production build passed. Exact dependency pins and the npm lockfile were checked; a clean offline npm ci also passed.',
        f'- {len(protected):,} protected baseline files remain byte-identical. All 15 selected bank/calibration identities and the pinned extractor/policy passed the existing deep backend validator.',
        '- Eight historical notebooks / 205 saved executed code cells passed format, syntax, saved-output and byte-identity checks. Notebooks were not rerun.',
        '- The saved OpenAPI prediction fields, literal model/version/seed and default upload limits agree with the frontend contracts.',
        '- Desktop and 390 x 844 mobile axe checks found zero scoped WCAG A/AA violations; keyboard tabs, zoom and no mobile horizontal overflow passed. This is a scoped automated check, not full accessibility certification.',
        '- Actual documented backend and npm development-server commands launched successfully. Owned service processes were stopped after checks.', '',
        '## Genuine backend inspections', '',
        'Four unmocked CPU inspections use existing M10 bottle/carpet normal/defect parity examples. They verify API-to-display fidelity, not new benchmark accuracy. The server remained authoritative for decisions. M11 already verified all 15 category paths on CPU/CUDA; those original results remain unchanged. No new GPU-browser timing claim is made.', '',
        '| Category | Recorded case | Raw score | Original threshold | Research decision | PNG bytes |',
        '|---|---|---:|---:|---|---|']
    for row in rows:
        lines.append(f'| {row["category"]} | {row["case"]} | {row["score"]:.9f} | {row["threshold"]:.9f} | {"Anomaly" if row["is_anomaly"] else "Normal"} | Unchanged |')
    lines += ['', 'Main UI numbers match the API rounded to six decimals; expanded details retain full numeric precision. Each displayed heatmap/overlay blob hash matched the corresponding cached backend PNG. The expiry test faults only image retrieval and preserves a genuine accepted numeric prediction.', '',
        '## Performance boundaries', '', result['methodology'], '',
        'Hardware/environment: Windows NT 10.0.26300.0, Intel64 Family 6 Model 191 Stepping 2, CPU backend with four PyTorch threads; installed Microsoft Edge, Node 26.13.0/npm 11.20.0. Browser tests ran serially with no backend regression tests competing. Timings are milliseconds; n=4 and linearly interpolated percentiles are descriptive only. Different image dimensions and visualization costs contribute variability. These are not equivalent to M10 model-only GPU timings.', '',
        '| Boundary | p50 ms | p95 ms | n |', '|---|---:|---:|---:|']
    for key, item in timing.items():
        lines.append(f'| {key} | {item["p50_ms"]:.3f} | {item["p95_ms"]:.3f} | {item["n"]} |')
    lines += ['', '`app:initial-render` measures module entry to first mounted effect, not backend readiness. First contentful paint comes from browser paint entries. Preview includes client validation/decode and the original-image load event. Submission is local synchronous preparation before fetch; response ends at parsed JSON. PNG measures include retrieval/decode until image load, not guaranteed screen presentation. Complete is the UI request-to-both-PNG-resolution boundary; driver turnaround also includes automation waits/assertions. Backend latency includes image decoding, inference, visualization and response preparation; it is included within the client response duration, not added to it.', '',
        '### Production bundle', '', '| File | Raw bytes | Gzip bytes |', '|---|---:|---:|']
    for asset in assets:
        lines.append(f'| {asset["path"]} | {asset["bytes"]:,} | {asset["gzip_bytes"]:,} |')
    lines += ['', 'Gzip sizes are measured locally with Python gzip; hosting compression is not configured by this milestone. The real browser measurements use the development server, not these production assets.', '',
        '## Safe public captures', '', 'The desktop/mobile screenshots are from the actual running UI and an actual backend response to an original procedural canvas fixture. No MVTec pixels are published. The out-of-distribution illustration is not a model-quality claim. Raw genuine-image screenshots, absolute input paths and service logs remain under ignored outputs/frontend. Screenshot and fixture SHA-256 values are in validation.json.', '',
        '## Dependency and operational limitations', '',
        'The production npm audit has zero vulnerabilities. The full audit reported 12 high-severity affected development-tool packages caused by the unpatched picomatch advisory [GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf), checked 2026-11-17. Latest 4.0.7 remained affected and no patched version was listed. A major forced downgrade was not applied as an unverified fix. Keep development tooling on loopback with trusted sources/glob patterns and revisit before deployment.', '',
        'Client cancellation prevents stale UI results but does not terminate server inference. Readiness is a point-in-time default-category check. Backend PNGs expire or are evicted; downloaded browser blobs last only for the current page inspection. No inspection history is persisted. Original frozen cutoffs retain category-dependent false alarms and misses. The application is a local research prototype; hosting, authentication, production security review and broader assistive-technology testing remain future work.', '',
        '## Reproduction', '', 'Follow [the frontend guide](../../docs/frontend_guide.md). Run frontend checks, scripts/verify_frontend_browser.py, the focused pytest command and scripts/validate_frontend.py using the existing project environment. Source identities, exact case values, PNG hashes and aggregates are in [validation.json](validation.json). No raw data or model weights are needed in Git.', '']
    (ROOT/'reports/frontend/results.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({k: result[k] for k in ['status','protected_files','prediction_fields','unit_tests','backend_regression','timings']},indent=2))


if __name__ == '__main__':
    validate()
