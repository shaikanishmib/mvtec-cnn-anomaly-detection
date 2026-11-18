"""Render release documentation from verified compact evidence; never infer results."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def pair(item):return f"{item['p50_ms']:.2f} / {item['p95_ms']:.2f}"


def main():
    validation=read('reports/deployment/validation.json');assert validation['status']=='passed'
    builds=read('reports/deployment/builds.json');env=read('reports/deployment/environment.json');audit=read('reports/deployment/dependency_audit.json')
    reports={d:read(f'reports/deployment/container-{d}.json') for d in validation['measurements']}
    metrics=pd.read_csv(ROOT/'reports/final_comparison/model_macro.csv')
    names=dict(basic_cnn='Basic CNN',autoencoder='Convolutional AE',patchcore='Simplified PatchCore',supersimplenet='SuperSimpleNet Lite',padim='PaDiM',efficientad='EfficientAD-S reduced-budget')
    table=['| Model | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AU-PRO | Recall | FPR | Precision | F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics.itertuples(index=False,name=None):table.append('| '+names[row[0]]+' | '+' | '.join(f'{v:.4f}' for v in row[1:])+' |')
    tests=validation['python_tests'];unit=validation['frontend_tests'];browser=validation['browser_tests']
    lines=['# Milestone 13: deployment and final-release preparation','',
        'Implementation and local CPU deployment verification are complete. GPU application status is reported below independently from hardware/container pass-through. No public tagged release or image-registry publication was performed. This is a local research release, not an authenticated or security-approved public service.','',
        '## Verification summary','',
        f'- Python: {tests["tests"]} passed; {tests["failures"]} failed, {tests["errors"]} errors, {tests["skipped"]} skipped. Includes prior research/inference/backend and new deployment contracts.',
        f'- Frontend: {unit["passed"]} unit tests and {browser["expected"]} real compiled-asset browser tests passed; no failures/skips. Typecheck, lint and build passed.',
        f'- Protected baseline: {validation["protected_files"]:,} unchanged files. Eight notebooks / 205 saved executed code cells and all 270 canonical category-seed research records remain intact. The existing comparison validator reconciled original sources without retraining or rerunning benchmark evaluations.',
        '- CPU and CUDA Compose schemas validated. Backend is non-root/read-only, private-network-only and single-worker; nginx is non-root/read-only with a loopback-only published port. All 33 external artifact files and original thresholds were checked.',
        '- The CPU stack serves genuine object and texture inspections with correct score/threshold/decision, native PNGs, reset/repeat, refresh and connection recovery. Only existing authored-fixture screenshots remain public.','',
        '## Hardware and software','',
        f"{env['host_os']}; {env['processor']}. {env['docker_platform']}, Engine {env['docker_server']}, Compose {env['compose']}; Docker VM {env['docker_cpus']} CPUs, {env['docker_memory_bytes']/1024**3:.2f} GiB. GPU: {env['gpu']}. WSL2 and NVIDIA container nvidia-smi pass-through were actually checked. The Docker engine initially required startup; no host CUDA/Python recreation occurred.",
        '', 'Runtime Python/Pillow differences from the historical fitting environment are explicit in validation.json. Container-only Pillow 11.3.0 and setuptools 78.1.1 security updates leave the original project environment unchanged. All-category parity constrains their observed numerical impact; it is not a proof for every possible input.','',
        '## Builds and startup','', '| Image | Runtime user | Uncompressed local size bytes | Final build seconds |','|---|---|---:|---:|']
    for key,value in builds.items():lines.append(f"| {key} | {value['runtime_user']} | {value['image_size_bytes']:,} | {value['duration_seconds']:.3f} |")
    lines += ['', 'Final build time may be cached: the CPU final identity check reused layers, and the successful GPU build reused its already downloaded official base. build_history.jsonl retains the earlier measured 93.065-second dependency-lock refresh; the initial frontend build took 17.823 seconds. GPU preparation also included two timed-out 657.9-MB cuDNN downloads and a later official-runtime layer download of 4.10 GB taking 263.2 seconds before extraction. Those failed/preparatory stages are not included in the last successful build duration. Uncompressed image sizes differ from registry transfer bytes. Detailed attempts are preserved only in ignored build logs. Images contain runtime code/dependencies, not pretrained weights, banks or MVTec images. The large optional GPU runtime is not presented as lightweight.', '',
        '| Device | Stopped-container live ms | Backend ready ms | Frontend/proxy ready ms |','|---|---:|---:|---:|']
    for device in reports:
        startup=read(f'reports/deployment/startup-{device}.json')
        lines.append(f"| {device} | {startup['backend_live_ms']:.2f} | {startup['backend_ready_ms']:.2f} | {startup['frontend_and_proxy_ready_ms']:.2f} |")
    lines += ['', 'Single controlled stopped-container starts use existing images/artifacts and potentially warm host page caches. docker exec polling overhead is included; Docker Desktop boot/build time and Compose health cadence are excluded. up.json separately records Compose wait-to-health elapsed time for its last successful CPU launch.', '',
        '## Bounded inference and HTTP measurements','',
        'Values are p50 / p95 milliseconds. Every device/category uses five warmups and 30 measured serial no-visual requests, 30 tensor-path measurements after five warmups, five visual HTTP requests and ten PNG GETs. Builds and tests do not overlap these measured phases. Only bottle and carpet provide performance samples; all 15 categories provide parity coverage.', '',
        '| Device/category | Tensor model wall | Adapter inference | Backend processing | HTTP round trip | Visual HTTP | PNG GET |','|---|---:|---:|---:|---:|---:|---:|']
    for device,report in reports.items():
        for record in report['records']:
            lines.append(f"| {device}/{record['category']} | {pair(record['model_only']['model_tensor_wall'])} | {pair(record['adapter_inference'])} | {pair(record['backend_processing'])} | {pair(record['http'])} | {pair(record['visual_http'])} | {pair(record['png_retrieval'])} |")
    lines += ['', 'Tensor model wall includes the frozen feature extractor/alignment and nearest-neighbor distances on a preprocessed, device-resident batch-one tensor. It excludes decoding, transfers and native map assembly. Adapter inference includes preprocessing and native-map generation; backend adds decode/response work; HTTP adds nginx/transport. Visual HTTP includes PNG generation; PNG GET is separate. These boundaries must not be compared as interchangeable latency improvements over previous milestones.', '',
        '### Container memory and reliability','']
    for device,report in reports.items():
        resource=report['reliability']['process_resources'];samples=report['container_stats_samples'];reliability=read(f'reports/deployment/reliability-{device}.json')
        def mib(value):
            value=value.split(' / ')[0]
            return float(value[:-3])*(1024 if value.endswith('GiB') else 1)
        maximum_memory=max(mib(sample['MemUsage']) for sample in samples)
        maximum_cpu=max(float(sample['CPUPerc'].rstrip('%')) for sample in samples)
        lines += [f"- {device}: backend RSS snapshot {resource['cpu_rss_bytes']/1024**2:.2f} MiB; CUDA allocated/reserved/peak allocated {resource['gpu_allocated_bytes']:,}/{resource['gpu_reserved_bytes']:,}/{resource['gpu_peak_allocated_bytes']:,} bytes. {len(samples)} docker stats samples retain CPU %, memory and PID observations in container-{device}.json; they are sampled bounds rather than exhaustive peaks.",
            f"- {device}: maximum sampled container memory {maximum_memory:.2f} MiB and CPU {maximum_cpu:.2f}% (Docker 100% represents one CPU core). The sampling window also includes the separate tensor-profiling process, category swaps and recovery; this is not a pure steady-worker memory estimate.",
            f"- {device}: four simultaneous requests returned {report['reliability']['concurrent_4']['statuses']}; expected busy rejections are distinct from the zero unexpected errors among asserted successful serial operations. Cache remained at two categories and evicted as categories changed.",
            f"- {device}: empty/corrupt/oversized images rejected; missing artifacts fail fast; copied corrupted bank rejected. Short-TTL authored-image PNG expiry and API-context cleanup passed in an isolated CPU container. No live retention or frozen model policy was changed.",
            f"- {device}: graceful backend/frontend restarts and post-restart prediction passed. A real uvicorn crash produced HTTP {reliability['unexpected_shutdown']['observed_http_status']} then automatically recovered in {reliability['unexpected_shutdown']['recovered_ms']:.2f} ms. Docker CLI kill was initially found to be a manual stop, so the corrected probe kills only the owned uvicorn child inside its namespace."]
    lines += ['', 'RSS before/after each 30-call tensor probe is retained. Short bounded runs show allocation/cache behavior; they cannot rule out long-duration leaks or establish production availability. CUDA peak allocation is the PyTorch allocator peak, not total driver/context/display VRAM. Resource sampling is deliberately concurrent observation; direct parity checks occur outside the timed serial loops. benchmark_images.json preserves the CPU image measured before a diagnostics-only preflight message refinement; model source, runtime dependency pins and artifact identities did not change.', '',
        '## Browser turnaround and production bundle','',
        'The compiled nginx frontend was exercised by Edge against the CPU stack. Two genuine normal-image categories provide descriptive browser timings, not a statistically stable UI p95. Inspection complete ends when both PNGs resolve/load, before guaranteed compositor presentation; driver timing includes automation waits/assertions.', '',
        '| Category | Server processing ms | UI request-to-PNG completion ms | Driver turnaround ms |','|---|---:|---:|---:|']
    for row in validation['browser_cases']:
        complete=next(x['duration_ms'] for x in row['measures'] if x['name']=='inspection:complete')
        lines.append(f"| {row['category']} | {row['backend_latency_ms']:.2f} | {complete:.2f} | {row['driver_turnaround_ms']:.2f} |")
    assets=read('reports/frontend/validation.json')['build_assets']
    lines += ['', '| Static asset | Raw bytes | Gzip reference bytes |','|---|---:|---:|']
    for asset in assets:lines.append(f"| {asset['path']} | {asset['bytes']:,} | {asset['gzip_bytes']:,} |")
    front=read('reports/deployment/frontend_resources.json')
    stopped=read('reports/deployment/down-cuda.json');assert stopped['exit_code']==0
    lines += ['', f"Frontend idle resource snapshot: {front['MemUsage']}, CPU {front['CPUPerc']}, {front['PIDs']} PIDs. This is a single observation, not a peak. Final owned-stack shutdown completed successfully in {stopped['elapsed_seconds']:.3f} seconds; external artifacts and local images were retained.", '',
        'These unchanged production assets are also built inside the frontend image. Reference gzip sizes are local compressed sizes, not a claim that nginx gzip is enabled. Browser navigation, initial-render, preview, image-load and submission measures are retained in validation.json.', '',
        '## GPU status','']
    if 'cuda' in reports:
        lines += ['The NVIDIA pass-through probe, actual CUDA-enabled PyTorch container, all-category model inference/parity, bounded CUDA benchmarks and recovery checks passed. CPU/CUDA were explicit configurations; no silent CPU fallback occurred. CUDA event timings and allocated/reserved/peak device memory are recorded separately from synchronized tensor-wall timing.']
    else:
        blocker=read('reports/deployment/gpu_blocker.json')
        lines += [blocker['summary'], 'GPU pass-through alone does not prove application inference. The opt-in configuration is retained for a separately verified retry; no CUDA application timing or score is fabricated.']
    lines += ['', '## Security audit and release limits','',
        f"OSV audited {audit['python']['packages']} exact release Python package versions. After compatible container-only updates, {len(audit['python']['findings'])} advisory records remain (records can share CVE aliases and are not a count of independent exploitable vulnerabilities). Findings and fixed-version ranges are retained in dependency_audit.json; baseline findings are retained separately. Larger Pillow fixes require a new major version; PyTorch's reported JIT issue is fixed in a later release. This adapter does not call torch.jit.script. Remaining setuptools records include platform/build-tool cases, with fixes incompatible with the frozen torch constraint. None of these scope observations is a claim of proven unreachability.",
        f"npm production dependency audit: {audit['npm']['production']['counts']['total']} findings. Full frontend toolchain: {audit['npm']['all']['counts']['total']} affected packages, from the known picomatch ReDoS advisory. Its published range remains unpatched in the pinned toolchain; forced major downgrades were not treated as a verified fix.",
        'Docker Scout 1.27.0 is installed, but its local-image CVE scan requires Docker Hub authentication and was blocked. No account credentials were requested or embedded. OS/image CVE scanning therefore remains unverified; base-image digest pinning is not proof that the images are vulnerability-free. The official GPU base includes additional inherited packages outside the 35 selected Python pins; that broader image surface has not been cleared by this application dependency audit.',
        'No public image push or GitHub tagged release occurred. The code is prepared and verified for bounded local research use; authenticated hardened public deployment, complete image CVE clearance, broader hardware/OS validation and prospective industrial calibration remain outside demonstrated readiness. No project-wide redistribution license is invented; review code, weight and dataset rights before reuse.', '',
        '## Research preservation and interpretation','',
        'The original six-model comparison remains unchanged. Simplified PatchCore was selected through the documented quality/efficiency/provenance tradeoff, not a test-picked deployment seed. Seed 2026 is fixed for every category. The model omits parts of the published PatchCore pipeline; no paper-equivalence claim is made. Small normal-calibration sets, distribution changes and border/texture responses retain false-alarm and missed-defect risks. Nominal calibration FPR does not guarantee official-test or manufacturing FPR.', '',
        '## Recovery and reproduction','',
        'Follow [the deployment guide](../../docs/deployment_guide.md). Restore legitimate byte-identical artifacts, validate/export them, build/up the CPU stack, run tests/benchmarks/browser checks and scripts/validate_release.py. Inspect existing final JSON, registry/history and local logs before repeating work. Benchmark category progress and build identities permit reuse after interruption. Stop only the owned Compose project; do not delete artifacts or change frozen thresholds.', '']
    (ROOT/'reports/deployment/results.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    readme=['# Fieldwork · Industrial visual anomaly inspection','',
        'A reproducible MVTec AD research project and local inspection application: **eight notebooks, six architectures, 15 categories, three seeds and 270 final evaluations**. Frozen simplified PatchCore powers a reusable CPU/CUDA inference adapter, FastAPI backend and responsive React dashboard.', '',
        '> Research use only. An anomaly distance is not a probability, and an anomaly decision is not a verified manufacturing defect. This release is not approved for unauthenticated public deployment.', '',
        '![Actual Fieldwork dashboard with an authored synthetic illustration](docs/images/inspection-dashboard-desktop.png)', '',
        'The screenshot contains an original procedural fixture and its genuine backend response, with no MVTec pixels. It illustrates the UI, not model accuracy. [Mobile view](docs/images/inspection-dashboard-mobile.png).', '',
        '## Run the packaged application','',
        'Prerequisites: running Docker Desktop/WSL2 or a compatible Linux Docker host, the existing project `.venv`, and legitimately retained byte-identical model artifacts. Weights, banks and dataset images are intentionally not included. From the repository root in PowerShell:', '',
        '```powershell', '.\\.venv\\Scripts\\python.exe -m deployment.artifacts --root . --export outputs/deployment/artifacts', '.\\.venv\\Scripts\\python.exe scripts/release_deployment.py build', '.\\.venv\\Scripts\\python.exe scripts/release_deployment.py up', '```', '',
        'Open **http://127.0.0.1:8080**. CPU is the portable default. See the [deployment guide](docs/deployment_guide.md) for artifact restoration, environment settings, verified commands, CUDA status, health checks, tests and shutdown. The backend has no published host port; only nginx binds loopback.', '',
        '```mermaid','flowchart LR','    B[Browser / React] --> N[nginx: loopback 8080]','    N --> A[FastAPI / guarded worker]','    A --> P[Frozen PatchCore adapter]','    P --> M[Read-only verified artifacts]','```','',
        '## Inspection features','',
        '- Actual model/category discovery; PatchCore supports all 15 original MVTec categories.',
        '- PNG/JPEG drag/drop, preview and server-authoritative size/format checks: 8 MiB, four million pixels, 4,096 per side.',
        '- Raw score, original threshold and strict research decision; provenance, device, preprocessing and latency.',
        '- Original/heatmap/overlay comparison, zoom, keyboard controls, responsive layouts, reset and clear failure states.',
        '- Lazy two-category cache, guarded inference and ephemeral visualizations. No browser inference or persisted inspection history.', '',
        '## Research methodology and findings','',
        'Normal-only fitting/development/calibration partitions are disjoint from official test evaluation. Category means average three seeds; macro means weight all 15 categories equally. Frozen selection/calibration policies and exact historical metric values are preserved. All scores increase with anomaly; cutoffs use strict comparisons. Simplified and reduced-budget architectures are labeled explicitly.', '',
        *table, '',
        'Values are fractions rounded to four decimals from [the canonical macro table](reports/final_comparison/model_macro.csv). [The full research report](reports/final_comparison/report.md) documents protocol differences, category/seed uncertainty and paired comparisons. Small macro differences do not establish universal superiority.', '',
        'Pretrained methods performed more strongly than the scratch baselines in this benchmark. PatchCore balances image ranking and localization; SuperSimpleNet Lite offers a competitive faster alternative, and PaDiM localizes well despite weaker image detection. The reduced-budget EfficientAD result does not represent full paper-length EfficientAD. These observations are not causal proof or a guarantee beyond MVTec.', '',
        'Threshold transfer remains a key limitation: PatchCore macro test FPR is about 17.2% despite a nominal 5% normal-calibration target. Fine defects, screw image detection, texture false alarms and broad localization responses remain documented failure modes. No deployment recalibration or test-driven tuning was performed.', '',
        '## Verified engineering and benchmarks','',
        f"The release verification includes **{tests['tests']} Python tests, {unit['passed']} frontend unit tests and {browser['expected']} production-browser tests**, with zero failures/skips in final runs. Typecheck, lint, build, artifact identities, all-category parity and research protection passed. CPU container inference and real object/texture dashboard flows are verified.", '',
        '| Device/category | Tensor model p50/p95 ms | HTTP p50/p95 ms |','|---|---:|---:|']
    for device,report in reports.items():
        for entry in report['records']:readme.append(f"| {device}/{entry['category']} | {pair(entry['model_only']['model_tensor_wall'])} | {pair(entry['http'])} |")
    readme += ['', 'Five warmups and 30 samples per category; model and HTTP boundaries differ. These Linux-container measurements are not directly interchangeable with earlier Windows, model-only GPU or browser timings. [Complete release evidence](reports/deployment/results.md) includes memory, startup, visualization, browser, reliability and GPU outcomes.', '',
        'Known security findings remain in frozen-compatible Python dependencies and the development toolchain. Production React dependencies audit clear. Docker Scout image-CVE scanning is blocked by required authentication. This is a bounded local research release, not a claim of production security or availability.', '',
        '## Notebooks and repository map','',
        '| Notebook | Research story |','|---|---|',
        '| [00](notebooks/00_dataset_eda.ipynb) | Dataset integrity and exploratory analysis |',
        '| [01](notebooks/01_basic_cnn.ipynb) | Basic CNN baseline |',
        '| [02](notebooks/02_convolutional_autoencoder.ipynb) | Reconstruction baseline |',
        '| [03](notebooks/03_patchcore.ipynb) | Simplified PatchCore |',
        '| [04](notebooks/04_supersimplenet.ipynb) | SuperSimpleNet Lite |',
        '| [05](notebooks/05_padim.ipynb) | PaDiM |',
        '| [06](notebooks/06_efficientad.ipynb) | EfficientAD-S reduced-budget |',
        '| [07](notebooks/07_model_comparison.ipynb) | Final six-model research comparison |', '',
        '- `src/anomaly_detection/`: reusable research, inference and backend modules.',
        '- `frontend/`: pinned React/TypeScript/Vite application and unit tests.',
        '- `deployment/`, `compose*.yaml`: pinned containers, private routing and artifact preflight.',
        '- `configs/`, `experiments/`, `reports/`: frozen policies, append-only history and compact evidence.',
        '- `tests/`, `scripts/`: focused tests, validators, bounded benchmarks and recovery tooling.',
        '- `docs/`: [inference](docs/patchcore_inference.md), [API](docs/backend_api.md), [frontend](docs/frontend_guide.md), [deployment](docs/deployment_guide.md) and [milestone status](docs/milestone_status.md).', '',
        'Raw MVTec images, model weights/banks, large maps, caches, environment secrets and build outputs stay outside Git and public container images. Detailed local measurements and any licensed screenshots remain under ignored outputs.', '',
        '## Local development and verification','',
        'The existing CUDA Python environment is preserved. In separate PowerShell terminals:', '',
        '```powershell', '$env:ANOMALY_DEVICE = \'cpu\'', '.\\.venv\\Scripts\\python.exe -m uvicorn anomaly_detection.backend.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log', '# Second terminal: Node 26.13.0 / npm 11.20.0', 'npm.cmd --prefix frontend ci', 'npm.cmd --prefix frontend run dev', '```', '',
        'Development UI: http://127.0.0.1:5173; backend OpenAPI: http://127.0.0.1:8000/docs. Deployment uses compiled nginx assets on port 8080 instead.', '',
        '```powershell', '.\\.venv\\Scripts\\python.exe -m pytest -q --basetemp=.pytest-tmp-release', 'npm.cmd --prefix frontend run typecheck', 'npm.cmd --prefix frontend run lint', 'npm.cmd --prefix frontend test', 'npm.cmd --prefix frontend run build', '```', '',
        'See the deployment guide for real-model integration, source protection, audit and benchmark commands. Do not retrain models to run the application.', '',
        '## References and rights','',
        'Research builds on MVTec AD, ImageNet/torchvision representations, PatchCore, PaDiM, SuperSimpleNet and EfficientAD; primary references and variant differences are documented in the notebooks and [final report](reports/final_comparison/report.md). Thanks to the dataset authors and open-source PyTorch, FastAPI, React, Vite, nginx and Docker communities.', '',
        'Review each dataset, pretrained weight, dependency and tool license before redistribution or commercial use. This repository does not invent a grant for MVTec images or ImageNet-derived weights, and no project-wide redistribution license is asserted here. See [weight provenance and reproduction](docs/patchcore_reproduction.md). No public release tag or Docker-registry publication was created.', '']
    (ROOT/'README.md').write_text('\n'.join(readme),encoding='utf-8',newline='\n')
    print('Rendered README and deployment report from verified evidence')


if __name__=='__main__':main()
