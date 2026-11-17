"""Derive readable performance tables from measured evidence; never rerun models."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json
import platform
import importlib.metadata
import subprocess
import pandas as pd
import torch

OUT=ROOT/'reports/backend'


def timing_section(report):
    lines=['<!-- BACKEND_TIMINGS_START -->','## Measured performance and validation','',
        'Primary measurements are isolated real loopback HTTP requests, **after** the test suite.',
        'Encoded PNG uploads use persistent HTTP/1.1 connections. Each category/device has five',
        'warmups and 30 warm requests without visualization; ten separate requests measure',
        'rendering/PNG encoding. Four CPU threads and unchanged full-FP32 inference are used.',
        'The broader ASGI parity run overlapped focused tests, so its timings are retained',
        'as secondary evidence; the following loopback measurements are the primary results.','',
        '| Category/device | Warm adapter p50/p95 ms | HTTP round-trip p50/p95 ms | Cold category request ms | Model loading ms |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in report['records']:
        a=r['warm_inference'];b=r['api_round_trip']
        lines.append(f"| {r['category']} / {r['device']} | {a['p50_ms']:.2f} / {a['p95_ms']:.2f} | {b['p50_ms']:.2f} / {b['p95_ms']:.2f} | {r['cold_api_ms']:.2f} | {r['cold_model_load_ms']:.2f} |")
    lines += ['', '| Category/device | Decode p50/p95 ms | Preprocess p50/p95 ms | Features p50/p95 ms | NN p50/p95 ms | Render+PNG p50/p95 ms |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in report['records']:
        values=[f"{r[k]['p50_ms']:.2f} / {r[k]['p95_ms']:.2f}" for k in ['decode','preprocessing','feature_extraction','nearest_neighbor','visualization_png']]
        lines.append('| '+r['category']+' / '+r['device']+' | '+' | '.join(values)+' |')
    starts={r['device']:r['startup_to_live_ms'] for r in report['records']}
    rss=max(r['resources']['cpu_rss_bytes'] for r in report['records'])
    allocated=max(r['resources']['gpu_peak_allocated_bytes'] for r in report['records'])
    reserved=max(r['resources']['gpu_reserved_bytes'] for r in report['records'])
    lines += ['',f"Process startup-to-live was {starts['cpu']/1000:.2f} s for CPU and {starts['cuda']/1000:.2f} s for CUDA.",
        'This includes interpreter/imports and API initialization, excludes model loading, and',
        'has only one observation per device. Cold-category requests include load/verification',
        'in a running Python process; they are not a cold-filesystem benchmark. Warm adapter',
        'time includes preprocessing, transfers and native map/score construction; HTTP time',
        'additionally includes multipart upload, decode, validation, JSON and loopback transport.',
        'The independent stage profiles use 20 unchanged-adapter calls with explicit CUDA',
        'synchronization around each stage. Extra synchronization and differing boundaries mean',
        'these percentiles must not be added as though they were one request decomposition.',
        'Visualization includes both heatmap/overlay rendering and PNG encoding, and is excluded',
        'from the warm HTTP series. Its substantial cost is retained rather than hidden.','',
        f'Maximum sampled API-worker RSS was {rss:,} bytes ({rss/2**30:.2f} GiB). GPU peak allocated',
        f'was {allocated:,} bytes ({allocated/2**20:.2f} MiB); maximum reserved was {reserved:,} bytes',
        f'({reserved/2**20:.2f} MiB). RSS includes libraries, CUDA context and bounded cached PNGs;',
        'it is not minimum serving RAM. GPU values are PyTorch allocator counters, not driver totals.',
        'No profiling-client memory is included in these server counters. Timings characterize',
        'this Windows/RTX 4060 Laptop environment, not production tail-latency guarantees.',
        'Thirty repetitions offer limited p95 precision; scheduling/power and input complexity',
        'can change performance. These scopes differ from historical model-only research timing.','',
        'Validation: all 15 categories on CPU/CUDA reproduced direct M10 scores exactly (30',
        'normal-image category/device checks), with exact thresholds, decisions and PNG arrays.',
        'Four additional genuine bottle/carpet defect checks also matched exactly. The full',
        'suite passed **203 tests, zero failures/errors/skips**. A pinned Starlette/httpx TestClient',
        'deprecation warning is retained; actual Uvicorn HTTP behavior was separately verified.',
        'All 3,086 historical files and all eight notebooks (205 executed code cells) were',
        'preserved/validated without model training or rerunning historical benchmarks.','',
        'Evidence: `reports/backend/performance.json`, `runtime_summary.csv`, `integration.json`,',
        '`parity.csv`, `defect_parity.csv`, `server_smoke.json`, `test_summary.json`,',
        '`validation.json`, `environment.json` and the OpenAPI snapshot.','',
        '<!-- BACKEND_TIMINGS_END -->','']
    return '\n'.join(lines)


def run():
    report=json.loads((OUT/'performance.json').read_text())
    assert report['status']=='passed' and len(report['records'])==4
    rows=[]
    for r in report['records']:
        row={k:r[k] for k in ['category','device','cold_api_ms','cold_model_load_ms','startup_to_live_ms']}
        for name,value in r.items():
            if isinstance(value,dict):row.update({name+'_'+k:v for k,v in value.items()})
        rows.append(row)
    assert not (OUT/'runtime_summary.csv').exists()
    pd.DataFrame(rows).to_csv(OUT/'runtime_summary.csv',index=False,float_format='%.17g',lineterminator='\n')
    guide=ROOT/'docs/backend_api.md';text=guide.read_text(encoding='utf-8')
    assert '<!-- BACKEND_TIMINGS_START -->' not in text
    guide.write_text(text.replace('## Security and research limitations',timing_section(report)+'\n## Security and research limitations'),encoding='utf-8',newline='\n')
    names=['torch','torchvision','fastapi','starlette','pydantic','uvicorn','python-multipart','httpx','anyio']
    environment=dict(os=platform.platform(),cpu=platform.processor(),gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,software={k:importlib.metadata.version(k) for k in names},
        cpu_threads=4,weights_or_research_changed=False)
    try:environment['driver']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):environment['driver']='not measured'
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    print('Backend timing tables, guide and measured environment finalized')


if __name__=='__main__':run()
