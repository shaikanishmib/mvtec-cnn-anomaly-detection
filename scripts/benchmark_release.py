"""Bounded real-model Compose benchmarks and reliability checks; no fitting."""
from pathlib import Path
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import httpx
import numpy as np
import pandas as pd
from anomaly_detection.data import autodiscover_data_root, CATEGORIES
from anomaly_detection.inference import PatchCorePredictor
from release_deployment import compose_command, docker_json

OUT=ROOT/'reports/deployment'


def percentiles(values):
    return dict(n=len(values),p50_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),min_ms=min(values),max_ms=max(values))


def wait_ready(client,timeout=180):
    start=time.perf_counter()
    while time.perf_counter()-start<timeout:
        try:
            if client.get('/health/ready').status_code==200:return (time.perf_counter()-start)*1000
        except httpx.TransportError:pass
        time.sleep(.25)
    raise RuntimeError('Readiness deadline exceeded')


def main():
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output-dir',type=Path,default=OUT,help='Use a fresh ignored directory for independent rechecks; never overwrite release evidence')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(exist_ok=True,parents=True)
    target=out/f'container-{args.device}.json'
    compose=compose_command(args.device=='cuda')
    if target.exists():raise SystemExit('Completed measurements exist; reuse them rather than rerunning')
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=autodiscover_data_root(ROOT)
    frame=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
    samples={c:root/frame.query('category==@c and device=="cpu"').iloc[0].path for c in CATEGORIES}
    container=subprocess.check_output([*compose,'ps','-q','backend'],cwd=ROOT,text=True).strip()
    assert container
    inspected=docker_json('inspect',container)[0]
    assert inspected['Config']['User']=='10001:10001'
    assert not inspected['HostConfig']['PortBindings']
    assert inspected['HostConfig']['ReadonlyRootfs']
    mount=next(m for m in inspected['Mounts'] if m['Destination']=='/artifacts');assert not mount['RW']
    resources=[];stop=threading.Event()
    def poll_resources():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['docker','stats','--no-stream','--format','{{json .}}',container],text=True)
                item=json.loads(raw);resources.append({k:item[k] for k in ['CPUPerc','MemUsage','MemPerc','PIDs']})
            except (subprocess.CalledProcessError,ValueError):pass
            stop.wait(.5)
    thread=threading.Thread(target=poll_resources,daemon=True);thread.start()
    progress=(ROOT/'outputs/deployment' if out==OUT else out)/f'benchmark-{args.device}-progress.json'
    saved=json.loads(progress.read_text()) if progress.exists() else {}
    records=saved.get('records',[]);parity=[];reliability={}
    try:
        with httpx.Client(base_url='http://127.0.0.1:8080',timeout=120,trust_env=False) as client:
            ready=wait_ready(client)
            assert client.get('/health/live').json()['status']=='live'
            assert len(client.get('/api/v1/models').json()['models'])==1
            assert set(client.get('/api/v1/categories').json()['categories'])==set(CATEGORIES)
            assert client.get('/').headers.get('x-content-type-options')=='nosniff'
            def predict(category,data=None,visual=False):
                data=data if data is not None else samples[category].read_bytes()
                return client.post('/api/v1/predict',data=dict(category=category,visualize=str(visual).lower()),files={'image':('inspection.png',data,'image/png')})
            # Cold carpet occurs before the all-category loader/parity sweep.
            for category in ['bottle','carpet']:
                if any(item['category']==category for item in records):continue
                started=time.perf_counter();response=predict(category);cold_ms=(time.perf_counter()-started)*1000
                assert response.status_code==200; cold=response.json();assert cold['device']==args.device
                for _ in range(5):assert predict(category).status_code==200
                http_times=[];model=[];processing=[]
                for _ in range(30):
                    started=time.perf_counter();response=predict(category);http_times.append((time.perf_counter()-started)*1000)
                    assert response.status_code==200
                    value=response.json();model.append(value['timings_ms']['inference']);processing.append(value['latency_ms'])
                visual_times=[];retrieval=[];generation=[]
                for _ in range(5):
                    started=time.perf_counter();response=predict(category,visual=True);assert response.status_code==200
                    value=response.json();visual_times.append((time.perf_counter()-started)*1000);generation.append(value['timings_ms']['visualization'])
                    for kind in ['heatmap','overlay']:
                        started=time.perf_counter();png=client.get(value['visualizations'][kind+'_url']);retrieval.append((time.perf_counter()-started)*1000)
                        assert png.status_code==200 and png.content.startswith(b'\x89PNG\r\n\x1a\n')
                source=(ROOT/'deployment/profile_model.py').read_text(encoding='utf-8')
                probe=subprocess.run(['docker','exec','-i',container,'python','-c',source],input=json.dumps(dict(category=category,device=args.device,image=base64.b64encode(samples[category].read_bytes()).decode())),capture_output=True,text=True,check=True)
                model_only=json.loads(probe.stdout)
                records.append(dict(category=category,cold_http_ms=cold_ms,cold_model_load_ms=cold['timings_ms']['model_load'],model_only=model_only,
                    warmups=5,http=percentiles(http_times),adapter_inference=percentiles(model),backend_processing=percentiles(processing),
                    visual_http=percentiles(visual_times),visual_generation=percentiles(generation),png_retrieval=percentiles(retrieval)))
                progress.write_text(json.dumps(dict(records=records),indent=2)+'\n',encoding='utf-8')
            # Cross-platform/package parity: compare fresh direct historical-policy predictions,
            # not rounded metric CSV values. This does not recreate research evaluations.
            for category in CATEGORIES:
                response=predict(category);assert response.status_code==200;value=response.json()
                direct=PatchCorePredictor(ROOT/'configs/patchcore_deployment_v1.json',ROOT,category,args.device)
                expected=direct.predict(samples[category],visualize=False)
                error=abs(expected.score-value['anomaly_score'])
                assert error<=1e-4+1e-4*abs(expected.score),(category,error)
                assert expected.threshold==value['threshold'] and expected.is_anomaly==value['is_anomaly']
                parity.append(dict(category=category,score=value['anomaly_score'],direct_score=expected.score,absolute_error=error,threshold=value['threshold'],decision_matches=True))
                del direct
            info=client.get('/api/v1/model-info',params={'category':'zipper'}).json()
            assert len(info['model_cache']['loaded_categories'])<=2
            reliability['cache']=info['model_cache'];reliability['process_resources']=info['resources']
            # Expected input rejections, distinct from successful-request error rate.
            errors={}
            for name,data in [('empty',b''),('corrupt',b'not an image'),('oversized',b'x'*(8*1024*1024+1))]:
                response=predict('bottle',data);assert 400<=response.status_code<500
                errors[name]=dict(status=response.status_code,code=response.json()['error']['code'])
            reliability['input_errors']=errors
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses=list(pool.map(lambda _:predict('bottle'),range(4)))
            statuses=[r.status_code for r in responses]
            assert 200 in statuses and all(s in [200,503] for s in statuses)
            assert all(r.status_code==200 or r.json()['error']['code']=='server_busy' for r in responses)
            reliability['concurrent_4']=dict(statuses=statuses,bounded=503 in statuses)
            assert reliability['concurrent_4']['bounded']
            # A short-lived, separate preflight verifies missing/corrupted mounts without
            # changing the live banks. Unit tests cover individual checksum corruption too.
            missing=subprocess.run(['docker','run','--rm','--network','none','fieldwork-backend:local'],capture_output=True,text=True)
            assert missing.returncode!=0 and 'preflight failed' in missing.stderr
            reliability['missing_artifacts_fail_fast']=True
            before=client.get('/health/live').status_code
            start=time.perf_counter();subprocess.run([*compose,'restart','backend'],cwd=ROOT,check=True,capture_output=True)
            restart_ready=wait_ready(client)
            reliability['backend_restart']=dict(before=before,ready_wait_ms=restart_ready,total_ms=(time.perf_counter()-start)*1000)
            start=time.perf_counter();subprocess.run([*compose,'restart','frontend'],cwd=ROOT,check=True,capture_output=True)
            wait_ready(client)
            reliability['frontend_restart_ms']=(time.perf_counter()-start)*1000
            assert predict('bottle').status_code==200
            reliability['post_restart_prediction']=True
    finally:
        stop.set();thread.join(timeout=10)
    report=dict(status='passed',device=args.device,image_id=inspected['Image'],readiness_probe_ms=ready,records=records,parity=parity,reliability=reliability,
        container_stats_samples=resources,successful_request_errors=0,methodology='Linux Docker Desktop, one guarded worker, 4 CPU threads. nginx loopback HTTP; 5 warmups and 30 serial no-visual predictions per category; 5 visual predictions and 10 PNG GETs per category. Adapter inference includes preprocessing, extraction, nearest-neighbor scoring and native map creation; it is not model-only CUDA event latency. Warm readiness loads bottle first. CPU percentage and memory are sampled docker stats, not exhaustive peaks. Direct parity tolerance atol=1e-4, rtol=1e-4 across OS/Pillow runtime; original thresholds/decisions must match exactly.')
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','device','records','successful_request_errors']},indent=2))


if __name__=='__main__':main()
