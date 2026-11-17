"""Isolated loopback HTTP timing; run after tests, not concurrently with them."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import logging
import subprocess
import socket
import time
from unittest.mock import patch
import httpx
import numpy as np
import pandas as pd
import torch
from anomaly_detection.data import autodiscover_data_root
from anomaly_detection.inference import PatchCorePredictor
from anomaly_detection.inference import patchcore as core
from backend_server_process import stop_owned_server

OUT=ROOT/'reports/backend'


def pct(a):return dict(p50_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95)))


def benchmark():
    target=OUT/'performance.json'
    assert not target.exists(),'Reuse completed performance evidence'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    samples=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
    root=autodiscover_data_root(ROOT)
    progress=ROOT/'outputs/backend/performance-progress.json'
    records=json.loads(progress.read_text()) if progress.exists() else []
    for device in ('cpu','cuda'):
        done={x['category'] for x in records if x['device']==device}
        if done=={'bottle','carpet'}:continue
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',8000))==0:
                raise RuntimeError('Port 8000 is occupied; existing process preserved')
        args=[sys.executable,'-m','uvicorn','anomaly_detection.backend.app:app','--host','127.0.0.1','--port','8000','--workers','1','--no-access-log']
        env=dict(os.environ,ANOMALY_DEVICE=device,ANOMALY_CPU_THREADS='4')
        folder=ROOT/'outputs/backend';folder.mkdir(parents=True,exist_ok=True)
        attempt=len(list(folder.glob(f'benchmark-{device}*.log')))+1
        log=folder/f'benchmark-{device}-attempt{attempt}.log'
        with log.open('w',encoding='utf-8') as stream:
            begin=time.perf_counter()
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                with httpx.Client(base_url='http://127.0.0.1:8000',timeout=90,trust_env=False) as client:
                    deadline=time.monotonic()+90
                    while True:
                        if process.poll() is not None:raise RuntimeError('Benchmark server exited; inspect local log')
                        try:
                            if client.get('/health/live').status_code==200:break
                        except httpx.TransportError:pass
                        if time.monotonic()>deadline:raise RuntimeError('Server startup deadline')
                        time.sleep(.25)
                    startup=(time.perf_counter()-begin)*1000
                    for category in ('bottle','carpet'):
                        if category in done:continue
                        row=samples.query('category == @category and device == "cpu"').iloc[0]
                        data=(root/row.path).read_bytes()
                        def request(visual=False):
                            response=client.post('/api/v1/predict',files={'image':('inspection.png',data,'image/png')},data={'category':category,'visualize':str(visual).lower()})
                            assert response.status_code==200,response.text
                            return response.json()
                        begin=time.perf_counter();cold=request();cold_wall=(time.perf_counter()-begin)*1000
                        if device=='cpu' and category=='bottle':
                            (OUT/'example_response.json').write_text(json.dumps(cold,indent=2)+'\n')
                        for _ in range(5):request()
                        network=[];inference=[];decode=[];server=[]
                        for _ in range(30):
                            begin=time.perf_counter();r=request();network.append((time.perf_counter()-begin)*1000)
                            inference.append(r['timings_ms']['inference']);decode.append(r['timings_ms']['decode']);server.append(r['latency_ms'])
                        visualization=[]
                        for _ in range(10):visualization.append(request(True)['timings_ms']['visualization'])
                        info=client.get('/api/v1/model-info',params={'category':category}).json()
                        model=PatchCorePredictor(ROOT/'configs/patchcore_deployment_v1.json',ROOT,category,device)
                        stages={name:[] for name in ['preprocessing','feature_extraction','nearest_neighbor']}
                        def measured(name,fn):
                            def call(*args,**kwargs):
                                if device=='cuda':torch.cuda.synchronize()
                                begin=time.perf_counter();value=fn(*args,**kwargs)
                                if device=='cuda':torch.cuda.synchronize()
                                stages[name].append((time.perf_counter()-begin)*1000)
                                return value
                            return call
                        for _ in range(5):model.predict(data)
                        with patch.object(core,'preprocess',measured('preprocessing',core.preprocess)), \
                             patch.object(model.extractor,'forward',measured('feature_extraction',model.extractor.forward)), \
                             patch.object(core,'nearest_neighbors',measured('nearest_neighbor',core.nearest_neighbors)):
                            for _ in range(20):model.predict(data)
                        records.append(dict(device=device,category=category,startup_to_live_ms=startup,cold_api_ms=cold_wall,
                            cold_model_load_ms=cold['timings_ms']['model_load'],api_round_trip=pct(network),server_processing=pct(server),
                            warm_inference=pct(inference),decode=pct(decode),visualization_png=pct(visualization),
                            **{k:pct(v) for k,v in stages.items()},resources=info['resources']))
                        del model
                        progress.write_text(json.dumps(records,indent=2)+'\n')
                        print(device+'/'+category+': 30 loopback warm HTTP requests and stage profiles measured',flush=True)
            finally:
                stop_owned_server(process)
    result=dict(status='passed',records=records,warmup=5,api_repetitions=30,visualization_repetitions=10,stage_repetitions=20,
        methodology='Sequential isolated benchmark after test suite. Real loopback HTTP/1.1 keepalive via httpx, encoded PNG bytes resident in client, multipart body included. No visualization in warm API series. Cold request is first category load; startup-to-live includes interpreter/imports but no model. Four CPU threads, FP32. Independent synchronized direct-adapter stage profiles use unchanged M10 functions; additional sync overhead and different boundaries mean components are not additive. Memory comes from API worker allocator/RSS, not the profiling client. Parent profiling model performs no computation during HTTP series. Original broader ASGI validation overlapped focused tests; these isolated loopback numbers are primary.')
    target.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':benchmark()
