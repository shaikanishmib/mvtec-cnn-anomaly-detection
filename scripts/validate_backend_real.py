"""Bounded real-model HTTP/direct parity plus stage-scoped measurements."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import time
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch
import importlib.metadata
import logging
import numpy as np
import pandas as pd
from PIL import Image
import psutil
import torch
from fastapi.testclient import TestClient
from anomaly_detection.backend.app import create_app
from anomaly_detection.backend.contracts import Settings
from anomaly_detection.inference import PatchCorePredictor
from anomaly_detection.inference import patchcore as inference_module
from anomaly_detection.inference.patchcore import sha256
from anomaly_detection.data import CATEGORIES,autodiscover_data_root

OUT=ROOT/'reports/backend'
logging.getLogger('anomaly.backend').setLevel(logging.WARNING)


def percentiles(values):return dict(p50_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)))


def run():
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    target=OUT/'integration.json'
    if target.exists():raise FileExistsError('Completed backend measurements exist; validate and reuse them')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available(),'CPU and CUDA validation required for this machine'
    source_table=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
    root=autodiscover_data_root(ROOT);rows=[];timings=[]
    for device in ('cpu','cuda'):
        settings=replace(Settings(),device=device)
        app=create_app(settings)
        with TestClient(app) as client:
            assert not app.state.manager.cache
            for category in CATEGORIES:
                saved=source_table.query('category == @category and device == "cpu"').iloc[0]
                path=root/saved.path;data=path.read_bytes()
                def request(visualize=False):
                    r=client.post('/api/v1/predict',files={'image':('inspection.png',data,'image/png')},
                        data={'category':category,'visualize':str(visualize).lower()})
                    assert r.status_code==200,r.text
                    return r
                begin=time.perf_counter();r=request(True);cold=(time.perf_counter()-begin)*1000;result=r.json()
                direct=PatchCorePredictor(settings.manifest,ROOT,category,device).predict(data,visualize=True)
                assert abs(result['anomaly_score']-direct.score)<1e-6
                assert result['threshold']==direct.threshold and result['is_anomaly']==direct.is_anomaly
                for name,expected in [('heatmap',direct.heatmap),('overlay',direct.overlay)]:
                    response=client.get(result['visualizations'][name+'_url']);assert response.status_code==200
                    actual=np.asarray(Image.open(BytesIO(response.content)))
                    assert np.array_equal(actual,expected)
                assert len(app.state.manager.cache)<=settings.model_cache_size
                rows.append(dict(category=category,device=device,source_path=saved.path,source_sha256=sha256(path),
                    api_score=result['anomaly_score'],direct_score=direct.score,absolute_score_error=abs(result['anomaly_score']-direct.score),
                    threshold=result['threshold'],decision=result['is_anomaly'],direct_decision=direct.is_anomaly,
                    heatmap_overlay_equal=True,cold_api_ms=cold,model_load_ms=result['timings_ms']['model_load']))
                if category in ('bottle','carpet'):
                    for _ in range(5):request(False)
                    api=[];warm=[];decode=[];visual=[]
                    for _ in range(30):
                        begin=time.perf_counter();value=request(False).json();api.append((time.perf_counter()-begin)*1000)
                        warm.append(value['timings_ms']['inference']);decode.append(value['timings_ms']['decode'])
                    for _ in range(10):visual.append(request(True).json()['timings_ms']['visualization'])
                    # Offline observers wrap the exact frozen adapter functions;
                    # the HTTP service never reimplements feature/NN scoring.
                    model=app.state.manager.cache[category].predictor
                    stages={name:[] for name in ['preprocessing','feature_extraction','nearest_neighbor']}
                    def observe(name,original):
                        def call(*a,**kw):
                            if device=='cuda':torch.cuda.synchronize()
                            begin=time.perf_counter();output=original(*a,**kw)
                            if device=='cuda':torch.cuda.synchronize()
                            stages[name].append((time.perf_counter()-begin)*1000)
                            return output
                        return call
                    with patch.object(inference_module,'preprocess',observe('preprocessing',inference_module.preprocess)), \
                         patch.object(model.extractor,'forward',observe('feature_extraction',model.extractor.forward)), \
                         patch.object(inference_module,'nearest_neighbors',observe('nearest_neighbor',inference_module.nearest_neighbors)):
                        for _ in range(20):model.predict(data)
                    memory=psutil.Process().memory_info()
                    timings.append(dict(category=category,device=device,cold_api_ms=cold,model_load_ms=result['timings_ms']['model_load'],
                        warm_prediction=percentiles(warm),api_request=percentiles(api),image_decode=percentiles(decode),
                        visualization_png=percentiles(visual),**{k:percentiles(v) for k,v in stages.items()},
                        cpu_rss_bytes=memory.rss,cpu_peak_rss_bytes=getattr(memory,'peak_wset',memory.rss),
                        gpu_allocated_bytes=torch.cuda.memory_allocated() if device=='cuda' else 0,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
                del direct
                pd.DataFrame(rows).to_csv(OUT/'parity.csv',index=False,float_format='%.17g',lineterminator='\n')
                print(device+'/'+category+': exact HTTP/direct score, thresholds, decisions and PNG arrays passed',flush=True)
            ready=client.get('/health/ready');assert ready.status_code==200
        assert not app.state.manager.cache and app.state.visualizations.bytes==0
    report=dict(status='passed',categories=15,devices=['cpu','cuda'],real_images=15,category_device_checks=30,
        maximum_api_direct_score_error=max(x['absolute_score_error'] for x in rows),all_decisions_match=True,
        all_visualization_arrays_equal=True,timings=timings,
        methodology='TestClient in-process ASGI round-trip, encoded PNG upload; 5 warmups, 30 warm requests without visualizations; 10 visualization requests. Stage observers: 20 direct adapter calls, synchronized CPU/GPU wall times; extra sync overhead, not additive latency components. Cold is first category request in a warm Python process, including validation/loading and visualization. Upload bytes already resident; no network transport. Four CPU threads. Model cache max2; PNG cache max8/32MiB/300s.',
        software={x:importlib.metadata.version(x) for x in ['fastapi','starlette','pydantic','uvicorn','python-multipart','httpx','torch','torchvision']},
        gpu=torch.cuda.get_device_name(0),manifest_sha256=settings.manifest_sha256,model_modified=False)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print('Real backend validation complete',flush=True)


if __name__=='__main__':run()
