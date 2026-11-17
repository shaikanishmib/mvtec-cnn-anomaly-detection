"""Verify the actual uvicorn import/launch command on loopback, then shut it down."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json
import os
import socket
import subprocess
import time
import httpx
import pandas as pd
from anomaly_detection.data import autodiscover_data_root
from backend_server_process import stop_owned_server


def run():
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    target=ROOT/'reports/backend/server_smoke.json'
    assert not target.exists(),'Loopback launch already verified; reuse evidence'
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8000))==0:raise RuntimeError('Port 8000 is already occupied; existing process preserved')
    log=ROOT/'outputs/backend/uvicorn-smoke.log';log.parent.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-m','uvicorn','anomaly_detection.backend.app:app','--host','127.0.0.1','--port','8000','--workers','1','--no-access-log']
    env=dict(os.environ,ANOMALY_DEVICE='cpu')
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            with httpx.Client(base_url='http://127.0.0.1:8000',timeout=90,trust_env=False) as client:
                deadline=time.monotonic()+90
                while True:
                    if process.poll() is not None:raise RuntimeError('uvicorn exited; inspect local smoke log')
                    try:
                        if client.get('/health/live').status_code==200:break
                    except httpx.TransportError:pass
                    if time.monotonic()>deadline:raise RuntimeError('uvicorn startup deadline exceeded')
                    time.sleep(.25)
                assert client.get('/health/ready').status_code==200
                schema=client.get('/openapi.json');assert schema.status_code==200
                table=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
                sample=table.query('category == "bottle" and device == "cpu"').iloc[0]
                data=(autodiscover_data_root(ROOT)/sample.path).read_bytes()
                response=client.post('/api/v1/predict',files={'image':('inspection.png',data,'image/png')},data={'category':'bottle'})
                assert response.status_code==200,response.text
                assert abs(response.json()['anomaly_score']-sample.score)<2e-4
                report=dict(status='passed',launch_module='anomaly_detection.backend.app:app',host='127.0.0.1',port=8000,
                    worker_count=1,device='cpu',live=200,ready=200,openapi=200,prediction=200,
                    real_score=response.json()['anomaly_score'],direct_score=float(sample.score))
        finally:
            stop_owned_server(process)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print('Real loopback uvicorn startup and HTTP prediction passed; smoke process stopped')


if __name__=='__main__':run()
