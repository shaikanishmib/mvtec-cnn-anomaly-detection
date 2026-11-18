"""Launch only owned loopback services, run real-browser checks, preserve evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import json
import os
import shutil
import socket
import subprocess
import time
import httpx
import pandas as pd
from anomaly_detection.data import autodiscover_data_root
from backend_server_process import stop_owned_server


def verify():
    sys.stdout.reconfigure(encoding='utf-8')
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    out=ROOT/'outputs/frontend';out.mkdir(exist_ok=True,parents=True)
    root=autodiscover_data_root(ROOT)
    old=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
    inputs=[]
    for category in ['bottle','carpet']:
        for row in old.query('category == @category and device == "cpu"').itertuples():
            case='normal' if '/test/good/' in row.path else 'defect'
            path=(root/row.path).resolve();assert path.is_relative_to(ROOT)
            inputs.append(dict(category=category,case=case,path=str(path),name=f'{category}-{case}.png'))
    (out/'e2e-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    for port in [8000,5173]:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} occupied; existing process preserved')
    attempt=len(list(out.glob('browser-attempt*.log')))+1
    backend_log=(out/f'backend-attempt{attempt}.log').open('w',encoding='utf-8')
    vite_log=(out/f'vite-attempt{attempt}.log').open('w',encoding='utf-8')
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    node=shutil.which('node');assert node
    backend=None;vite=None
    try:
        backend=subprocess.Popen([sys.executable,'-m','uvicorn','anomaly_detection.backend.app:app','--host','127.0.0.1','--port','8000','--workers','1','--no-access-log'],cwd=ROOT,env=dict(os.environ,ANOMALY_DEVICE='cpu',ANOMALY_CPU_THREADS='4'),stdout=backend_log,stderr=subprocess.STDOUT,creationflags=flags)
        # Exercise the exact documented npm command, including its package script.
        vite=subprocess.Popen([os.environ.get('COMSPEC','cmd.exe'),'/d','/c','npm.cmd run dev'],cwd=ROOT/'frontend',stdout=vite_log,stderr=subprocess.STDOUT,creationflags=flags)
        with httpx.Client(timeout=10,trust_env=False) as client:
            for url,process in [('http://127.0.0.1:8000/health/live',backend),('http://127.0.0.1:5173/',vite)]:
                deadline=time.monotonic()+120
                while True:
                    if process.poll() is not None:raise RuntimeError('Service exited; inspect local logs')
                    try:
                        if client.get(url).status_code==200:break
                    except httpx.TransportError:pass
                    if time.monotonic()>deadline:raise RuntimeError('Service startup deadline exceeded')
                    time.sleep(.25)
        cli=ROOT/'frontend/node_modules/@playwright/test/cli.js'
        with (out/f'browser-attempt{attempt}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([node,str(cli),'test'],cwd=ROOT/'frontend',stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        print((out/f'browser-attempt{attempt}.log').read_text(encoding='utf-8'))
        assert result.returncode==0,'Browser checks failed; preserved logs and screenshots are under outputs/frontend'
    finally:
        if vite is not None:stop_owned_server(vite,5173)
        if backend is not None:stop_owned_server(backend,8000)
        backend_log.close();vite_log.close()
    print('Owned Vite/backend workers stopped; real browser integration passed')


if __name__=='__main__':verify()
