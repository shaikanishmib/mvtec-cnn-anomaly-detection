"""Short failure/recovery probes against only the owned M13 Compose project."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import httpx
from release_deployment import compose_command
from benchmark_release import wait_ready


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    out=ROOT/'reports/deployment'/f'reliability-{args.device}.json'
    if out.exists():raise SystemExit('Verified reliability evidence exists; reuse it')
    compose=compose_command(args.device=='cuda')
    container=subprocess.check_output([*compose,'ps','-q','backend'],cwd=ROOT,text=True).strip();assert container
    with httpx.Client(base_url='http://127.0.0.1:8080',timeout=20,trust_env=False) as client:
        assert client.get('/health/ready').status_code==200
        start=time.perf_counter()
        # Docker's own `kill` is a manual stop and can suppress restart policy.
        # Kill only the owned uvicorn child from inside its PID namespace instead.
        crash="import os,signal,psutil; targets=[p.pid for p in psutil.process_iter(['cmdline']) if 'anomaly_detection.backend.app:app' in (p.info['cmdline'] or [])]; assert len(targets)==1; os.kill(targets[0],signal.SIGKILL)"
        subprocess.run(['docker','exec',container,'python','-c',crash],capture_output=True)
        response=client.get('/health/live')
        during=response.status_code
        wait_ready(client)
        recovered=(time.perf_counter()-start)*1000
        assert client.get('/health/live').status_code==200
    bad=ROOT/'outputs/deployment/corrupt-artifacts'
    if not bad.exists():
        shutil.copytree(ROOT/'outputs/deployment/artifacts',bad)
        manifest=json.loads((bad/'configs/patchcore_deployment_v1.json').read_text())
        path=bad/manifest['categories']['bottle']['bank']['path']
        with path.open('r+b') as stream:
            first=stream.read(1);stream.seek(0);stream.write(bytes([first[0]^255]))
    failed=subprocess.run(['docker','run','--rm','--network','none','--mount',f'type=bind,source={bad},target=/artifacts,readonly','fieldwork-backend:local'],capture_output=True,text=True)
    assert failed.returncode==1 and 'preflight failed' in failed.stderr
    # A separate CPU API uses tiny authored pixels and a short configurable TTL;
    # the live service's five-minute retention policy is not altered.
    code="""
from fastapi.testclient import TestClient
from anomaly_detection.backend.app import create_app
from anomaly_detection.backend.contracts import Settings
from PIL import Image
from io import BytesIO
import time,json
settings=Settings.from_env()
with TestClient(create_app(settings)) as client:
    stream=BytesIO();Image.new('RGB',(24,20),(80,110,95)).save(stream,format='PNG')
    response=client.post('/api/v1/predict',files={'image':('authored.png',stream.getvalue(),'image/png')},data={'category':'bottle','visualize':'true'})
    assert response.status_code==200
    url=response.json()['visualizations']['heatmap_url']
    time.sleep(.3)
    assert client.get(url).status_code==404
assert not client.app.state.manager.cache and client.app.state.visualizations.bytes==0
print(json.dumps(dict(ttl_expiry=True,clean_resources=True)))
"""
    mounted=ROOT/'outputs/deployment/artifacts'
    checked=subprocess.run(['docker','run','--rm','--network','none','--mount',f'type=bind,source={mounted},target=/artifacts,readonly',
        '-e','ANOMALY_VISUALIZATION_TTL_SECONDS=0.1','--entrypoint','python','fieldwork-backend:local','-c',code],capture_output=True,text=True)
    assert checked.returncode==0,checked.stderr
    result=dict(status='passed',device=args.device,unexpected_shutdown=dict(observed_http_status=during,recovered_ms=recovered),
        corrupted_bank_fails_preflight=True,short_ttl_expiration=True,clean_api_context_resources=True,
        scope='Only owned Compose backend killed once; automatic on-failure recovery verified. Separate CPU container validates corrupted bank and configurable short TTL with authored pixels; no frozen or live retention policy changed.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
