"""Measure owned stopped-container startup separately from Compose health cadence."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import httpx
from release_deployment import compose_command


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    target=ROOT/'reports/deployment'/f'startup-{args.device}.json'
    if target.exists():raise SystemExit('Completed startup measurement exists')
    compose=compose_command(args.device=='cuda')
    subprocess.run([*compose,'stop','--timeout','30'],cwd=ROOT,check=True,capture_output=True)
    container=subprocess.check_output([*compose,'ps','-a','-q','backend'],cwd=ROOT,text=True).strip();assert container
    start=time.perf_counter()
    subprocess.run([*compose,'start','backend'],cwd=ROOT,check=True,capture_output=True)
    def probe(path):
        code=f"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/{path}',timeout=30).status)"
        return subprocess.run(['docker','exec',container,'python','-c',code],capture_output=True).returncode==0
    deadline=time.monotonic()+120
    while not probe('health/live'):
        if time.monotonic()>deadline:raise RuntimeError('Live startup deadline exceeded')
        time.sleep(.25)
    live=(time.perf_counter()-start)*1000
    while not probe('health/ready'):
        if time.monotonic()>deadline:raise RuntimeError('Model readiness deadline exceeded')
        time.sleep(.25)
    ready=(time.perf_counter()-start)*1000
    frontend_start=time.perf_counter();subprocess.run([*compose,'start','frontend'],cwd=ROOT,check=True,capture_output=True)
    with httpx.Client(base_url='http://127.0.0.1:8080',timeout=10,trust_env=False) as client:
        while True:
            try:
                if client.get('/frontend-health').status_code==200 and client.get('/health/ready').status_code==200:break
            except httpx.TransportError:pass
            if time.monotonic()>deadline:raise RuntimeError('Frontend startup deadline exceeded')
            time.sleep(.25)
    result=dict(status='passed',device=args.device,backend_live_ms=live,backend_ready_ms=ready,frontend_and_proxy_ready_ms=(time.perf_counter()-frontend_start)*1000,
        scope='One stopped-container start with images/artifacts already present. Fresh Python/model processes; host filesystem/page caches may be warm. docker exec polling adds overhead. Excludes image builds, Docker Desktop boot and Compose health-check wait cadence. Both services stopped gracefully before measurement.')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
