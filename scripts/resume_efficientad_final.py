"""Safe restart wrapper: detect a live manager or orphan worker before launching."""
from pathlib import Path
import argparse
import json
import subprocess
import time
import psutil
from anomaly_detection.efficientad_final import ROOT,REPORT


def active():
    found=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            args=process.info['cmdline'] or []
            if any(Path(a).name=='run_efficientad_final.py' for a in args):
                found.append(dict(pid=process.pid,worker='--worker' in args,manager='--all' in args))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found


def main(status=False):
    import sys
    assert Path(sys.executable).resolve()==ROOT/'.venv/Scripts/python.exe'
    processes=active();manifest=REPORT/'benchmark_manifest.json'
    saved=json.loads(manifest.read_text()) if manifest.exists() else None
    print(json.dumps(dict(active_processes=processes,verified_complete=saved.get('verified_complete',0) if saved else 0),indent=2),flush=True)
    if status:return
    if any(p['manager'] for p in processes):
        print('Live manager already owns execution; no duplicate started.',flush=True);return
    while any(p['worker'] for p in processes):
        print('Orphan worker still active; waiting for durable completion before restarting manager.',flush=True)
        time.sleep(30);processes=active()
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/run_efficientad_final.py'),'--all'],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/reconcile_efficientad_registry.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--status',action='store_true');args=parser.parse_args();main(args.status)
