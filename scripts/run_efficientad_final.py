"""Manifest-driven final benchmark manager; project interpreter only."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import psutil
import torch
from anomaly_detection.data import sha256
from anomaly_detection.efficientad_research import configure
from anomaly_detection.efficientad_final import (ROOT,REPORT,LOCAL,freeze,atomic_json,identity_for,
    choose_attempt,verify_complete,run_one,FREEZE,utc)


def setup():
    f=freeze();REPORT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    identity_path=REPORT/'evaluator_identity.json'
    sources=['src/anomaly_detection/efficientad_final.py','scripts/run_efficientad_final.py',
        'src/anomaly_detection/metrics.py','src/anomaly_detection/pixel_metrics.py','src/anomaly_detection/cnn_evaluation.py']
    identity=dict(freeze_sha256=FREEZE,source_sha256={p:sha256(ROOT/p) for p in sources},
        protocol_sha256=sha256(ROOT/'docs/efficientad_final_protocol.md'))
    if identity_path.exists(): assert json.loads(identity_path.read_text())==identity,'Evaluator changed after final reservation'
    else: atomic_json(identity_path,identity)
    manifest_path=REPORT/'benchmark_manifest.json'
    if manifest_path.exists(): manifest=json.loads(manifest_path.read_text());assert manifest['freeze_sha256']==FREEZE
    else:
        manifest=dict(schema=1,freeze_sha256=FREEZE,evaluator_sha256=sha256(identity_path),created_at_utc=utc(),
            models=[dict(category=c,seed=s,status='pending',result=None) for s in f['seeds'] for c in f['categories']])
        atomic_json(manifest_path,manifest)
    preservation=REPORT/'baseline_preservation.json'
    if not preservation.exists():
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only','664e106dbdd25104ea4e3c3be192fc8be7a0b772'],cwd=ROOT,text=True).splitlines()
        allowed={'.gitattributes','README.md','docs/milestone_status.md','notebooks/06_efficientad.ipynb'}
        atomic_json(preservation,dict(baseline='664e106dbdd25104ea4e3c3be192fc8be7a0b772',
            files={p:sha256(ROOT/p) for p in paths if p not in allowed}))
    return manifest


def progress(manifest,message):
    complete=[m for m in manifest['models'] if m['status']=='complete']
    manifest.update(updated_at_utc=utc(),verified_complete=len(complete),message=message)
    atomic_json(REPORT/'benchmark_manifest.json',manifest)
    path=ROOT/'docs/milestone_status.md';text=path.read_text(encoding='utf-8')
    marker='<!-- efficientad-final-live -->'
    if marker in text: text=text[:text.index(marker)].rstrip()+'\n'
    text+='\n'+marker+'\n\n## Live final benchmark progress\n\n'
    text+=f"UTC {utc()}: {len(complete)}/45 verified complete. {message}\n\n"
    text+='Manifest: reports/efficientad/final/benchmark_manifest.json. Recovery: project Python scripts/run_efficientad_final.py --all. Complete outputs are hashed before skipping; each epoch saves model/optimizer/RNG state.\n'
    path.write_text(text,encoding='utf-8')


def resource_gate(manifest):
    completed=[json.loads((ROOT/m['result']).read_text()) for m in manifest['models'] if m['status']=='complete']
    estimate=json.loads((ROOT/'reports/efficientad/development/feasibility.json').read_text())
    expected={r['category']:r['conservative_three_seed_seconds']/3 for r in estimate['workload']}
    actual=sum(r['wall_seconds'] for r in completed);budget=sum(expected[r['category']] for r in completed)
    ratio=actual/budget if budget else 0.
    report=dict(completed=len(completed),actual_pipeline_hours=actual/3600,conservative_expected_to_date_hours=budget/3600,
        actual_to_conservative_ratio=ratio,disk_free_bytes=shutil.disk_usage(ROOT).free,
        projected_complete_hours=estimate['estimated_conservative_total_elapsed_hours']*ratio if ratio else None)
    stop=(actual>12*3600 or report['disk_free_bytes']<20*1024**3 or
        (len(completed)>=3 and ratio>1.25) or any(r['wall_seconds']>2*expected[r['category']] for r in completed))
    report['stop_before_next_run']=stop;atomic_json(REPORT/'resource_gate.json',report)
    if stop: raise RuntimeError('Measured resource gate exceeded; completed runs preserved. Inspect resource_gate.json before further execution')


def manager():
    configure(ROOT);manifest=setup();lock=LOCAL/'manager.lock.json'
    if lock.exists():
        owner=json.loads(lock.read_text())
        if psutil.pid_exists(owner['pid']):
            process=psutil.Process(owner['pid'])
            if abs(process.create_time()-owner['created'])<1: raise RuntimeError('A benchmark manager is already running')
        lock.unlink()
    with lock.open('x') as stream: json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
    try:
        for item in manifest['models']:
            category,seed=item['category'],item['seed']
            complete,identifier=choose_attempt(category,seed)
            if complete is not None:
                item.update(status='complete',result=f"reports/efficientad/final/runs/{complete['experiment_id']}-result.json")
                continue
            resource_gate(manifest)
            item.update(status='running',attempt=identifier)
            progress(manifest,f'Running {category}, seed {seed}; frozen nine epochs, then calibration/test/artifact checks.')
            command=[str(ROOT/'.venv/Scripts/python.exe'),str(Path(__file__).resolve()),'--worker',category,str(seed)]
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode:
                item['status']='interrupted';progress(manifest,f'Worker {category}/{seed} failed; durable stages retained. Inspect registry failure and resume.')
                raise RuntimeError('Worker failed with exit code '+str(result.returncode))
            complete,_=choose_attempt(category,seed);assert complete is not None
            item.update(status='complete',result=f"reports/efficientad/final/runs/{complete['experiment_id']}-result.json")
            progress(manifest,f'Verified {category}/{seed}, including official test evaluation and artifact hashes.')
            resource_gate(manifest)
        progress(manifest,'All 45 run artifacts verified. Aggregation, failure analysis, notebook, final tests and publication remain.')
    finally:
        if lock.exists(): lock.unlink()


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');parser.add_argument('--worker',nargs=2)
    args=parser.parse_args()
    if args.worker:
        run_one(args.worker[0],int(args.worker[1]))
    elif args.all: manager()
    else: parser.error('Use --all or --worker category seed')
