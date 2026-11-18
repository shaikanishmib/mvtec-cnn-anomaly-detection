"""Bounded, resumable local release operations; no public registry publication."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/deployment'
LOCAL = ROOT/'outputs/deployment'
COMPOSE = ['docker','compose','-p','fieldwork-m13','--env-file','deployment/.env.example']


def compose_command(cuda=False):
    return [*COMPOSE,'-f','compose.yaml','-f','compose.cuda.yaml'] if cuda else COMPOSE


def docker_json(*args):
    return json.loads(subprocess.check_output(['docker',*args],cwd=ROOT))


def source_hash(service,cuda=False):
    paths = [ROOT/'deployment'/('Dockerfile.cuda' if cuda and service=='backend' else f'Dockerfile.{service}')]
    if service == 'backend':
        paths += [ROOT/'deployment'/name for name in ['__init__.py','artifacts.py','runtime.py','healthcheck.py','requirements-runtime.txt','requirements-cuda-runtime.txt']]
        paths += list((ROOT/'src').rglob('*.py'))
    else:
        paths += [ROOT/'deployment/nginx.conf']
        paths += [p for p in (ROOT/'frontend').glob('*') if p.is_file() and p.suffix not in {'.tsbuildinfo'}]
        paths += list((ROOT/'frontend/src').rglob('*'))
    digest = hashlib.sha256()
    for path in sorted(set(p for p in paths if p.is_file())):
        digest.update(path.relative_to(ROOT).as_posix().encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def build(service,cuda=False):
    OUT.mkdir(exist_ok=True,parents=True); LOCAL.mkdir(exist_ok=True,parents=True)
    target = OUT/'builds.json'
    results = json.loads(target.read_text()) if target.exists() else {}
    key=service+'-cuda' if cuda and service=='backend' else service
    expected = source_hash(service,cuda)
    if key in results and results[key]['source_sha256'] == expected:
        try:
            current = docker_json('image','inspect',f'fieldwork-{key}:local')[0]
        except subprocess.CalledProcessError:
            current = None  # Fresh checkout: published evidence is not a local image.
        if current and current['Id'] == results[key]['image_id']:
            print(f'{service}: verified completed build reused'); return
    attempt = len(list(LOCAL.glob(f'build-{key}-*.log')))+1
    log = LOCAL/f'build-{key}-{attempt}.log'
    start = time.perf_counter()
    with log.open('w',encoding='utf-8') as stream:
        result = subprocess.run([*compose_command(cuda),'--progress','plain','build',service],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    duration = time.perf_counter()-start
    if result.returncode:
        print(log.read_text(encoding='utf-8')[-6000:]); raise SystemExit(result.returncode)
    image = docker_json('image','inspect',f'fieldwork-{key}:local')[0]
    if key in results:
        with (OUT/'build_history.jsonl').open('a',encoding='utf-8') as history:
            history.write(json.dumps(dict(service=key,**results[key]))+'\n')
    results[key] = dict(status='passed',duration_seconds=duration,source_sha256=expected,image_id=image['Id'],
        image_size_bytes=image['Size'],runtime_user=image['Config']['User'],platform=image['Os']+'/'+image['Architecture'],attempt=attempt)
    target.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results[key],indent=2))


def main():
    assert Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','up','down','browser'])
    parser.add_argument('--service',choices=['backend','frontend'])
    parser.add_argument('--cuda',action='store_true',help='Opt into separately verified CUDA build/Compose override')
    args = parser.parse_args()
    if args.action == 'browser':
        sys.path.insert(0,str(ROOT/'src'))
        import pandas as pd
        from anomaly_detection.data import autodiscover_data_root
        root=autodiscover_data_root(ROOT)
        rows=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
        inputs=[]
        for category in ['bottle','carpet']:
            for row in rows.query('category==@category and device=="cpu"').itertuples():
                inputs.append(dict(category=category,case='normal' if '/test/good/' in row.path else 'defect',path=str(root/row.path)))
        LOCAL.mkdir(exist_ok=True,parents=True)
        (LOCAL/'e2e-inputs.json').write_text(json.dumps(inputs),encoding='utf-8')
        cli=ROOT/'frontend/node_modules/@playwright/test/cli.js'
        raise SystemExit(subprocess.call(['node',str(cli),'test','--config',str(ROOT/'deployment/browser/playwright.config.mjs')],cwd=ROOT))
    elif args.action == 'build':
        for service in ([args.service] if args.service else ['backend','frontend']):
            build(service,args.cuda)
    else:
        command = ['up','-d','--wait','--wait-timeout','180'] if args.action == 'up' else ['down','--timeout','30']
        start=time.perf_counter()
        result=subprocess.call([*compose_command(args.cuda),*command],cwd=ROOT)
        key=args.action+('-cuda' if args.cuda else '')
        (OUT/f'{key}.json').write_text(json.dumps(dict(action=args.action,exit_code=result,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n',encoding='utf-8')
        raise SystemExit(result)


if __name__ == '__main__':
    main()
