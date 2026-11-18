"""Record non-sensitive runtime facts and a predetermined benchmark protocol."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.data import autodiscover_data_root
import pandas as pd
from PIL import Image


def main():
    out=ROOT/'reports/deployment';out.mkdir(exist_ok=True,parents=True)
    version=json.loads(subprocess.check_output(['docker','version','--format','{{json .}}']))
    info=json.loads(subprocess.check_output(['docker','info','--format','{{json .}}']))
    environment=dict(host_os=platform.platform(),processor=os.environ.get('PROCESSOR_IDENTIFIER'),python=platform.python_version(),
        node=subprocess.check_output(['node','--version'],text=True).strip(),npm=subprocess.check_output(['npm.cmd','--version'],text=True).strip(),
        docker_client=version['Client']['Version'],docker_server=version['Server']['Version'],docker_platform=version['Server']['Platform']['Name'],
        compose=subprocess.check_output(['docker','compose','version','--short'],text=True).strip(),docker_cpus=info['NCPU'],docker_memory_bytes=info['MemTotal'],
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        wsl='Version 2 default confirmed by elevated wsl --status; Docker Desktop Linux engine successfully started')
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n',encoding='utf-8')
    root=autodiscover_data_root(ROOT);frame=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv');fixtures=[]
    for category in ['bottle','carpet']:
        row=frame.query('category==@category and device=="cpu"').iloc[0];path=root/row.path
        with Image.open(path) as image:shape=list(image.size)
        fixtures.append(dict(category=category,source='Retained M10 normal-image parity sample',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,width_height=shape))
    plan=dict(policy='Frozen before measured loops in benchmark_release.py; no hyperparameter search',devices=['cpu','cuda if actual container inference succeeds'],categories=['bottle','carpet'],
        serial_warmups=5,serial_samples_per_category=30,tensor_warmups=5,tensor_samples_per_category=30,visual_samples_per_category=5,png_gets_per_category=10,
        concurrency=4,parity_categories=15,parity_atol=1e-4,parity_rtol=1e-4,exact_threshold_and_decision=True,fixtures=fixtures,
        boundary_note='Tensor model, adapter inference, backend, HTTP, image retrieval and browser completion are separate. Builds/tests must not overlap measured phases. Browser sample count is two genuine categories, descriptive only.')
    (out/'benchmark_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print('Environment and frozen benchmark protocol captured without local paths or dataset pixels')


if __name__=='__main__':main()
