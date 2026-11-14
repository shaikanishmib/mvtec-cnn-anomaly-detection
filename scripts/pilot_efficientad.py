"""Bounded, real fitting-normal GPU engineering pilot; no official test access."""
from pathlib import Path
import json
import os
import sys
import time
import subprocess
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from anomaly_detection.data import sha256,project_data_root
from anomaly_detection.efficientad import EfficientAD,save_checkpoint,load_checkpoint
from anomaly_detection.tracking import record

ROOT=Path(__file__).resolve().parents[1]
ID='06-pilot-bottle-seed2026-v1'


def main():
    assert Path(sys.executable).resolve()==ROOT/'.venv/Scripts/python.exe'
    report=ROOT/'reports/efficientad'; report.mkdir(exist_ok=True)
    assert not (ROOT/f'experiments/{ID}-start.json').exists(), 'Inspect existing run before resuming'
    torch.set_num_threads(4); torch.manual_seed(2026); np.random.seed(2026)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    source=ROOT/'checkpoints/efficientad/assets/teacher_small.pth'
    manifest=ROOT/'reports/partitions/seed2026-fit.csv'
    fit=pd.read_csv(manifest)
    rows=fit[fit.category=='bottle'].sort_values('path').head(8)
    assert rows.partition.eq('fit').all() and rows.split.eq('train').all()
    root=project_data_root(ROOT)
    images=[]; augmented=[]
    def tensor(im):
        return TF.normalize(TF.to_tensor(TF.resize(im,[256,256])),[.485,.456,.406],[.229,.224,.225]).unsqueeze(0).cuda()
    for row in rows.itertuples():
        assert sha256(root/row.path)==row.sha256
        with Image.open(root/row.path) as im:
            im=im.convert('RGB'); images.append(tensor(im)); augmented.append(tensor(TF.adjust_brightness(im,1.1)))
    base=json.loads((ROOT/'configs/experiment_template.json').read_text())
    base.update(experiment_id=ID+'-start',model_family='EfficientAD-S-no-penalty-pilot',
        categories=['bottle'],hypothesis='Pretrained PDN permits finite student and AE learning on fitting normals',
        preprocessing=dict(size=256,normalization='ImageNet'),hyperparameters=dict(steps=32,batch=1,lr=.0001,hard_quantile=.999),
        environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True),
        training=dict(partition_manifest_sha256=sha256(manifest),teacher_sha256=sha256(source),fit_paths=rows.path.tolist(),external_penalty=False),
        status='started',decision='pilot only')
    record(ROOT/'experiments',base)
    started=time.perf_counter(); peak_rss=psutil.Process().memory_info().rss
    model=EfficientAD().cuda()
    model.teacher.load_state_dict(torch.load(source,map_location='cuda',weights_only=True),strict=True)
    model.fit_feature_normalization(images)
    optimizer=torch.optim.Adam(list(model.student.parameters())+list(model.autoencoder.parameters()),lr=1e-4,weight_decay=1e-5)
    def probe():
        model.eval()
        with torch.no_grad(): return {k:float(v) for k,v in model.losses(images[0],augmented[0]).items()}
    before=probe(); history=[]
    branch_before={name:next(branch.parameters()).detach().clone() for name,branch in [('student',model.student),('autoencoder',model.autoencoder)]}
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); train_start=time.perf_counter()
    for step in range(32):
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses=model.losses(images[step%8],augmented[step%8])
        assert all(torch.isfinite(v) for v in losses.values())
        losses['total'].backward()
        for branch in (model.student,model.autoencoder):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        optimizer.step()
        history.append(dict(step=step,**{k:float(v.detach()) for k,v in losses.items()}))
        peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
    torch.cuda.synchronize(); training_seconds=time.perf_counter()-train_start
    after=probe()
    assert after['hard']<before['hard'] and after['autoencoder']<before['autoencoder'], (before,after)
    for name,branch in [('student',model.student),('autoencoder',model.autoencoder)]:
        assert not torch.equal(branch_before[name],next(branch.parameters()))
    # Pilot-only map fit uses pilot fitting normals, not reserved calibration.
    model.fit_map_normalization(images)
    with torch.no_grad():
        maps,score=model(images[0]); local,global_map=model.raw_maps(images[0])
        assert maps.shape==(1,1,256,256) and torch.isfinite(maps).all() and torch.isfinite(score).all()
    checkpoint=ROOT/f'checkpoints/efficientad/{ID}.pt'
    save_checkpoint(checkpoint,model,dict(teacher_sha256=sha256(source),pilot=True))
    loaded,metadata=load_checkpoint(checkpoint,'cuda')
    with torch.no_grad(): assert torch.equal(model(images[0])[0],loaded(images[0])[0])
    del loaded
    latency=[]
    with torch.inference_mode():
        for _ in range(20): model(images[0])
        torch.cuda.synchronize()
        for _ in range(100):
            t=time.perf_counter(); model(images[0]); torch.cuda.synchronize(); latency.append((time.perf_counter()-t)*1000)
    result=dict(experiment_id=ID,status='passed',before=before,after=after,history=history,
        training_seconds=training_seconds,seconds_per_step=training_seconds/32,
        model_latency_ms=dict(p50=float(np.percentile(latency,50)),p95=float(np.percentile(latency,95)),warmup=20,iterations=100),
        peak_process_rss_bytes=max(peak_rss,psutil.Process().memory_info().rss),
        process_peak_working_set_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        parameters={k:sum(p.numel() for p in getattr(model,k).parameters()) for k in ('teacher','student','autoencoder')},
        feature_shape=list(model.teacher(images[0]).shape),map_shape=list(maps.shape),score=float(score[0]),
        checkpoint=dict(path=checkpoint.relative_to(ROOT).as_posix(),sha256=sha256(checkpoint),bytes=checkpoint.stat().st_size),
        teacher=dict(source='nelson1425/EfficientAD',revision='fcab5146f84ae17597044ad5ddf1656ccf805401',sha256=sha256(source),bytes=source.stat().st_size,license='repository Apache-2.0; no separate weight license/model card'),
        partition_manifest_sha256=sha256(manifest),input_paths=rows.path.tolist(),
        elapsed_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['32-step engineering pilot only','No external-image penalty','Pilot-subset feature and map statistics; not final calibration','No test evaluation','Brightness fixed at 1.1 for branch learning probe'])
    (report/'pilot.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    base.update(experiment_id=ID+'-result',status='completed',metrics={k:result[k] for k in ('before','after','model_latency_ms')},
        paths=dict(checkpoint=result['checkpoint'],results='reports/efficientad/pilot.json'),
        interpretation='Both branches learned on real fitting normals; engineering gate only.',decision='Review measured cost and bounded development budget before further training')
    record(ROOT/'experiments',base)
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
