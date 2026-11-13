"""Bounded normal-only reference feasibility and Lite learning pilot."""
import json,time,sys,os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from anomaly_detection.supersimplenet import SSNExtractor,SuperSimpleHeads,training_loss,optimizer_for
from anomaly_detection.ssn_research import experiment
from anomaly_detection.cnn_research import read_partition,load_resized
from anomaly_detection.patchcore_research import Resources
from anomaly_detection.basic_cnn import seed_everything
from anomaly_detection.data import sha256

def main():
    folder=ROOT/'reports/supersimplenet';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'reference_feasibility.json'
    if path.exists():raise FileExistsError('Preserve completed feasibility measurements')
    seed_everything(2026)
    extractor=SSNExtractor('wide_resnet50_2').cuda()
    rows=read_partition('fit','bottle').iloc[:4]
    images=load_resized(rows,256).cuda().float()/255
    model=SuperSimpleHeads(1536,1024,128).cuda();optimizer=optimizer_for(model)
    mask=torch.Generator().manual_seed(2027);noise=torch.Generator(device='cuda').manual_seed(2028)
    with Resources() as extraction:
        first,second=extractor.feature_maps(images);features=extractor(images)
    losses=[];times=[]
    with Resources() as training:
        for step in range(12):
            torch.cuda.synchronize();tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            seg,det,target,label=model.synthetic_forward(features,mask,noise)
            loss,parts=training_loss(seg,det,target,label);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            times.append(time.perf_counter()-tick);losses.append(dict(step=step,loss=float(loss.detach()),**parts))
            print(json.dumps(dict(reference_step=step,seconds=times[-1],**losses[-1])),flush=True)
    result=dict(backbone='wide_resnet50_2',weights_sha256=extractor.weight_sha256,
        layer_shapes=[list(first.shape),list(second.shape)],aggregated_shape=list(features.shape),
        segmentation_shape=list(seg.shape),detection_shape=list(det.shape),
        head_parameters=sum(p.numel() for p in model.parameters()),batch_size=4,steps=12,
        warmup_steps=2,mean_step_seconds=sum(times[2:])/len(times[2:]),step_seconds=times,losses=losses,
        extraction_resources=extraction.result,training_resources=training.result,
        projected_40_epoch_45_fit_training_seconds=sum(times[2:])/len(times[2:])*40*3*sum((n+3)//4 for n in [147,158,155,196,186,275,173,154,187,224,162,42,151,173,168]),
        provenance='12 optimization steps on four fitting-normal bottle images; feasibility only, not a convergence experiment',
        source_sha256=sha256(ROOT/'src/anomaly_detection/supersimplenet.py'))
    with path.open('x') as stream:json.dump(result,stream,indent=2)
    del model,optimizer,extractor,features,images,first,second,seg,det,loss,target,label
    torch.cuda.empty_cache()
    config=dict(backbone='resnet18',channels=384,seg_width=256,det_width=32,adapter_trainable=True,
        noise_std=.015,loss='margin_focal',seed=2026,batch_size=8,extraction_batch=8,epochs=20,
        validation_every=2,hypothesis='Lite ICPR components can learn masked feature perturbations on fitting normals without real anomaly supervision')
    experiment('04-pilot-lite-bottle-s2026-v1','bottle',config,SSNExtractor('resnet18').cuda(),pilot=True,phase='pilot')

if __name__=='__main__':main()
