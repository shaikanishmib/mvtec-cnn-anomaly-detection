"""Explicit bounded development; final phases require a separately frozen policy."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from anomaly_detection.supersimplenet import SSNExtractor
from anomaly_detection.ssn_research import experiment

def development_identifier(candidate,category):
    recovery=ROOT/'reports/supersimplenet/recovery.json'
    mapping=json.loads(recovery.read_text()) if recovery.exists() else {}
    return mapping.get(candidate+'/'+category,f'04-dev-{candidate}-{category}-s2026-v1')

def candidates():
    base=dict(backbone='resnet18',channels=384,seg_width=256,det_width=32,adapter_trainable=True,
        noise_std=.015,loss='margin_focal',seed=2026,batch_size=8,extraction_batch=8,epochs=40,validation_every=10)
    return dict(reference_noise=dict(base,hypothesis='Reference noise is learnable with more fitting normals and the fixed 40-epoch Lite budget'),
        noise05=dict(base,noise_std=.05,hypothesis='Stronger noise improves synthetic discrimination over the same fixed Lite budget'),
        frozen_adapter=dict(base,adapter_trainable=False,hypothesis='Freezing the random adapter tests whether adaptation improves stable discrimination'),
        focal_only=dict(base,loss='focal',hypothesis='Removing the balanced margin tests whether it impedes short-budget segmentation learning'))

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if sys.argv[1]!='development':raise ValueError('Only bounded development is authorized by this runner until a measured freeze exists')
    reference=json.loads((ROOT/'reports/supersimplenet/reference_feasibility.json').read_text())
    assert reference['projected_40_epoch_45_fit_training_seconds']>90*60
    extractor=SSNExtractor('resnet18').cuda()
    for name,config in candidates().items():
        for category in ['bottle','cable','carpet','tile']:
            identifier=development_identifier(name,category)
            if (ROOT/f'reports/supersimplenet/runs/{identifier}-result.json').exists():continue
            experiment(identifier,category,config,extractor)

if __name__=='__main__':main()
