"""Reconcile actual pilot measurements and register retrospective feasibility."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.tracking import record
from anomaly_detection.data import sha256
from anomaly_detection.cnn_research import read_partition

folder=ROOT/'reports/supersimplenet'
r=json.loads((folder/'reference_feasibility.json').read_text())
trials=[json.loads((folder/f'runs/{name}-result.json').read_text()) for name in ['04-pilot-lite-bottle-s2026-v1','04-pilot-lite-noise05-bottle-s2026-v1']]
weights=ROOT/'checkpoints/pretrained/wide_resnet50_2-95faca4d.pth'
report=dict(status='architecture and pilots complete; development pending; no official test access',
    reference=r,pilots=[{k:t[k] for k in ['experiment_id','configuration','development','resources','parameters','checkpoint_replay_exact','wall_seconds','adapter_relative_weight_drift']} for t in trials],
    weight_acquisition=dict(url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',sha256=sha256(weights),bytes=weights.stat().st_size,
        rationale='Official ICPR code uses ImageNet V1; existing WideResNet cache was V2 and was not substituted',storage='ignored project checkpoint directory'),
    decision='Reference excluded by measured workload; .05 Lite pilot improves synthetic discrimination but tiny probe uncertainty precludes quality claims. Proceed only to preregistered four-category development.',
    proposed_final_workload=dict(seeds=[2026,2027,2028],epochs=40,categories=15,
        lite_training_seconds_estimate=sum(h['train_epoch_seconds'] for h in trials[1]['history'])/20/4*40*3*sum((n+7)//8 for n in [147,158,155,196,186,275,173,154,187,224,162,42,151,173,168]),
        note='Estimate uses measured four-batch pilot epochs; IO, validation, cache extraction and native evaluation are additional. Final gate must use development measurements.'),
    core_source_sha256=sha256(ROOT/'src/anomaly_detection/supersimplenet.py'),notebook_sha256=sha256(ROOT/'notebooks/04_supersimplenet.ipynb'))
with (folder/'pilot_report.json').open('x') as stream:json.dump(report,stream,indent=2)
record(ROOT/'experiments',dict(experiment_id='04-reference-feasibility-s2026-v1',model_family='SuperSimpleNetReferenceFeasibility',categories=['bottle'],
    hypothesis='Reference-width normal-only feature discrimination fits safely within measured compute and VRAM budgets',
    preprocessing='Whole RGB256, ImageNet V1, layers2/3 align64 before pool3',hyperparameters=dict(batch_size=4,steps=12,warmup_steps=2,noise_std=.015),seed=2026,
    environment=dict(weight_acquisition=report['weight_acquisition'],source_sha256=report['core_source_sha256']),
    training=dict(phase='feasibility',source_paths=read_partition('fit','bottle').iloc[:4][['path','sha256']].to_dict('records'),
        recording='Retrospective registration of completed measured feasibility; protocol was recorded before execution'),metrics=r,calibration={},
    paths=dict(report='reports/supersimplenet/reference_feasibility.json'),interpretation='Memory fits, projected training exceeds the prespecified 90-minute budget',
    decision='Reject reference width for broad study; retain complete measurements',status='completed'))
print(json.dumps(dict(decision=report['decision'],workload=report['proposed_final_workload']),indent=2))
