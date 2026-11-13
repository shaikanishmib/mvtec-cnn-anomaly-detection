"""Normal-only CUDA extraction replay, backward and exact durable recovery."""
import os,sys,json,copy
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from anomaly_detection.supersimplenet import SSNExtractor,optimizer_for
from anomaly_detection.ssn_research import feature_cache,model_for,train_epoch
from anomaly_detection.ssn_final import save_progress,restore_progress
from anomaly_detection.cnn_research import read_partition,require_fitting,load_resized
from anomaly_detection.basic_cnn import seed_everything
from anomaly_detection.data import sha256
from anomaly_detection.tracking import record,config_hash
torch.set_num_threads(4);seed_everything(2026);torch.backends.cuda.matmul.allow_tf32=False
identifier='04-cuda-durable-verification-v1';output=ROOT/'reports/supersimplenet/cuda_verification.json'
assert not output.exists()
rows=read_partition('fit','bottle').iloc[:8];require_fitting(rows);extractor=SSNExtractor().cuda()
features=feature_cache(extractor,rows,8);images=load_resized(rows,256).cuda().float()/255
with torch.no_grad():
    first=extractor(images);second=extractor(images);cached=features.batch(range(8))
torch.testing.assert_close(first,second,rtol=0,atol=0)
torch.testing.assert_close(first,cached,rtol=1e-5,atol=1e-6)
config=dict(channels=384,seg_width=256,det_width=32,adapter_trainable=True,batch_size=8,noise_std=.05,loss='margin_focal')
entry=dict(experiment_id=identifier+'-start',model_family='SuperSimpleNetVerification',categories=['bottle'],
    hypothesis='Frozen extraction is repeatable and durable GPU recovery reproduces the next complete optimization epoch exactly',
    preprocessing='Fitting-normal RGB256 and same compatible per-image cache',hyperparameters=config,seed=2026,
    environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),source_sha256=sha256(Path(__file__)),weight_sha256=extractor.weight_sha256),
    training=dict(phase='correctness only; not a selection candidate'),metrics={},calibration={},paths={},
    interpretation='Eight fitting normals; no calibration or test access',decision='Verify before freeze',status='started')
record(ROOT/'experiments',entry)
def setup():
    model=model_for(config).cuda();optimizer=optimizer_for(model)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[2],gamma=.4)
    generators=dict(shuffle=torch.Generator().manual_seed(1),mask=torch.Generator().manual_seed(2),noise=torch.Generator(device='cuda').manual_seed(3))
    return model,optimizer,scheduler,generators
def epoch(model,opt,sch,gen):
    values=train_epoch(model,features,opt,gen['shuffle'],gen['mask'],gen['noise'],config);sch.step();return values
model,opt,sch,gen=setup();history=[epoch(model,opt,sch,gen)]
path=ROOT/'checkpoints/supersimplenet'/identifier/'progress.pt';path.parent.mkdir(parents=True,exist_ok=False)
identity=dict(category='bottle',rows_sha256=config_hash(rows[['path','sha256']].to_dict('records')),configuration=config)
save_progress(path,model,opt,sch,gen,history,identity,{})
expected=epoch(model,opt,sch,gen);resumed,ropt,rsch,rgen=setup()
restore_progress(path,identity,resumed,ropt,rsch,rgen);actual=epoch(resumed,ropt,rsch,rgen)
assert expected==actual
for key,value in model.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
model.eval();resumed.eval()
with torch.no_grad():
    a=model(cached);b=resumed(cached)
for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
result=dict(feature_shape=list(first.shape),extraction_repeat_exact=True,cache_replay_max_abs_difference=float((first-cached).abs().max()),
    gpu_forward_backward_finite=True,durable_next_epoch_exact=True,clean_inference_exact=True,
    first_epoch=history[0],second_epoch=actual,checkpoint_sha256=sha256(path),checkpoint=path.relative_to(ROOT).as_posix(),
    segmentation_shape=list(a[0].shape),image_score_shape=list(a[1].shape))
with output.open('x') as stream:json.dump(result,stream,indent=2)
entry.update(experiment_id=identifier+'-result',metrics=result,status='completed',decision='GPU correctness gate passed')
record(ROOT/'experiments',entry);print(json.dumps(result,indent=2))
