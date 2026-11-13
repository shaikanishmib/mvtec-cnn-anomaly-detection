"""Durable frozen fitting: identical epoch computation, recoverable RNG state."""
import json,time,copy,os,traceback,subprocess
from datetime import datetime,timezone
import torch,pandas as pd
from .ssn_research import ROOT,feature_cache,model_for,train_epoch
from .supersimplenet import optimizer_for
from .cnn_research import read_partition,require_fitting
from .basic_cnn import seed_everything
from .patchcore_research import Resources
from .data import sha256
from .tracking import config_hash,record


def save_progress(path,model,optimizer,scheduler,generators,history,identity,resources):
    """Atomic latest-state replacement; preserve an interrupted temporary file."""
    temporary=path.with_suffix('.pending.pt')
    if temporary.exists():
        retained=path.with_name('interrupted-write-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.pt')
        assert temporary.resolve().is_relative_to(path.parent.resolve()) and retained.resolve().is_relative_to(path.parent.resolve())
        temporary.rename(retained)
    state=dict(identity=identity,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        generators={k:g.get_state() for k,g in generators.items()},torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,history=history,resources=resources)
    torch.save(state,temporary)
    assert path.resolve().is_relative_to((ROOT/'checkpoints/supersimplenet').resolve())
    os.replace(temporary,path)


def restore_progress(path,identity,model,optimizer,scheduler,generators):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['identity']!=identity:raise ValueError('Durable checkpoint configuration/source/partition identity mismatch')
    model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    for key,generator in generators.items():generator.set_state(state['generators'][key])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:torch.cuda.set_rng_state(state['cuda_rng'])
    return state


def fit_frozen(identifier,category,config,extractor,freeze_sha256):
    frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())
    assert sha256(ROOT/'configs/supersimplenet_frozen.json')==freeze_sha256
    assert dict(frozen['configuration'],seed=config['seed'])==config and config['seed'] in frozen['seeds']
    for path,value in frozen['source_sha256'].items():assert sha256(ROOT/path)==value,path
    assert extractor.weight_sha256==frozen['weight_sha256'] and category in frozen['categories']
    output=ROOT/'reports/supersimplenet/runs';output.mkdir(parents=True,exist_ok=True)
    result_path=output/(identifier+'-result.json')
    if result_path.exists():return json.loads(result_path.read_text())
    local=ROOT/'outputs/supersimplenet'/identifier;local.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'checkpoints/supersimplenet'/identifier/'last.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    progress=checkpoint.parent/'progress.pt';history_path=output/(identifier+'-history.csv')
    rows=read_partition('fit',category);require_fitting(rows)
    identity=dict(category=category,configuration=config,freeze_sha256=freeze_sha256,
        fit_rows_sha256=config_hash(rows[['path','sha256']].to_dict('records')))
    reservation=ROOT/f'experiments/{identifier}-start.json'
    attempt=1+len(list((ROOT/'experiments').glob(identifier+'-resume-*.json')))
    if reservation.exists():
        if not progress.exists():raise RuntimeError('Incomplete fit has no durable checkpoint; preserve and explicitly recover')
        entry=json.loads(reservation.read_text())
        for key in ['recorded_at','config_sha256']:entry.pop(key)
        resumed=copy.deepcopy(entry);resumed.update(experiment_id=identifier+f'-resume-{attempt:02d}',
            interpretation='Resume exact model, optimizer, scheduler and all RNG states; completed epochs are not repeated',status='resumed')
        if history_path.exists():
            retained=history_path.with_name(identifier+f'-history-before-resume-{attempt:02d}.csv')
            with retained.open('xb') as stream:stream.write(history_path.read_bytes())
            resumed['paths']['retained_pre_resume_history']=retained.relative_to(ROOT).as_posix()
        record(ROOT/'experiments',resumed)
    else:
        entry=dict(experiment_id=identifier+'-start',model_family='SuperSimpleNetLite',categories=[category],hypothesis=config['hypothesis'],
            preprocessing='Whole RGB256, ImageNet V1; layer2/3 aligned64 before pooling; no RGB training corruption',hyperparameters=config,seed=config['seed'],
            environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),weight_sha256=extractor.weight_sha256,
                source_sha256=frozen['source_sha256'],git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),
            training=dict(phase='final',freeze_sha256=freeze_sha256),metrics={},calibration={},paths=dict(local=local.relative_to(ROOT).as_posix()),
            interpretation='Frozen normal-only fit; original development normals used solely for checkpoint replay',decision='Fixed final epoch; persist recovery after every epoch',status='started')
        record(ROOT/'experiments',entry)
    started=time.perf_counter()
    try:
        features=feature_cache(extractor,rows,config['extraction_batch'])
        normals=feature_cache(extractor,read_partition('development',category),config['extraction_batch'])
        seed_everything(config['seed']);model=model_for(config).cuda();initial=model.adapter.weight.detach().cpu().clone()
        optimizer=optimizer_for(model)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(.8*config['epochs']),int(.9*config['epochs'])],gamma=.4)
        generators=dict(shuffle=torch.Generator().manual_seed(config['seed']),mask=torch.Generator().manual_seed(config['seed']+1),
            noise=torch.Generator(device='cuda').manual_seed(config['seed']+2))
        history=[];previous={}
        if progress.exists():
            restored=restore_progress(progress,identity,model,optimizer,scheduler,generators)
            history=restored['history'];previous=restored['resources']
        with Resources() as monitor:
            for epoch in range(len(history)+1,config['epochs']+1):
                tick=time.perf_counter()
                values=train_epoch(model,features,optimizer,generators['shuffle'],generators['mask'],generators['noise'],config)
                torch.cuda.synchronize();values['train_epoch_seconds']=time.perf_counter()-tick
                scheduler.step();values['epoch']=epoch;history.append(values)
                pd.DataFrame(history).to_csv(history_path,index=False,lineterminator='\n')
                partial=dict(seconds=previous.get('seconds',0)+time.perf_counter()-monitor.started,
                    cpu_peak_rss_bytes=max(previous.get('cpu_peak_rss_bytes',0),monitor.peak),
                    peak_gpu_allocated_bytes=max(previous.get('peak_gpu_allocated_bytes',0),torch.cuda.max_memory_allocated()),
                    peak_gpu_reserved_bytes=max(previous.get('peak_gpu_reserved_bytes',0),torch.cuda.max_memory_reserved()))
                save_progress(progress,model,optimizer,scheduler,generators,history,identity,partial)
                print(json.dumps(dict(experiment=identifier,**values)),flush=True)
                if torch.cuda.max_memory_allocated()>6*2**30:raise RuntimeError('Frozen fit exceeded6GiB allocated VRAM guard')
        resources=monitor.result
        resources['seconds']+=previous.get('seconds',0)
        for key in ['cpu_peak_rss_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes']:
            resources[key]=max(resources[key],previous.get(key,0))
        state=dict(state_dict=model.state_dict(),configuration=config,category=category,phase='final',epoch=config['epochs'],
            weight_sha256=extractor.weight_sha256,fit_rows_sha256=identity['fit_rows_sha256'])
        if checkpoint.exists():raise FileExistsError('Preserve existing final checkpoint; completion needs explicit reconciliation')
        torch.save(state,checkpoint)
        restored=model_for(config).cuda().eval();restored.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['state_dict'])
        model.eval()
        with torch.no_grad():
            sample=normals.batch([0,1]);before=model(sample);after=restored(sample)
        for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
        result=dict(experiment_id=identifier,category=category,phase='final',configuration=config,history=history,development={},fit_normals=len(rows),probe_sources=0,
            checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha256(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
            history_path=history_path.relative_to(ROOT).as_posix(),history_sha256=sha256(history_path),
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            resources=resources,feature_caches=dict(fit=features.metadata,normal_development=normals.metadata,probes=None),
            adapter_relative_weight_drift=float((model.adapter.weight.detach().cpu()-initial).norm()/initial.norm()),checkpoint_replay_exact=True,
            wall_seconds=time.perf_counter()-started,durable_progress=dict(path=progress.relative_to(ROOT).as_posix(),sha256=sha256(progress),resume_attempts=attempt-1),
            timing_scope='Active final invocation wall time; training resources sum active recovered segments, excluding interruption downtime')
        with result_path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        event=copy.deepcopy(entry);event.update(experiment_id=identifier+'-result',status='completed',metrics=result,decision='Frozen category fit complete; no test inspection')
        record(ROOT/'experiments',event);return result
    except BaseException as exc:
        (local/f'failure-attempt-{attempt:02d}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        event=copy.deepcopy(entry);event.update(experiment_id=identifier+f'-failure-{attempt:02d}',status='failed',interpretation=str(exc),decision='Preserve exact progress and diagnose; no automatic retuning')
        record(ROOT/'experiments',event);raise
