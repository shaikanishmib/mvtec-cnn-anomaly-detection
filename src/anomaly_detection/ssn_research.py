"""Normal-only SuperSimpleNet feature caches, explicit fitting and development."""
import copy,json,time,traceback,math,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from .supersimplenet import SSNExtractor,SuperSimpleHeads,training_loss,optimizer_for,anomaly_maps,image_scores
from .cnn_research import ROOT,read_partition,require_fitting,inner_fit_split,load_resized
from .patchcore_research import Resources
from .basic_cnn import seed_everything
from .data import sha256
from .tracking import record,config_hash
from .metrics import binary_metrics,calibrate_normal


class FeatureSet:
    """Per-image immutable float32 caches avoid duplicate extraction across splits.

    SSN upsamples BEFORE pooling; PatchCore feature caches cannot be substituted.
    Random-access batches use bounded CPU stacks, never an all-category GPU tensor.
    """
    def __init__(self,paths,metadata):self.paths=paths;self.metadata=metadata
    def __len__(self):return len(self.paths)
    def batch(self,indices,device='cuda'):
        values=np.stack([np.load(self.paths[int(i)],mmap_mode='r') for i in indices])
        return torch.from_numpy(values).to(device)


@torch.no_grad()
def feature_cache(extractor,rows,batch_size=4):
    if rows.empty or rows.category.nunique()!=1:raise ValueError('Single nonempty category cache required')
    profile=dict(backbone=extractor.backbone,weight_sha256=extractor.weight_sha256,
        source_sha256=sha256(ROOT/'src/anomaly_detection/supersimplenet.py'),
        preprocessing='PIL-RGB-bilinear-256 + V1 ImageNet normalization; whole image',
        aggregation='layer2/3 bilinear align_corners=True to 64, concatenate, then avg3 pad1',dtype='float32')
    key=config_hash(profile);folder=ROOT/'outputs/supersimplenet/features'/key
    folder.mkdir(parents=True,exist_ok=True)
    paths=[folder/(r.sha256+'.npy') for r in rows.itertuples()]
    missing=[];entries=[]
    for i,path in enumerate(paths):
        meta=path.with_suffix('.json')
        if meta.exists():
            data=json.loads(meta.read_text())
            if not path.exists() or sha256(path)!=data['sha256']:raise ValueError('Feature cache identity changed')
            entries.append(data)
        else:
            if path.exists():raise FileExistsError('Incomplete feature cache retained: '+str(path))
            missing.append(i);entries.append(None)
    with Resources() as resources:
        if missing:
            images=load_resized(rows,256,allow_test=rows.partition.eq('test').all())
            for start in range(0,len(missing),batch_size):
                indices=missing[start:start+batch_size]
                x=images[indices].cuda().float()/255
                features=extractor(x).cpu().numpy()
                for index,array in zip(indices,features):
                    path=paths[index];np.save(path,array,allow_pickle=False)
                    entry=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),
                        shape=list(array.shape),bytes=path.stat().st_size,source_image=rows.path.iloc[index],
                        source_image_sha256=rows.sha256.iloc[index],profile=profile)
                    with path.with_suffix('.json').open('x') as stream:json.dump(entry,stream,indent=2)
                    entries[index]=entry
    metadata=dict(category=rows.category.iloc[0],profile=profile,images=entries,
        input_rows_sha256=config_hash(rows[['path','sha256']].to_dict('records')),
        cache_misses=len(missing),cache_hits=len(rows)-len(missing),resources=resources.result)
    return FeatureSet(paths,metadata)


def model_for(config):
    return SuperSimpleHeads(channels=config['channels'],seg_width=config['seg_width'],det_width=config['det_width'],
        adapter_trainable=config['adapter_trainable'],stop_grad=True)


@torch.inference_mode()
def predict(model,features,batch_size=8):
    model.eval();maps=[];scores=[];raw=[]
    for start in range(0,len(features),batch_size):
        x=features.batch(range(start,min(start+batch_size,len(features))))
        segmentation,detection=model(x)
        maps.append(anomaly_maps(segmentation).cpu());scores.append(detection.cpu());raw.append(segmentation.cpu())
    return torch.cat(maps),torch.cat(scores),torch.cat(raw)


@torch.inference_mode()
def development_metrics(model,normals,probes,config):
    model.eval();normal_maps,normal_scores,normal_raw=predict(model,normals,config['batch_size'])
    mask_gen=torch.Generator().manual_seed(9901)
    noise_gen=torch.Generator(device='cuda').manual_seed(9902)
    maps=[];scores=[];targets=[];labels=[];losses=[]
    for start in range(0,len(probes),config['batch_size']):
        x=probes.batch(range(start,min(start+config['batch_size'],len(probes))))
        seg,score,mask,y=model.synthetic_forward(x,mask_gen,noise_gen,noise_std=config['noise_std'],threshold=.2)
        loss,_=training_loss(seg,score,mask,y,segmentation_loss=config['loss'])
        losses.append((float(loss),len(y)))
        maps.append(anomaly_maps(seg).cpu());scores.append(score.cpu())
        targets.append(torch.nn.functional.interpolate(mask,size=(256,256),mode='nearest').cpu());labels.append(y.cpu())
    maps=torch.cat(maps);scores=torch.cat(scores);targets=torch.cat(targets);labels=torch.cat(labels)
    pixel_ap=float(average_precision_score(targets.numpy().ravel(),maps.numpy().ravel()))
    metrics={}
    for aggregation in ['head','max','top1pct']:
        threshold=calibrate_normal(image_scores(normal_maps,normal_scores,aggregation).numpy())
        value=binary_metrics(labels.numpy(),image_scores(maps,scores,aggregation).numpy(),threshold)
        value.update(pixel_ap=pixel_ap,normal_development_q95=threshold,
            objective=(value['auroc']+pixel_ap)/2-.25*max(0,value['fpr']-.05))
        metrics[aggregation]=value
    return dict(candidates=metrics,synthetic_loss=sum(x*n for x,n in losses)/sum(n for _,n in losses),
        proxy_images=len(labels),proxy_positives=int(labels.sum()),
        normal_map_focal=float(torch.nn.functional.softplus(normal_raw).mul(normal_raw.sigmoid().pow(4)).mean()),
        normal_head_focal=float(torch.nn.functional.softplus(normal_scores).mul(normal_scores.sigmoid().pow(4)).mean()))


def train_epoch(model,features,optimizer,shuffle,mask_gen,noise_gen,config):
    model.train();totals={};seen=0
    for indices in torch.randperm(len(features),generator=shuffle).split(config['batch_size']):
        x=features.batch(indices)
        optimizer.zero_grad(set_to_none=True)
        seg,score,mask,y=model.synthetic_forward(x,mask_gen,noise_gen,noise_std=config['noise_std'],threshold=.2)
        loss,parts=training_loss(seg,score,mask,y,segmentation_loss=config['loss'])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        values=dict(train_loss=float(loss.detach()),gradient_norm=float(norm),synthetic_fraction=float(y.mean()),
            target_pixel_fraction=float(mask.mean()),**parts)
        for key,value in values.items():totals[key]=totals.get(key,0)+value*len(x)
        seen+=len(x)
    return {key:value/seen for key,value in totals.items()}


def experiment(identifier,category,config,extractor,phase='development',pilot=False):
    output=ROOT/'reports/supersimplenet/runs';output.mkdir(parents=True,exist_ok=True)
    local=ROOT/'outputs/supersimplenet'/identifier;local.mkdir(parents=True,exist_ok=False)
    checkpoint=ROOT/'checkpoints/supersimplenet'/identifier/'last.pt';checkpoint.parent.mkdir(parents=True,exist_ok=False)
    entry=dict(experiment_id=identifier+'-start',model_family='SuperSimpleNetVariant',categories=[category],
        hypothesis=config['hypothesis'],preprocessing='Whole RGB PIL256, ImageNet V1 normalization; upscale layer2/3 to64 before pooling',
        hyperparameters=config,seed=config['seed'],environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            weight_sha256=extractor.weight_sha256,lock_sha256=sha256(ROOT/'requirements.lock.txt'),
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha256(ROOT/p) for p in ['src/anomaly_detection/supersimplenet.py','src/anomaly_detection/ssn_research.py']},
            partitions=json.loads((ROOT/'reports/dataset_validation.json').read_text())['partitions']),
        training=dict(phase=phase),metrics={},calibration={},paths=dict(local=local.relative_to(ROOT).as_posix()),
        interpretation='Only normal fitting/development and fitting-derived synthetic proxies; no calibration/test access',
        decision='retain every epoch; fixed last epoch',status='started')
    record(ROOT/'experiments',entry);started=time.perf_counter()
    try:
        rows=read_partition('fit',category);require_fitting(rows)
        if pilot:learning=rows.iloc[:32];probe_rows=rows.iloc[32:40]
        elif phase=='final':learning=rows;probe_rows=None
        else:learning,probe_rows=inner_fit_split(rows)
        features=feature_cache(extractor,learning,config['extraction_batch'])
        normals=feature_cache(extractor,read_partition('development',category).iloc[:16] if pilot else read_partition('development',category),config['extraction_batch'])
        probes=feature_cache(extractor,probe_rows,config['extraction_batch']) if probe_rows is not None else None
        seed_everything(config['seed']);model=model_for(config).cuda()
        initial_adapter=model.adapter.weight.detach().cpu().clone()
        optimizer=optimizer_for(model)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(config['epochs']*.8),int(config['epochs']*.9)],gamma=.4)
        shuffle=torch.Generator().manual_seed(config['seed'])
        mask_gen=torch.Generator().manual_seed(config['seed']+1)
        noise_gen=torch.Generator(device='cuda').manual_seed(config['seed']+2)
        history=[]
        with Resources() as resources:
            for epoch in range(1,config['epochs']+1):
                tick=time.perf_counter();values=train_epoch(model,features,optimizer,shuffle,mask_gen,noise_gen,config)
                torch.cuda.synchronize();values['train_epoch_seconds']=time.perf_counter()-tick
                if probes is not None and (epoch%config['validation_every']==0 or epoch==config['epochs']):
                    dev=development_metrics(model,normals,probes,config)
                    values.update(proxy_objective=dev['candidates']['head']['objective'],proxy_image_auroc=dev['candidates']['head']['auroc'],
                        proxy_pixel_ap=dev['candidates']['head']['pixel_ap'],proxy_loss=dev['synthetic_loss'],
                        normal_map_focal=dev['normal_map_focal'],normal_head_focal=dev['normal_head_focal'])
                scheduler.step();values.update(epoch=epoch,elapsed_seconds=time.perf_counter()-started)
                history.append(values)
                pd.DataFrame(history).to_csv(output/(identifier+'-history.csv'),index=False,lineterminator='\n')
                print(json.dumps(dict(experiment=identifier,**values)),flush=True)
                if time.perf_counter()-started>1800 or torch.cuda.max_memory_allocated()>6*2**30:raise RuntimeError('Per-run 30-minute/6GiB budget exceeded')
        state=dict(state_dict=model.state_dict(),configuration=config,category=category,phase=phase,epoch=config['epochs'],
            weight_sha256=extractor.weight_sha256,fit_rows_sha256=config_hash(learning[['path','sha256']].to_dict('records')))
        torch.save(state,checkpoint)
        torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),shuffle_rng=shuffle.get_state(),
            mask_rng=mask_gen.get_state(),noise_rng=noise_gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),checkpoint.parent/'training-state.pt')
        model.eval();sample=normals.batch([0,1]);before=model(sample)
        restored=model_for(config).cuda().eval();restored.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['state_dict'])
        with torch.no_grad():after=restored(sample)
        for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
        drift=float((model.adapter.weight.detach().cpu()-initial_adapter).norm()/initial_adapter.norm())
        result=dict(experiment_id=identifier,category=category,phase=phase,configuration=config,history=history,
            development=dev if probes is not None else {},fit_normals=len(learning),probe_sources=0 if probes is None else len(probe_rows),
            checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha256(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
            history_path=(output/(identifier+'-history.csv')).relative_to(ROOT).as_posix(),history_sha256=sha256(output/(identifier+'-history.csv')),
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            resources=resources.result,feature_caches=dict(fit=features.metadata,normal_development=normals.metadata,probes=None if probes is None else probes.metadata),
            adapter_relative_weight_drift=drift,checkpoint_replay_exact=True,wall_seconds=time.perf_counter()-started)
        with (output/(identifier+'-result.json')).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        event=copy.deepcopy(entry);event.update(experiment_id=identifier+'-result',status='completed',metrics=result,
            decision='await measured bounded comparison' if phase!='final' else 'frozen model fitted')
        record(ROOT/'experiments',event);return result
    except BaseException as exc:
        (local/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        failed=copy.deepcopy(entry);failed.update(experiment_id=identifier+'-failure',status='failed',decision='preserve and diagnose',interpretation=str(exc))
        record(ROOT/'experiments',failed);raise
