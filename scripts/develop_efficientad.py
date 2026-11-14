"""Execute the predeclared four-fit study. Never launches final benchmark."""
from pathlib import Path
import copy
import json
import os
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from torchvision.transforms import functional as TF
from anomaly_detection.data import sha256
from anomaly_detection.tracking import record,config_hash
from anomaly_detection.efficientad import EfficientAD,save_checkpoint,load_checkpoint
from anomaly_detection.efficientad_research import (configure,read_normals,normalized,proxy_edits,
    latency_audit,gpu_snapshot,collect_raw_maps,map_quantiles,score_maps,distribution,
    calibration_diagnostics,select_configuration)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/efficientad/development'
CKPT=ROOT/'checkpoints/efficientad/development'
PLAN=ROOT/'configs/efficientad_development.json'
ASSET=ROOT/'checkpoints/efficientad/assets/teacher_small.pth'
START=time.perf_counter()


def write_json(path,value):
    """Atomic report publication, preserving a complete prior report on interruption."""
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def event(identifier,hypothesis,parameters,status='started',**extra):
    item=json.loads((ROOT/'configs/experiment_template.json').read_text())
    item.update(experiment_id=identifier,model_family='EfficientAD-S reduced-budget variant',
        categories=parameters.get('categories',[]),hypothesis=hypothesis,
        preprocessing=dict(size=256,normalization='ImageNet'),hyperparameters=parameters,
        environment=dict(python=str(ROOT/'.venv/Scripts/python.exe'),torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True),
        training=dict(plan_sha256=sha256(PLAN),teacher_sha256=sha256(ASSET),source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in (Path(__file__),ROOT/'src/anomaly_detection/efficientad.py',ROOT/'src/anomaly_detection/efficientad_research.py')},
            partition_sha256={part:sha256(ROOT/f'reports/partitions/seed2026-{part}.csv') for part in ('fit','development','calibration')}),
        status=status)
    item.update(extra)
    return record(ROOT/'experiments',item)


def pilot_latency(plan):
    path=OUT/'latency.json'
    if path.exists(): return json.loads(path.read_text())
    identifier='06-latency-audit-v1'
    assert not (ROOT/f'experiments/{identifier}-start.json').exists(),'Incomplete latency audit; inspect before recovery'
    event(identifier+'-start','Separate resident compute, transfer and Windows scheduling latency',plan['latency'])
    model,_=load_checkpoint(ROOT/'checkpoints/efficientad/06-pilot-bottle-seed2026-v1.pt','cuda')
    rows,images=read_normals(ROOT,'bottle','fit')
    t=time.perf_counter()
    result=latency_audit(model,images[0],ROOT/'data/mvtec_ad'/rows.iloc[0].path,plan['latency'])
    result['elapsed_seconds']=time.perf_counter()-t
    result['pilot_checkpoint_sha256']=sha256(ROOT/'checkpoints/efficientad/06-pilot-bottle-seed2026-v1.pt')
    write_json(path,result)
    event(identifier+'-result','Separate resident compute, transfer and Windows scheduling latency',plan['latency'],status='completed',
        metrics={k:result[k] for k in ('resident_wall','resident_gpu_event','host_to_device','end_to_end')},paths=dict(results=path.relative_to(ROOT).as_posix()),
        interpretation='Same model semantics; CUDA event times include stream launch gaps. Telemetry snapshots do not prove a causal OS or power-state explanation.',decision='Use separated synchronized methods for subsequent reporting')
    print('Latency audit',result['resident_wall'],result['resident_gpu_event'],flush=True)
    del model
    return result


@torch.no_grad()
def probe(model,images):
    model.eval(); values=[]
    for x in images[:4]:
        values.append({k:float(v) for k,v in model.losses(normalized(x).cuda(),normalized(TF.adjust_brightness(x,1.1)).cuda()).items()})
    return {k:float(np.mean([v[k] for v in values])) for k in values[0]}


def fit(category,epochs,rows,images,dev_rows,dev_images,plan):
    identifier=f'06-dev-{category}-e{epochs}-seed2026-v1'
    result_path=OUT/f'{identifier}.json'; checkpoint=CKPT/f'{identifier}.pt'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        assert sha256(checkpoint)==result['checkpoint']['sha256']
        assert result['plan_sha256']==sha256(PLAN)
        assert (ROOT/f'experiments/{identifier}-result.json').exists()
        print('Reuse verified fit',identifier,flush=True)
        return result
    assert not (ROOT/f'experiments/{identifier}-start.json').exists(), 'Incomplete fit: inspect saved events/checkpoints before recovery'
    params=dict(categories=[category],epochs=epochs,steps=epochs*len(images),seed=2026,plan_sha256=sha256(PLAN))
    event(identifier+'-start','Full-epoch budget controls learning, normal score stability and paired synthetic discrimination',params)
    try:
        begin=time.perf_counter(); torch.manual_seed(2026); np.random.seed(2026)
        rng=np.random.default_rng(2026)
        model=EfficientAD().cuda()
        model.teacher.load_state_dict(torch.load(ASSET,map_location='cuda',weights_only=True))
        torch.cuda.synchronize(); prep_start=time.perf_counter()
        cache=CKPT/f'{category}-feature-statistics.pt'
        cache_identity=dict(teacher=sha256(ASSET),manifest=sha256(ROOT/'reports/partitions/seed2026-fit.csv'),category=category,core=sha256(ROOT/'src/anomaly_detection/efficientad.py'))
        if cache.exists():
            payload=torch.load(cache,weights_only=True)
            assert payload['identity']==cache_identity
            model.mean.copy_(payload['mean']); model.std.copy_(payload['std'])
            normalization_seconds=payload['normalization_seconds']; normalization_reused=True
        else:
            model.fit_feature_normalization(normalized(x).cuda() for x in images)
            torch.cuda.synchronize(); normalization_seconds=time.perf_counter()-prep_start
            torch.save(dict(identity=cache_identity,mean=model.mean.cpu(),std=model.std.cpu(),normalization_seconds=normalization_seconds),cache)
            normalization_reused=False
        torch.cuda.synchronize(); preparation_seconds=time.perf_counter()-begin
        before=probe(model,images)
        optimizer=torch.optim.Adam(list(model.student.parameters())+list(model.autoencoder.parameters()),lr=1e-4,weight_decay=1e-5)
        torch.cuda.reset_peak_memory_stats(); curves=[]; epoch_scores=[]; train_seconds=0.
        telemetry=[]; process=psutil.Process(); peak_rss=process.memory_info().rss
        for epoch in range(epochs):
            if time.perf_counter()-START > plan['development_wall_time_cap_hours']*3600:
                raise RuntimeError('Predeclared development wall-time cap exceeded')
            model.train(); torch.cuda.synchronize(); start=time.perf_counter()
            cpu_before=process.cpu_times(); totals={}; grad_max=0.
            for index in rng.permutation(len(images)):
                raw=images[index]
                factor=float(rng.uniform(.8,1.2))
                augmentation=(TF.adjust_brightness,TF.adjust_contrast,TF.adjust_saturation)[int(rng.integers(3))]
                x=normalized(raw).cuda(); aug=normalized(augmentation(raw,factor)).cuda()
                optimizer.zero_grad(set_to_none=True)
                losses=model.losses(x,aug)
                losses['total'].backward()
                gradients=[p.grad for p in model.parameters() if p.requires_grad]
                assert all(g is not None for g in gradients)
                assert bool(torch.stack([torch.isfinite(v).all() for v in list(losses.values())+gradients]).all())
                optimizer.step()
                for k,v in losses.items(): totals[k]=totals.get(k,0.)+float(v.detach())
            torch.cuda.synchronize(); seconds=time.perf_counter()-start; train_seconds+=seconds
            cpu_after=process.cpu_times(); peak_rss=max(peak_rss,process.memory_info().rss)
            # A fixed set of clean held-out normals tracks scores without refitting map scaling.
            model.eval()
            maps=collect_raw_maps(model,dev_images[:8])
            stability=[float(m.flatten(1).amax(1).median()) for m in maps]
            curve=dict(epoch=epoch+1,steps=(epoch+1)*len(images),seconds=seconds,
                **{k:v/len(images) for k,v in totals.items()},raw_local_dev_max_median=stability[0],raw_global_dev_max_median=stability[1])
            curves.append(curve); telemetry.append(dict(epoch=epoch+1,gpu=gpu_snapshot(),
                cpu_seconds=(cpu_after.user+cpu_after.system)-(cpu_before.user+cpu_before.system)))
            pd.DataFrame(curves).to_csv(OUT/f'{identifier}-epochs.csv',index=False,lineterminator='\n')
            print(identifier,'epoch',epoch+1,'/',epochs,'seconds',round(seconds,2),'loss',round(curve['total'],4),flush=True)
        after=probe(model,images)
        model.eval(); diag_start=time.perf_counter()
        dev_pairs=collect_raw_maps(model,dev_images); q=map_quantiles(dev_pairs)
        assert (q[:,1]>q[:,0]).all()
        model.map_quantiles.copy_(q.to('cuda'))
        dev_scores=score_maps(dev_pairs,q)
        assert all(np.isfinite(a).all() for a in dev_scores.values())
        chosen=np.linspace(0,len(images)-1,16,dtype=int)
        proxy_images=[]; proxy_names=[]; proxy_paths=[]
        for j,index in enumerate(chosen):
            proxy_images.append(images[index]); proxy_names.append('normal'); proxy_paths.append(rows.iloc[index].path)
            for name,img in proxy_edits(images[index],j).items():
                proxy_images.append(img); proxy_names.append(name); proxy_paths.append(rows.iloc[index].path)
        proxy_pairs=collect_raw_maps(model,proxy_images); proxy_scores=score_maps(proxy_pairs,q)
        labels=np.array([name!='normal' for name in proxy_names],int)
        proxy={}
        for name,values in proxy_scores.items():
            proxy[name]=dict(auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)),
                per_edit={edit:float(roc_auc_score(labels[np.isin(proxy_names,['normal',edit])],values[np.isin(proxy_names,['normal',edit])])) for edit in ('scratch','block','color_patch')},
                mean_paired_score_delta={edit:float(np.mean(values[np.array(proxy_names)==edit]-values[np.array(proxy_names)=='normal'])) for edit in ('scratch','block','color_patch')})
        dev_loss=probe(model,dev_images)
        # Leave-one-image-out map-scaling sensitivity, no extra model forwards.
        loo_quantiles=[]
        for i in range(len(dev_images)):
            loo_quantiles.append(map_quantiles(tuple(torch.cat((a[:i],a[i+1:])) for a in dev_pairs)).numpy())
        loo=np.stack(loo_quantiles)
        diagnostic_seconds=time.perf_counter()-diag_start
        serialization_start=time.perf_counter()
        save_checkpoint(checkpoint,model,dict(experiment_id=identifier,plan_sha256=sha256(PLAN),teacher_sha256=sha256(ASSET),epochs=epochs,category=category,seed=2026,final=False))
        checkpoint_info=dict(path=checkpoint.relative_to(ROOT).as_posix(),sha256=sha256(checkpoint),bytes=checkpoint.stat().st_size)
        # Integrity/reproducible inference; validates every completed fit before result event.
        loaded,_=load_checkpoint(checkpoint,'cuda')
        with torch.no_grad():
            test=normalized(dev_images[0]).cuda()
            assert torch.equal(model(test)[0],loaded(test)[0])
        del loaded
        serialization_validation_seconds=time.perf_counter()-serialization_start
        dev_table=pd.DataFrame(dict(path=dev_rows.path,**dev_scores))
        proxy_table=pd.DataFrame(dict(path=proxy_paths,edit=proxy_names,label=labels,**proxy_scores))
        dev_table.to_csv(OUT/f'{identifier}-development.csv',index=False,lineterminator='\n')
        proxy_table.to_csv(OUT/f'{identifier}-proxies.csv',index=False,lineterminator='\n')
        result=dict(experiment_id=identifier,category=category,epochs=epochs,seed=2026,fit_n=len(images),development_n=len(dev_images),steps=epochs*len(images),
            plan_sha256=sha256(PLAN),finite=True,probe_before=before,probe_after=after,development_probe_losses=dev_loss,
            training_seconds=train_seconds,seconds_per_image_epoch=train_seconds/(epochs*len(images)),
            preparation_seconds=preparation_seconds,normalization_seconds=normalization_seconds,normalization_reused=normalization_reused,
            diagnostic_seconds=diagnostic_seconds,serialization_validation_seconds=serialization_validation_seconds,
            elapsed_seconds=time.perf_counter()-begin,curves=curves,telemetry=telemetry,proxy=proxy,
            development_score_distributions={k:distribution(v) for k,v in dev_scores.items()},map_quantiles=q.tolist(),
            leave_one_development_image_out_quantile_range=dict(min=loo.min(0).tolist(),max=loo.max(0).tolist()),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_bytes=max(peak_rss,process.memory_info().rss),process_peak_working_set_bytes=getattr(process.memory_info(),'peak_wset',None),
            checkpoint=checkpoint_info)
        result['artifacts']=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p),bytes=p.stat().st_size) for p in (OUT/f'{identifier}-epochs.csv',OUT/f'{identifier}-development.csv',OUT/f'{identifier}-proxies.csv')]
        write_json(result_path,result)
        event(identifier+'-result','Full-epoch budget controls learning, normal score stability and paired synthetic discrimination',params,status='completed',
            metrics=dict(proxy=proxy,probe_before=before,probe_after=after,training_seconds=train_seconds),
            paths=dict(checkpoint=checkpoint_info,results=result_path.relative_to(ROOT).as_posix()),
            interpretation='Training losses and paired fitting-background proxy sensitivity must be assessed jointly; not real-defect performance.',decision='Retain for predeclared selection')
        del model,optimizer
        return result
    except BaseException:
        trace=traceback.format_exc()
        write_json(OUT/f'{identifier}-failure.json',dict(traceback=trace))
        event(identifier+'-failure','Preserve failed bounded fit',params,status='failed',interpretation=trace,decision='Inspect before recovery')
        raise


def evaluate_overhead_probe():
    """Measure exact native metric sweep using synthetic truth, never test pixels."""
    from anomaly_detection.pixel_metrics import exact_pixel_metrics
    import scipy.ndimage as ndi
    rng=np.random.default_rng(8606); side=1024
    truth=np.zeros((side,side),np.uint8); truth[300:400,400:500]=1
    scores=rng.normal(0,.1,(side,side)).astype(np.float32)+truth*.1
    weights=truth.astype(np.float64)/truth.sum()
    t=time.perf_counter(); metric=exact_pixel_metrics(scores.ravel(),truth.ravel(),weights.ravel(),1)
    return dict(seconds=time.perf_counter()-t,pixels=side*side,metrics=metric,
                purpose='CPU exact-native-metric cost probe on synthetic arrays only; not a model metric')


def main():
    configure(ROOT); OUT.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PLAN.read_text()); assert sha256(ASSET)==plan['teacher_sha256']
    latency=pilot_latency(plan)
    results=[]; load_times={}
    for category in plan['categories']:
        t=time.perf_counter(); rows,images=read_normals(ROOT,category,'fit'); dev_rows,dev_images=read_normals(ROOT,category,'development')
        load_times[category]=time.perf_counter()-t
        for epochs in plan['epoch_budgets']:
            results.append(fit(category,epochs,rows,images,dev_rows,dev_images,plan))
    selection_path=OUT/'selection.json'
    selection=select_configuration(results,plan)
    selection.update(plan_sha256=sha256(PLAN),threshold_policy=plan['threshold_policy'],
        rationale='Apply predeclared proxy eligibility, shorter-budget tolerance and combined-map preference. Calibration data not read before this file is locked.')
    if selection_path.exists(): assert json.loads(selection_path.read_text())==selection
    else: write_json(selection_path,selection)
    # Selection is durably fixed before the first reserved calibration image is opened.
    selected=selection['selected']; calibration=[]
    for category in plan['categories']:
        path=OUT/f'calibration-{category}.json'
        if path.exists(): calibration.append(json.loads(path.read_text())); continue
        identifier=f'06-calibration-{category}-v1'
        event(identifier+'-start','Check predeclared reserved-normal threshold uncertainty after configuration selection',dict(categories=[category],selection_sha256=sha256(selection_path)))
        selected_run=next(r for r in results if r['category']==category and r['epochs']==selected['epochs'])
        t=time.perf_counter(); model,_=load_checkpoint(ROOT/selected_run['checkpoint']['path'],'cuda')
        rows,images=read_normals(ROOT,category,'calibration')
        torch.cuda.synchronize(); infer_start=time.perf_counter()
        pairs=collect_raw_maps(model,images)
        q=model.map_quantiles.cpu(); scores=score_maps(pairs,q)[selected['score']]
        torch.cuda.synchronize(); inference_seconds=time.perf_counter()-infer_start
        dev=pd.read_csv(OUT/f"{selected_run['experiment_id']}-development.csv")[selected['score']].to_numpy()
        diagnosis=calibration_diagnostics(scores,dev,plan['threshold_policy'])
        diagnosis.update(category=category,selection_sha256=sha256(selection_path),checkpoint_sha256=selected_run['checkpoint']['sha256'],
            calibration_inference_seconds=inference_seconds,elapsed_seconds=time.perf_counter()-t)
        table=OUT/f'calibration-{category}-scores.csv'
        pd.DataFrame(dict(path=rows.path,score=scores)).to_csv(table,index=False,lineterminator='\n')
        diagnosis['scores_sha256']=sha256(table)
        write_json(path,diagnosis); calibration.append(diagnosis)
        event(identifier+'-result','Check predeclared reserved-normal threshold uncertainty after configuration selection',dict(categories=[category],selection_sha256=sha256(selection_path)),
            status='completed',calibration=diagnosis,paths=dict(results=path.relative_to(ROOT).as_posix()),
            interpretation='Small calibration tails imply uncertain future FPR. Calibration findings do not alter configuration or threshold policy.',decision='Retain fixed higher-quantile strict-greater policy')
        del model
    metric_path=OUT/'metric-cost-probe.json'
    if metric_path.exists(): metric_probe=json.loads(metric_path.read_text())
    else: metric_probe=evaluate_overhead_probe(); write_json(metric_path,metric_probe)
    counts=pd.read_csv(ROOT/'reports/partition_counts.csv')
    # Only existing dimensions/count metadata; no official test pixels/labels/masks.
    inventory=pd.read_csv(ROOT/'outputs/eda/inventory.csv',usecols=['category','split','width','height'])
    native_pixels=inventory[inventory.split=='test'].assign(pixels=lambda f:f.width*f.height).groupby('category').pixels.sum().to_dict()
    selected_runs=[r for r in results if r['epochs']==selected['epochs']]
    train_rate=max(r['seconds_per_image_epoch'] for r in selected_runs)
    prep_rate=max((r['preparation_seconds']+r['normalization_seconds']*(r['normalization_reused']))/r['fit_n'] for r in results)
    map_rate=max(r['diagnostic_seconds']/r['development_n'] for r in results) # Includes proxy/LOO diagnostics: intentionally conservative.
    cal_rate=max(c['elapsed_seconds']/c['calibration_distribution']['n'] for c in calibration)
    inference_rate=latency['end_to_end']['p95_ms']/1000
    pixel_rate=metric_probe['seconds']/metric_probe['pixels']
    workloads=[]
    for row in counts.itertuples():
        pixel_count=native_pixels[row.category]
        metric_seconds=pixel_count*pixel_rate*max(1,np.log2(pixel_count)/np.log2(metric_probe['pixels']))
        components=dict(training=row.fit*selected['epochs']*train_rate,
            preparation=row.fit*prep_rate+max(load_times.values()),
            map_scaling=row.development*map_rate,calibration=row.calibration*cal_rate,
            inference=row.test*inference_rate,native_metrics=metric_seconds,
            serialization_validation=max(r['serialization_validation_seconds'] for r in results),
            fixed_cpu_artifact_overhead=60.)
        workloads.append(dict(category=row.category,models=3,per_model_seconds=components,
            conservative_three_seed_seconds=3*2*sum(components.values())))
    total=sum(r['conservative_three_seed_seconds'] for r in workloads)
    training_total=3*sum(counts.fit)*selected['epochs']*train_rate
    gate=dict(recommendation='GO' if selection['eligible'] and total<=12*3600 else 'NO-GO',
        target_hours=12,estimated_conservative_total_elapsed_hours=total/3600,
        estimated_training_hours_no_margin=training_total/3600,models=45,categories=15,seeds=[2026,2027,2028],
        selected_configuration=selected,selection_eligible=selection['eligible'],
        methodology='Slower measured selected-budget full-epoch rate, p95 decode-through-CPU-map inference, synthetic native-metric CPU cost scaled by N log N, 60s/model fixed overhead, factor-two total margin. Conservative sequential elapsed estimate includes CPU work; not measured final GPU-hours.',
        workload=workloads,metric_probe=metric_probe,
        development_training_seconds=sum(r['training_seconds'] for r in results),
        development_fit_elapsed_seconds=sum(r['elapsed_seconds'] for r in results),
        latency_audit_seconds=latency['elapsed_seconds'],data_load_seconds=load_times,
        calibration_elapsed_seconds=sum(c['elapsed_seconds'] for c in calibration),
        current_invocation_elapsed_seconds=time.perf_counter()-START,
        limitations=['Two categories, one development seed; no assurance of real-defect discrimination','Unseen sustained workload, OS and power variability','No ImageNet penalty','Small calibration partitions; nominal 5% is not a guarantee','Metric probe is synthetic and cost-scaled; native final evaluation has not run'],
        final_benchmark_started=False)
    write_json(OUT/'feasibility.json',gate)
    print(json.dumps({k:v for k,v in gate.items() if k not in ('workload','metric_probe')},indent=2),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
