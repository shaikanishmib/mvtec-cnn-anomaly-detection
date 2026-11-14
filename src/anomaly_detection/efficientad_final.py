"""Durable frozen EfficientAD training/evaluation. No configuration selection."""
from pathlib import Path
from datetime import datetime,timezone
import copy
import gc
import json
import os
import shutil
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
import psutil
from PIL import Image
from scipy.ndimage import label
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from .data import sha256,CATEGORIES
from .tracking import record,config_hash
from .metrics import binary_metrics,calibrate_normal
from .pixel_metrics import exact_pixel_metrics,histogram_metrics
from .cnn_evaluation import bootstrap_evaluation
from .patchcore_research import Resources
from .efficientad import EfficientAD,load_checkpoint,save_checkpoint
from .efficientad_research import configure,read_normals,normalized,collect_raw_maps,map_quantiles,score_maps,gpu_snapshot
from .efficientad_calibration import qualified_calibration

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/efficientad/final'
LOCAL=ROOT/'outputs/efficientad/final'
CHECKPOINT=ROOT/'checkpoints/efficientad/final'
FROZEN=ROOT/'configs/efficientad_reduced_frozen.json'
FREEZE='d728cf544ebc6f0c3d012adbca044a7aa5ddf4c0ced2e3210bef19b77a71d321'
METRICS=['image_auroc','image_ap','pixel_auroc','pixel_ap','au_pro','recall','fpr','precision','f1']


def utc(): return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(pending,path)


def atomic_torch(path,value):
    pending=path.with_suffix('.pending.pt');torch.save(value,pending);os.replace(pending,path)


def freeze():
    assert sha256(FROZEN)==FREEZE,'Frozen configuration changed'
    f=json.loads(FROZEN.read_text())
    for p,h in f['source_sha256'].items(): assert sha256(ROOT/p)==h,p
    for part,h in f['partition_sha256'].items(): assert sha256(ROOT/f'reports/partitions/seed2026-{part}.csv')==h,part
    assert sha256(ROOT/f['teacher']['path'])==f['teacher']['sha256']
    assert f['categories']==list(CATEGORIES) and f['epochs']==9 and f['score']=='local_max'
    return f


def artifact(path):
    path=Path(path)
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha256(path))


def check_artifact(item):
    p=ROOT/item['path']
    assert p.is_file() and p.stat().st_size==item['bytes'] and sha256(p)==item['sha256'],str(p)


def read_rows(partition,category):
    assert partition in ('fit','development','calibration','test')
    rows=pd.read_csv(ROOT/f'reports/partitions/seed2026-{partition}.csv')
    rows=rows[rows.category.eq(category)].sort_values('path').reset_index(drop=True)
    assert len(rows) and rows.partition.eq(partition).all()
    assert rows.split.eq('test' if partition=='test' else 'train').all()
    if partition!='test': assert rows.path.str.contains('/train/good/').all()
    return rows


def source_identity(rows): return config_hash(rows[['path','sha256']].to_dict('records'))


def registry(identifier,suffix,identity,status,**extra):
    event=dict(experiment_id=identifier+'-'+suffix,model_family='EfficientAD-S reduced-budget variant',categories=[identity['category']],
        hypothesis='Frozen 45-model confirmation; no test-guided selection',preprocessing=dict(size=256,normalization='ImageNet',maps='signed; native bilinear align_corners=False'),
        hyperparameters=identity,seed=identity['seed'],environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True),
        training=dict(freeze_sha256=FREEZE),metrics={},calibration={},paths={},interpretation=None,decision='frozen final evaluation',status=status)
    event.update(extra);record(ROOT/'experiments',event)


def identity_for(category,seed):
    f=freeze(); assert category in f['categories'] and seed in f['seeds']
    evaluator=REPORT/'evaluator_identity.json'
    if evaluator.exists():
        for p,h in json.loads(evaluator.read_text())['source_sha256'].items(): assert sha256(ROOT/p)==h,p
    return dict(category=category,seed=int(seed),freeze_sha256=FREEZE,teacher_sha256=f['teacher']['sha256'],
        partition_sha256=f['partition_sha256'],epochs=f['epochs'],score=f['score'],
        evaluator_sha256=sha256(evaluator) if evaluator.exists() else None)


def csv_exact(path,frame):
    frame=frame.copy()
    for column in frame.select_dtypes(include=['float32']).columns: frame[column]=frame[column].astype(np.float64)
    frame.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')
    return pd.read_csv(path,float_precision='round_trip')


def verify_complete(path,expected=None,deep=True):
    result=json.loads(Path(path).read_text())
    assert result['status']=='complete' and result['freeze_sha256']==FREEZE
    if expected is not None: assert result['identity']==expected
    if deep:
        for item in result['artifacts']: check_artifact(item)
    images=pd.read_csv(ROOT/result['images_csv'],float_precision='round_trip')
    assert images.path.tolist()==read_rows('test',result['category']).path.tolist()
    expected_metrics=binary_metrics(images.label,images.score,result['calibration']['threshold'])
    for key in ('recall','fpr','precision','f1'): assert np.isclose(result[key],expected_metrics[key],atol=1e-12)
    assert np.isclose(result['image_auroc'],expected_metrics['auroc'],atol=1e-12)
    assert np.isclose(result['image_ap'],expected_metrics['ap'],atol=1e-12)
    assert result['confusion_matrix']==expected_metrics['confusion_matrix']
    assert np.array_equal(images.predicted.to_numpy(),images.score.to_numpy()>result['calibration']['threshold'])
    assert all(np.isfinite(result[k]) and 0<=result[k]<=1+1e-9 for k in METRICS)
    return result


def choose_attempt(category,seed):
    expected=identity_for(category,seed)
    paths=sorted((REPORT/'runs').glob(f'06-final-{category}-s{seed}-a*-result.json'))
    for p in reversed(paths):
        try: return verify_complete(p,expected),None
        except (AssertionError,OSError,ValueError,KeyError): pass # preserve corrupt attempt; never overwrite it
    starts=sorted((ROOT/'experiments').glob(f'06-final-{category}-s{seed}-a*-start.json'))
    for p in reversed(starts):
        identifier=p.name.removesuffix('-start.json')
        if (REPORT/'runs'/f'{identifier}-result.json').exists(): continue
        entry=json.loads(p.read_text())
        if entry['hyperparameters']==expected: return None,identifier
    attempts=[int(p.name.split('-a')[-1].split('-')[0]) for p in starts]
    return None,f'06-final-{category}-s{seed}-a{max(attempts,default=0)+1:02d}'


def train(identifier,identity,f):
    category,seed=identity['category'],identity['seed']
    ckdir=CHECKPOINT/identifier;ckdir.mkdir(parents=True,exist_ok=True)
    local=LOCAL/identifier;local.mkdir(parents=True,exist_ok=True)
    checkpoint=ckdir/'model.pt';progress=ckdir/'progress.pt';stage=local/'trained.json'
    if stage.exists():
        saved=json.loads(stage.read_text()); assert saved['identity']==identity
        check_artifact(saved['checkpoint'])
        return saved
    rows,images=read_normals(ROOT,category,'fit')
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    model=EfficientAD().cuda()
    model.teacher.load_state_dict(torch.load(ROOT/f['teacher']['path'],map_location='cuda',weights_only=True))
    optimizer=torch.optim.Adam(list(model.student.parameters())+list(model.autoencoder.parameters()),lr=f['optimizer']['learning_rate'],weight_decay=f['optimizer']['weight_decay'])
    history=[];prep_start=time.perf_counter()
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);assert saved['identity']==identity
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        prep_seconds=saved['preparation_seconds']
    else:
        # Cache only teacher moments, keyed by full fitting and teacher/source identity, independent of seed.
        cache=CHECKPOINT/f'{category}-teacher-normalization.pt'
        cache_identity=dict(fit=source_identity(rows),teacher=f['teacher']['sha256'],source=f['source_sha256']['src/anomaly_detection/efficientad.py'])
        if cache.exists():
            saved=torch.load(cache,map_location='cpu',weights_only=True);assert saved['identity']==cache_identity
            model.mean.copy_(saved['mean']);model.std.copy_(saved['std'])
        else:
            model.fit_feature_normalization(normalized(x).cuda() for x in images)
            atomic_torch(cache,dict(identity=cache_identity,mean=model.mean.cpu(),std=model.std.cpu()))
        torch.cuda.synchronize();prep_seconds=time.perf_counter()-prep_start
    with Resources() as resources:
        for epoch in range(len(history)+1,f['epochs']+1):
            model.train();torch.cuda.synchronize();start=time.perf_counter();totals={}
            for index in rng.permutation(len(images)):
                raw=images[index];factor=float(rng.uniform(.8,1.2))
                augmentation=(TF.adjust_brightness,TF.adjust_contrast,TF.adjust_saturation)[int(rng.integers(3))]
                optimizer.zero_grad(set_to_none=True)
                losses=model.losses(normalized(raw).cuda(),normalized(augmentation(raw,factor)).cuda(),quantile=f['hard_quantile'])
                losses['total'].backward()
                gradients=[p.grad for p in model.parameters() if p.requires_grad]
                assert all(g is not None for g in gradients)
                assert bool(torch.stack([torch.isfinite(v).all() for v in list(losses.values())+gradients]).all())
                optimizer.step()
                for k,v in losses.items(): totals[k]=totals.get(k,0.)+float(v.detach())
            torch.cuda.synchronize()
            history.append(dict(epoch=epoch,steps=epoch*len(images),seconds=time.perf_counter()-start,**{k:v/len(images) for k,v in totals.items()}))
            atomic_torch(progress,dict(identity=identity,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),preparation_seconds=prep_seconds))
            csv_exact(REPORT/'runs'/f'{identifier}-history.csv',pd.DataFrame(history))
            print(identifier,'epoch',epoch,'/',f['epochs'],'seconds',round(history[-1]['seconds'],2),flush=True)
        model.eval();dev_rows,dev_images=read_normals(ROOT,category,'development')
        torch.cuda.synchronize();normalization_start=time.perf_counter()
        dev_pairs=collect_raw_maps(model,dev_images);q=map_quantiles(dev_pairs)
        assert (q[:,1]>q[:,0]).all();model.map_quantiles.copy_(q.cuda())
        dev_scores=score_maps(dev_pairs,q)[f['score']].astype(np.float64)
        normalization_seconds=time.perf_counter()-normalization_start
        start=time.perf_counter()
        save_checkpoint(checkpoint,model,dict(identity=identity,final=True,trained_epochs=len(history)))
        reloaded,meta=load_checkpoint(checkpoint,'cuda');assert meta['identity']==identity
        with torch.inference_mode():
            x=normalized(dev_images[0]).cuda();assert torch.equal(model(x)[0],reloaded(x)[0])
        serialization_seconds=time.perf_counter()-start
    saved=dict(identity=identity,checkpoint=artifact(checkpoint),fit_n=len(rows),fit_source_sha256=source_identity(rows),
        development_n=len(dev_rows),development_source_sha256=source_identity(dev_rows),development_scores=dev_scores.tolist(),
        training_seconds=sum(r['seconds'] for r in history),preparation_seconds=prep_seconds,map_normalization_seconds=normalization_seconds,
        serialization_seconds=serialization_seconds,history=history,resources=resources.result,map_quantiles=q.tolist(),trained_at_utc=utc())
    atomic_json(stage,saved)
    del model,optimizer,reloaded,dev_pairs,images,dev_images;gc.collect();torch.cuda.empty_cache()
    return saved


def histogram_indices(values,center,scale,bins=4096):
    transformed=.5+np.arctan((np.asarray(values,dtype=np.float64)-center)/scale)/np.pi
    return np.minimum((transformed*bins).astype(np.int64),bins-1)


def native_evaluation(rows,maps,scores,calibration,category,local):
    inventory=pd.read_csv(ROOT/'outputs/eda/inventory.csv',usecols=['path','height','width']).set_index('path')
    sizes=[(int(inventory.loc[p,'height']),int(inventory.loc[p,'width'])) for p in rows.path]
    count=sum(h*w for h,w in sizes)
    scratch=[local/f'native-{name}.dat' for name in ('scores','truth','regions')]
    vectors=[np.memmap(p,mode='w+',dtype=dtype,shape=(count,)) for p,dtype in zip(scratch,(np.float32,np.uint8,np.float64))]
    positive=np.zeros((len(rows),4096),np.uint32);negative=np.zeros_like(positive);region=np.zeros_like(positive,dtype=np.float64)
    region_counts=[];cursor=0;pixel_cm=np.zeros((2,2),np.int64);per_image=[]
    for i,(row,size) in enumerate(zip(rows.itertuples(),sizes)):
        native=F.interpolate(torch.from_numpy(np.array(maps[i:i+1,None])),size=size,mode='bilinear',align_corners=False).numpy()[0,0]
        if isinstance(row.mask,str):
            mask_path=ROOT/'data/mvtec_ad'/row.mask;assert sha256(mask_path)==row.mask_sha256
            with Image.open(mask_path) as im: truth=(np.asarray(im)>0).astype(np.uint8)
        else: truth=np.zeros(size,np.uint8)
        assert truth.shape==native.shape
        components,n=label(truth,structure=np.ones((3,3)))
        areas=np.bincount(components.ravel());areas[0]=1
        weight=np.where(truth,1/areas[components],0.)
        end=cursor+native.size
        for target,values in zip(vectors,(native,truth,weight)): target[cursor:end]=values.ravel()
        cursor=end;region_counts.append(n)
        idx=histogram_indices(native,calibration['histogram_center'],calibration['histogram_scale'])
        positive[i]=np.bincount(idx[truth.astype(bool)],minlength=4096)
        negative[i]=np.bincount(idx[~truth.astype(bool)],minlength=4096)
        region[i]=np.bincount(idx.ravel(),weights=weight.ravel(),minlength=4096)
        predicted=native>calibration['pixel_threshold'];pixel_cm+=np.bincount((2*truth+predicted).ravel(),minlength=4).reshape(2,2)
        overlap=predicted&truth.astype(bool);peak=np.unravel_index(native.argmax(),native.shape)
        per_image.append(dict(path=row.path,defect_pixels=int(truth.sum()),defect_fraction=float(truth.mean()),
            pixel_detected_defect_pixels=int(overlap.sum()),pixel_predicted_pixels=int(predicted.sum()),regions=int(n),
            native_peak_y=int(peak[0]),native_peak_x=int(peak[1]),peak_in_mask=bool(truth[peak]) if truth.any() else None))
    for v in vectors: v.flush()
    print(category,'exact native metric sweep',count,'pixels',flush=True)
    exact=exact_pixel_metrics(*vectors,sum(region_counts))
    labels=rows['mask'].notna().astype(int).to_numpy()
    intervals,boots=bootstrap_evaluation(labels,scores,calibration['threshold'],positive,negative,region,region_counts,category)
    intervals['pixel_ci_method']='Whole-image bootstrap; 4096 monotone arctangent bins from calibration center/IQR; primary native metrics exact'
    np.savez_compressed(local/'bootstrap_vectors.npz',**boots)
    np.savez_compressed(local/'native_image_histograms.npz',positive=positive,negative=negative,region=region,region_counts=region_counts)
    binned=histogram_metrics(positive.sum(0),negative.sum(0),region.sum(0),sum(region_counts))
    # Scratch only, confined to this run. Completed maps and bootstrap data are retained.
    for v in vectors: v._mmap.close()
    del vectors;gc.collect()
    for p in scratch:
        assert p.resolve().is_relative_to(local.resolve()) and p.name.startswith('native-');p.unlink()
    return dict(**exact,uncertainty=intervals,pixel_confusion_matrix=pixel_cm.tolist(),
        bootstrap_binned_pixel_point_estimates=dict(zip(('pixel_auroc','pixel_ap','au_pro'),binned))),per_image


@torch.inference_mode()
def benchmark(model,category,local):
    rows=read_rows('calibration',category);path=ROOT/'data/mvtec_ad'/rows.iloc[0].path
    with Image.open(path) as im: raw=im.convert('RGB').copy()
    q=model.map_quantiles.cpu();names=['preprocessing','host_to_device','model','postprocessing','end_to_end']
    def once(image):
        torch.cuda.synchronize();start=time.perf_counter()
        x=normalized(TF.to_tensor(TF.resize(image,[256,256])).unsqueeze(0));pre=time.perf_counter()
        x=x.cuda();torch.cuda.synchronize();transfer=time.perf_counter()
        pair=model.raw_maps(x);torch.cuda.synchronize();forward=time.perf_counter()
        pair=tuple(m.cpu() for m in pair);scores=score_maps(pair,q)['local_max']
        local_map=.1*(pair[0]-q[0,0])/(q[0,1]-q[0,0])
        native=F.interpolate(local_map,size=(raw.height,raw.width),mode='bilinear',align_corners=False).numpy()
        float(native.max());float(scores[0]);finish=time.perf_counter()
        return [(pre-start)*1000,(transfer-pre)*1000,(forward-transfer)*1000,(finish-forward)*1000,(finish-start)*1000]
    for _ in range(50): once(raw)
    torch.cuda.reset_peak_memory_stats();times=np.array([once(raw) for _ in range(200)])
    csv_exact(local/'latency_ms.csv',pd.DataFrame(times,columns=names))
    decode=[]
    for _ in range(50):
        start=time.perf_counter()
        with Image.open(path) as im: once(im.convert('RGB'))
        decode.append((time.perf_counter()-start)*1000)
    return {name:dict(p50_ms=float(np.quantile(times[:,i],.5)),p95_ms=float(np.quantile(times[:,i],.95))) for i,name in enumerate(names)}|dict(
        warmup=50,iterations=200,decode_iterations=50,decode_inclusive_p50_ms=float(np.quantile(decode,.5)),decode_inclusive_p95_ms=float(np.quantile(decode,.95)),
        precision='deterministic FP32',batch_size=1,scope='decoded RGB to native CPU signed map; disk decode excluded except separate series; GPU model is full raw local/global forward',
        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),peak_inference_reserved_bytes=torch.cuda.max_memory_reserved(),power_context=gpu_snapshot())


def evaluate(identifier,identity,trained,f):
    local=LOCAL/identifier;checkpoint=ROOT/trained['checkpoint']['path']
    model,_=load_checkpoint(checkpoint,'cuda');q=model.map_quantiles.cpu()
    start=time.perf_counter();cal_rows,cal_images=read_normals(ROOT,identity['category'],'calibration')
    cal_pairs=collect_raw_maps(model,cal_images);cal_values=score_maps(cal_pairs,q)[f['score']].astype(np.float64)
    calibration=qualified_calibration(cal_values,trained['development_scores'],f['threshold_policy'])
    local_cal=.1*(cal_pairs[0]-q[0,0])/(q[0,1]-q[0,0]);flat=local_cal.flatten().numpy().astype(np.float64)
    calibration.update(source_sha256=source_identity(cal_rows),paths=cal_rows.path.tolist(),scores=cal_values.tolist(),
        pixel_threshold=float(np.quantile(flat,.995,method='higher')),pixel_quantile=.995,
        histogram_center=float(np.median(flat)),histogram_scale=max(float(np.quantile(flat,.75)-np.quantile(flat,.25)),1e-6))
    calibration_seconds=time.perf_counter()-start
    cal_csv=REPORT/'runs'/f'{identifier}-calibration-scores.csv';csv_exact(cal_csv,pd.DataFrame(dict(path=cal_rows.path,score=cal_values)))
    cal_path=REPORT/'runs'/f'{identifier}-calibration.json';atomic_json(cal_path,calibration)
    del cal_pairs,cal_images,flat,local_cal
    # From this point only: official test images, then masks/labels in frozen evaluation.
    start=time.perf_counter();rows=read_rows('test',identity['category']);n=len(rows)
    raw_path=local/'raw_branch_maps.npy';map_path=local/'selected_maps.npy'
    raw_maps=np.lib.format.open_memmap(raw_path,mode='w+',dtype=np.float32,shape=(n,2,256,256))
    maps=np.lib.format.open_memmap(map_path,mode='w+',dtype=np.float32,shape=(n,256,256));scores=[]
    with torch.inference_mode():
        for i,row in enumerate(rows.itertuples()):
            path=ROOT/'data/mvtec_ad'/row.path;assert sha256(path)==row.sha256
            with Image.open(path) as im: x=normalized(TF.to_tensor(TF.resize(im.convert('RGB'),[256,256])).unsqueeze(0)).cuda()
            pair=tuple(m.cpu() for m in model.raw_maps(x))
            for j,m in enumerate(pair): raw_maps[i,j]=m.numpy()[0,0]
            selected=.1*(pair[0]-q[0,0])/(q[0,1]-q[0,0]);maps[i]=selected.numpy()[0,0]
            scores.append(float(selected.max()))
    maps.flush();raw_maps.flush();scores=np.asarray(scores,dtype=np.float64)
    labels=rows['mask'].notna().astype(int).to_numpy()
    image=binary_metrics(labels,scores,calibration['threshold'])
    table=rows[['path','category','mask']].copy();table['defect']=rows.path.str.split('/').str[-2]
    table['label']=labels;table['score']=scores;table['predicted']=scores>calibration['threshold']
    table['outcome']=np.select([(labels==0)&~table.predicted,(labels==0)&table.predicted,(labels==1)&table.predicted],['TN','FP','TP'],default='FN')
    inference_seconds=time.perf_counter()-start;start=time.perf_counter()
    pixel,per_image=native_evaluation(rows,maps,scores,calibration,identity['category'],local)
    native_seconds=time.perf_counter()-start
    table=table.merge(pd.DataFrame(per_image),on='path',validate='one_to_one')
    image_path=REPORT/'runs'/f'{identifier}-images.csv';replay=csv_exact(image_path,table)
    assert np.array_equal(replay.score.to_numpy(),scores)
    assert np.array_equal(replay.predicted.to_numpy(),scores>calibration['threshold'])
    defects=[]
    for defect in sorted(table.loc[table.label.eq(1),'defect'].unique()):
        group=table[table.defect.isin(['good',defect])]
        defects.append(dict(category=identity['category'],seed=identity['seed'],defect=defect,**binary_metrics(group.label,group.score,calibration['threshold'])))
    peak_evaluation_gpu=torch.cuda.max_memory_allocated();start=time.perf_counter()
    latency=benchmark(model,identity['category'],local);latency_seconds=time.perf_counter()-start
    tn,fp,fn,tp=map(int,np.array(image['confusion_matrix']).ravel())
    result=dict(identity=identity,experiment_id=identifier,category=identity['category'],seed=identity['seed'],freeze_sha256=FREEZE,
        image_auroc=image.pop('auroc'),image_ap=image.pop('ap'),**image,**pixel,tn=tn,fp=fp,fn=fn,tp=tp,
        calibration=calibration,training=trained,training_seconds=trained['training_seconds'],calibration_seconds=calibration_seconds,
        inference_seconds=inference_seconds,native_metrics_bootstrap_seconds=native_seconds,latency_seconds=latency_seconds,
        evaluation_seconds=inference_seconds+native_seconds+latency_seconds,latency=latency,
        evaluation_peak_allocated_bytes=peak_evaluation_gpu,checkpoint=trained['checkpoint'],
        local_maps=map_path.relative_to(ROOT).as_posix(),raw_branch_maps=raw_path.relative_to(ROOT).as_posix(),
        images_csv=image_path.relative_to(ROOT).as_posix(),test_source_sha256=source_identity(rows),per_defect=defects,evaluated_at_utc=utc())
    del model,maps,raw_maps;gc.collect();torch.cuda.empty_cache()
    result['artifacts']=[artifact(p) for p in (checkpoint,REPORT/'runs'/f'{identifier}-history.csv',cal_csv,cal_path,image_path,
        raw_path,map_path,local/'bootstrap_vectors.npz',local/'native_image_histograms.npz',local/'latency_ms.csv',local/'trained.json')]
    return result


def run_one(category,seed):
    configure(ROOT);f=freeze()
    for p in (REPORT/'runs',LOCAL,CHECKPOINT):p.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(ROOT).free>20*1024**3,'Insufficient free disk'
    complete,identifier=choose_attempt(category,seed)
    if complete is not None:
        print('Verified complete, skip',complete['experiment_id'],flush=True);return complete
    identity=identity_for(category,seed);reservation=ROOT/f'experiments/{identifier}-start.json'
    if reservation.exists():
        count=len(list((ROOT/'experiments').glob(identifier+'-resume-*.json')))+1
        registry(identifier,f'resume-{count:02d}',identity,'resumed',interpretation='Recover durable epochs or trained checkpoint; do not change model policy')
    else: registry(identifier,'start',identity,'started',interpretation='Reserved before fitting/calibration/test access')
    start=time.perf_counter()
    try:
        with Resources() as resources:
            trained=train(identifier,identity,f)
            result=evaluate(identifier,identity,trained,f)
        result.update(status='complete',resources=resources.result,wall_seconds=time.perf_counter()-start,
            disk_free_bytes=shutil.disk_usage(ROOT).free,completed_at_utc=utc())
        # Validate staged result before immutable publication and completion registry event.
        pending=LOCAL/identifier/'candidate-result.json';atomic_json(pending,result)
        verify_complete(pending,identity)
        destination=REPORT/'runs'/f'{identifier}-result.json'
        assert not destination.exists(),'Never overwrite a completed result'
        atomic_json(destination,result)
        registry(identifier,'result',identity,'completed',metrics={k:result[k] for k in METRICS},calibration=result['calibration'],
            paths=dict(result=destination.relative_to(ROOT).as_posix(),result_sha256=sha256(destination),checkpoint=result['checkpoint']),
            interpretation='All test metrics and required artifacts verified before completion')
        print('COMPLETE',identifier,{k:result[k] for k in METRICS},flush=True)
        return result
    except BaseException:
        index=len(list((ROOT/'experiments').glob(identifier+'-failure-*.json')))+1
        registry(identifier,f'failure-{index:02d}',identity,'failed',interpretation=traceback.format_exc(),decision='Preserve stage and resume; no policy change')
        raise
