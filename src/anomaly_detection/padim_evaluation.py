"""Frozen PaDiM confirmation with shared exact native metrics and bootstrap."""
import gc
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label
import torch
from torch.nn import functional as F
from .padim import PaDiMExtractor, SpatialGaussian, PaDiMPipeline
from .padim_research import ROOT, read_partition, load_resized, extract, score, write_json, experiment, source_identity
from .patchcore_research import Resources
from .cnn_evaluation import bootstrap_evaluation
from .data import autodiscover_data_root, sha256
from .metrics import binary_metrics, calibrate_normal
from .pixel_metrics import exact_pixel_metrics, histogram_metrics


@torch.inference_mode()
def benchmark(model, row, native_shape, path):
    with Image.open(autodiscover_data_root(ROOT)/row.path) as image:
        raw=image.convert('RGB').copy()
    def once():
        start=time.perf_counter()
        array=np.asarray(raw.resize((256,256),Image.Resampling.BILINEAR)).copy()
        x=torch.from_numpy(array).permute(2,0,1)[None].float()/255
        pre=time.perf_counter();x=x.cuda();torch.cuda.synchronize();transfer=time.perf_counter()
        features=model.extractor(x);torch.cuda.synchronize();extracted=time.perf_counter()
        distances=model.gaussian.score(features);torch.cuda.synchronize();gaussian=time.perf_counter()
        from .padim import anomaly_maps
        maps=anomaly_maps(distances,model.sigma)
        F.interpolate(maps[:,None],size=native_shape,mode='bilinear',align_corners=False).cpu().numpy()
        float(maps.max());torch.cuda.synchronize();finish=time.perf_counter()
        return np.array([pre-start,transfer-pre,extracted-transfer,gaussian-extracted,gaussian-transfer,finish-gaussian,finish-start])*1000
    for _ in range(50):once()
    torch.cuda.reset_peak_memory_stats()
    times=np.stack([once() for _ in range(200)])
    names=['preprocessing','host_to_device','feature_extraction','gaussian_scoring','model','postprocessing','end_to_end']
    pd.DataFrame(times,columns=names).to_csv(path,index=False,lineterminator='\n')
    return {name:dict(p50_ms=float(np.median(times[:,i])),p95_ms=float(np.quantile(times[:,i],.95))) for i,name in enumerate(names)} | dict(
        warmup=50,iterations=200,batch_size=1,precision='float32',
        scope='Decoded RGB through native CPU map; file decode excluded; model=features+Mahalanobis; smoothing in postprocessing',
        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),peak_inference_reserved_bytes=torch.cuda.max_memory_reserved())


def native_evaluation(rows, maps, scores, calibration, category, local):
    """Reuse established exact pixel implementation; disk-backed native vectors."""
    root=autodiscover_data_root(ROOT)
    inventory=pd.read_csv(ROOT/'outputs/eda/inventory.csv').set_index('path')
    sizes=[(int(inventory.loc[p,'height']),int(inventory.loc[p,'width'])) for p in rows.path]
    count=sum(h*w for h,w in sizes)
    paths=[local/f'native-{name}.dat' for name in ('scores','truth','regions')]
    vectors=[np.memmap(p,mode='w+',dtype=dtype,shape=(count,)) for p,dtype in zip(paths,(np.float32,np.uint8,np.float64))]
    positive=np.zeros((len(rows),4096),np.uint32);negative=np.zeros_like(positive);region=np.zeros_like(positive,dtype=np.float64)
    region_counts=[];cursor=0;cm=np.zeros((2,2),np.int64)
    for i,(row,size) in enumerate(zip(rows.itertuples(),sizes)):
        native=F.interpolate(torch.from_numpy(maps[i:i+1,None]),size=size,mode='bilinear',align_corners=False).numpy()[0,0]
        if isinstance(row.mask,str):
            if sha256(root/row.mask)!=row.mask_sha256:raise ValueError('Mask identity changed')
            with Image.open(root/row.mask) as im:truth=(np.asarray(im)>0).astype(np.uint8)
        else:truth=np.zeros(size,np.uint8)
        components,n=label(truth,structure=np.ones((3,3)))
        areas=np.bincount(components.ravel());areas[0]=1
        weight=np.where(truth,1/areas[components],0.)
        end=cursor+native.size
        for target,values in zip(vectors,(native,truth,weight)):target[cursor:end]=values.ravel()
        cursor=end;region_counts.append(n)
        idx=np.minimum((native/(native+calibration['histogram_scale'])*4096).astype(np.int64),4095)
        positive[i]=np.bincount(idx[truth.astype(bool)],minlength=4096)
        negative[i]=np.bincount(idx[~truth.astype(bool)],minlength=4096)
        region[i]=np.bincount(idx.ravel(),weights=weight.ravel(),minlength=4096)
        pred=native>calibration['pixel_threshold']
        cm+=np.bincount((2*truth+pred).ravel(),minlength=4).reshape(2,2)
    for vector in vectors:vector.flush()
    print(category,'exact native sweep',count,flush=True)
    point=exact_pixel_metrics(*vectors,sum(region_counts))
    labels=rows['mask'].notna().astype(int).to_numpy()
    intervals,boots=bootstrap_evaluation(labels,scores,calibration['image_threshold'],positive,negative,region,region_counts,category)
    intervals['pixel_ci_method']='Whole-image bootstrap; 4096 bins of d/(d+calibration median map value); primary native metrics exact'
    np.savez_compressed(local/'bootstrap_vectors.npz',**boots)
    np.savez_compressed(local/'native_image_histograms.npz',positive=positive,negative=negative,region=region,region_counts=region_counts)
    binned=histogram_metrics(positive.sum(0),negative.sum(0),region.sum(0),sum(region_counts))
    tn,fp,fn,tp=map(int,cm.ravel());precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn)
    operating=dict(tn=tn,fp=fp,fn=fn,tp=tp,precision=precision,recall=recall,fpr=fp/(fp+tn),f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    for vector in vectors:vector._mmap.close()
    del vectors;gc.collect()
    for path in paths:
        assert path.resolve().is_relative_to((ROOT/'outputs/padim').resolve())
        path.unlink()
    return dict(**point,pixel_confusion_matrix=cm.tolist(),pixel_operating_point=operating,uncertainty=intervals,
                bootstrap_binned_pixel_point_estimates=dict(zip(('pixel_auroc','pixel_ap','au_pro'),binned))),sizes


def verify_freeze():
    path=ROOT/'configs/padim_frozen.json'
    identity=json.loads((ROOT/'reports/padim/freeze_identity.json').read_text())['sha256']
    if sha256(path)!=identity:raise ValueError('Frozen configuration changed')
    frozen=json.loads(path.read_text())
    for source,expected in frozen['source_sha256'].items():
        if sha256(ROOT/source)!=expected:raise ValueError('Frozen source changed: '+source)
    if sha256(ROOT/'reports/padim/environment.json')!=frozen['environment_sha256']:raise ValueError('Environment identity changed')
    if sha256(ROOT/'reports/dataset_validation.json')!=frozen['dataset_validation_sha256']:raise ValueError('Dataset identity changed')
    return frozen,identity


def confirmation():
    frozen,identity=verify_freeze()
    for seed in frozen['seeds']:
        config=dict(frozen['configuration'],seed=seed)
        extractor=PaDiMExtractor(config['dimension'],seed).cuda()
        if extractor.weight_sha256!=frozen['weight_sha256']:raise ValueError('Weights changed')
        for category in frozen['categories']:
            identifier=f'05-test-{category}-s{seed}-v1'
            destination=ROOT/f'reports/padim/evaluation/{identifier}-result.json'
            if destination.exists():continue
            verify_freeze()
            with experiment(identifier,category,config,'frozen confirmation; fit then normal calibration then test') as result:
                local=ROOT/f'outputs/padim/{identifier}';local.mkdir(parents=True,exist_ok=False)
                with Resources() as resources:
                    fit=read_partition('fit',category)
                    started=time.perf_counter()
                    features=extract(extractor,load_resized(fit,256))
                    gaussian=SpatialGaussian(config['epsilon']).fit(features,'cuda')
                    torch.cuda.synchronize();fit_seconds=time.perf_counter()-started
                    fit_peak_allocated=torch.cuda.max_memory_allocated()
                    fit_peak_reserved=torch.cuda.max_memory_reserved()
                    checkpoint=ROOT/f'checkpoints/padim/{identifier}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    gaussian.save(checkpoint,category,config,dict(phase='final',fit_source_sha256=source_identity(fit),freeze_sha256=identity))
                    # save() uses CPU copies and leaves the model on the GPU.
                    calibration_rows=read_partition('calibration',category)
                    calibration_maps=score(gaussian,extract(extractor,load_resized(calibration_rows,256)),config['sigma']).numpy()
                    calibration_scores=calibration_maps.reshape(len(calibration_maps),-1).max(1)
                    calibration=dict(images=len(calibration_rows),image_threshold=calibrate_normal(calibration_scores),
                        pixel_threshold=float(np.quantile(calibration_maps,.995,method='higher')),
                        image_quantile=.95,pixel_quantile=.995,strict_greater=True,
                        histogram_scale=max(float(np.median(calibration_maps)),1e-6),
                        scores=calibration_scores.tolist(),source_sha256=source_identity(calibration_rows))
                    write_json(ROOT/f'reports/padim/evaluation/{identifier}-calibration.json',calibration)
                    del features,calibration_maps
                    rows=read_partition('test',category)
                    maps=score(gaussian,extract(extractor,load_resized(rows,256,allow_test=True)),config['sigma']).numpy()
                    np.save(local/'model_resolution_maps.npy',maps,allow_pickle=False)
                    scores=maps.reshape(len(maps),-1).max(1)
                    labels=rows['mask'].notna().astype(int).to_numpy()
                    image=binary_metrics(labels,scores,calibration['image_threshold'])
                    tn,fp,fn,tp=map(int,np.asarray(image['confusion_matrix']).ravel())
                    table=rows[['path','category','mask']].copy()
                    table['defect']=rows.path.str.split('/').str[-2];table['label']=labels;table['score']=scores
                    table['predicted']=scores>calibration['image_threshold']
                    table['outcome']=np.select([(labels==0)&~table.predicted,(labels==0)&table.predicted,(labels==1)&table.predicted],['TN','FP','TP'],default='FN')
                    table.to_csv(destination.with_name(identifier+'-images.csv'),index=False,lineterminator='\n')
                    pixel,sizes=native_evaluation(rows,maps,scores,calibration,category,local)
                    evaluation_peak_allocated=torch.cuda.max_memory_allocated()
                    evaluation_peak_reserved=torch.cuda.max_memory_reserved()
                    latency=benchmark(PaDiMPipeline(extractor,gaussian,config['sigma']),calibration_rows.iloc[0],sizes[0],local/'latency_ms.csv')
                    defects=[]
                    for defect in sorted(table.loc[table.label.eq(1),'defect'].unique()):
                        group=table[table.defect.isin(['good',defect])]
                        defects.append(dict(category=category,seed=seed,defect=defect,**binary_metrics(group.label,group.score,calibration['image_threshold'])))
                result.update(experiment_id=identifier,category=category,seed=seed,configuration=config,
                    image_auroc=image.pop('auroc'),image_ap=image.pop('ap'),**image,tn=tn,fp=fp,fn=fn,tp=tp,**pixel,
                    calibration=calibration,latency=latency,training_seconds=fit_seconds,resources=resources.result,
                    fitting_peak_allocated_bytes=fit_peak_allocated,fitting_peak_reserved_bytes=fit_peak_reserved,
                    evaluation_peak_allocated_bytes=evaluation_peak_allocated,evaluation_peak_reserved_bytes=evaluation_peak_reserved,
                    checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha256(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                    local_maps=(local/'model_resolution_maps.npy').relative_to(ROOT).as_posix(),maps_sha256=sha256(local/'model_resolution_maps.npy'),
                    freeze_sha256=identity,fit_source_sha256=source_identity(fit),test_source_sha256=source_identity(rows),
                    evaluated_at_utc=datetime.now(timezone.utc).isoformat(),image_rows=len(rows),per_defect=defects)
                write_json(destination,result)
                print(identifier,{k:result[k] for k in ('image_auroc','pixel_auroc','recall','fpr')},flush=True)
            del gaussian,maps;torch.cuda.empty_cache()
