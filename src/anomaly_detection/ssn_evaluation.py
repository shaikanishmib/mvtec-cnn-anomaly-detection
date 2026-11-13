"""Frozen normal calibration and native exact SuperSimpleNet metrics.

Raw logits define exact point estimates. Fixed sigmoid bins approximate only
pixel bootstrap intervals; no test-derived normalization or thresholds.
"""
from pathlib import Path
from datetime import datetime,timezone
import gc,json,time,traceback,subprocess
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label
from scipy.special import expit
import torch
from torch.nn import functional as F
from .supersimplenet import anomaly_maps,image_scores
from .ssn_research import ROOT,feature_cache,predict,model_for
from .patchcore_research import Resources
from .cnn_research import read_partition
from .data import autodiscover_data_root,sha256
from .metrics import binary_metrics,calibrate_normal
from .pixel_metrics import exact_pixel_metrics,histogram_metrics
from .tracking import config_hash,record
from .cnn_evaluation import bootstrap_evaluation as shared_bootstrap


def load_heads(path,category,configuration):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['category']!=category or saved['configuration']!=configuration:
        raise ValueError('Category/configuration checkpoint isolation failed')
    model=model_for(configuration).cuda().eval();model.load_state_dict(saved['state_dict'])
    return model,saved


def bootstrap_evaluation(*args):
    result,vectors=shared_bootstrap(*args)
    result['pixel_ci_method']='Whole-image bootstrap; fixed sigmoid(raw logits), 4096 bins; primary native raw-logit metrics exact'
    return result,vectors


class SSNPipeline(torch.nn.Module):
    def __init__(self,extractor,heads,config):
        super().__init__();self.extractor=extractor;self.heads=heads;self.config=config

    def forward(self,images):
        segmentation,detection=self.heads(self.extractor(images))
        maps=anomaly_maps(segmentation)
        return maps,image_scores(maps,detection,self.config['aggregation'])


def calibrate_model(model,category,resolution,seed=2026):
    rows=read_partition('calibration',category)
    if not rows.partition.eq('calibration').all() or rows['mask'].notna().any():
        raise ValueError('Calibration must contain only reserved unmodified normals')
    features=feature_cache(model.extractor,rows,model.config['extraction_batch'])
    maps,det,_=predict(model.heads,features,model.config['batch_size'])
    scores=image_scores(maps,det,model.config['aggregation']).numpy()
    maps=maps.numpy();threshold=calibrate_normal(scores)
    draws=np.random.default_rng(seed).choice(scores,size=(2000,len(scores)),replace=True)
    intervals=np.quantile(np.quantile(draws,.95,axis=1,method='higher'),[.025,.975]).tolist()
    return dict(category=category,images=len(rows),image_threshold=threshold,
        secondary_image_threshold_q99=calibrate_normal(scores,.01),
        pixel_threshold=float(np.quantile(maps,.995,method='higher')),
        image_quantile=.95,pixel_quantile=.995,pixel_calibration_samples=maps.size,
        pixel_score_resolution=256,calibration_scores=scores.tolist(),
        image_threshold_bootstrap_95ci=intervals,
        histogram_transform='fixed sigmoid(raw logits), 4096 bins; primary raw scores never clipped',
        method='Reserved unmodified normals; higher image q95 and pooled pixel q99.5; strict >',
        source_paths_sha256=config_hash(rows[['path','sha256']].to_dict('records')),feature_cache=features.metadata)


@torch.inference_mode()
def benchmark_model(model,calibration_path,resolution,native_shape,output):
    with Image.open(autodiscover_data_root(ROOT)/calibration_path) as image:raw=image.convert('RGB').copy()
    def once():
        start=time.perf_counter()
        array=np.asarray(raw.resize((256,256),Image.Resampling.BILINEAR)).copy()
        x=torch.from_numpy(array).permute(2,0,1)[None].float()/255
        pre=time.perf_counter();x=x.cuda();torch.cuda.synchronize();transfer=time.perf_counter()
        features=model.extractor(x);torch.cuda.synchronize();extracted=time.perf_counter()
        seg,det=model.heads(features);torch.cuda.synchronize();learned=time.perf_counter()
        maps=anomaly_maps(seg)
        result=F.interpolate(maps[:,None],size=native_shape,mode='bilinear',align_corners=False).cpu().numpy()
        float(image_scores(maps,det,model.config['aggregation']).item())
        torch.cuda.synchronize();finish=time.perf_counter()
        assert np.isfinite(result).all()
        return np.array([pre-start,transfer-pre,extracted-transfer,learned-extracted,learned-transfer,finish-learned,finish-start])*1000
    for _ in range(50):once()
    torch.cuda.reset_peak_memory_stats();times=np.stack([once() for _ in range(200)])
    names=['preprocessing','host_to_device','feature_extraction','learned_heads','model','postprocessing','end_to_end']
    pd.DataFrame(times,columns=names).to_csv(output,index=False,lineterminator='\n')
    context={}
    for key,command in [('windows_scheme',['powercfg','/getactivescheme']),('gpu',['nvidia-smi','--query-gpu=name,driver_version,pstate,clocks.gr,temperature.gpu','--format=csv,noheader'])]:
        try:context[key]=subprocess.check_output(command,text=True,timeout=10).strip()
        except (OSError,subprocess.SubprocessError) as exc:context[key]=str(exc)
    return {name:dict(p50_ms=float(np.median(times[:,i])),p95_ms=float(np.quantile(times[:,i],.95))) for i,name in enumerate(names)} | dict(warmup=50,iterations=200,batch_size=1,precision='float32',scope='Decoded RGB -> native raw-logit CPU map + image score; file decode excluded; model=frozen extractor+learned heads',power_context_at_end_of_timing=context,peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),peak_inference_reserved_bytes=torch.cuda.max_memory_reserved())


def evaluate_frozen(identifier, training_result, freeze_sha256, extractor):
    """One immutable evaluation reservation per category/seed; never retries itself."""
    from .cnn_research import code_identity
    freeze_path=ROOT/"configs/supersimplenet_frozen.json"
    if sha256(freeze_path)!=freeze_sha256:raise ValueError("Frozen confirmation identity mismatch")
    frozen=json.loads(freeze_path.read_text())
    for path, expected_hash in frozen["source_sha256"].items():
        if sha256(ROOT/path)!=expected_hash:raise ValueError("Frozen model/training source changed")
    expected=dict(frozen["configuration"],seed=training_result["configuration"]["seed"])
    if expected!=training_result["configuration"] or expected["seed"] not in frozen["seeds"]:
        raise ValueError("Training configuration differs from frozen policy")
    if training_result["category"] not in frozen["categories"]:
        raise ValueError("Unregistered category")
    if sha256(Path(__file__))!=frozen["evaluation_source_sha256"]:
        raise ValueError("Evaluation implementation changed after freeze")
    if sha256(Path(__file__).with_name("pixel_metrics.py"))!=frozen["pixel_metric_source_sha256"]:
        raise ValueError("Pixel metric implementation changed after freeze")
    result_dir=ROOT/"reports/supersimplenet/evaluation"
    result_dir.mkdir(parents=True,exist_ok=True)
    local=ROOT/"outputs/supersimplenet"/identifier
    local.mkdir(parents=True,exist_ok=False)
    category=training_result["category"]
    config=training_result["configuration"]
    entry={"experiment_id":identifier+"-start","model_family":"SuperSimpleNetLite","categories":[category],
           "hypothesis":"Frozen feature-synthetic discrimination confirmation; no further selection",
           "preprocessing":{"resolution":config["resolution"],"native_map":"bilinear, align_corners=False"},
           "hyperparameters":config,"seed":config["seed"],
           "environment":json.loads((ROOT/"reports/environment.json").read_text()),
           "training":{"training_experiment":training_result["experiment_id"],"freeze_sha256":freeze_sha256,
                       "code":{"source_sha256":frozen["source_sha256"],"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()},"evaluation_source_sha256":sha256(Path(__file__)),
                       "pixel_metric_source_sha256":sha256(Path(__file__).with_name("pixel_metrics.py"))},
           "metrics":{},"calibration":{},"paths":{"local":local.relative_to(ROOT).as_posix()},
           "interpretation":"Reserved before calibration/test inference","decision":"frozen evaluation","status":"started"}
    record(ROOT/"experiments",entry)
    started=time.perf_counter()
    monitor=Resources();monitor.__enter__()
    try:
        checkpoint=ROOT/training_result["checkpoint"]
        if sha256(checkpoint)!=training_result["checkpoint_sha256"]:raise ValueError("Checkpoint identity changed")
        model_heads,saved=load_heads(checkpoint,category,config)
        if extractor.weight_sha256!=frozen['weight_sha256']:raise ValueError('Pretrained weights changed')
        model=SSNPipeline(extractor,model_heads,config)
        if saved["category"]!=category or saved['phase']!="final" or saved['configuration']!=config:
            raise ValueError("Expected a fresh category-specific final checkpoint")
        # Calibration occurs and is saved BEFORE test rows/images are loaded.
        calibration=calibrate_model(model,category,config["resolution"],config["seed"])
        with (result_dir/f"{identifier}-calibration.json").open("x") as stream:json.dump(calibration,stream,indent=2)
        rows=read_partition("test",category)
        data_root=autodiscover_data_root(ROOT)
        features=feature_cache(extractor,rows,config['extraction_batch'])
        maps,raw_detection,raw_segmentation=predict(model.heads,features,config['batch_size'])
        scores=image_scores(maps,raw_detection,config['aggregation']).numpy()
        np.save(local/'raw_detection.npy',raw_detection.numpy(),allow_pickle=False)
        np.save(local/'raw_segmentation.npy',raw_segmentation.numpy(),allow_pickle=False)
        maps=maps.numpy()
        labels=rows["mask"].notna().astype(int).to_numpy()
        image_result=binary_metrics(labels,scores,calibration["image_threshold"])
        tn,fp,fn,tp=np.asarray(image_result["confusion_matrix"]).ravel()
        image_counts={"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
        secondary=binary_metrics(labels,scores,calibration["secondary_image_threshold_q99"])
        table=rows[["path","category","mask"]].copy()
        table["defect"]=rows.path.str.split("/").str[-2]
        table["label"]=labels;table["score"]=scores
        table["predicted"]=scores>calibration["image_threshold"]
        table["outcome"]=np.select([(labels==0)&~table.predicted,(labels==0)&table.predicted,
                                    (labels==1)&table.predicted],["TN","FP","TP"],default="FN")
        table.to_csv(result_dir/f"{identifier}-images.csv",index=False,lineterminator="\n")
        np.save(local/"model_resolution_maps.npy",maps,allow_pickle=False)
        # Original dimensions from verified inventory, not estimated from resized masks.
        inventory=pd.read_csv(ROOT/"outputs/eda/inventory.csv").set_index("path")
        sizes=[(int(inventory.loc[p,"height"]),int(inventory.loc[p,"width"])) for p in rows.path]
        pixel_count=sum(h*w for h,w in sizes)
        scratch=[]
        for name,dtype in (("scores",np.float32),("truth",np.uint8),("regions",np.float64)):
            path=local/f"native-{name}.dat";scratch.append(path)
        flat_scores=np.memmap(scratch[0],mode="w+",dtype=np.float32,shape=(pixel_count,))
        flat_truth=np.memmap(scratch[1],mode="w+",dtype=np.uint8,shape=(pixel_count,))
        flat_regions=np.memmap(scratch[2],mode="w+",dtype=np.float64,shape=(pixel_count,))
        bins=4096
        positive_hist=np.zeros((len(rows),bins),np.uint32)
        negative_hist=np.zeros_like(positive_hist)
        region_hist=np.zeros((len(rows),bins),np.float64)
        region_counts=[];cursor=0
        pixel_cm=np.zeros((2,2),np.int64)
        for i,(row,size) in enumerate(zip(rows.itertuples(),sizes)):
            with torch.inference_mode():
                native=F.interpolate(torch.from_numpy(maps[i:i+1,None]),size=size,mode="bilinear",align_corners=False).numpy()[0,0]
            if isinstance(row.mask,str):
                mask_path=data_root/row.mask
                if sha256(mask_path)!=row.mask_sha256:raise ValueError("Official mask changed")
                with Image.open(mask_path) as mask_image:truth=(np.asarray(mask_image)>0).astype(np.uint8)
            else:truth=np.zeros(size,np.uint8)
            components,count=label(truth,structure=np.ones((3,3)))
            areas=np.bincount(components.ravel());areas[0]=1
            region_weight=np.where(truth,1/areas[components],0.)
            end=cursor+native.size
            flat_scores[cursor:end]=native.ravel();flat_truth[cursor:end]=truth.ravel();flat_regions[cursor:end]=region_weight.ravel()
            cursor=end;region_counts.append(count)
            idx=np.minimum((expit(native)*bins).astype(np.int64),bins-1)
            positive_hist[i]=np.bincount(idx[truth.astype(bool)],minlength=bins)
            negative_hist[i]=np.bincount(idx[~truth.astype(bool)],minlength=bins)
            region_hist[i]=np.bincount(idx.ravel(),weights=region_weight.ravel(),minlength=bins)
            pred=native>calibration["pixel_threshold"]
            pixel_cm+=np.bincount((2*truth+pred).ravel(),minlength=4).reshape(2,2)
        flat_scores.flush();flat_truth.flush();flat_regions.flush()
        print(identifier,"exact pixel sweep",pixel_count,flush=True)
        pixel_result=exact_pixel_metrics(flat_scores,flat_truth,flat_regions,sum(region_counts))
        ptn,pfp,pfn,ptp=pixel_cm.ravel()
        prec=float(ptp/(ptp+pfp)) if ptp+pfp else 0.
        rec=float(ptp/(ptp+pfn))
        pixel_operating={"precision":prec,"recall":rec,"fpr":float(pfp/(pfp+ptn)),
                         "f1":2*prec*rec/(prec+rec) if prec+rec else 0.,
                         "tp":int(ptp),"fp":int(pfp),"tn":int(ptn),"fn":int(pfn)}
        intervals,bootstrap_vectors=bootstrap_evaluation(labels,scores,calibration["image_threshold"],
                        positive_hist,negative_hist,region_hist,region_counts,category)
        np.savez_compressed(local/"bootstrap_vectors.npz",**bootstrap_vectors)
        np.savez_compressed(local/"native_image_histograms.npz",positive=positive_hist,negative=negative_hist,
                            region=region_hist,region_counts=region_counts)
        binned=histogram_metrics(positive_hist.sum(0),negative_hist.sum(0),region_hist.sum(0),sum(region_counts))
        calibration_rows=read_partition("calibration",category)
        latency=benchmark_model(model,calibration_rows.path.iloc[0],config["resolution"],sizes[0],local/"latency_ms.csv")
        defects=[]
        for defect in sorted(table.loc[table.label.eq(1),"defect"].unique()):
            subset=table[table.defect.isin(["good",defect])]
            values=binary_metrics(subset.label,subset.score,calibration["image_threshold"])
            defects.append({"category":category,"seed":config["seed"],"defect":defect,
                            "defective_images":int(subset.label.sum()),**values})
        monitor.__exit__(None,None,None)
        result={"experiment_id":identifier,"category":category,"seed":config["seed"],
                "training_experiment":training_result["experiment_id"],"checkpoint_sha256":training_result["checkpoint_sha256"],
                "image_auroc":image_result.pop("auroc"),"image_ap":image_result.pop("ap"),**image_result,**image_counts,
                **pixel_result,"pixel_confusion_matrix":pixel_cm.tolist(),"pixel_operating_point":pixel_operating,"secondary_image_q99":secondary,
                "calibration":calibration,"latency":latency,"uncertainty":intervals,
                "bootstrap_binned_pixel_point_estimates":dict(zip(["pixel_auroc","pixel_ap","au_pro"],binned)),
                "training_seconds":training_result["wall_seconds"],
                "training_loop_seconds":training_result['resources']['seconds'],
                "test_cache":features.metadata,"cpu_peak_rss_bytes":monitor.result['cpu_peak_rss_bytes'],
                "training_peak_allocated_bytes":training_result["resources"]["peak_gpu_allocated_bytes"],
                "checkpoint_bytes":training_result["checkpoint_bytes"],"parameters":sum(p.numel() for p in model.parameters()),
                "evaluation_seconds":time.perf_counter()-started,"per_defect":defects,
                "local_maps":(local/"model_resolution_maps.npy").relative_to(ROOT).as_posix(),
                "maps_sha256":sha256(local/"model_resolution_maps.npy"),
                "freeze_sha256":freeze_sha256,"image_rows":len(rows),"evaluated_at_utc":datetime.now(timezone.utc).isoformat()}
        with (result_dir/f"{identifier}-result.json").open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
        entry.update(experiment_id=identifier+"-result",status="completed",metrics=result,calibration=calibration,
                     interpretation="Frozen real-test confirmation; no post-test changes permitted",decision="retain even if disappointing")
        record(ROOT/"experiments",entry)
        # Scratch native vectors are reproducible from retained low-resolution
        # maps and original masks; remove only these three explicitly named files.
        for mapping in (flat_scores,flat_truth,flat_regions):mapping._mmap.close()
        del flat_scores,flat_truth,flat_regions;gc.collect()
        for path in scratch:
            assert path.resolve().is_relative_to((ROOT/"outputs/supersimplenet").resolve())
            path.unlink()
        return result
    except BaseException as exc:
        if not monitor.done.is_set():monitor.__exit__(None,None,None)
        (local/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        entry.update(experiment_id=identifier+"-failure",status="failed",interpretation=f"{type(exc).__name__}: {exc}",decision="preserve; investigate without retuning")
        record(ROOT/"experiments",entry)
        raise
