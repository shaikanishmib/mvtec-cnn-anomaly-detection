"""Frozen calibration, one-shot official evaluation and native pixel metrics."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import time
import traceback
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label
import torch
from torch.nn import functional as F

from .autoencoder import ConvAutoencoder, anomaly_map, image_scores
from .cnn_research import ROOT, read_partition, load_resized
from .ae_research import reconstruct
from .data import CATEGORIES, autodiscover_data_root, sha256
from .metrics import binary_metrics, calibrate_normal
from .pixel_metrics import exact_pixel_metrics, histogram_metrics
from .tracking import config_hash, record


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = ConvAutoencoder(**{k:saved["config"][k] for k in ["width","depth","latent_channels"]})
    model.load_state_dict(saved["state_dict"])
    model = model.cuda().eval()
    model.map_kind = saved["config"]["map_kind"]
    return model, saved


def calibrate_model(model, category, resolution, seed=2026):
    """Only the calibration manifest is read; API takes no test labels/masks."""
    rows = read_partition("calibration", category)
    images = load_resized(rows, resolution).cuda()
    maps = anomaly_map(images.cpu().float()/255, reconstruct(model, images), model.map_kind)
    scores = image_scores(maps).numpy()
    primary, secondary = calibrate_normal(scores), calibrate_normal(scores, .01)
    # Conservative upper edge for the fixed quantile's bin.
    probabilities = maps.numpy().ravel()
    bins = 65536
    histogram = np.bincount(np.minimum((probabilities*bins).astype(np.int64), bins-1), minlength=bins)
    rank = int(np.ceil(.995*(len(probabilities)-1))) + 1
    index = int(np.searchsorted(np.cumsum(histogram), rank))
    pixel_threshold = min(1., (index+1)/bins)
    rng = np.random.default_rng(seed)
    draws = rng.choice(scores, size=(2000,len(scores)), replace=True)
    boot_thresholds = np.quantile(draws,.95,axis=1,method="higher")
    return {"category": category, "images": len(rows), "image_threshold": primary,
            "secondary_image_threshold_q99": secondary, "pixel_threshold": pixel_threshold,
            "image_quantile": .95, "pixel_quantile": .995, "pixel_bins": bins,
            "pixel_calibration_samples": len(probabilities), "pixel_score_resolution": resolution,
            "image_threshold_bootstrap_95ci": np.quantile(boot_thresholds,[.025,.975]).tolist(),
            "calibration_scores": scores.tolist(),
            "method": "normal calibration only; higher order statistic; strict score > threshold",
            "source_paths_sha256": config_hash(rows[["path","sha256"]].to_dict("records"))}


@torch.inference_mode()
def benchmark_model(model, calibration_path, resolution, native_shape, output):
    """Memory-resident RGB input; disk decode excluded and explicitly disclosed."""
    root = autodiscover_data_root(ROOT)
    with Image.open(root / calibration_path) as image:
        raw = image.convert("RGB").copy()
    def once():
        start = time.perf_counter()
        array = np.asarray(raw.resize((resolution,resolution), Image.Resampling.BILINEAR)).copy()
        x = torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
        pre = time.perf_counter()
        x = x.cuda(); torch.cuda.synchronize()
        transfer = time.perf_counter()
        reconstruction = model(x)
        maps = anomaly_map(x, reconstruction, model.map_kind).unsqueeze(1); torch.cuda.synchronize()
        forward = time.perf_counter()
        native = F.interpolate(maps, size=native_shape, mode="bilinear", align_corners=False)
        result = native.cpu().numpy()
        float(image_scores(maps[:,0]).item())
        torch.cuda.synchronize()
        finish = time.perf_counter()
        return np.array([pre-start,transfer-pre,forward-transfer,finish-forward,finish-start])*1000
    for _ in range(50): once()
    torch.cuda.reset_peak_memory_stats()
    times = np.stack([once() for _ in range(200)])
    names = ["preprocessing", "host_to_device", "model", "postprocessing", "end_to_end"]
    pd.DataFrame(times,columns=names).to_csv(output,index=False,lineterminator="\n")
    power_context={}
    for key,command in (("windows_scheme",["powercfg","/getactivescheme"]),
                        ("gpu",["nvidia-smi","--query-gpu=name,driver_version,power.limit,pstate,clocks.gr,temperature.gpu","--format=csv,noheader"])):
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=10,check=True)
            power_context[key]=process.stdout.strip()
        except (OSError,subprocess.SubprocessError) as exc:power_context[key]=str(exc)
    return {name: {"p50_ms":float(np.median(times[:,i])),"p95_ms":float(np.quantile(times[:,i],.95))}
            for i,name in enumerate(names)} | {
                "warmup":50,"iterations":200,"batch_size":1,"precision":"float32",
                "scope":"memory-resident RGB -> native residual map on CPU; file read/decode excluded",
                "power_context_at_end_of_timing":power_context,
                "peak_inference_allocated_bytes":torch.cuda.max_memory_allocated(),
                "peak_inference_reserved_bytes":torch.cuda.max_memory_reserved()}


def image_bootstrap(labels, scores, threshold, weights):
    """Whole-image resampling, exact weighted score ties; no threshold refitting."""
    order=np.argsort(scores,kind="stable")[::-1]
    values=np.asarray(scores)[order]
    y=np.asarray(labels)[order]
    w=weights[:,order]
    endpoints=np.r_[np.flatnonzero(values[1:] != values[:-1]),len(values)-1]
    tp=np.cumsum(w*y,axis=1)[:,endpoints]
    fp=np.cumsum(w*(1-y),axis=1)[:,endpoints]
    total_p,total_n=tp[:,-1],fp[:,-1]
    valid=(total_p>0)&(total_n>0)
    tp,fp=tp[valid],fp[valid]
    total_p,total_n=total_p[valid],total_n[valid]
    recall=tp/total_p[:,None]; fpr=fp/total_n[:,None]
    roc=np.trapezoid(np.c_[np.zeros(len(tp)),recall],np.c_[np.zeros(len(tp)),fpr],axis=1)
    increments=np.diff(np.c_[np.zeros(len(tp)),tp],axis=1)
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0)
    ap=(increments/total_p[:,None]*precision).sum(1)
    predictions=(np.asarray(scores)>threshold).astype(int)
    true_p=(weights*(np.asarray(labels)*predictions)).sum(1)[valid]
    false_p=(weights*((1-np.asarray(labels))*predictions)).sum(1)[valid]
    rec=true_p/total_p; false_rate=false_p/total_n
    prec=np.divide(true_p,true_p+false_p,out=np.zeros_like(true_p,dtype=float),where=true_p+false_p>0)
    f1=np.divide(2*prec*rec,prec+rec,out=np.zeros_like(rec),where=prec+rec>0)
    return {"image_auroc":roc,"image_ap":ap,"recall":rec,"fpr":false_rate,"precision":prec,"f1":f1},valid


def bootstrap_evaluation(labels, scores, threshold, positive_hist, negative_hist, region_hist, region_counts, category):
    # Same resamples for all model seeds within a category preserve dependence
    # when averaging seed results; category draws are independent.
    rng=np.random.default_rng(2026+CATEGORIES.index(category))
    n=len(labels)
    weights=rng.multinomial(n,np.full(n,1/n),size=2000)
    vectors,valid=image_bootstrap(labels,scores,threshold,weights)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        pos=weights[valid]@positive_hist.astype(np.float64)
        neg=weights[valid]@negative_hist.astype(np.float64)
        region=weights[valid]@region_hist.astype(np.float64)
    counts=weights[valid]@np.asarray(region_counts)
    pixel=np.array([histogram_metrics(p,n,r,c) for p,n,r,c in zip(pos,neg,region,counts)])
    vectors.update(pixel_auroc=pixel[:,0],pixel_ap=pixel[:,1],au_pro=pixel[:,2])
    intervals={name:np.quantile(v,[.025,.975]).tolist() for name,v in vectors.items()}
    return {"replicates":2000,"invalid_single_class_replicates":int((~valid).sum()),
            "95ci":intervals,"pixel_ci_method":"whole-image bootstrap, 4096 fixed [0,1] residual bins; primary metrics exact"},vectors


def evaluate_frozen(identifier, training_result, freeze_sha256):
    """One immutable evaluation reservation per category/seed; never retries itself."""
    from .cnn_research import code_identity
    freeze_path=ROOT/"configs/autoencoder_frozen.json"
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
    result_dir=ROOT/"reports/autoencoder/evaluation"
    result_dir.mkdir(parents=True,exist_ok=True)
    local=ROOT/"outputs/autoencoder"/identifier
    local.mkdir(parents=True,exist_ok=False)
    category=training_result["category"]
    config=training_result["configuration"]
    entry={"experiment_id":identifier+"-start","model_family":"ConvAutoencoder","categories":[category],
           "hypothesis":"Frozen normal-only reconstruction baseline; no further selection",
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
    try:
        checkpoint=ROOT/training_result["checkpoint"]
        if sha256(checkpoint)!=training_result["checkpoint_sha256"]:raise ValueError("Checkpoint identity changed")
        model,saved=load_checkpoint(checkpoint)
        if saved["category"]!=category or saved["training_phase"]!="final":
            raise ValueError("Expected a fresh category-specific final checkpoint")
        # Calibration occurs and is saved BEFORE test rows/images are loaded.
        calibration=calibrate_model(model,category,config["resolution"],config["seed"])
        with (result_dir/f"{identifier}-calibration.json").open("x") as stream:json.dump(calibration,stream,indent=2)
        rows=read_partition("test",category)
        data_root=autodiscover_data_root(ROOT)
        images=load_resized(rows,config["resolution"],allow_test=True).cuda()
        maps=anomaly_map(images.cpu().float()/255,reconstruct(model,images,config["batch_size"]),model.map_kind).numpy()
        # Frozen top-1% mean at model resolution.
        scores=image_scores(torch.from_numpy(maps)).numpy()
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
            idx=np.minimum((native*bins).astype(np.int64),bins-1)
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
        result={"experiment_id":identifier,"category":category,"seed":config["seed"],
                "training_experiment":training_result["experiment_id"],"checkpoint_sha256":training_result["checkpoint_sha256"],
                "image_auroc":image_result.pop("auroc"),"image_ap":image_result.pop("ap"),**image_result,**image_counts,
                **pixel_result,"pixel_confusion_matrix":pixel_cm.tolist(),"pixel_operating_point":pixel_operating,"secondary_image_q99":secondary,
                "calibration":calibration,"latency":latency,"uncertainty":intervals,
                "bootstrap_binned_pixel_point_estimates":dict(zip(["pixel_auroc","pixel_ap","au_pro"],binned)),
                "training_seconds":training_result["training_seconds_including_loading_validation"],
                "training_peak_allocated_bytes":training_result["peak_allocated_bytes"],
                "checkpoint_bytes":training_result["checkpoint_bytes"],"parameters":training_result["parameters"],
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
            assert path.resolve().is_relative_to((ROOT/"outputs/autoencoder").resolve())
            path.unlink()
        return result
    except BaseException as exc:
        (local/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        entry.update(experiment_id=identifier+"-failure",status="failed",interpretation=f"{type(exc).__name__}: {exc}",decision="preserve; investigate without retuning")
        record(ROOT/"experiments",entry)
        raise
