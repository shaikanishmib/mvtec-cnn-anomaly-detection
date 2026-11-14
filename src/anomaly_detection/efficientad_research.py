"""Bounded EfficientAD research helpers; never open official test data."""
from contextlib import contextmanager
from pathlib import Path
import json
import subprocess
import time
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image
from scipy.stats import beta, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.transforms import functional as TF
from anomaly_detection.data import sha256
from anomaly_detection.metrics import calibrate_normal

MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)


def configure(root):
    import sys
    assert Path(sys.executable).resolve()==root/'.venv/Scripts/python.exe'
    torch.set_num_threads(4)
    torch.manual_seed(2026); np.random.seed(2026)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()


def read_normals(root,category,partition,verify=True):
    if partition not in ('fit','development','calibration'):
        raise ValueError('Only official training-normal partitions permitted')
    table=pd.read_csv(root/f'reports/partitions/seed2026-{partition}.csv')
    rows=table[table.category==category].sort_values('path').reset_index(drop=True)
    assert len(rows) and rows.partition.eq(partition).all() and rows.split.eq('train').all()
    images=[]
    for row in rows.itertuples():
        assert '/train/good/' in row.path
        path=root/'data/mvtec_ad'/row.path
        if verify: assert sha256(path)==row.sha256
        with Image.open(path) as im:
            images.append(TF.to_tensor(TF.resize(im.convert('RGB'),[256,256])).unsqueeze(0))
    return rows,images


def normalized(x):
    return (x-MEAN)/STD


def proxy_edits(x,index):
    """Three deterministic edits from fitting normals; no test data/templates."""
    if x.shape != (1,3,256,256): raise ValueError('Expected batch-one RGB')
    rng=np.random.default_rng(9006+index)
    y,z=map(int,rng.integers(48,176,size=2))
    scratch=x.clone(); scratch[:,:,y:y+2,z:z+48]=1-scratch[:,:,y:y+2,z:z+48]
    block=x.clone(); block[:,:,y:y+32,z:z+32]=torch.flip(x[:,:,y:y+32,z:z+32],[-1,-2])*.3
    color=x.clone(); color[:,0,y:y+32,z:z+32]=1.; color[:,1:,y:y+32,z:z+32]*=.25
    return dict(scratch=scratch,block=block,color_patch=color)


def gpu_snapshot():
    fields='name,pstate,temperature.gpu,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.current.sm,memory.used'
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        return dict(zip(fields.split(','),[v.strip() for v in output.split(',')]))
    except Exception as exc:
        return dict(unavailable=repr(exc))


def percentiles(values):
    a=np.asarray(values,dtype=float)
    return dict(p50_ms=float(np.quantile(a,.5)),p95_ms=float(np.quantile(a,.95)),count=len(a))


@torch.inference_mode()
def latency_audit(model,raw,path,plan):
    """Identical FP32 forward; events bracket stream work, wall includes host waits."""
    model.eval(); cpu=normalized(raw).contiguous(); x=cpu.cuda()
    samples=[]; repeats=[]
    process=psutil.Process()
    for repeat in range(plan['repeats']):
        for _ in range(plan['warmup']): model(x)
        torch.cuda.synchronize()
        before_gpu=gpu_snapshot(); start_cpu=process.cpu_times(); start=time.perf_counter()
        for index in range(plan['iterations_per_repeat']):
            torch.cuda.synchronize()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin=time.perf_counter(); a.record(); model(x); b.record(); b.synchronize()
            samples.append(dict(repeat=repeat,index=index,scope='resident',wall_ms=(time.perf_counter()-begin)*1000,gpu_ms=a.elapsed_time(b)))
        elapsed=time.perf_counter()-start; end_cpu=process.cpu_times()
        repeats.append(dict(repeat=repeat,gpu_before=before_gpu,gpu_after=gpu_snapshot(),
            elapsed_seconds=elapsed,cpu_process_seconds=(end_cpu.user+end_cpu.system)-(start_cpu.user+start_cpu.system),
            cpu_percent_one_core=100*((end_cpu.user+end_cpu.system)-(start_cpu.user+start_cpu.system))/elapsed))
    # Diagnostic burst keeps CPU from deliberately waiting between each forward.
    burst=[]
    for _ in range(3):
        torch.cuda.synchronize(); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        t=time.perf_counter(); a.record()
        for _ in range(50): model(x)
        b.record(); b.synchronize()
        burst.append(dict(images=50,wall_ms_per_image=(time.perf_counter()-t)*1000/50,gpu_stream_ms_per_image=a.elapsed_time(b)/50))
    transfer=[]; e2e=[]
    for i in range(100):
        torch.cuda.synchronize(); t=time.perf_counter(); device=cpu.cuda(); torch.cuda.synchronize()
        transfer.append((time.perf_counter()-t)*1000)
        torch.cuda.synchronize(); t=time.perf_counter()
        with Image.open(path) as im: decoded=TF.to_tensor(TF.resize(im.convert('RGB'),[256,256])).unsqueeze(0)
        m,s=model(normalized(decoded).cuda()); m.cpu(); s.cpu(); torch.cuda.synchronize()
        e2e.append((time.perf_counter()-t)*1000)
    return dict(methodology='Same deterministic FP32 model, no semantic changes. 50 warmups/repeat, resident 1x3x256x256 input. Explicit synchronization and CUDA events; event time includes stream launch gaps. End-to-end includes file decode, CPU preprocessing, H2D, forward, D2H map and score.',
        resident_wall=percentiles([s['wall_ms'] for s in samples]),resident_gpu_event=percentiles([s['gpu_ms'] for s in samples]),
        per_repeat=[dict(repeat=i,wall=percentiles([s['wall_ms'] for s in samples if s['repeat']==i]),gpu=percentiles([s['gpu_ms'] for s in samples if s['repeat']==i])) for i in range(3)],
        host_to_device=percentiles(transfer),end_to_end=percentiles(e2e),bursts=burst,telemetry=repeats,samples=samples,
        end_to_end_samples_ms=e2e,transfer_samples_ms=transfer)


@torch.inference_mode()
def collect_raw_maps(model,images):
    model.eval(); pairs=[model.raw_maps(normalized(x).cuda()) for x in images]
    # Move each image promptly: memory bounded by category, no native test maps.
    return tuple(torch.cat([p[i].cpu() for p in pairs]) for i in range(2))


def map_quantiles(pairs):
    return torch.stack([torch.quantile(a.flatten(),torch.tensor([.9,.995])) for a in pairs])


def score_maps(pairs,quantiles):
    local,global_map=[.1*(a-q[0])/(q[1]-q[0]).clamp_min(1e-12) for a,q in zip(pairs,quantiles)]
    combined=(local+global_map)*.5
    return dict(combined_max=combined.flatten(1).amax(1).numpy(),local_max=local.flatten(1).amax(1).numpy(),
        global_max=global_map.flatten(1).amax(1).numpy(),
        raw_combined_max=((pairs[0]+pairs[1])*.5).flatten(1).amax(1).numpy(),
        combined_top1pct=combined.flatten(1).topk(656,dim=1).values.mean(1).numpy())


def distribution(values):
    a=np.asarray(values)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),std=float(a.std()),
                q90=float(np.quantile(a,.9)),q95=float(np.quantile(a,.95)),max=float(a.max()))


def calibration_diagnostics(scores,development,policy):
    """Fixed policy only: no threshold/model selection from these observations."""
    a=np.asarray(scores); n=len(a); threshold=calibrate_normal(a,policy['target_fpr'])
    rng=np.random.default_rng(policy['bootstrap_seed'])
    boot=np.array([calibrate_normal(rng.choice(a,n,replace=True),policy['target_fpr']) for _ in range(policy['bootstrap_replicates'])])
    exceed=int((a>threshold).sum())
    lo=0. if exceed==0 else float(beta.ppf(.025,exceed,n-exceed+1))
    hi=1. if exceed==n else float(beta.ppf(.975,exceed+1,n-exceed))
    loo=np.array([calibrate_normal(np.delete(a,i),policy['target_fpr']) for i in range(n)])
    index=int(np.ceil((n-1)*(1-policy['target_fpr'])))
    return dict(threshold=threshold,nominal_fpr=policy['target_fpr'],strict_greater=True,quantile_method='higher',
        calibration_distribution=distribution(a),observed_calibration_fpr=exceed/n,exceedances=exceed,
        calibration_exceedance_descriptive_cp95=[lo,hi],
        uncertainty_note='CP interval is descriptive; threshold fitted to these same samples. IID continuous future FPR follows the order-statistic Beta distribution; bootstrap is unstable with small tails.',
        future_fpr_order_statistic_mean=(n-index)/(n+1),future_fpr_order_statistic_95=beta.ppf([.025,.975],n-index,index+1).tolist(),
        bootstrap_threshold_p025_p50_p975=np.quantile(boot,[.025,.5,.975]).tolist(),bootstrap_replicates=len(boot),
        leave_one_out_threshold_range=[float(loo.min()),float(loo.max())],
        sensitivity_diagnostic_only={str(fpr):calibrate_normal(a,fpr) for fpr in (.01,.05,.1)},
        development_exceedance_fraction=float((np.asarray(development)>threshold).mean()),
        threshold_policy=policy)


def select_configuration(results,plan):
    """Predeclared rule. Only development-fit results, never calibration inputs."""
    candidates=[]
    for epochs in plan['epoch_budgets']:
        group=[r for r in results if r['epochs']==epochs]
        if set(r['category'] for r in group)!=set(plan['categories']): raise ValueError('Incomplete category coverage')
        for score in plan['candidate_scores']:
            eligible=all(r['finite'] and r['probe_after']['hard']<r['probe_before']['hard'] and
                r['probe_after']['autoencoder']<r['probe_before']['autoencoder'] and r['proxy'][score]['auroc']>=.65 for r in group)
            candidates.append(dict(epochs=epochs,score=score,eligible=eligible,
                macro_proxy_auroc=float(np.mean([r['proxy'][score]['auroc'] for r in group])),
                worst_category_proxy_auroc=min(r['proxy'][score]['auroc'] for r in group)))
    eligible=[c for c in candidates if c['eligible']]
    if not eligible: return dict(eligible=False,candidates=candidates,selected=max(candidates,key=lambda c:c['macro_proxy_auroc']))
    best=max(eligible,key=lambda c:c['macro_proxy_auroc'])
    short=[c for c in eligible if c['epochs']==min(plan['epoch_budgets']) and c['macro_proxy_auroc']>=best['macro_proxy_auroc']-.02]
    budget=min(plan['epoch_budgets']) if short else best['epochs']
    pool=[c for c in eligible if c['epochs']==budget]
    winner=max(pool,key=lambda c:c['macro_proxy_auroc'])
    combined=[c for c in pool if c['score']=='combined_max' and c['macro_proxy_auroc']>=winner['macro_proxy_auroc']-.02]
    return dict(eligible=True,candidates=candidates,selected=combined[0] if combined else winner)
