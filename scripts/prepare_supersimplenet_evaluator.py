"""Adapt only model orchestration; retain verified shared native metric logic."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old=(root/'src/anomaly_detection/patchcore_evaluation.py').read_text(encoding='utf-8')
source=old[old.index('def evaluate_frozen('):]
def replace(a,b):
    global source
    assert a in source,a[:100]
    source=source.replace(a,b)
source=source.replace('patchcore_frozen.json','supersimplenet_frozen.json').replace('reports/patchcore','reports/supersimplenet').replace('outputs/patchcore','outputs/supersimplenet')
source=source.replace('PatchCoreVariant','SuperSimpleNetLite').replace('Frozen normal patch-memory confirmation; no further selection','Frozen feature-synthetic discrimination confirmation; no further selection')
replace("        bank,saved=load_bank(checkpoint,category,'cuda')","        model_heads,saved=load_heads(checkpoint,category,config)")
replace('model=PatchPipeline(extractor,bank,config)','model=SSNPipeline(extractor,model_heads,config)')
replace("saved['provenance']['phase']!=\"final\" or saved['config']!=config", "saved['phase']!=\"final\" or saved['configuration']!=config")
replace('''        features,test_cache=feature_cache(extractor,rows)
        patch_distances=score_embeddings(features,bank)
        maps=pixel_maps(patch_distances).numpy()
        np.save(local/'patch_distances.npy',patch_distances.numpy(),allow_pickle=False)
        scores=image_scores(patch_distances,config['aggregation']).numpy()''','''        features=feature_cache(extractor,rows,config['extraction_batch'])
        maps,raw_detection,raw_segmentation=predict(model.heads,features,config['batch_size'])
        scores=image_scores(maps,raw_detection,config['aggregation']).numpy()
        np.save(local/'raw_detection.npy',raw_detection.numpy(),allow_pickle=False)
        np.save(local/'raw_segmentation.npy',raw_segmentation.numpy(),allow_pickle=False)
        maps=maps.numpy()''')
replace('idx=np.minimum((native/(native+calibration["histogram_scale"])*bins).astype(np.int64),bins-1)',
    'idx=np.minimum((expit(native)*bins).astype(np.int64),bins-1)')
replace('''                "coreset_seconds":training_result['coreset_seconds'],
                "original_extraction_seconds":training_result['original_extraction_seconds'],
                "feature_cache_hit_during_fit":training_result['feature_cache']['cache_hit'],
                "test_cache":test_cache,"cpu_peak_rss_bytes":monitor.result['cpu_peak_rss_bytes'],
                "bank_vector_bytes":training_result['bank_vector_bytes'],''','''                "training_loop_seconds":training_result['resources']['seconds'],
                "test_cache":features.metadata,"cpu_peak_rss_bytes":monitor.result['cpu_peak_rss_bytes'],''')
replace('training_result["bank_resources"]["peak_gpu_allocated_bytes"]','training_result["resources"]["peak_gpu_allocated_bytes"]')
replace('"parameters":sum(p.numel() for p in extractor.parameters())','"parameters":sum(p.numel() for p in model.parameters())')
header='''"""Frozen normal calibration and native exact SuperSimpleNet metrics.

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
    pd.DataFrame(times,columns=names).to_csv(output,index=False,lineterminator='\\n')
    context={}
    for key,command in [('windows_scheme',['powercfg','/getactivescheme']),('gpu',['nvidia-smi','--query-gpu=name,driver_version,pstate,clocks.gr,temperature.gpu','--format=csv,noheader'])]:
        try:context[key]=subprocess.check_output(command,text=True,timeout=10).strip()
        except (OSError,subprocess.SubprocessError) as exc:context[key]=str(exc)
    return {name:dict(p50_ms=float(np.median(times[:,i])),p95_ms=float(np.quantile(times[:,i],.95))) for i,name in enumerate(names)} | dict(warmup=50,iterations=200,batch_size=1,precision='float32',scope='Decoded RGB -> native raw-logit CPU map + image score; file decode excluded; model=frozen extractor+learned heads',power_context_at_end_of_timing=context,peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),peak_inference_reserved_bytes=torch.cuda.max_memory_reserved())


'''
(root/'src/anomaly_detection/ssn_evaluation.py').write_bytes((header+source).encode())
