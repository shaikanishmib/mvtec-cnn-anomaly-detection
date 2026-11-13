"""Adapt the verified native evaluator; leave earlier models untouched."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old=(root/'src/anomaly_detection/ae_evaluation.py').read_text(encoding='utf-8')
source=old[old.index('def evaluate_frozen('):]
def replace(old,new):
    global source
    if old not in source:raise ValueError('Missing evaluator adaptation: '+old[:80])
    source=source.replace(old,new)
replace('def evaluate_frozen(identifier, training_result, freeze_sha256):','def evaluate_frozen(identifier, training_result, freeze_sha256, extractor):')
source=source.replace('autoencoder_frozen.json','patchcore_frozen.json').replace('reports/autoencoder','reports/patchcore').replace('outputs/autoencoder','outputs/patchcore')
source=source.replace('ConvAutoencoder','PatchCoreVariant').replace('Frozen normal-only reconstruction baseline; no further selection','Frozen normal patch-memory confirmation; no further selection')
replace('    started=time.perf_counter()','    started=time.perf_counter()\n    monitor=Resources();monitor.__enter__()')
replace('        model,saved=load_checkpoint(checkpoint)\n        if saved["category"]!=category or saved["training_phase"]!="final":',
'''        bank,saved=load_bank(checkpoint,category,'cuda')
        if extractor.weight_sha256!=frozen['weight_sha256']:raise ValueError('Pretrained weights changed')
        model=PatchPipeline(extractor,bank,config)
        if saved["category"]!=category or saved['provenance']['phase']!="final" or saved['config']!=config:''')
replace('''        images=load_resized(rows,config["resolution"],allow_test=True).cuda()
        maps=anomaly_map(images.cpu().float()/255,reconstruct(model,images,config["batch_size"]),model.map_kind).numpy()
        # Frozen top-1% mean at model resolution.
        scores=image_scores(torch.from_numpy(maps)).numpy()''',
'''        features,test_cache=feature_cache(extractor,rows)
        patch_distances=score_embeddings(features,bank)
        maps=pixel_maps(patch_distances).numpy()
        np.save(local/'patch_distances.npy',patch_distances.numpy(),allow_pickle=False)
        scores=image_scores(patch_distances,config['aggregation']).numpy()''')
replace('idx=np.minimum((native*bins).astype(np.int64),bins-1)',
    'idx=np.minimum((native/(native+calibration["histogram_scale"])*bins).astype(np.int64),bins-1)')
replace('        result={','        monitor.__exit__(None,None,None)\n        result={')
replace('"training_seconds":training_result["training_seconds_including_loading_validation"],',
'''"training_seconds":training_result["wall_seconds"],
                "coreset_seconds":training_result['coreset_seconds'],
                "original_extraction_seconds":training_result['original_extraction_seconds'],
                "feature_cache_hit_during_fit":training_result['feature_cache']['cache_hit'],
                "test_cache":test_cache,"cpu_peak_rss_bytes":monitor.result['cpu_peak_rss_bytes'],
                "bank_vector_bytes":training_result['bank_vector_bytes'],''')
replace('"training_peak_allocated_bytes":training_result["peak_allocated_bytes"],',
    '"training_peak_allocated_bytes":training_result["bank_resources"]["peak_gpu_allocated_bytes"],')
replace('"parameters":training_result["parameters"],',
    '"parameters":sum(p.numel() for p in extractor.parameters()),')
replace('    except BaseException as exc:',
    '    except BaseException as exc:\n        if not monitor.done.is_set():monitor.__exit__(None,None,None)')
header='''"""Frozen distance calibration and exact native PatchCore evaluation.

Primary metrics use raw L2 distances. Binned uncertainty uses only a monotone
calibration-scaled transform, never clipping raw distances to probability range.
"""
from pathlib import Path
from datetime import datetime,timezone
import gc,json,time,traceback,subprocess
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label
import torch
from torch.nn import functional as F
from .patchcore import nearest_neighbors,patch_embeddings,pixel_maps,image_scores,load_bank
from .patchcore_research import ROOT,Resources,feature_cache,score_embeddings
from .cnn_research import read_partition
from .data import autodiscover_data_root,sha256
from .metrics import binary_metrics,calibrate_normal
from .pixel_metrics import exact_pixel_metrics,histogram_metrics
from .tracking import config_hash,record
from .cnn_evaluation import bootstrap_evaluation as shared_bootstrap


def bootstrap_evaluation(*args):
    result,vectors=shared_bootstrap(*args)
    result['pixel_ci_method']='Whole-image bootstrap; 4096 bins of d/(d+calibration median patch distance); exact raw-distance primary metrics'
    return result,vectors


class PatchPipeline(torch.nn.Module):
    def __init__(self,extractor,bank,config):
        super().__init__()
        self.extractor=extractor;self.bank=bank;self.config=config

    @torch.inference_mode()
    def patches(self,images):
        features=self.extractor(images)
        distances,_=nearest_neighbors(patch_embeddings(features),self.bank)
        return distances.reshape(len(images),32,32)

    def forward(self,images):return pixel_maps(self.patches(images))[:,None]


def calibrate_model(model,category,resolution,seed=2026):
    rows=read_partition('calibration',category)
    features,cache=feature_cache(model.extractor,rows)
    patches=score_embeddings(features,model.bank)
    scores=image_scores(patches,model.config['aggregation']).numpy()
    maps=pixel_maps(patches).numpy()
    threshold=calibrate_normal(scores)
    rng=np.random.default_rng(seed)
    draws=rng.choice(scores,size=(2000,len(scores)),replace=True)
    intervals=np.quantile(np.quantile(draws,.95,axis=1,method='higher'),[.025,.975]).tolist()
    return dict(category=category,images=len(rows),image_threshold=threshold,
        secondary_image_threshold_q99=calibrate_normal(scores,.01),
        pixel_threshold=float(np.quantile(maps,.995,method='higher')),
        image_quantile=.95,pixel_quantile=.995,pixel_calibration_samples=maps.size,
        pixel_score_resolution=256,calibration_scores=scores.tolist(),
        image_threshold_bootstrap_95ci=intervals,
        histogram_scale=max(float(np.median(patches.numpy())),1e-6),
        histogram_transform='d/(d+normal calibration median patch distance), 4096 bins; no raw-score clipping',
        method='Reserved normals only; higher empirical image q95 and pooled pixel q99.5; strict >',
        source_paths_sha256=config_hash(rows[['path','sha256']].to_dict('records')),feature_cache=cache)


@torch.inference_mode()
def benchmark_model(model,calibration_path,resolution,native_shape,output):
    with Image.open(autodiscover_data_root(ROOT)/calibration_path) as image:raw=image.convert('RGB').copy()
    def once():
        start=time.perf_counter()
        array=np.asarray(raw.resize((256,256),Image.Resampling.BILINEAR)).copy()
        x=torch.from_numpy(array).permute(2,0,1)[None].float()/255
        pre=time.perf_counter();x=x.cuda();torch.cuda.synchronize();transfer=time.perf_counter()
        features=model.extractor(x);torch.cuda.synchronize();extracted=time.perf_counter()
        distances,_=nearest_neighbors(patch_embeddings(features),model.bank)
        patches=distances.reshape(1,32,32);torch.cuda.synchronize();searched=time.perf_counter()
        maps=pixel_maps(patches)
        result=F.interpolate(maps[:,None],size=native_shape,mode='bilinear',align_corners=False).cpu().numpy()
        float(image_scores(patches,model.config['aggregation']).item())
        torch.cuda.synchronize();finish=time.perf_counter()
        return np.array([pre-start,transfer-pre,extracted-transfer,searched-extracted,
            searched-transfer,finish-searched,finish-start])*1000
    for _ in range(50):once()
    torch.cuda.reset_peak_memory_stats()
    times=np.stack([once() for _ in range(200)])
    names=['preprocessing','host_to_device','feature_extraction','nearest_neighbor','model','postprocessing','end_to_end']
    pd.DataFrame(times,columns=names).to_csv(output,index=False,lineterminator='\\n')
    context={}
    for key,command in [('windows_scheme',['powercfg','/getactivescheme']),
        ('gpu',['nvidia-smi','--query-gpu=name,driver_version,pstate,clocks.gr,temperature.gpu','--format=csv,noheader'])]:
        try:context[key]=subprocess.check_output(command,text=True,timeout=10).strip()
        except (OSError,subprocess.SubprocessError) as exc:context[key]=str(exc)
    return {name:dict(p50_ms=float(np.median(times[:,i])),p95_ms=float(np.quantile(times[:,i],.95)))
            for i,name in enumerate(names)} | dict(warmup=50,iterations=200,batch_size=1,precision='float32',
        scope='Memory-resident decoded RGB -> native raw-distance CPU map + image score; file decode excluded; model=feature extraction+NN',
        power_context_at_end_of_timing=context,peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_inference_reserved_bytes=torch.cuda.max_memory_reserved())


'''
(root/'src/anomaly_detection/patchcore_evaluation.py').write_bytes((header+source).encode('utf-8'))
