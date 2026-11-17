"""Real-image parity and bounded runtime validation, never research refitting."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import platform
import time
import importlib.metadata
import numpy as np
import pandas as pd
import psutil
import torch
from anomaly_detection.data import CATEGORIES,autodiscover_data_root
from anomaly_detection.inference import PatchCorePredictor
from anomaly_detection.inference.patchcore import sha256,decode_image,preprocess

OUT=ROOT/'reports/engineering/patchcore'
MANIFEST=ROOT/'configs/patchcore_deployment_v1.json'
ATOL=.005
RTOL=.01


def percentiles(values):
    return dict(p50_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)))


def run():
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    target=OUT/'integration.json'
    if target.exists():raise FileExistsError('Integration already recorded; inspect and reuse completed evidence')
    assert torch.cuda.is_available(),'CUDA required for the requested parity validation'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data_root=autodiscover_data_root(ROOT)
    manifest=json.loads(MANIFEST.read_text())
    pinned=sha256(MANIFEST)
    rows=[];timings=[]
    warmup=20;iterations=100
    for category in CATEGORIES:
        entry=manifest['categories'][category]
        table=pd.read_csv(ROOT/entry['predictions']['path'])
        samples=pd.concat([table[table.label.eq(label)].sort_values('path').head(1) for label in (0,1)])
        result=json.loads((ROOT/entry['result']['path']).read_text())
        maps=np.load(ROOT/result['local_maps'],mmap_mode='r')
        patches=np.load(ROOT/'outputs/patchcore'/result['experiment_id']/'patch_distances.npy',mmap_mode='r')
        predictions={}
        for device in ('cpu','cuda'):
            model=PatchCorePredictor(MANIFEST,ROOT,category,device,pinned)
            for sample in samples.itertuples():
                source=data_root/sample.path
                expected_sha=pd.read_csv(ROOT/'reports/partitions/seed2026-test.csv').set_index('path').loc[sample.path,'sha256']
                assert sha256(source)==expected_sha
                prediction=model.predict(source,visualize=False)
                index=int(table.index[table.path.eq(sample.path)][0])
                reference_score=float(np.float32(sample.score))
                np.testing.assert_allclose(prediction.score,reference_score,atol=ATOL,rtol=RTOL)
                np.testing.assert_allclose(prediction.model_map,maps[index],atol=ATOL,rtol=RTOL)
                np.testing.assert_allclose(prediction.patch_distances,patches[index],atol=ATOL,rtol=RTOL)
                assert prediction.is_anomaly==bool(sample.predicted)
                if device=='cpu':predictions[sample.path]=prediction
                else:
                    cpu=predictions[sample.path]
                    np.testing.assert_allclose(prediction.anomaly_map,cpu.anomaly_map,atol=2e-4,rtol=2e-4)
                    assert prediction.is_anomaly==cpu.is_anomaly
                repeated=model.predict(source)
                assert prediction.score==repeated.score and np.array_equal(prediction.model_map,repeated.model_map)
                rows.append(dict(category=category,device=device,path=sample.path,source_sha256=expected_sha,
                    reference_score=reference_score,score=prediction.score,absolute_score_error=abs(prediction.score-reference_score),
                    maximum_model_map_error=float(np.abs(prediction.model_map-maps[index]).max()),
                    maximum_patch_error=float(np.abs(prediction.patch_distances-patches[index]).max()),
                    decision=prediction.is_anomaly,reference_decision=bool(sample.predicted),reproducible=True,
                    maximum_cpu_gpu_map_error=float(np.abs(prediction.anomaly_map-predictions[sample.path].anomaly_map).max())))
            if category in ('bottle','carpet'):
                source=data_root/samples.iloc[0].path
                image=decode_image(source)
                tensor=preprocess(image).to(device)
                for _ in range(warmup):
                    model.model_patches(tensor);model.predict(image)
                if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                wall=[];events=[];decoded=[];encoded=[]
                for _ in range(iterations):
                    if device=='cuda':
                        torch.cuda.synchronize()
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin=time.perf_counter()
                    if device=='cuda':start.record()
                    model.model_patches(tensor)
                    if device=='cuda':end.record();torch.cuda.synchronize();events.append(start.elapsed_time(end))
                    wall.append((time.perf_counter()-begin)*1000)
                    begin=time.perf_counter();model.predict(image)
                    decoded.append((time.perf_counter()-begin)*1000)
                    begin=time.perf_counter();model.predict(source)
                    encoded.append((time.perf_counter()-begin)*1000)
                memory=psutil.Process().memory_info()
                timings.append(dict(category=category,device=device,model_wall=percentiles(wall),
                    model_cuda_event=percentiles(events) if events else None,
                    decoded_end_to_end=percentiles(decoded),file_end_to_end=percentiles(encoded),
                    cpu_rss_bytes=memory.rss,cpu_peak_rss_bytes=getattr(memory,'peak_wset',memory.rss),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0))
            del model
        print(category+': CPU/CUDA parity, original decisions, reference maps and repeatability passed',flush=True)
        pd.DataFrame(rows).to_csv(OUT/'parity.csv',index=False,float_format='%.17g',lineterminator='\n')
    pd.DataFrame(rows).to_csv(OUT/'parity.csv',index=False,float_format='%.17g',lineterminator='\n')
    report=dict(status='passed',manifest_sha256=pinned,categories=15,real_images=30,device_predictions=60,
        repeats=60,sample_policy='Lexicographically first test normal and first defect per category, seed fixed before inspection',
        atol=ATOL,rtol=RTOL,tolerance_note='Historical cuDNN TF32/batch-16 cache versus full-FP32 batch-one: atol .005 + rtol .01. CPU/CUDA portability: atol/rtol .0002. Decisions must match exactly; no score adjustment.',
        max_score_error=max(x['absolute_score_error'] for x in rows),max_model_map_error=max(x['maximum_model_map_error'] for x in rows),
        max_cpu_gpu_native_map_error=max(x['maximum_cpu_gpu_map_error'] for x in rows),
        timings=timings,warmup=warmup,iterations=iterations,precision='deterministic FP32; TF32 off',threads=4,
        methodology='Batch-one 3x256x256; model wall includes extractor+NN, resident input; CUDA events exclude CPU dispatch/transfers. Decoded E2E includes resize/transfer/CPU-native map/score. File E2E additionally opens/decodes a warm filesystem-cache PNG. Both exclude visualization and artifact loading. Synchronize CUDA at model boundaries; output transfer synchronizes E2E.',
        hardware=dict(cpu=platform.processor(),gpu=torch.cuda.get_device_name(0),os=platform.platform()),
        software={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','pillow','psutil']},
        frozen_research_modified=False)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='timings'},indent=2))


if __name__=='__main__':run()
