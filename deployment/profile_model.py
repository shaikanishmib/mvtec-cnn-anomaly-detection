"""Read an encoded benchmark fixture on stdin; profile the unchanged tensor path.

Executed via docker exec python -c by benchmark_release.py, not an API endpoint.
"""
import base64
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch
from anomaly_detection.inference import PatchCorePredictor
from anomaly_detection.inference.patchcore import decode_image,preprocess

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
fixture=json.load(sys.stdin)
device=fixture['device'];category=fixture['category']
model=PatchCorePredictor(Path('/artifacts/configs/patchcore_deployment_v1.json'),Path('/artifacts'),category,device)
image=decode_image(base64.b64decode(fixture['image']))
tensor=preprocess(image).to(device)
for _ in range(5):model.model_patches(tensor)
if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
wall=[];events=[];before=psutil.Process().memory_info().rss
for _ in range(30):
    if device=='cuda':
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    start=time.perf_counter();model.model_patches(tensor)
    if device=='cuda':b.record();torch.cuda.synchronize();events.append(a.elapsed_time(b))
    wall.append((time.perf_counter()-start)*1000)
def summarize(x):return dict(n=len(x),p50_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)))
print(json.dumps(dict(device=device,category=category,warmups=5,model_tensor_wall=summarize(wall),
    cuda_events=summarize(events) if events else None,cpu_rss_before=before,cpu_rss_after=psutil.Process().memory_info().rss,
    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
    scope='Actual RGB image resized/preprocessed before timing; fixed batch-one device-resident 1x3x256x256 tensor; frozen extractor, feature alignment and nearest-neighbor distances. Excludes file decode, preprocessing, device transfer, CPU/native map assembly and visualization. CUDA wall uses explicit synchronization; CUDA events report GPU stream elapsed separately.')))
