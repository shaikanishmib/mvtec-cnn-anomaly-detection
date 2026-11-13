"""PatchCore provenance, deterministic feature caches and scoped resources."""
import threading,time,os,json
from pathlib import Path
import numpy as np
import psutil
import torch
from .cnn_research import ROOT,read_partition,require_fitting,load_resized
from .data import sha256
from .tracking import config_hash
from .patchcore import patch_embeddings,nearest_neighbors


class Resources:
    """Sample process RSS at 20 Hz; CUDA peaks use PyTorch's allocator counters."""
    def __enter__(self):
        self.process=psutil.Process(os.getpid())
        self.initial=self.process.memory_info().rss;self.peak=self.initial
        self.done=threading.Event()
        def sample():
            while not self.done.wait(.05):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        self.started=time.perf_counter()
        return self

    def __exit__(self,*args):
        if torch.cuda.is_available():torch.cuda.synchronize()
        self.done.set();self.thread.join()
        self.peak=max(self.peak,self.process.memory_info().rss)
        self.result=dict(seconds=time.perf_counter()-self.started,cpu_initial_rss_bytes=self.initial,
            cpu_peak_rss_bytes=self.peak,cpu_peak_increment_bytes=self.peak-self.initial,
            cpu_sampling_seconds=.05,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())


@torch.inference_mode()
def feature_cache(extractor,rows,batch_size=16):
    """Immutable once-completed cache, keyed by exact rows, weights and transform.

    This function can load official test images only when called by the frozen
    evaluator. Memory builders independently require fitting-normal provenance.
    """
    if rows.category.nunique()!=1:raise ValueError('A feature cache belongs to one category')
    identity=dict(records=rows[['path','sha256']].to_dict('records'),
        extractor_source_sha256=sha256(ROOT/'src/anomaly_detection/patchcore.py'),
        weight_sha256=extractor.weight_sha256,preprocessing='whole-PIL-bilinear-256; V1-transform-256-identity-geometry',
        layers=['layer2','layer3'],local_pool=3,alignment='bilinear-align_corners_false',dtype='float32')
    key=config_hash(identity)
    folder=ROOT/'outputs/patchcore/features';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.npy');meta=folder/(key+'.json')
    if meta.exists():
        saved=json.loads(meta.read_text())
        if not path.exists() or sha256(path)!=saved['sha256']:raise ValueError('Feature cache changed')
        return np.load(path,mmap_mode='r'),dict(saved,cache_hit=True)
    if path.exists():raise FileExistsError('Incomplete cache preserved; register recovery: '+str(path))
    with Resources() as resource:
        images=load_resized(rows,256,allow_test=rows.partition.eq('test').all())
        output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows),1024,384))
        for start in range(0,len(images),batch_size):
            features=extractor(images[start:start+batch_size].cuda().float()/255)
            output[start:start+len(features)]=patch_embeddings(features).reshape(len(features),1024,384).cpu().numpy()
        output.flush();del output
    saved=dict(identity=identity,sha256=sha256(path),path=path.relative_to(ROOT).as_posix(),
        shape=[len(rows),1024,384],bytes=path.stat().st_size,resources=resource.result)
    with meta.open('x') as stream:json.dump(saved,stream,indent=2)
    return np.load(path,mmap_mode='r'),dict(saved,cache_hit=False)


@torch.inference_mode()
def score_embeddings(features,bank,normalize=False,image_batch=4):
    maps=[]
    for start in range(0,len(features),image_batch):
        block=torch.from_numpy(np.array(features[start:start+image_batch])).cuda().float()
        if normalize:block=torch.nn.functional.normalize(block,dim=-1)
        distances,_=nearest_neighbors(block.reshape(-1,block.shape[-1]),bank)
        maps.append(distances.reshape(len(block),32,32).cpu())
    return torch.cat(maps)


def fitting_embeddings(rows,features):
    require_fitting(rows)
    if rows.category.nunique()!=1 or len(rows)!=len(features):raise ValueError('Category/feature provenance mismatch')
    return torch.from_numpy(np.array(features)).reshape(-1,384).cuda()
