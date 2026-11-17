"""Bounded model/visualization lifetime; authoritative inference remains in M10."""
from collections import OrderedDict
from io import BytesIO
import gc
import json
import threading
import time
import warnings
import numpy as np
from PIL import Image
import torch
from ..data import CATEGORIES
from ..inference import PatchCorePredictor,ArtifactError,InputError
from ..inference.patchcore import decode_image,render,sha256
from .contracts import Settings


class ServiceError(Exception):
    def __init__(self,status,code,message):
        self.status,self.code,self.message=status,code,message
        super().__init__(message)


class PatchCoreAdapter:
    def __init__(self,settings,category):
        self.predictor=PatchCorePredictor(settings.manifest,settings.artifact_root,category,
            settings.device,settings.manifest_sha256)

    def predict(self,image):return self.predictor.predict(image,visualize=False)

    def verify_serving(self):
        self.predict(Image.new('RGB',(16,16),(127,127,127)))

    def close(self):
        cuda=self.predictor is not None and self.predictor.device.type=='cuda'
        self.predictor=None
        gc.collect()
        if cuda:
            torch.cuda.synchronize();torch.cuda.empty_cache()


class ModelManager:
    def __init__(self,settings:Settings,factory=PatchCoreAdapter):
        self.settings=settings;self.factory=factory
        self.cache=OrderedDict();self.lock=threading.RLock()
        self.loads=0;self.evictions=0;self.manifest=None

    def metadata(self):
        if self.manifest is None:
            if sha256(self.settings.manifest)!=self.settings.manifest_sha256:
                raise ArtifactError('Manifest identity mismatch')
            self.manifest=json.loads(self.settings.manifest.read_text())
        return self.manifest

    def get(self,category):
        if category not in CATEGORIES:raise ServiceError(422,'unsupported_category','Unsupported MVTec category')
        if category in self.cache:
            self.cache.move_to_end(category)
            return self.cache[category],0.
        start=time.perf_counter()
        # Evict before allocating another GPU model, respecting the memory bound.
        if len(self.cache)>=self.settings.model_cache_size:
            _,old=self.cache.popitem(last=False);old.close();self.evictions+=1
        model=self.factory(self.settings,category)
        self.cache[category]=model;self.loads+=1
        return model,(time.perf_counter()-start)*1000

    def close(self):
        with self.lock:
            for model in self.cache.values():model.close()
            self.cache.clear()


class VisualizationStore:
    def __init__(self,settings,clock=time.monotonic):
        self.settings=settings;self.clock=clock;self.items=OrderedDict();self.bytes=0;self.lock=threading.Lock()

    def purge(self):
        for key in list(self.items):
            if self.items[key][0]<=self.clock():self.remove(key)

    def remove(self,key):
        _,content=self.items.pop(key);self.bytes-=sum(map(len,content.values()))

    def put(self,key,content):
        size=sum(map(len,content.values()))
        if size>self.settings.visualization_bytes:
            raise ServiceError(413,'visualization_limit','Encoded visualizations exceed configured memory limit')
        with self.lock:
            self.purge()
            while self.items and (len(self.items)>=self.settings.visualization_items or self.bytes+size>self.settings.visualization_bytes):
                self.remove(next(iter(self.items)))
            self.items[key]=(self.clock()+self.settings.visualization_ttl_seconds,content)
            self.bytes+=size

    def get(self,key,kind):
        with self.lock:
            self.purge()
            if key not in self.items:raise ServiceError(404,'visualization_not_found','Visualization expired, evicted or unavailable')
            return self.items[key][1][kind]

    def close(self):
        with self.lock:self.items.clear();self.bytes=0


def validated_image(data,filename,mime,settings):
    extensions={'.png':('image/png','PNG'),'.jpg':('image/jpeg','JPEG'),'.jpeg':('image/jpeg','JPEG')}
    # Extract only a suffix; never open, log, return or save the client filename.
    suffix='.'+filename.rsplit('.',1)[-1].lower() if '.' in filename else ''
    if suffix not in extensions or extensions[suffix][0]!=mime:
        raise ServiceError(415,'unsupported_image_type','Use a matching PNG/.png or JPEG/.jpg/.jpeg content type and extension')
    if not data:raise ServiceError(422,'empty_image','The uploaded image is empty')
    if len(data)>settings.max_upload_bytes:raise ServiceError(413,'upload_too_large','Image exceeds configured upload limit')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as opened:
                if opened.format!=extensions[suffix][1]:
                    raise ServiceError(415,'format_mismatch','Decoded format does not match MIME type and extension')
                if opened.width>settings.max_dimension or opened.height>settings.max_dimension or opened.width*opened.height>settings.max_pixels:
                    raise ServiceError(413,'image_dimensions','Image dimensions or pixel count exceed configured limits')
                if getattr(opened,'n_frames',1)!=1:
                    raise ServiceError(422,'animated_image','Animated images are unsupported')
        return decode_image(data,max_pixels=settings.max_pixels,max_bytes=settings.max_upload_bytes)
    except ServiceError:raise
    except (OSError,ValueError,InputError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise ServiceError(422,'invalid_image','The uploaded image cannot be decoded safely') from exc


def png(array):
    stream=BytesIO();Image.fromarray(np.asarray(array,dtype=np.uint8)).save(stream,format='PNG')
    return stream.getvalue()


def execute(manager,store,data,filename,mime,category,visualize,prediction_id):
    with manager.lock:
        start=time.perf_counter()
        image=validated_image(data,filename,mime,manager.settings)
        decoded=time.perf_counter()
        metadata=manager.metadata()
        model,load_ms=manager.get(category)
        before=time.perf_counter();prediction=model.predict(image);after=time.perf_counter()
        visual={}
        if visualize:
            heat,overlay,scaling=render(image,prediction.anomaly_map)
            store.put(prediction_id,dict(heatmap=png(heat),overlay=png(overlay)))
            base=f'/api/v1/predictions/{prediction_id}'
            visual=dict(heatmap_url=base+'/heatmap',overlay_url=base+'/overlay',
                expires_in_seconds=manager.settings.visualization_ttl_seconds,scaling=scaling)
        finish=time.perf_counter();entry=metadata['categories'][category]
        return dict(prediction_id=prediction_id,category=category,anomaly_score=prediction.score,
            threshold=prediction.threshold,is_anomaly=prediction.is_anomaly,pixel_threshold=prediction.pixel_threshold,
            input_width=image.width,input_height=image.height,device=prediction.device,visualizations=visual,
            timings_ms=dict(decode=(decoded-start)*1000,model_load=load_ms,inference=(after-before)*1000,
                visualization=(finish-after)*1000,service_total=(finish-start)*1000),
            provenance=dict(manifest_sha256=prediction.manifest_sha256,bank_sha256=entry['bank']['sha256'],
                extractor_sha256=metadata['extractor']['sha256'],pretrained_weight_sha256=metadata['extractor']['original_weight_sha256'],
                calibration_sha256=entry['calibration']['sha256'],preprocessing_sha256=metadata['preprocessing_sha256'],seed=2026))
