"""Stable frontend-facing contracts and deployment-only settings."""
from dataclasses import dataclass,field
import os
import math
from pathlib import Path
from typing import Literal,Protocol,Any
from pydantic import BaseModel,ConfigDict,Field

ROOT=Path(__file__).resolve().parents[3]
PIN='58be2eafb16ac012ac18e0feb69f11b2f510e7417464eaba5bd9797b423272fa'
NOTICE='Research use only. Raw anomaly distances are not probabilities; frozen normal-only cutoffs can have substantial category-dependent false alarms and misses.'


@dataclass(frozen=True)
class Settings:
    artifact_root:Path=ROOT
    manifest:Path=ROOT/'configs/patchcore_deployment_v1.json'
    manifest_sha256:str=PIN
    device:str='cpu'
    cpu_threads:int=4
    ready_category:str='bottle'
    model_cache_size:int=2
    max_upload_bytes:int=8*1024*1024
    max_pixels:int=4_000_000
    max_dimension:int=4096
    upload_timeout_seconds:float=15.
    visualization_bytes:int=32*1024*1024
    visualization_items:int=8
    visualization_ttl_seconds:float=300.
    cors_origins:tuple[str,...]=field(default_factory=tuple)

    def __post_init__(self):
        if not 1<=self.model_cache_size<=15:raise ValueError('Model cache size must be 1..15')
        if not 1<=self.cpu_threads<=32:raise ValueError('CPU threads must be 1..32')
        if '*' in self.cors_origins:raise ValueError('Explicit CORS origins are required')
        for value in (self.model_cache_size,self.max_upload_bytes,self.max_pixels,self.max_dimension,
                      self.upload_timeout_seconds,self.visualization_bytes,self.visualization_items,self.visualization_ttl_seconds):
            if not math.isfinite(value) or value<=0:raise ValueError('Resource limits must be finite and positive')
        if self.model_cache_size>15 or self.max_upload_bytes>32*1024*1024 or self.max_pixels>32_000_000:
            raise ValueError('Resource limit exceeds inference contract')

    @classmethod
    def from_env(cls):
        values={}
        for key,convert in [('artifact_root',Path),('manifest',Path),('device',str),('manifest_sha256',str),
            ('ready_category',str),('cpu_threads',int),('model_cache_size',int),('max_upload_bytes',int),('max_pixels',int),
            ('max_dimension',int),('upload_timeout_seconds',float),('visualization_bytes',int),
            ('visualization_items',int),('visualization_ttl_seconds',float)]:
            name='ANOMALY_'+key.upper()
            if name in os.environ:values[key]=convert(os.environ[name])
        if 'ANOMALY_CORS_ORIGINS' in os.environ:
            values['cors_origins']=tuple(x.strip() for x in os.environ['ANOMALY_CORS_ORIGINS'].split(',') if x.strip())
        return cls(**values)


class Adapter(Protocol):
    def predict(self,image:Any):...
    def close(self)->None:...


class ErrorDetail(BaseModel):
    code:str
    message:str
    request_id:str


class ErrorResponse(BaseModel):
    error:ErrorDetail


class Provenance(BaseModel):
    manifest_sha256:str
    bank_sha256:str
    extractor_sha256:str
    pretrained_weight_sha256:str
    calibration_sha256:str
    preprocessing_sha256:str
    seed:Literal[2026]


class Visualizations(BaseModel):
    heatmap_url:str|None=None
    overlay_url:str|None=None
    expires_in_seconds:float|None=None
    scaling:dict|None=None


class PredictionResponse(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False,extra='forbid')
    schema_version:Literal['1.0']='1.0'
    prediction_id:str
    request_id:str
    model:Literal['patchcore']='patchcore'
    model_version:Literal['patchcore-inference-v1']='patchcore-inference-v1'
    category:str
    anomaly_score:float
    threshold:float
    is_anomaly:bool
    pixel_threshold:float
    decision_rule:Literal['score > threshold']='score > threshold'
    score_semantics:str='Raw L2 distance; higher is more anomalous; not a probability'
    input_width:int
    input_height:int
    device:str
    latency_ms:float=Field(ge=0)
    timings_ms:dict[str,float]
    provenance:Provenance
    visualizations:Visualizations
    limitations:list[str]=Field(default_factory=lambda:[NOTICE])
