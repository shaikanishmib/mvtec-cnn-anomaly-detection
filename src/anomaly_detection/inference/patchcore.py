"""Inference adapter preserving the original research math and artifact identities."""
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import pickle
from pathlib import Path
import warnings
from threading import RLock
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.nn import functional as F
from ..data import CATEGORIES
from ..patchcore import PatchExtractor, nearest_neighbors, patch_embeddings, pixel_maps, image_scores
from ..tracking import config_hash

_PRECISION_LOCK=RLock()


class ArtifactError(ValueError):
    """A trusted artifact is missing, corrupt or inconsistent."""


class InputError(ValueError):
    """The input cannot be decoded into a permitted RGB image."""


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verified_path(root,entry):
    root=Path(root).resolve()
    relative=Path(entry['path'])
    path=(root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ArtifactError('Artifact path must remain inside artifact_root')
    if not path.is_file():raise ArtifactError(f'Missing artifact: {relative.as_posix()}')
    if path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
        raise ArtifactError(f'Artifact checksum mismatch: {relative.as_posix()}')
    return path


def decode_image(source,max_pixels=32_000_000,max_bytes=32*1024*1024):
    """No EXIF rotation/cropping: match research RGB conversion exactly."""
    try:
        if isinstance(source,np.ndarray):
            if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3:
                raise InputError('Array input must be H x W x 3 uint8 RGB')
            if source.shape[0]*source.shape[1]>max_pixels:raise InputError('Decoded image exceeds pixel limit')
            image=Image.fromarray(source)
        elif isinstance(source,Image.Image):image=source
        else:
            if isinstance(source,(bytes,bytearray)):
                if len(source)>max_bytes:raise InputError('Encoded image exceeds byte limit')
                stream=BytesIO(source)
            elif isinstance(source,(str,Path)):
                path=Path(source)
                if not path.is_file():raise InputError('Image file does not exist')
                if path.stat().st_size>max_bytes:raise InputError('Encoded image exceeds byte limit')
                stream=path
            else:raise InputError('Expected image path, encoded bytes, PIL image or uint8 RGB array')
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(stream) as opened:
                    if opened.width*opened.height>max_pixels:raise InputError('Decoded image exceeds pixel limit')
                    if getattr(opened,'n_frames',1)!=1:raise InputError('Animated/multipage images are unsupported')
                    image=opened.convert('RGB').copy()
        if image.width<1 or image.height<1 or image.width*image.height>max_pixels:
            raise InputError('Invalid image size or pixel limit exceeded')
        return image.convert('RGB').copy()
    except InputError:raise
    except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise InputError('Image is corrupt, unsafe or cannot be decoded') from exc


def preprocess(image):
    array=np.asarray(image.resize((256,256),Image.Resampling.BILINEAR)).copy()
    return torch.from_numpy(array).permute(2,0,1)[None].float()/255


def render(image,anomaly_map,alpha=.45):
    """Display-only min/max scaling; never mutate raw scoring arrays."""
    if not 0<=alpha<=1:raise ValueError('Overlay alpha must be in [0,1]')
    scores=np.asarray(anomaly_map)
    if scores.shape!=(image.height,image.width) or not np.isfinite(scores).all():
        raise ValueError('Expected finite native-resolution anomaly map')
    lo,hi=float(scores.min()),float(scores.max())
    normalized=np.zeros_like(scores,dtype=np.float32) if hi==lo else (scores-lo)/(hi-lo)
    # Fixed RGB color ramp: black/blue through yellow/red; visualization only.
    heat=np.stack([np.clip(2*normalized,0,1),np.clip(2-abs(4*normalized-2),0,1),
                   np.clip(1-2*normalized,0,1)],axis=-1)
    heat=np.round(heat*255).astype(np.uint8)
    overlay=np.round((1-alpha)*np.asarray(image,dtype=float)+alpha*heat).astype(np.uint8)
    return heat,overlay,dict(method='per-image min/max display only',minimum=lo,maximum=hi,alpha=alpha)


@dataclass(frozen=True)
class Prediction:
    category:str
    seed:int
    device:str
    manifest_sha256:str
    score:float
    threshold:float
    is_anomaly:bool
    pixel_threshold:float
    anomaly_map:np.ndarray
    model_map:np.ndarray
    patch_distances:np.ndarray
    pixel_decisions:np.ndarray
    heatmap:np.ndarray | None=None
    overlay:np.ndarray | None=None
    visualization:dict | None=None

    def to_dict(self,include_maps=False):
        result=dict(schema_version=1,model_family='Simplified PatchCore',category=self.category,
                    seed=self.seed,device=self.device,manifest_sha256=self.manifest_sha256,
                    score=self.score,image_threshold=self.threshold,is_anomaly=self.is_anomaly,
                    decision_rule='score > threshold',score_semantics='raw L2 distance; larger is more anomalous; not a probability',
                    pixel_threshold=self.pixel_threshold,native_shape=list(self.anomaly_map.shape),
                    model_map_shape=list(self.model_map.shape),visualization=self.visualization)
        if include_maps:
            result.update(anomaly_map=self.anomaly_map.tolist(),pixel_decisions=self.pixel_decisions.tolist())
        return result


class PatchCorePredictor:
    """One category per instance. Construct once, then reuse read-only weights.

    No implicit device fallback, weight download, coreset fitting or calibration.
    Checksum verification occurs before deserialization on every construction.
    Treat the release manifest / optional expected hash as the trust root.
    """
    def __init__(self,manifest_path,artifact_root,category,device='cpu',expected_manifest_sha256=None):
        if category not in CATEGORIES:raise ValueError(f'Unsupported category {category!r}; expected one of {CATEGORIES}')
        manifest_path=Path(manifest_path)
        try:
            self.manifest_sha256=sha256(manifest_path)
            if expected_manifest_sha256 and self.manifest_sha256!=expected_manifest_sha256:
                raise ArtifactError('Deployment manifest checksum mismatch')
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['schema_version']!=1 or manifest['seed']!=2026 or set(manifest['categories'])!=set(CATEGORIES):
                raise ArtifactError('Unsupported manifest version, seed policy or categories')
            if config_hash(manifest['preprocessing'])!=manifest['preprocessing_sha256']:
                raise ArtifactError('Preprocessing identity mismatch')
            root=Path(artifact_root)
            policy=json.loads(verified_path(root,manifest['frozen_policy']).read_text())
            if manifest['configuration']!=policy['configuration']:
                raise ArtifactError('Deployment configuration differs from frozen research')
            entry=manifest['categories'][category]
            calibration=json.loads(verified_path(root,entry['calibration']).read_text())
            thresholds=entry['thresholds']
            if thresholds!={'image':calibration['image_threshold'],'pixel':calibration['pixel_threshold'],'strict_greater':True}:
                raise ArtifactError('Threshold identity differs from original calibration')
            extractor_path=verified_path(root,manifest['extractor'])
            bank_path=verified_path(root,entry['bank'])
        except ArtifactError:raise
        except (KeyError,TypeError,ValueError,OSError) as exc:
            raise ArtifactError('Deployment manifest is missing, malformed or incomplete') from exc
        # torch.device may wrap very large indices internally; validate the
        # original text before construction rather than trusting that index.
        if isinstance(device,str) and device.startswith('cuda:'):
            try:requested_index=int(device.split(':',1)[1])
            except ValueError as exc:raise ValueError('Invalid CUDA device index') from exc
            if requested_index<0 or requested_index>=torch.cuda.device_count():
                raise ValueError('Requested CUDA device is unavailable; select cpu explicitly')
        try:self.device=torch.device(device)
        except (RuntimeError,ValueError,TypeError) as exc:raise ValueError('Expected cpu or an available cuda device') from exc
        if self.device.type not in ('cpu','cuda') or (self.device.type=='cpu' and self.device.index is not None):
            raise ValueError('Only cpu or cuda[:index] devices are supported')
        if self.device.type=='cuda' and (not torch.cuda.is_available() or (self.device.index or 0)<0 or (self.device.index or 0)>=torch.cuda.device_count()):
            raise ValueError('Requested CUDA device is unavailable; select cpu explicitly')
        try:
            weights=torch.load(extractor_path,map_location='cpu',weights_only=True)
            saved=torch.load(bank_path,map_location='cpu',weights_only=True)
            if weights['original_weight_sha256']!=manifest['extractor']['original_weight_sha256']:
                raise ArtifactError('Pretrained extractor provenance mismatch')
            if saved['category']!=category or saved['config']!=manifest['configuration'] or saved['provenance']['phase']!='final':
                raise ArtifactError('Category/seed/configuration/final-bank provenance mismatch')
            bank=saved['bank']
            if bank.shape!=(2048,384) or bank.dtype!=torch.float32 or not torch.isfinite(bank).all():
                raise ArtifactError('Expected finite float32 2048 x 384 memory bank')
            self.extractor=PatchExtractor(pretrained=False)
            self.extractor.load_state_dict(weights['state_dict'],strict=True)
            self.extractor.weight_sha256=weights['original_weight_sha256']
            self.extractor.to(self.device).eval()
            self.bank=bank.to(self.device)
        except ArtifactError:raise
        except (RuntimeError,KeyError,TypeError,ValueError,OSError,EOFError,pickle.UnpicklingError) as exc:
            raise ArtifactError('Cannot deserialize a compatible frozen extractor or bank') from exc
        self.category=category
        self.seed=manifest['seed']
        self.threshold=float(thresholds['image']);self.pixel_threshold=float(thresholds['pixel'])
        self.configuration=dict(manifest['configuration'])
        if not np.isfinite([self.threshold,self.pixel_threshold]).all():raise ArtifactError('Nonfinite calibration thresholds')

    @torch.inference_mode()
    def model_patches(self,tensor):
        # Preserve full FP32 and restore cuDNN flags after this call. Package
        # contexts are serialized; unrelated clients must coordinate global flags.
        with _PRECISION_LOCK, torch.backends.cudnn.flags(benchmark=False,deterministic=True,allow_tf32=False):
            if self.device.type=='cuda' and torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError('Disable torch.backends.cuda.matmul.allow_tf32 for full-FP32 inference')
            features=self.extractor(tensor)
            distances,_=nearest_neighbors(patch_embeddings(features),self.bank,
                query_chunk=self.configuration['query_chunk'],bank_chunk=self.configuration['bank_chunk'])
        return distances.reshape(1,32,32)

    @torch.inference_mode()
    def predict(self,source,visualize=False):
        image=decode_image(source)
        tensor=preprocess(image).to(self.device)
        # Research exports patches to CPU before map interpolation / score reduction.
        patches=self.model_patches(tensor).cpu()
        model_map=pixel_maps(patches)
        native=F.interpolate(model_map[:,None],size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0].numpy()
        score=float(image_scores(patches,self.configuration['aggregation']).item())
        if not np.isfinite(score) or not np.isfinite(native).all():raise RuntimeError('Inference produced nonfinite anomaly scores')
        heat,overlay,description=render(image,native) if visualize else (None,None,None)
        return Prediction(self.category,self.seed,str(self.device),self.manifest_sha256,score,self.threshold,
            score>self.threshold,self.pixel_threshold,native,model_map[0].numpy(),patches[0].numpy(),
            native>self.pixel_threshold,heat,overlay,description)
