"""Engineering adapter tests: immutable real artifacts plus small input fixtures."""
from io import BytesIO
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch
from anomaly_detection.data import CATEGORIES
from anomaly_detection.inference import PatchCorePredictor,ArtifactError,InputError
from anomaly_detection.inference.patchcore import decode_image,preprocess,render,verified_path,sha256
from anomaly_detection.patchcore import image_scores,nearest_neighbors

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'configs/patchcore_deployment_v1.json'


@pytest.fixture(scope='module')
def cpu():
    threads=torch.get_num_threads();torch.set_num_threads(4)
    model=PatchCorePredictor(MANIFEST,ROOT,'bottle',expected_manifest_sha256=sha256(MANIFEST))
    yield model
    torch.set_num_threads(threads)


@pytest.mark.parametrize('category',CATEGORIES)
def test_all_real_category_artifacts(category):
    model=PatchCorePredictor(MANIFEST,ROOT,category)
    assert model.seed==2026 and model.bank.shape==(2048,384)
    assert not model.extractor.training
    assert all(not p.requires_grad for p in model.extractor.parameters())
    assert model.configuration['aggregation']=='top1pct'


def test_exact_preprocessing_and_input_forms():
    array=np.random.default_rng(2026).integers(0,256,(71,93,3),dtype=np.uint8)
    image=Image.fromarray(array);stream=BytesIO();image.save(stream,format='PNG')
    expected=torch.tensor(np.asarray(image.resize((256,256),Image.Resampling.BILINEAR)).copy()).permute(2,0,1)[None].float()/255
    for source in (array,image,stream.getvalue()):
        assert torch.equal(preprocess(decode_image(source)),expected)
    assert decode_image(image.convert('L')).mode=='RGB'
    assert decode_image(image.convert('RGBA')).mode=='RGB'


@pytest.mark.parametrize('source',[b'not a PNG',None,17,np.ones((4,4,3),np.float32),np.ones((4,4),np.uint8)])
def test_invalid_inputs(source):
    with pytest.raises(InputError):decode_image(source)


def test_size_limit_and_animated_input():
    with pytest.raises(InputError):decode_image(np.zeros((10,10,3),np.uint8),max_pixels=20)
    with pytest.raises(InputError):decode_image(b'1234',max_bytes=2)
    stream=BytesIO();Image.new('RGB',(3,3)).save(stream,format='GIF',save_all=True,
        append_images=[Image.new('RGB',(3,3),'white')],duration=10,loop=0)
    with pytest.raises(InputError):decode_image(stream.getvalue())


def test_checksum_missing_and_traversal(tmp_path):
    p=tmp_path/'weight.pt';p.write_bytes(b'good')
    entry=dict(path='weight.pt',bytes=4,sha256=sha256(p))
    assert verified_path(tmp_path,entry)==p
    p.write_bytes(b'evil')
    with pytest.raises(ArtifactError,match='checksum'):verified_path(tmp_path,entry)
    with pytest.raises(ArtifactError):verified_path(tmp_path,dict(entry,path='../outside.pt'))
    with pytest.raises(ArtifactError):verified_path(tmp_path,dict(entry,path='missing.pt'))


def test_manifest_threshold_and_seed_mismatches(tmp_path):
    with pytest.raises(ArtifactError,match='manifest checksum'):
        PatchCorePredictor(MANIFEST,ROOT,'bottle',expected_manifest_sha256='0'*64)
    m=json.loads(MANIFEST.read_text());m['categories']['bottle']['thresholds']['image']+=1
    target=tmp_path/'manifest.json';target.write_text(json.dumps(m))
    with pytest.raises(ArtifactError,match='Threshold identity'):PatchCorePredictor(target,ROOT,'bottle')
    m['seed']=2027;target.write_text(json.dumps(m))
    with pytest.raises(ArtifactError):PatchCorePredictor(target,ROOT,'bottle')
    with pytest.raises(ValueError,match='Unsupported category'):PatchCorePredictor(MANIFEST,ROOT,'other')
    with pytest.raises(ValueError,match='CUDA device'):PatchCorePredictor(MANIFEST,ROOT,'bottle','cuda:999')


def test_raw_scoring_top_percent_and_orientation():
    patch=torch.arange(1024,dtype=torch.float32).reshape(1,32,32)
    assert image_scores(patch,'top1pct').item()==torch.arange(1013,1024,dtype=torch.float32).mean().item()
    distance,_=nearest_neighbors(torch.tensor([[0.,0.],[3.,4.]]),torch.tensor([[0.,0.]]))
    assert distance.tolist()==[0.,5.]


def test_visualizations_leave_raw_distances_unchanged():
    image=Image.new('RGB',(9,7),'gray');raw=np.linspace(2,9,63,dtype=np.float32).reshape(7,9);before=raw.copy()
    heat,overlay,meta=render(image,raw)
    assert np.array_equal(raw,before) and heat.shape==overlay.shape==(7,9,3)
    assert heat.dtype==np.uint8 and meta['minimum']==2 and meta['maximum']==9
    heat,_,_=render(image,np.full((7,9),5.,np.float32));assert np.isfinite(heat).all()
    with pytest.raises(ValueError):render(image,raw,alpha=2)


def test_prediction_schema_maps_threshold_and_reproducibility(cpu):
    image=np.full((48,65,3),127,dtype=np.uint8)
    first=cpu.predict(image,visualize=True);second=cpu.predict(image)
    assert first.anomaly_map.shape==(48,65) and first.model_map.shape==(256,256)
    assert first.patch_distances.shape==(32,32) and first.overlay.shape==(48,65,3)
    assert np.array_equal(first.model_map,second.model_map) and first.score==second.score
    assert first.is_anomaly==(first.score>first.threshold)
    assert np.array_equal(first.pixel_decisions,first.anomaly_map>first.pixel_threshold)
    assert first.to_dict()['seed']==2026 and 'anomaly_map' not in first.to_dict()
    assert len(first.to_dict(include_maps=True)['anomaly_map'])==48
    json.dumps(first.to_dict(),allow_nan=False)


def test_cpu_cuda_integration(cpu):
    if not torch.cuda.is_available():pytest.skip('CUDA hardware unavailable')
    gpu=PatchCorePredictor(MANIFEST,ROOT,'bottle','cuda')
    image=np.full((41,59,3),127,dtype=np.uint8)
    before=torch.backends.cudnn.allow_tf32
    a=cpu.predict(image);b=gpu.predict(image)
    assert torch.backends.cudnn.allow_tf32==before
    np.testing.assert_allclose(a.anomaly_map,b.anomaly_map,atol=2e-4,rtol=2e-4)
    assert a.is_anomaly==b.is_anomaly
