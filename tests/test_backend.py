"""Lightweight HTTP/security/resource tests use an explicitly mocked adapter."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import json
import threading
import time
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from anomaly_detection.backend.app import create_app
from anomaly_detection.backend.contracts import Settings,PIN,PredictionResponse
from anomaly_detection.backend.service import VisualizationStore
from anomaly_detection.inference import ArtifactError


def image_bytes(format='PNG',size=(24,20)):
    stream=BytesIO();Image.new('RGB',size,(42,71,93)).save(stream,format=format)
    return stream.getvalue()


class FakeAdapter:
    def __init__(self,settings,category):self.category=category;self.closed=False
    def predict(self,image):
        return SimpleNamespace(score=1.25,threshold=2.,is_anomaly=False,pixel_threshold=.7,device='cpu',
            manifest_sha256=PIN,anomaly_map=np.full((image.height,image.width),.5,np.float32))
    def close(self):self.closed=True


@pytest.fixture
def client():
    app=create_app(factory=FakeAdapter)
    with TestClient(app) as c:yield c
    assert len(app.state.manager.cache)==0 and app.state.visualizations.bytes==0


def post(client,**fields):
    return client.post('/api/v1/predict',files={'image':('input.png',image_bytes(),'image/png')},data=dict(category='bottle',**fields))


def test_startup_health_discovery_openapi(client):
    assert not client.app.state.manager.cache
    assert client.get('/health/live').json()['status']=='live'
    assert client.get('/health/ready').status_code==200
    assert client.app.state.manager.loads==1
    assert client.get('/api/v1/models').json()['models'][0]['id']=='patchcore'
    assert len(client.get('/api/v1/categories').json()['categories'])==15
    schema=client.get('/openapi.json').json()
    assert 'multipart/form-data' in schema['paths']['/api/v1/predict']['post']['requestBody']['content']
    assert 'PredictionResponse' in schema['components']['schemas']
    info=client.get('/api/v1/model-info').json();assert info['seed']==2026 and 'bank_sha256' in info
    assert info['model_cache']['capacity']==2 and info['resources']['cpu_rss_bytes']>0
    assert 'path' not in info


def test_success_schema_visualization_and_reuse(client):
    r=post(client,visualize='true');assert r.status_code==200,r.text
    result=PredictionResponse.model_validate(r.json())
    assert result.anomaly_score==1.25 and result.threshold==2 and not result.is_anomaly
    assert result.request_id==r.headers['x-request-id']
    assert not any(k in r.json() for k in ('anomaly_map','probability','confidence'))
    for url in [result.visualizations.heatmap_url,result.visualizations.overlay_url]:
        response=client.get(url);assert response.status_code==200
        assert response.headers['content-type']=='image/png'
        assert Image.open(BytesIO(response.content)).size==(24,20)
    assert post(client).status_code==200 and client.app.state.manager.loads==1


@pytest.mark.parametrize('filename,mime,data,status,code',[
    ('x.png','image/png',b'',422,'empty_image'),('x.png','image/png',b'bad',422,'invalid_image'),
    ('x.exe','image/png',image_bytes(),415,'unsupported_image_type'),
    ('x.png','text/plain',image_bytes(),415,'unsupported_image_type'),
    ('x.jpg','image/jpeg',image_bytes(),415,'format_mismatch'),
    ('x.webp','image/webp',image_bytes(),415,'unsupported_image_type')])
def test_bad_images(client,filename,mime,data,status,code):
    r=client.post('/api/v1/predict',files={'image':(filename,data,mime)},data={'category':'bottle'})
    assert r.status_code==status and r.json()['error']['code']==code
    assert r.json()['error']['request_id']==r.headers['x-request-id']


@pytest.mark.parametrize('fields,code',[
    ({'category':'other'},'unsupported_category'),({'category':'bottle','model':'padim'},'unsupported_model'),
    ({'category':'bottle','visualize':'yes'},'invalid_parameters'),({},'invalid_parameters'),
    ({'category':'bottle','device':'cuda'},'invalid_parameters')])
def test_bad_parameters(client,fields,code):
    r=client.post('/api/v1/predict',files={'image':('x.png',image_bytes(),'image/png')},data=fields)
    assert r.status_code==422 and r.json()['error']['code']==code


def test_body_and_pixel_limits():
    app=create_app(replace(Settings(),max_upload_bytes=80,max_pixels=100),FakeAdapter)
    with TestClient(app) as c:
        r=c.post('/api/v1/predict',files={'image':('x.png',image_bytes(),'image/png')},data={'category':'bottle'})
        assert r.status_code==413
        r=c.post('/api/v1/predict',content=b'x'*66000,headers={'content-type':'multipart/form-data; boundary=x'})
        assert r.status_code==413 and r.json()['error']['code']=='request_too_large'
    with TestClient(create_app(replace(Settings(),max_pixels=10),FakeAdapter)) as c:
        assert post(c).json()['error']['code']=='image_dimensions'


def test_bomb_and_malformed_multipart(client,monkeypatch):
    monkeypatch.setattr(Image,'MAX_IMAGE_PIXELS',10)
    assert post(client).status_code==422
    r=client.post('/api/v1/predict',content=b'bad',headers={'content-type':'multipart/form-data'})
    assert r.status_code==400
    assert client.post('/api/v1/predict',json={'image':'x'}).status_code==415


def test_artifact_failure_sanitized_and_live():
    def fail(*args):raise ArtifactError('secret C:/Users/private/bank.pt')
    with TestClient(create_app(factory=fail)) as c:
        r=c.get('/health/ready');assert r.status_code==503 and 'secret' not in r.text
        r=post(c);assert r.status_code==503 and 'private' not in r.text
        assert c.get('/health/live').status_code==200
    with TestClient(create_app(replace(Settings(),manifest_sha256='0'*64),FakeAdapter)) as c:
        assert c.get('/health/ready').status_code==503


def test_eviction_and_cleanup():
    app=create_app(replace(Settings(),model_cache_size=1),FakeAdapter)
    with TestClient(app) as c:
        assert post(c).status_code==200;old=app.state.manager.cache['bottle']
        r=c.post('/api/v1/predict',files={'image':('x.png',image_bytes(),'image/png')},data={'category':'carpet'})
        assert r.status_code==200 and old.closed and app.state.manager.evictions==1
        assert len(app.state.manager.cache)==1
    assert not app.state.manager.cache


def test_concurrent_requests_rejected_without_parallel_inference():
    entered=threading.Event();release=threading.Event()
    class Slow(FakeAdapter):
        def predict(self,image):
            entered.set();assert release.wait(5)
            return super().predict(image)
    with TestClient(create_app(factory=Slow)) as c,ThreadPoolExecutor(2) as pool:
        future=pool.submit(post,c);assert entered.wait(5)
        busy=post(c);assert busy.status_code==503 and busy.json()['error']['code']=='server_busy'
        assert c.get('/health/live').status_code==200
        release.set();assert future.result().status_code==200


def test_visualization_ttl_byte_item_limits():
    now=[0.];store=VisualizationStore(replace(Settings(),visualization_items=1,visualization_bytes=8,visualization_ttl_seconds=2),lambda:now[0])
    a,b=str(uuid4()),str(uuid4());store.put(a,{'heatmap':b'a','overlay':b'b'})
    store.put(b,{'heatmap':b'c','overlay':b'd'})
    from anomaly_detection.backend.service import ServiceError
    with pytest.raises(ServiceError):store.get(a,'heatmap')
    now[0]=3
    with pytest.raises(ServiceError):store.get(b,'heatmap')
    assert store.bytes==0
    with pytest.raises(ServiceError):store.put(a,{'heatmap':b'x'*9})


def test_cors_paths_and_request_logging(client,caplog):
    assert 'access-control-allow-origin' not in client.get('/health/live',headers={'Origin':'https://evil.invalid'}).headers
    assert client.get('/api/v1/predictions/'+str(uuid4())+'/overlay').status_code==404
    bad=client.get('/api/v1/predictions/not-a-uuid/overlay')
    assert bad.status_code==422 and bad.json()['error']['code']=='invalid_parameters'
    schema=client.get('/openapi.json').json()
    assert schema['paths']['/api/v1/predictions/{prediction_id}/overlay']['get']['responses']['422']['content']['application/json']['schema']['$ref'].endswith('ErrorResponse')
    with caplog.at_level('INFO',logger='anomaly.backend'):
        r=client.post('/api/v1/predict',files={'image':('../../sensitive.png',image_bytes(),'image/png')},data={'category':'bottle'})
    assert r.status_code==200 and 'sensitive' not in caplog.text
    with pytest.raises(ValueError):replace(Settings(),cors_origins=('*',))


def test_chunked_body_limit_and_upload_deadline():
    async def check(delay,body,expected):
        app=create_app(replace(Settings(),max_upload_bytes=16,upload_timeout_seconds=.01),FakeAdapter)
        sent=[]
        async def receive():
            if delay:await asyncio.sleep(delay)
            return dict(type='http.request',body=body,more_body=False)
        async def send(message):sent.append(message)
        scope=dict(type='http',asgi={'version':'3.0'},method='POST',path='/api/v1/predict',raw_path=b'/api/v1/predict',
                   query_string=b'',headers=[],scheme='http',server=('test',80),client=('test',1),http_version='1.1')
        await app(scope,receive,send)
        assert next(x['status'] for x in sent if x['type']=='http.response.start')==expected
    asyncio.run(check(0,b'x'*66000,413));asyncio.run(check(.1,b'',408))


def test_large_upload_is_memory_only_and_closed(client,monkeypatch):
    import starlette.formparsers as parser
    original=parser.SpooledTemporaryFile;opened=[]
    def tracked(*args,**kwargs):
        stream=original(*args,**kwargs);opened.append(stream);return stream
    monkeypatch.setattr(parser,'SpooledTemporaryFile',tracked)
    buffer=BytesIO();Image.fromarray(np.random.default_rng(1).integers(0,256,(768,768,3),dtype=np.uint8)).save(buffer,format='PNG')
    assert len(buffer.getvalue())>1024*1024
    r=client.post('/api/v1/predict',files={'image':('large.png',buffer.getvalue(),'image/png')},data={'category':'bottle'})
    assert r.status_code==200
    assert opened and all(x.closed and not x._rolled for x in opened)


def test_failure_releases_admission_and_jpeg_support(client):
    bad=client.post('/api/v1/predict',files={'image':('bad.png',b'bad','image/png')},data={'category':'bottle'})
    assert bad.status_code==422
    good=client.post('/api/v1/predict',files={'image':('photo.jpeg',image_bytes('JPEG'),'image/jpeg')},data={'category':'bottle'})
    assert good.status_code==200


def test_real_missing_artifact_root_sanitized(tmp_path):
    with TestClient(create_app(replace(Settings(),artifact_root=tmp_path))) as c:
        assert c.get('/health/ready').status_code==503
        r=post(c);assert r.status_code==503 and str(tmp_path) not in r.text


def test_expired_visualizations_are_cleaned_without_a_retrieval():
    app=create_app(replace(Settings(),visualization_ttl_seconds=.05),FakeAdapter)
    with TestClient(app) as client:
        assert post(client,visualize='true').status_code==200
        time.sleep(.2)
        assert app.state.visualizations.bytes==0 and not app.state.visualizations.items
