"""Single-worker research API. Bounded, memory-only multipart ingestion."""
import asyncio
from contextlib import asynccontextmanager,suppress
import json
import logging
import threading
import time
import torch
import psutil
from uuid import uuid4,UUID
from fastapi import FastAPI,Request,Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,Response
from starlette.exceptions import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartParser,MultiPartException
from starlette.datastructures import UploadFile
from starlette.middleware.cors import CORSMiddleware
from ..data import CATEGORIES
from ..inference import ArtifactError
from .contracts import Settings,PredictionResponse,ErrorResponse,NOTICE
from .service import ModelManager,PatchCoreAdapter,VisualizationStore,ServiceError,execute

LOG=logging.getLogger('anomaly.backend')
if not LOG.handlers:
    handler=logging.StreamHandler();handler.setFormatter(logging.Formatter('%(message)s'));LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


def error(status,code,message,request_id):
    return JSONResponse(status_code=status,content=dict(error=dict(code=code,message=message,request_id=request_id)),
        headers={'X-Request-ID':request_id,'Cache-Control':'no-store'})


class GuardMiddleware:
    """Count actual body bytes before multipart parsing, including chunked input.

    One admitted prediction (upload through completion) per process. No unsafe
    cancellation/release of a still-running GPU thread on an inference deadline.
    """
    def __init__(self,app,settings):self.app=app;self.settings=settings;self.gate=threading.Lock()

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        request_id=str(uuid4());begin=time.perf_counter();scope.setdefault('state',{})
        scope['state'].update(request_id=request_id,started=begin)
        status=500;held=False;started=False
        async def wrapped(message):
            nonlocal status,started
            if message['type']=='http.response.start':
                started=True;status=message['status']
                message['headers']=[(k,v) for k,v in message['headers'] if k.lower() not in (b'x-request-id',b'cache-control')]
                message['headers'] += [(b'x-request-id',request_id.encode()),(b'cache-control',b'no-store'),(b'x-content-type-options',b'nosniff')]
            await send(message)
        try:
            if scope['path']=='/api/v1/predict' and scope['method']=='POST':
                if not self.gate.acquire(blocking=False):
                    return await error(503,'server_busy','One inspection is already in progress; retry later',request_id)(scope,receive,wrapped)
                held=True
                headers=dict(scope['headers']);limit=self.settings.max_upload_bytes+65536
                try:content_length=int(headers.get(b'content-length',b'0'))
                except ValueError:raise ServiceError(400,'invalid_content_length','Invalid Content-Length')
                if content_length<0:raise ServiceError(400,'invalid_content_length','Invalid Content-Length')
                if content_length>limit:raise ServiceError(413,'request_too_large','Request exceeds configured body limit')
                chunks=[];size=0
                try:
                    async with asyncio.timeout(self.settings.upload_timeout_seconds):
                        while True:
                            message=await receive()
                            if message['type']=='http.disconnect':raise ServiceError(400,'upload_disconnected','Upload disconnected')
                            block=message.get('body',b'');size+=len(block)
                            if size>limit:raise ServiceError(413,'request_too_large','Request exceeds configured body limit')
                            chunks.append(block)
                            if not message.get('more_body',False):break
                except TimeoutError:raise ServiceError(408,'upload_timeout','Upload exceeded the configured deadline')
                body=b''.join(chunks);del chunks
                consumed=False
                original_receive=receive
                async def replay():
                    nonlocal consumed
                    if not consumed:
                        consumed=True
                        return dict(type='http.request',body=body,more_body=False)
                    return await original_receive()
                receive=replay
            await self.app(scope,receive,wrapped)
        except ServiceError as exc:
            if not started:await error(exc.status,exc.code,exc.message,request_id)(scope,receive,wrapped)
        except Exception as exc:
            LOG.error(json.dumps(dict(event='request_failed',request_id=request_id,error_type=type(exc).__name__)))
            if not started:await error(500,'internal_error','Inspection failed unexpectedly',request_id)(scope,receive,wrapped)
        finally:
            if held:self.gate.release()
            known=scope['path'] if scope['path'] in ('/health/live','/health/ready','/api/v1/predict','/api/v1/models','/api/v1/categories','/api/v1/model-info') else 'other'
            LOG.info(json.dumps(dict(event='request',request_id=request_id,route=known,status=status,latency_ms=round((time.perf_counter()-begin)*1000,3))))


def create_app(settings=None,factory=PatchCoreAdapter):
    settings=settings or Settings.from_env()
    manager=ModelManager(settings,factory);store=VisualizationStore(settings)
    @asynccontextmanager
    async def lifespan(app):
        previous_threads=torch.get_num_threads();torch.set_num_threads(settings.cpu_threads)
        async def cleanup_expired():
            while True:
                await asyncio.sleep(max(.05,min(5.,settings.visualization_ttl_seconds)))
                with store.lock:store.purge()
        janitor=asyncio.create_task(cleanup_expired())
        try:yield
        finally:
            janitor.cancel()
            with suppress(asyncio.CancelledError):await janitor
            await run_in_threadpool(manager.close);store.close();torch.set_num_threads(previous_threads)
    app=FastAPI(title='Industrial anomaly inspection research API',version='1.0.0',lifespan=lifespan,
        description=NOTICE,debug=False)
    app.state.manager=manager;app.state.visualizations=store;app.state.settings=settings
    app.add_middleware(GuardMiddleware,settings=settings)
    app.add_middleware(CORSMiddleware,allow_origins=list(settings.cors_origins),allow_credentials=False,
        allow_methods=['GET','POST'],allow_headers=['Content-Type'],expose_headers=['X-Request-ID'])
    @app.exception_handler(ServiceError)
    async def service_error(request,exc):return error(exc.status,exc.code,exc.message,request.state.request_id)
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):return error(422,'invalid_parameters','Missing or invalid request parameters',request.state.request_id)
    @app.exception_handler(HTTPException)
    async def http_error(request,exc):return error(exc.status_code,'http_error','Request cannot be served',request.state.request_id)
    common={code:{'model':ErrorResponse} for code in (400,408,413,415,422,500,503)}

    @app.get('/health/live')
    async def live():return dict(status='live',api_version='1.0.0')

    @app.get('/health/ready',responses={503:{'model':ErrorResponse}})
    async def ready():
        def check():
            if not manager.lock.acquire(blocking=False):raise ServiceError(503,'server_busy','Inspection worker is busy')
            try:
                manager.metadata();model,load_ms=manager.get(settings.ready_category)
                if not getattr(model,'serving_verified',False):
                    if hasattr(model,'verify_serving'):model.verify_serving()
                    model.serving_verified=True
                return dict(status='ready',model='patchcore',category=settings.ready_category,device=settings.device,
                    loaded_categories=list(manager.cache),scope='Configured readiness category; other categories validated on load')
            except (ArtifactError,OSError,ValueError,RuntimeError) as exc:
                raise ServiceError(503,'model_unavailable','Configured model artifacts or device are unavailable') from exc
            finally:manager.lock.release()
        return await run_in_threadpool(check)

    @app.get('/api/v1/models')
    async def models():return dict(models=[dict(id='patchcore',version='patchcore-inference-v1',family='Simplified PatchCore',artifact_policy='Verified on load')],research_notice=NOTICE)

    @app.get('/api/v1/categories')
    async def categories():return dict(model='patchcore',categories=list(CATEGORIES))

    @app.get('/api/v1/model-info',responses=common)
    async def info(category:str=Query(default='bottle'),model:str=Query(default='patchcore')):
        validate_selection(model,category)
        def metadata():
            with manager.lock:
                try:m=manager.metadata()
                except (ArtifactError,OSError,ValueError) as exc:raise ServiceError(503,'model_unavailable','Model metadata is unavailable') from exc
                e=m['categories'][category]
                return dict(model='patchcore',version=m['release'],category=category,seed=2026,
                    image_threshold=e['thresholds']['image'],pixel_threshold=e['thresholds']['pixel'],
                    calibration_normal_count=e['calibration_normal_count'],manifest_sha256=settings.manifest_sha256,
                    bank_sha256=e['bank']['sha256'],extractor_sha256=m['extractor']['sha256'],
                    preprocessing=m['preprocessing'],software=m['software'],research_notice=NOTICE,
                    device=settings.device,loaded=category in manager.cache,
                    model_cache=dict(capacity=settings.model_cache_size,loaded_categories=list(manager.cache),loads=manager.loads,evictions=manager.evictions),
                    resources=dict(cpu_rss_bytes=psutil.Process().memory_info().rss,
                        gpu_allocated_bytes=torch.cuda.memory_allocated(settings.device) if settings.device.startswith('cuda') and manager.cache and torch.cuda.is_available() else 0,
                        gpu_reserved_bytes=torch.cuda.memory_reserved(settings.device) if settings.device.startswith('cuda') and manager.cache and torch.cuda.is_available() else 0,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(settings.device) if settings.device.startswith('cuda') and manager.cache and torch.cuda.is_available() else 0))
        return await run_in_threadpool(metadata)

    body_schema=dict(required=True,content={'multipart/form-data':{'schema':dict(type='object',required=['image','category'],
        properties=dict(image=dict(type='string',format='binary'),category=dict(type='string',enum=list(CATEGORIES)),
            model=dict(type='string',enum=['patchcore'],default='patchcore'),visualize=dict(type='boolean',default=False)))}})

    @app.post('/api/v1/predict',response_model=PredictionResponse,responses=common,openapi_extra={'requestBody':body_schema})
    async def predict(request:Request):
        if request.headers.get('content-type','').split(';',1)[0].strip().lower()!='multipart/form-data':
            raise ServiceError(415,'multipart_required','Use multipart/form-data')
        form=None
        try:
            parser=MultiPartParser(request.headers,request.stream(),max_files=1,max_fields=3,max_part_size=1024)
            # Total body was already bounded by the ASGI guard. Never spill an upload to disk.
            parser.spool_max_size=settings.max_upload_bytes+65537
            form=await parser.parse()
            pairs=list(form.multi_items())
            if len(pairs)!=len({k for k,v in pairs}) or set(form)-{'image','category','model','visualize'}:
                raise ServiceError(422,'invalid_parameters','Duplicate or unknown form fields')
            image=form.get('image');category=form.get('category');model=form.get('model','patchcore');visual=form.get('visualize','false')
            if not isinstance(image,UploadFile) or not isinstance(category,str) or not isinstance(model,str) or not isinstance(visual,str):
                raise ServiceError(422,'invalid_parameters','Image and category are required; parameters must be text')
            validate_selection(model,category)
            if visual.lower() not in ('true','false'):raise ServiceError(422,'invalid_parameters','visualize must be true or false')
            data=await image.read(settings.max_upload_bytes+1)
            filename=image.filename or '';mime=(image.content_type or '').lower()
            prediction_id=str(uuid4())
            try:
                result=await run_in_threadpool(execute,manager,store,data,filename,mime,category,visual.lower()=='true',prediction_id)
            except (ArtifactError,OSError,RuntimeError,ValueError) as exc:
                raise ServiceError(503,'model_unavailable','Model artifacts, device or inference runtime are unavailable') from exc
            result.update(request_id=request.state.request_id,latency_ms=(time.perf_counter()-request.state.started)*1000)
            return PredictionResponse(**result)
        except MultiPartException as exc:raise ServiceError(400,'invalid_multipart','Malformed multipart data or excessive fields/files') from exc
        finally:
            if form is not None:await form.close()

    @app.get('/api/v1/predictions/{prediction_id}/heatmap',responses={200:{'content':{'image/png':{'schema':{'type':'string','format':'binary'}}}},404:{'model':ErrorResponse},422:{'model':ErrorResponse}},response_class=Response)
    async def heatmap(prediction_id:UUID):return Response(store.get(str(prediction_id),'heatmap'),media_type='image/png')

    @app.get('/api/v1/predictions/{prediction_id}/overlay',responses={200:{'content':{'image/png':{'schema':{'type':'string','format':'binary'}}}},404:{'model':ErrorResponse},422:{'model':ErrorResponse}},response_class=Response)
    async def overlay(prediction_id:UUID):return Response(store.get(str(prediction_id),'overlay'),media_type='image/png')
    return app


def validate_selection(model,category):
    if model!='patchcore':raise ServiceError(422,'unsupported_model','Only patchcore has a verified adapter')
    if category not in CATEGORIES:raise ServiceError(422,'unsupported_category','Unsupported MVTec category')


app=create_app()
