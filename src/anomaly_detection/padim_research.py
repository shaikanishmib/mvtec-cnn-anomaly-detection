"""PaDiM phase gates, immutable evidence and fit-only Gaussian provenance."""
import json
import time
import traceback
from contextlib import contextmanager
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from .cnn_research import ROOT, read_partition, require_fitting, load_resized, inner_fit_split, make_probes, code_identity
from .patchcore_research import Resources
from .padim import PaDiMExtractor, SpatialGaussian, anomaly_maps
from .data import sha256, CATEGORIES
from .tracking import record, config_hash


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@contextmanager
def experiment(identifier, category, configuration, phase):
    code=code_identity()
    code['source_sha256'].update({p.relative_to(ROOT).as_posix():sha256(p)
        for p in (ROOT/'src/anomaly_detection').glob('padim*.py')})
    event = dict(experiment_id=identifier+'-start', model_family='PaDiM', categories=[category],
        hypothesis=phase, preprocessing={'resolution':256,'normalization':'ImageNet V1'},
        hyperparameters=configuration, seed=configuration['seed'],
        environment=json.loads((ROOT/'reports/padim/environment.json').read_text()),
        training={'code':code,'phase':phase}, metrics={}, calibration={}, paths={},
        interpretation='Reserved before execution', decision='run registered phase', status='started')
    record(ROOT/'experiments', event)
    result = {}
    try:
        yield result
        write_json(ROOT/f'reports/padim/runs/{identifier}-result.json', result)
        event.update(experiment_id=identifier+'-result', metrics=result, status='completed',
                     interpretation='Actual measured execution', decision='retain')
        record(ROOT/'experiments', event)
    except BaseException:
        event.update(experiment_id=identifier+'-failure', status='failed',
                     interpretation=traceback.format_exc(), decision='preserve and investigate')
        record(ROOT/'experiments', event)
        raise


@torch.inference_mode()
def extract(extractor, images, batch=8):
    return torch.cat([extractor(images[i:i+batch].cuda().float()/255).cpu()
                      for i in range(0,len(images),batch)])


@torch.inference_mode()
def score(gaussian, features, sigma=4.):
    return torch.cat([anomaly_maps(gaussian.score(features[i:i+4]), sigma).cpu()
                      for i in range(0,len(features),4)])


def source_identity(rows):
    return config_hash(rows[['path','sha256','partition']].to_dict('records'))


def pilot():
    config = dict(dimension=100, epsilon=.01, sigma=4., seed=2026, resolution=256)
    identifier='05-pilot-bottle-s2026-v1'
    with experiment(identifier, 'bottle', config, 'real normal GPU pilot') as result:
        fit = read_partition('fit','bottle').iloc[:24]
        dev = read_partition('development','bottle').iloc[:8]
        require_fitting(fit)
        with Resources() as resources:
            extractor = PaDiMExtractor(100,2026).cuda()
            images = load_resized(fit,256)
            torch.cuda.synchronize(); start=time.perf_counter()
            maps=extractor.feature_maps(images[:1].cuda().float()/255)
            raw_shapes=[list(x.shape) for x in maps]
            features=extract(extractor,images)
            torch.cuda.synchronize(); extraction=time.perf_counter()-start
            start=time.perf_counter()
            gaussian=SpatialGaussian().fit(features,'cuda')
            torch.cuda.synchronize(); fitting=time.perf_counter()-start
            dev_features=extract(extractor,load_resized(dev,256))
            start=time.perf_counter(); scores=score(gaussian,dev_features)
            torch.cuda.synchronize(); scoring=time.perf_counter()-start
            image_scores=scores.flatten(1).max(1).values.numpy()
        assert np.isfinite(image_scores).all() and scores.shape==(8,256,256)
        assert resources.result['seconds']<600 and resources.result['peak_gpu_allocated_bytes']<6*1024**3
        assert resources.result['cpu_peak_rss_bytes']<16*1024**3
        result.update(configuration=config, raw_layer_shapes=raw_shapes, selected_fit_shape=list(features.shape),
            extraction_seconds=extraction, gaussian_fit_seconds=fitting, development_scoring_seconds=scoring,
            resources=resources.result, score_min=float(scores.min()),score_max=float(scores.max()),
            image_score_quantiles=np.quantile(image_scores,[0,.25,.5,.75,1]).tolist(),
            image_scores=image_scores.tolist(), fit_source_sha256=source_identity(fit),
            development_source_sha256=source_identity(dev),weight_sha256=extractor.weight_sha256,
            passed=True, gpu=torch.cuda.get_device_name(0))
    return result


def require_pilot():
    pilot=json.loads((ROOT/'reports/padim/runs/05-pilot-bottle-s2026-v1-result.json').read_text())
    if not pilot['passed']: raise ValueError('Pilot has not passed')


def development():
    require_pilot()
    for dimension in (50,100):
        for category in ('bottle','cable','carpet','tile'):
            fit, probe_rows=inner_fit_split(read_partition('fit',category))
            dev=read_partition('development',category)
            extractor=PaDiMExtractor(dimension,2026).cuda()
            fit_features=extract(extractor,load_resized(fit,256))
            dev_features=extract(extractor,load_resized(dev,256))
            probes,masks=make_probes(load_resized(probe_rows,256),probe_rows)
            probe_features=extract(extractor,(probes*255).round().to(torch.uint8))
            for epsilon in (.01,.1):
                config=dict(dimension=dimension,epsilon=epsilon,sigma=4.,seed=2026,resolution=256)
                identifier=f'05-dev-d{dimension}-e{int(epsilon*100):02d}-{category}-s2026-v1'
                if (ROOT/f'reports/padim/runs/{identifier}-result.json').exists():continue
                with experiment(identifier,category,config,'bounded synthetic development') as result:
                    with Resources() as resource:
                        gaussian=SpatialGaussian(epsilon).fit(fit_features,'cuda')
                        normal_scores=score(gaussian,dev_features).flatten(1).max(1).values.numpy()
                        maps=score(gaussian,probe_features)
                        scores=maps.flatten(1).max(1).values.numpy()
                        labels=masks.flatten(1).any(1).numpy().astype(int)
                        threshold=float(np.quantile(normal_scores,.95,method='higher'))
                        auroc=float(roc_auc_score(labels,scores))
                        pixel_ap=float(average_precision_score(masks.flatten().numpy(),maps.flatten().numpy()))
                    result.update(category=category,configuration=config,proxy_image_auroc=auroc,
                        proxy_pixel_ap=pixel_ap,selection_objective=(auroc+pixel_ap)/2,
                        proxy_recall=float((scores[labels==1]>threshold).mean()),
                        development_fpr=float((normal_scores>threshold).mean()),
                        fit_source_sha256=source_identity(fit),probe_source_sha256=source_identity(probe_rows),
                        development_source_sha256=source_identity(dev),resources=resource.result)
                    print(identifier,result['selection_objective'],flush=True)
            del fit_features,dev_features,probe_features,extractor,gaussian
            torch.cuda.empty_cache()


def freeze():
    require_pilot()
    results=[json.loads(p.read_text()) for p in (ROOT/'reports/padim/runs').glob('05-dev-*-result.json')]
    if len(results)!=16:raise ValueError('Expected all 16 registered development results')
    ranked=[]
    for dimension in (50,100):
        for epsilon in (.01,.1):
            group=[r for r in results if r['configuration']['dimension']==dimension and r['configuration']['epsilon']==epsilon]
            ranked.append(dict(dimension=dimension,epsilon=epsilon,objective=float(np.mean([r['selection_objective'] for r in group]))))
    selected=sorted(ranked,key=lambda r:(-r['objective'],r['dimension'],-r['epsilon']))[0]
    sources=['src/anomaly_detection/padim.py','src/anomaly_detection/padim_research.py',
             'src/anomaly_detection/padim_evaluation.py','src/anomaly_detection/pixel_metrics.py',
             'src/anomaly_detection/patchcore.py','src/anomaly_detection/cnn_research.py',
             'src/anomaly_detection/cnn_evaluation.py','docs/padim_protocol.md']
    frozen=dict(configuration=dict(dimension=selected['dimension'],epsilon=selected['epsilon'],sigma=4.,resolution=256),
        seeds=[2026,2027,2028],categories=list(CATEGORIES),development=ranked,
        source_sha256={p:sha256(ROOT/p) for p in sources},
        environment_sha256=sha256(ROOT/'reports/padim/environment.json'),
        dataset_validation_sha256=sha256(ROOT/'reports/dataset_validation.json'),
        weight_sha256=json.loads((ROOT/'reports/padim/runs/05-pilot-bottle-s2026-v1-result.json').read_text())['weight_sha256'],
        code=code_identity(),selection='mean synthetic objective; no real test access')
    write_json(ROOT/'configs/padim_frozen.json',frozen)
    write_json(ROOT/'reports/padim/freeze_identity.json',dict(sha256=sha256(ROOT/'configs/padim_frozen.json')))
    return frozen
