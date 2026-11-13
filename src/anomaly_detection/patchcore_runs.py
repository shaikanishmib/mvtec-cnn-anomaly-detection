"""Auditable category-specific normal banks and bounded development proxies."""
import copy,json,time,traceback,subprocess
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from .patchcore import projected_coreset,nearest_neighbors,pixel_maps,image_scores,save_bank
from .patchcore_research import ROOT,Resources,feature_cache,score_embeddings,fitting_embeddings
from .cnn_research import read_partition,inner_fit_split,load_resized,make_probes
from .data import sha256
from .tracking import record,config_hash
from .metrics import binary_metrics,calibrate_normal


@torch.inference_mode()
def proxy_cache(extractor,rows):
    identity=dict(rows=rows[['path','sha256']].to_dict('records'),weight=extractor.weight_sha256,
        core=sha256(ROOT/'src/anomaly_detection/patchcore.py'),synthetic_source=sha256(ROOT/'src/anomaly_detection/basic_cnn.py'),seed=9901)
    key=config_hash(identity)
    folder=ROOT/'outputs/patchcore/proxies';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.npz');meta=folder/(key+'.json')
    if meta.exists():
        saved=json.loads(meta.read_text())
        if sha256(path)!=saved['sha256']:raise ValueError('Changed synthetic feature cache')
        data=np.load(path,allow_pickle=False)
        return data['features'],data['masks'],dict(saved,cache_hit=True)
    if path.exists():raise FileExistsError('Incomplete proxy cache preserved')
    with Resources() as resources:
        original=load_resized(rows,256).cuda()
        probes,masks=make_probes(original,rows)
        features=torch.cat([extractor(x).permute(0,2,3,1).reshape(len(x),1024,384).cpu()
                            for x in probes.split(16)]).numpy()
        masks=masks[:,0].cpu().numpy().astype(np.uint8)
        np.savez(path,features=features,masks=masks)
    saved=dict(identity=identity,path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),resources=resources.result)
    with meta.open('x') as stream:json.dump(saved,stream,indent=2)
    return features,masks,dict(saved,cache_hit=False)


def build_bank(identifier,category,config,extractor,final=False):
    folder=ROOT/'reports/patchcore/runs';folder.mkdir(parents=True,exist_ok=True)
    local=ROOT/'outputs/patchcore'/identifier;local.mkdir(parents=True,exist_ok=False)
    checkpoint=ROOT/'checkpoints/patchcore'/identifier/'bank.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=False)
    entry=dict(experiment_id=identifier+'-start',model_family='PatchCoreVariant',categories=[category],
        hypothesis=f"A {config['bank_budget']}-patch projected coreset represents category normals within bounded search cost." if not final else 'Fresh category/seed bank confirms frozen policy without test-driven selection.',
        preprocessing='Whole PIL bilinear 256 + V1 transform resize/crop 256; ImageNet normalization',
        hyperparameters=config,seed=config['seed'],environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            weight_sha256=extractor.weight_sha256,lock_sha256=sha256(ROOT/'requirements.lock.txt'),
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha256(ROOT/p) for p in ['src/anomaly_detection/patchcore.py','src/anomaly_detection/patchcore_research.py','src/anomaly_detection/patchcore_runs.py']},
            partitions=json.loads((ROOT/'reports/dataset_validation.json').read_text())['partitions']),
        training=dict(phase='final' if final else 'development'),metrics={},calibration={},paths=dict(local=local.relative_to(ROOT).as_posix()),
        interpretation='No calibration/test data in bank construction or development',decision='retain actual results',status='started')
    record(ROOT/'experiments',entry)
    started=time.perf_counter()
    try:
        rows=read_partition('fit',category)
        learning,probe_rows=(rows,None) if final else inner_fit_split(rows)
        features,feature_metadata=feature_cache(extractor,learning)
        cache_original_extraction_seconds=feature_metadata['resources']['seconds']
        with Resources() as resources:
            embeddings=fitting_embeddings(learning,features)
            tick=time.perf_counter()
            indices,coverage=projected_coreset(embeddings,config['bank_budget'],config['seed'],
                config['candidate_limit'],config['projection_dim'])
            bank=embeddings[indices].clone();del embeddings
            torch.cuda.synchronize();selection_seconds=time.perf_counter()-tick
            provenance=dict(partition='fit',source_rows_sha256=config_hash(learning[['path','sha256']].to_dict('records')),
                feature_cache=feature_metadata,weight_sha256=extractor.weight_sha256,phase='final' if final else 'development')
            save_bank(checkpoint,bank,indices,category,config,provenance)
        metrics={};timing={}
        if not final:
            dev,dev_meta=feature_cache(extractor,read_partition('development',category))
            probes,masks,probe_meta=proxy_cache(extractor,probe_rows)
            with Resources() as search_resources:
                tick=time.perf_counter();dev_maps=score_embeddings(dev,bank)
                proxy_maps=score_embeddings(probes,bank)
                torch.cuda.synchronize();search_seconds=time.perf_counter()-tick
                spatial=pixel_maps(proxy_maps).numpy()
                pixel_ap=float(average_precision_score(masks.ravel(),spatial.ravel()))
                labels=masks.reshape(len(masks),-1).any(1).astype(int)
                for aggregation in ['max','top1pct']:
                    scores=image_scores(proxy_maps,aggregation).numpy()
                    threshold=calibrate_normal(image_scores(dev_maps,aggregation).numpy())
                    values=binary_metrics(labels,scores,threshold)
                    metrics[aggregation]=dict(**values,pixel_ap=pixel_ap,objective=(values['auroc']+pixel_ap)/2)
                q=torch.from_numpy(np.array(dev[:1])).cuda().reshape(-1,384)
                for _ in range(5):nearest_neighbors(q,bank)
                durations=[]
                for _ in range(20):
                    tick=time.perf_counter();nearest_neighbors(q,bank);torch.cuda.synchronize()
                    durations.append((time.perf_counter()-tick)*1000)
            timing=dict(development_search_seconds=search_seconds,search_p50_ms=float(np.median(durations)),
                search_p95_ms=float(np.quantile(durations,.95)),resources=search_resources.result,
                normal_development_patch_distance_quantiles=np.quantile(dev_maps.numpy(),[0,.5,.95,1]).tolist(),
                normal_cache=dev_meta,proxy_cache=probe_meta)
            np.savez(local/'development_maps.npz',normal=dev_maps.numpy(),proxy=proxy_maps.numpy(),masks=masks)
        result=dict(experiment_id=identifier,category=category,configuration=config,phase='final' if final else 'development',
            fitting_normals=len(learning),probe_sources=0 if final else len(probe_rows),
            candidate_coverage=coverage,coreset_seconds=selection_seconds,bank_resources=resources.result,
            checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha256(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size,bank_vector_bytes=bank.numel()*4,
            feature_cache=feature_metadata,original_extraction_seconds=cache_original_extraction_seconds,
            metrics=metrics,development=timing,wall_seconds=time.perf_counter()-started)
        with (folder/(identifier+'-result.json')).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        event=copy.deepcopy(entry)
        event.update(experiment_id=identifier+'-result',status='completed',metrics=result,
            decision='await bounded development selection' if not final else 'frozen category bank complete')
        record(ROOT/'experiments',event)
        print(identifier,'completed',round(result['wall_seconds'],2),metrics,flush=True)
        return result
    except BaseException as exc:
        (local/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        event=copy.deepcopy(entry)
        event.update(experiment_id=identifier+'-failure',status='failed',decision='preserve failure',interpretation=str(exc))
        record(ROOT/'experiments',event)
        raise
