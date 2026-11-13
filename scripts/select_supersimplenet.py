"""Common synthetic-proxy comparison and prospective final policy freeze."""
import os,sys,json,copy,math
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import torch,pandas as pd
from run_supersimplenet import candidates,development_identifier
from anomaly_detection.supersimplenet import SSNExtractor
from anomaly_detection.ssn_research import feature_cache,development_metrics
from anomaly_detection.ssn_evaluation import load_heads
from anomaly_detection.cnn_research import read_partition,inner_fit_split
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.tracking import config_hash,record
from anomaly_detection.basic_cnn import seed_everything

def main():
    torch.set_num_threads(4);seed_everything(2026);torch.backends.cuda.matmul.allow_tf32=False
    output=ROOT/'reports/supersimplenet';configurations=candidates();results={}
    # Require the entire bounded study before any selection or freeze.
    for name in configurations:
        for category in ['bottle','cable','carpet','tile']:
            path=output/f'runs/{development_identifier(name,category)}-result.json'
            results[(name,category)]=json.loads(path.read_text())
    extractor=SSNExtractor('resnet18').cuda();table=[]
    for category in ['bottle','cable','carpet','tile']:
        _,probes=inner_fit_split(read_partition('fit',category))
        normals=feature_cache(extractor,read_partition('development',category),8)
        features=feature_cache(extractor,probes,8)
        for name,config in configurations.items():
            result=results[(name,category)];identifier=f'04-common-{name}-{category}-s2026-v1'
            path=output/f'{identifier}.json'
            if path.exists():evaluation=json.loads(path.read_text())
            else:
                assert sha256(ROOT/result['checkpoint'])==result['checkpoint_sha256']
                model,_=load_heads(ROOT/result['checkpoint'],category,config)
                evaluation=development_metrics(model,normals,features,dict(config,noise_std=.05))
                evaluation.update(checkpoint_sha256=result['checkpoint_sha256'],training_experiment=result['experiment_id'],
                    proxy_noise_std=.05,probe_rows_sha256=features.metadata['input_rows_sha256'],
                    clean_normal_development_sha256=normals.metadata['input_rows_sha256'])
                with path.open('x') as stream:json.dump(evaluation,stream,indent=2)
                entry=json.loads((ROOT/f"experiments/{result['experiment_id']}-start.json").read_text())
                for key in ['recorded_at','config_sha256']:entry.pop(key)
                entry.update(experiment_id=identifier,hypothesis='All candidates face identical .05 synthetic proxies; easier perturbations cannot determine the winner',
                    hyperparameters=dict(training=config,common_proxy=dict(noise_std=.05,mask_seed=9901,noise_seed=9902)),
                    metrics=evaluation,status='completed',decision='include in prespecified common-proxy comparison',
                    interpretation='Synthetic-only held-out fitting proxies and clean development normals, not real-defect validation')
                record(ROOT/'experiments',entry)
                del model;torch.cuda.empty_cache()
            for aggregation,metrics in evaluation['candidates'].items():
                table.append(dict(candidate=name,category=category,aggregation=aggregation,
                    train_seconds=sum(h['train_epoch_seconds'] for h in result['history']),
                    parameters=result['parameters'],**{k:v for k,v in metrics.items() if k!='confusion_matrix'}))
    table=pd.DataFrame(table);macro=table.groupby(['candidate','aggregation']).mean(numeric_only=True).reset_index()
    eligible=macro[macro.objective>=macro.objective.max()-.005]
    chosen=eligible.sort_values(['train_seconds','parameters','candidate','aggregation']).iloc[0]
    table.to_csv(output/'development_common_proxy.csv',index=False,lineterminator='\n')
    macro.sort_values('objective',ascending=False).to_csv(output/'development_macro.csv',index=False,lineterminator='\n')
    # A proxy can establish learnability, not real-defect performance.
    if chosen.auroc<=.5:raise RuntimeError('Selected proxy image discrimination is not above chance; diagnose before final fitting')
    name=chosen.candidate;selected=[results[(name,c)] for c in ['bottle','cable','carpet','tile']]
    step_seconds=sum(sum(h['train_epoch_seconds'] for h in r['history']) for r in selected)/sum(40*math.ceil(r['fit_normals']/8) for r in selected)
    fit_counts=[len(read_partition('fit',c)) for c in CATEGORIES]
    single_seconds=step_seconds*40*sum(math.ceil(n/8) for n in fit_counts)
    seeds=[2026,2027,2028] if single_seconds*3<=90*60 else [2026]
    config=dict(configurations[name],aggregation=chosen.aggregation,resolution=256,
        hypothesis='Frozen selected Lite feature-space anomaly discrimination; no further development or test-driven changes')
    sources=['src/anomaly_detection/supersimplenet.py','src/anomaly_detection/ssn_research.py',
        'src/anomaly_detection/ssn_evaluation.py','src/anomaly_detection/ssn_final.py','src/anomaly_detection/cnn_evaluation.py',
        'src/anomaly_detection/metrics.py','src/anomaly_detection/pixel_metrics.py','src/anomaly_detection/basic_cnn.py',
        'src/anomaly_detection/cnn_research.py','scripts/benchmark_supersimplenet.py']
    frozen=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),configuration=config,selected_variant=name,
        seeds=seeds,categories=list(CATEGORIES),weight_sha256=extractor.weight_sha256,weight_url=extractor.weight_url,
        source_sha256={p:sha256(ROOT/p) for p in sources},
        evaluation_source_sha256=sha256(ROOT/'src/anomaly_detection/ssn_evaluation.py'),
        pixel_metric_source_sha256=sha256(ROOT/'src/anomaly_detection/pixel_metrics.py'),
        protocol_sha256=sha256(output/'protocol.md'),
        image_calibration='Reserved-normal higher q95; strict >. Separate q99 diagnostic only.',
        pixel_calibration='Reserved-normal pooled higher q99.5 at256; strict >.',
        uncertainty='2000 paired whole-image bootstrap draws; exact image metrics; approximate fixed sigmoid4096-bin pixel intervals; exact native point metrics',
        simplifications='ResNet18/384/256/32; normal-only, 40epochs and batch8; clip10; independent seeded Perlin; whole-image256; no test minmax; final fixed checkpoint')
    path=ROOT/'configs/supersimplenet_frozen.json'
    # Explicit LF bytes make the freeze SHA portable under configs/*.json attributes.
    with path.open('xb') as stream:stream.write(json.dumps(frozen,indent=2).encode())
    budget=dict(selected_configuration=config,seeds=seeds,models=len(seeds)*15,
        measured_train_step_seconds=step_seconds,projected_training_seconds=single_seconds*len(seeds),
        projected_three_seed_training_seconds=single_seconds*3,
        measured_development_peak_gpu_bytes=max(r['resources']['peak_gpu_allocated_bytes'] for r in selected),
        measured_development_peak_rss_bytes=max(r['resources']['cpu_peak_rss_bytes'] for r in selected),
        expected_total_feature_cache_bytes=5354*384*64*64*4,
        expected_native_evaluation_minutes=75*len(seeds)/3,
        decision='Use three seeds within90-minute measured training budget; otherwise retain one predefined seed2026 and report missing seed uncertainty',
        learning_evidence=dict(common_proxy_image_auroc=float(chosen.auroc),common_proxy_pixel_ap=float(chosen.pixel_ap),objective=float(chosen.objective)),
        official_test_accessed=False)
    with (output/'compute_budget.json').open('x') as stream:json.dump(budget,stream,indent=2)
    with (output/'freeze_identity.json').open('x') as stream:json.dump(dict(sha256=sha256(path)),stream,indent=2)
    entry=json.loads((ROOT/f"experiments/{selected[0]['experiment_id']}-start.json").read_text())
    for key in ['recorded_at','config_sha256']:entry.pop(key)
    entry.update(experiment_id='04-development-selection-v1',categories=list(CATEGORIES),hyperparameters=frozen,
        metrics=dict(common_proxy_macro=macro.to_dict('records'),compute_budget=budget),status='completed',
        decision=f'Freeze {name}/{chosen.aggregation}, seeds{seeds}',interpretation='Common synthetic probes only; selection cannot establish real-defect quality')
    record(ROOT/'experiments',entry)
    for (candidate,category),result in results.items():
        event=copy.deepcopy(entry);event.update(experiment_id=result['experiment_id']+'-decision',categories=[category],
            hyperparameters=result['configuration'],metrics=dict(common_proxy=table[table.candidate.eq(candidate)&table.category.eq(category)].to_dict('records')),
            decision='select' if candidate==name else 'reject',interpretation=f'Four-category common-proxy rule selected {name}/{chosen.aggregation}; all unsuccessful trials retained')
        record(ROOT/'experiments',event)
    print(macro.sort_values('objective',ascending=False).to_string(index=False));print(json.dumps(budget,indent=2))

if __name__=='__main__':main()
