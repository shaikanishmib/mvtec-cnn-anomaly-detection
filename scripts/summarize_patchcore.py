"""Reconcile native results, resources and paired image uncertainty, no new inference."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT
from anomaly_detection.data import CATEGORIES,TEXTURES,sha256
METRICS=['image_auroc','image_ap','pixel_auroc','pixel_ap','au_pro','recall','fpr','precision','f1']
IMAGE=['image_auroc','image_ap','recall','fpr','precision','f1']


def bootstrap_vectors(result):
    path=ROOT/result['local_maps']
    with np.load(path.parent/'bootstrap_vectors.npz') as data:return {key:data[key].copy() for key in METRICS}


def main():
    output=ROOT/'reports/patchcore';freeze=json.loads((ROOT/'configs/patchcore_frozen.json').read_text())
    rows=[];artifacts=[];all_results=[];vectors={k:[] for k in METRICS};cache_files={};bias={k:[] for k in ['pixel_auroc','pixel_ap','au_pro']}
    baseline_tables={name:pd.read_csv(ROOT/f'reports/{folder}/all_runs.csv').set_index(['category','seed'])
                     for name,folder in [('Basic CNN','basic_cnn'),('Autoencoder','autoencoder')]}
    paired={name:{k:[] for k in IMAGE} for name in baseline_tables};disagreements=[]
    for seed in freeze['seeds']:
        for category in CATEGORIES:
            identifier=f"03-test-{freeze['selected_variant']}-{category}-s{seed}-v1"
            path=output/f'evaluation/{identifier}-result.json';r=json.loads(path.read_text());all_results.append(r)
            run_path=output/f"runs/{r['training_experiment']}-result.json";run=json.loads(run_path.read_text())
            assert r['freeze_sha256']==sha256(ROOT/'configs/patchcore_frozen.json')
            assert sha256(ROOT/run['checkpoint'])==run['checkpoint_sha256']==r['checkpoint_sha256']
            assert sha256(ROOT/r['local_maps'])==r['maps_sha256']
            row={k:r[k] for k in METRICS+['tn','fp','fn','tp','training_seconds','checkpoint_bytes','parameters','coreset_seconds','bank_vector_bytes','cpu_peak_rss_bytes','evaluation_seconds']}
            row.update(category=category,seed=seed,experiment_id=identifier,kind='texture' if category in TEXTURES else 'object',
                calibration_images=r['calibration']['images'],image_threshold=r['calibration']['image_threshold'],pixel_threshold=r['calibration']['pixel_threshold'],
                model_p50_ms=r['latency']['model']['p50_ms'],model_p95_ms=r['latency']['model']['p95_ms'],
                extraction_p50_ms=r['latency']['feature_extraction']['p50_ms'],nn_p50_ms=r['latency']['nearest_neighbor']['p50_ms'],
                end_to_end_p50_ms=r['latency']['end_to_end']['p50_ms'],end_to_end_p95_ms=r['latency']['end_to_end']['p95_ms'],
                train_peak_vram_gib=r['training_peak_allocated_bytes']/2**30,
                inference_peak_vram_gib=r['latency']['peak_inference_allocated_bytes']/2**30,
                cpu_peak_rss_gib=r['cpu_peak_rss_bytes']/2**30,precision_defined=bool(r['tp']+r['fp']))
            rows.append(row)
            for key in ['recall','fpr','precision','f1','tn','fp','fn','tp']:row['pixel_'+key]=r['pixel_operating_point'][key]
            for key in bias:bias[key].append(r['bootstrap_binned_pixel_point_estimates'][key]-r[key])
            boot=bootstrap_vectors(r)
            table=pd.read_csv(output/f'evaluation/{identifier}-images.csv')
            # All benchmark category/seed results use identical official rows and
            # multinomial bootstrap draws; preserve dependence between models.
            for name,baseline in baseline_tables.items():
                folder='basic_cnn' if name=='Basic CNN' else 'autoencoder'
                base_id=baseline.loc[(category,seed),'experiment_id']
                base=json.loads((ROOT/f'reports/{folder}/evaluation/{base_id}-result.json').read_text())
                other=bootstrap_vectors(base)
                compare=pd.read_csv(ROOT/f'reports/{folder}/evaluation/{base_id}-images.csv')
                assert table.path.tolist()==compare.path.tolist() and table.label.tolist()==compare.label.tolist()
                for key in IMAGE:
                    assert boot[key].shape==other[key].shape
                    paired[name][key].append(boot[key]-other[key])
                truth=table.label.to_numpy().astype(bool)
                pc=table.predicted.to_numpy().astype(bool);old=compare.predicted.to_numpy().astype(bool)
                disagreements.append(dict(category=category,seed=seed,comparison=name,
                    pc_only_detected_defects=int((truth&pc&~old).sum()),baseline_only_detected_defects=int((truth&~pc&old).sum()),
                    pc_only_false_alarms=int((~truth&pc&~old).sum()),baseline_only_false_alarms=int((~truth&~pc&old).sum())))
            for key in METRICS:vectors[key].append(boot[key])
            for cache in [run['feature_cache'],r['calibration']['feature_cache'],r['test_cache']]:cache_files[cache['path']]=cache
            local=(ROOT/r['local_maps']).parent
            for artifact in [path,run_path,ROOT/run['checkpoint'],ROOT/r['local_maps'],local/'patch_distances.npy',
                local/'bootstrap_vectors.npz',local/'native_image_histograms.npz',local/'latency_ms.csv']:
                artifacts.append(dict(path=artifact.relative_to(ROOT).as_posix(),sha256=sha256(artifact),bytes=artifact.stat().st_size))
    table=pd.DataFrame(rows)
    assert len(table)==45 and table.groupby('category').size().eq(3).all()
    table.to_csv(output/'all_runs.csv',index=False,lineterminator='\n')
    category=table.groupby('category')[METRICS].agg(['mean','std']);category.columns=['_'.join(x) for x in category.columns]
    category.reindex(CATEGORIES).to_csv(output/'all_category_means.csv',lineterminator='\n')
    seeds=table.groupby('seed')[METRICS].mean();seeds.to_csv(output/'macro_by_seed.csv',lineterminator='\n')
    table.groupby('kind')[METRICS].mean().to_csv(output/'texture_object.csv',lineterminator='\n')
    pixel_operating=['pixel_recall','pixel_fpr','pixel_precision','pixel_f1']
    table.groupby('category')[pixel_operating].mean().to_csv(output/'pixel_operating_category_means.csv',lineterminator='\n')
    pd.DataFrame(disagreements).to_csv(output/'paired_prediction_disagreements.csv',index=False,lineterminator='\n')
    # Current category class counts yield all 2000 valid draws. Do not silently
    # align missing draws if a new dataset makes that assumption false.
    assert all(len(v)==2000 for values in vectors.values() for v in values)
    uncertainty={k:dict(mean=float(seeds[k].mean()),seed_std=float(seeds[k].std()),
        image_bootstrap_95ci=np.quantile(np.stack(vectors[k]).mean(0),[.025,.975]).tolist(),
        bootstrap_target='exact image metric' if k in IMAGE else 'calibration-scaled 4096-bin pixel approximation') for k in METRICS}
    summary=dict(categories=15,seeds=freeze['seeds'],models=45,development_runs=12,macro=uncertainty,
        frozen_configuration_sha256=sha256(ROOT/'configs/patchcore_frozen.json'),
        total_bank_fit_wall_seconds=float(table.training_seconds.sum()),total_coreset_seconds=float(table.coreset_seconds.sum()),
        total_evaluation_seconds=float(table.evaluation_seconds.sum()),
        deterministic_feature_caches=len(cache_files),deterministic_feature_cache_bytes=sum(c['bytes'] for c in cache_files.values()),
        unique_extraction_stage_seconds=sum(c['resources']['seconds'] for c in cache_files.values()),
        extraction_stage_scope='Once per compatible partition/category; includes image loading, transforms, GPU extraction, device transfer and cache writes; hash verification outside scope',
        mean_model_p50_ms=float(table.model_p50_ms.mean()),mean_model_p95_ms=float(table.model_p95_ms.mean()),
        mean_end_to_end_p50_ms=float(table.end_to_end_p50_ms.mean()),mean_end_to_end_p95_ms=float(table.end_to_end_p95_ms.mean()),
        mean_extraction_p50_ms=float(table.extraction_p50_ms.mean()),mean_nn_p50_ms=float(table.nn_p50_ms.mean()),
        peak_bank_gpu_allocated_gib=float(table.train_peak_vram_gib.max()),peak_inference_allocated_gib=float(table.inference_peak_vram_gib.max()),
        peak_evaluation_cpu_rss_gib=float(table.cpu_peak_rss_gib.max()),
        parameters=int(table.parameters.iloc[0]),trainable_parameters=0,max_bank_checkpoint_bytes=int(table.checkpoint_bytes.max()),
        bank_vector_bytes=int(table.bank_vector_bytes.iloc[0]),
        pixel_bootstrap_quantization_bias={k:dict(mean_binned_minus_exact=float(np.mean(v)),maximum_absolute_run_difference=float(np.max(np.abs(v)))) for k,v in bias.items()},
        metric_edge_cases=dict(image_runs_with_no_alarms=int((~table.precision_defined).sum()),precision_convention='shared zero_division=0; conditional defined-run precision separate',
            defined_run_macro_precision=float(table[table.precision_defined].precision.mean())),
        configuration_changed_after_test=False,test_evaluations_per_category_seed=1)
    summary['shared_backbone_storage']=json.loads((output/'backbone_storage.json').read_text())
    summary['pixel_operating_macro']={key:float(table[key].mean()) for key in pixel_operating}
    pixel_comparisons={'PatchCore variant':summary['pixel_operating_macro']}
    for name,baseline in baseline_tables.items():pixel_comparisons[name]=baseline[pixel_operating].mean().to_dict()
    pd.DataFrame(pixel_comparisons).to_csv(output/'pixel_operating_model_comparison.csv',lineterminator='\n')
    comparisons={};paired_rows=[]
    for name,folder in [('Basic CNN','basic_cnn'),('Autoencoder','autoencoder')]:
        base=json.loads((ROOT/f'reports/{folder}/summary.json').read_text())
        comparisons[name]={k:base['macro'][k]['mean'] for k in METRICS}
        for key in IMAGE:
            delta=np.stack(paired[name][key]).mean(0)
            paired_rows.append(dict(comparison=name,metric=key,point_difference=uncertainty[key]['mean']-base['macro'][key]['mean'],
                low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),
                method='Paired whole-image within category, same draws across seeds/models, 2000 replicates; conditional on fitted models'))
    comparisons['PatchCore variant']={k:uncertainty[k]['mean'] for k in METRICS}
    pd.DataFrame(comparisons).to_csv(output/'model_macro_comparison.csv',lineterminator='\n')
    pd.DataFrame(paired_rows).to_csv(output/'paired_image_difference_intervals.csv',index=False,lineterminator='\n')
    costs=['model_p50_ms','model_p95_ms','end_to_end_p50_ms','end_to_end_p95_ms','train_peak_vram_gib','checkpoint_bytes','training_seconds']
    for name,folder in [('cnn','basic_cnn'),('ae','autoencoder')]:
        baseline=pd.read_csv(ROOT/f'reports/{folder}/all_category_means.csv').set_index('category')
        category.join(baseline,lsuffix='_pc',rsuffix='_'+name).to_csv(output/f'{name}_category_comparison.csv',lineterminator='\n')
        baseline_runs=pd.read_csv(ROOT/f'reports/{folder}/all_runs.csv')
        table.groupby('category')[costs].mean().join(baseline_runs.groupby('category')[costs].mean(),lsuffix='_pc',rsuffix='_'+name).to_csv(output/f'{name}_category_costs.csv',lineterminator='\n')
    (output/'artifact_manifest.json').write_text(json.dumps(artifacts,indent=2)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
