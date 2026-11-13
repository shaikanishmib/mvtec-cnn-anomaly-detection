"""Reconcile frozen predictions and exact stored baselines; no new fitting."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from anomaly_detection.data import CATEGORIES,TEXTURES,sha256
from summarize_patchcore import METRICS,IMAGE,bootstrap_vectors

def main():
    output=ROOT/'reports/supersimplenet';frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())
    baseline_folders={'Basic CNN':'basic_cnn','Autoencoder':'autoencoder','PatchCore variant':'patchcore'}
    baselines={name:pd.read_csv(ROOT/f'reports/{folder}/all_runs.csv').set_index(['category','seed']) for name,folder in baseline_folders.items()}
    rows=[];artifacts=[];cache_files={};paired={name:{k:[] for k in IMAGE} for name in baselines};disagreements=[]
    vectors={k:[] for k in METRICS};bias={k:[] for k in ['pixel_auroc','pixel_ap','au_pro']};defects=[];cache_seconds=0
    for seed in frozen['seeds']:
        for category in CATEGORIES:
            identifier=f'04-test-{category}-s{seed}-v1';path=output/f'evaluation/{identifier}-result.json'
            r=json.loads(path.read_text());run_path=output/f"runs/{r['training_experiment']}-result.json";run=json.loads(run_path.read_text())
            assert r['freeze_sha256']==sha256(ROOT/'configs/supersimplenet_frozen.json')
            assert sha256(ROOT/run['checkpoint'])==run['checkpoint_sha256']==r['checkpoint_sha256']
            assert sha256(ROOT/r['local_maps'])==r['maps_sha256']
            row={k:r[k] for k in METRICS+['tn','fp','fn','tp','training_seconds','checkpoint_bytes','parameters','cpu_peak_rss_bytes','evaluation_seconds']}
            row.update(category=category,seed=seed,experiment_id=identifier,kind='texture' if category in TEXTURES else 'object',
                calibration_images=r['calibration']['images'],image_threshold=r['calibration']['image_threshold'],pixel_threshold=r['calibration']['pixel_threshold'],
                model_p50_ms=r['latency']['model']['p50_ms'],model_p95_ms=r['latency']['model']['p95_ms'],
                extraction_p50_ms=r['latency']['feature_extraction']['p50_ms'],heads_p50_ms=r['latency']['learned_heads']['p50_ms'],
                end_to_end_p50_ms=r['latency']['end_to_end']['p50_ms'],end_to_end_p95_ms=r['latency']['end_to_end']['p95_ms'],
                train_peak_vram_gib=r['training_peak_allocated_bytes']/2**30,inference_peak_vram_gib=r['latency']['peak_inference_allocated_bytes']/2**30,
                cpu_peak_rss_gib=r['cpu_peak_rss_bytes']/2**30,training_loop_seconds=r['training_loop_seconds'],precision_defined=bool(r['tp']+r['fp']))
            for key in ['recall','fpr','precision','f1','tn','fp','fn','tp']:row['pixel_'+key]=r['pixel_operating_point'][key]
            rows.append(row);defects.extend(r['per_defect'])
            boot=bootstrap_vectors(r)
            for key in METRICS:vectors[key].append(boot[key])
            for key in bias:bias[key].append(r['bootstrap_binned_pixel_point_estimates'][key]-r[key])
            table=pd.read_csv(output/f'evaluation/{identifier}-images.csv')
            for name,baseline in baselines.items():
                folder=baseline_folders[name];base_id=baseline.loc[(category,seed),'experiment_id']
                base=json.loads((ROOT/f'reports/{folder}/evaluation/{base_id}-result.json').read_text())
                other=bootstrap_vectors(base);compare=pd.read_csv(ROOT/f'reports/{folder}/evaluation/{base_id}-images.csv')
                assert table.path.tolist()==compare.path.tolist() and table.label.tolist()==compare.label.tolist()
                for key in IMAGE:
                    assert boot[key].shape==other[key].shape==(2000,)
                    paired[name][key].append(boot[key]-other[key])
                truth=table.label.to_numpy().astype(bool);current=table.predicted.to_numpy().astype(bool);old=compare.predicted.to_numpy().astype(bool)
                disagreements.append(dict(category=category,seed=seed,comparison=name,
                    ssn_only_detected_defects=int((truth&current&~old).sum()),baseline_only_detected_defects=int((truth&~current&old).sum()),
                    ssn_only_false_alarms=int((~truth&current&~old).sum()),baseline_only_false_alarms=int((~truth&~current&old).sum())))
            caches=[run['feature_caches']['fit'],run['feature_caches']['normal_development'],r['calibration']['feature_cache'],r['test_cache']]
            for cache in caches:
                for image in cache['images']:cache_files[image['path']]=image
                if cache['cache_misses']:cache_seconds+=cache['resources']['seconds']
            local=(ROOT/r['local_maps']).parent
            for artifact in [path,run_path,ROOT/run['checkpoint'],ROOT/r['local_maps'],local/'raw_detection.npy',local/'raw_segmentation.npy',
                local/'bootstrap_vectors.npz',local/'native_image_histograms.npz',local/'latency_ms.csv']:
                artifacts.append(dict(path=artifact.relative_to(ROOT).as_posix(),sha256=sha256(artifact),bytes=artifact.stat().st_size))
    table=pd.DataFrame(rows);assert len(table)==15*len(frozen['seeds']) and table.groupby('category').size().eq(len(frozen['seeds'])).all()
    table.to_csv(output/'all_runs.csv',index=False,lineterminator='\n')
    category=table.groupby('category')[METRICS].agg(['mean','std']);category.columns=['_'.join(x) for x in category.columns]
    category.reindex(CATEGORIES).to_csv(output/'all_category_means.csv',lineterminator='\n')
    seeds=table.groupby('seed')[METRICS].mean();seeds.to_csv(output/'macro_by_seed.csv',lineterminator='\n')
    table.groupby('kind')[METRICS].mean().to_csv(output/'texture_object.csv',lineterminator='\n')
    pd.DataFrame(defects).to_csv(output/'per_defect_results.csv',index=False,lineterminator='\n')
    pd.DataFrame(disagreements).to_csv(output/'paired_prediction_disagreements.csv',index=False,lineterminator='\n')
    assert all(len(v)==2000 for values in vectors.values() for v in values)
    uncertainty={k:dict(mean=float(seeds[k].mean()),seed_std=float(seeds[k].std()) if len(seeds)>1 else None,
        image_bootstrap_95ci=np.quantile(np.stack(vectors[k]).mean(0),[.025,.975]).tolist(),
        bootstrap_target='exact image metric; fitted models fixed' if k in IMAGE else 'fixed sigmoid4096-bin pixel approximation') for k in METRICS}
    summary=dict(categories=15,seeds=frozen['seeds'],models=len(table),development_runs=16,macro=uncertainty,
        frozen_configuration_sha256=sha256(ROOT/'configs/supersimplenet_frozen.json'),
        total_fit_wall_seconds=float(table.training_seconds.sum()),total_training_loop_seconds=float(table.training_loop_seconds.sum()),
        total_evaluation_seconds=float(table.evaluation_seconds.sum()),deterministic_feature_images=len(cache_files),
        deterministic_feature_cache_bytes=sum(c['bytes'] for c in cache_files.values()),new_final_cache_stage_seconds=cache_seconds,
        cache_scope='Per-image compatible SSN cache; seeds and overlapping splits reuse extraction; final-stage time excludes already completed pilot/development extraction',
        mean_model_p50_ms=float(table.model_p50_ms.mean()),mean_model_p95_ms=float(table.model_p95_ms.mean()),
        mean_end_to_end_p50_ms=float(table.end_to_end_p50_ms.mean()),mean_end_to_end_p95_ms=float(table.end_to_end_p95_ms.mean()),
        mean_extraction_p50_ms=float(table.extraction_p50_ms.mean()),mean_heads_p50_ms=float(table.heads_p50_ms.mean()),
        peak_training_gpu_allocated_gib=float(table.train_peak_vram_gib.max()),peak_inference_allocated_gib=float(table.inference_peak_vram_gib.max()),
        peak_evaluation_cpu_rss_gib=float(table.cpu_peak_rss_gib.max()),parameters=int(table.parameters.iloc[0]),
        max_head_checkpoint_bytes=int(table.checkpoint_bytes.max()),
        pixel_bootstrap_quantization_bias={k:dict(mean_binned_minus_exact=float(np.mean(v)),maximum_absolute_run_difference=float(np.max(np.abs(v)))) for k,v in bias.items()},
        image_runs_with_no_alarms=int((~table.precision_defined).sum()),precision_convention='shared zero_division=0',
        configuration_changed_after_test=False,test_evaluations_per_category_seed=1,
        uncertainty_limit='Image bootstrap conditions on fitted models and reserved-calibration thresholds. It does not quantify training-seed or calibration-sampling variation.',
        seed_budget_decision=json.loads((output/'compute_budget.json').read_text()),
        shared_backbone_storage=json.loads((output/'backbone_storage.json').read_text()))
    comparisons={};paired_rows=[];costrows=[]
    for name,folder in baseline_folders.items():
        base=json.loads((ROOT/f'reports/{folder}/summary.json').read_text());baseline=baselines[name]
        comparisons[name]={k:base['macro'][k]['mean'] for k in METRICS}
        matched=baseline[baseline.index.get_level_values('seed').isin(frozen['seeds'])]
        for key in IMAGE:
            delta=np.stack(paired[name][key]).mean(0)
            paired_rows.append(dict(comparison=name,metric=key,point_difference=uncertainty[key]['mean']-float(matched[key].mean()),
                low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),matched_seeds=str(frozen['seeds']),
                method='Paired same-category same-seed same-image draws, 2000 replicates; conditional on fitted models and thresholds'))
        category.join(pd.read_csv(ROOT/f'reports/{folder}/all_category_means.csv').set_index('category'),lsuffix='_ssn',rsuffix='_baseline').to_csv(output/f'{folder}_category_comparison.csv',lineterminator='\n')
        costrows.append(dict(model=name,seeds=len(base['seeds']),model_p50_ms=base['mean_model_p50_ms'],end_to_end_p50_ms=base['mean_end_to_end_p50_ms']))
    comparisons['SuperSimpleNet Lite']={k:v['mean'] for k,v in uncertainty.items()}
    costrows.append(dict(model='SuperSimpleNet Lite',seeds=len(frozen['seeds']),model_p50_ms=summary['mean_model_p50_ms'],end_to_end_p50_ms=summary['mean_end_to_end_p50_ms']))
    pd.DataFrame(comparisons).to_csv(output/'model_macro_comparison.csv',lineterminator='\n')
    pd.DataFrame(costrows).to_csv(output/'model_latency_comparison.csv',index=False,lineterminator='\n')
    pd.DataFrame(paired_rows).to_csv(output/'paired_image_difference_intervals.csv',index=False,lineterminator='\n')
    pixel=['pixel_recall','pixel_fpr','pixel_precision','pixel_f1'];summary['pixel_operating_macro']=table[pixel].mean().to_dict()
    table.groupby('category')[pixel].mean().to_csv(output/'pixel_operating_category_means.csv',lineterminator='\n')
    for filename,data in [('artifact_manifest.json',artifacts),('summary.json',summary)]:
        (output/filename).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
