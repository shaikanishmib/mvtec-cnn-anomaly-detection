"""Reconcile 45 frozen runs and exact stored six-model results; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from anomaly_detection.data import CATEGORIES,TEXTURES,sha256
from anomaly_detection.efficientad_final import ROOT,REPORT,LOCAL,FREEZE,METRICS,atomic_json,artifact,verify_complete,freeze

OUT=ROOT/'reports/efficientad'
BASELINES={'Basic CNN':'basic_cnn','Convolutional Autoencoder':'autoencoder','PatchCore variant':'patchcore','SuperSimpleNet Lite':'supersimplenet','PaDiM':'padim'}
NAME='EfficientAD-S reduced-budget'
IMAGE=['image_auroc','image_ap','recall','fpr','precision','f1']


def main():
    frozen=freeze();manifest=json.loads((REPORT/'benchmark_manifest.json').read_text())
    assert all(m['status']=='complete' for m in manifest['models']) and len(manifest['models'])==45
    runs=[];records=[];defects=[];all_images=[];vectors={k:[] for k in METRICS};bias={k:[] for k in METRICS[2:5]}
    baseline_tables={name:pd.read_csv(ROOT/f'reports/{folder}/all_runs.csv').set_index(['category','seed']) for name,folder in BASELINES.items()}
    paired={name:{k:[] for k in IMAGE} for name in BASELINES};disagreements=[];artifacts=[]
    baseline_evaluation_seconds={name:0. for name in BASELINES}
    baseline_inference_vram={name:[] for name in BASELINES}
    for item in manifest['models']:
        # Per-run checks already hashed all large files before marking complete; final deep validator does one final full hash pass.
        r=verify_complete(ROOT/item['result'],deep=False);records.append(r)
        category,seed=r['category'],r['seed'];local=ROOT/r['local_maps'];local=local.parent
        images=pd.read_csv(ROOT/r['images_csv'],float_precision='round_trip');images['seed']=seed
        all_images.append(images);defects.extend(r['per_defect'])
        row={k:r[k] for k in METRICS+['tn','fp','fn','tp','training_seconds','calibration_seconds','inference_seconds','native_metrics_bootstrap_seconds','latency_seconds','evaluation_seconds','wall_seconds']}
        row.update(category=category,seed=seed,experiment_id=r['experiment_id'],kind='texture' if category in TEXTURES else 'object',
            optimizer_steps=int(r['training']['history'][-1]['steps']),
            preparation_seconds=r['training']['preparation_seconds'],map_normalization_seconds=r['training']['map_normalization_seconds'],
            checkpoint_serialization_verification_seconds=r['training']['serialization_seconds'],
            checkpoint_bytes=r['checkpoint']['bytes'],image_threshold=r['calibration']['threshold'],pixel_threshold=r['calibration']['pixel_threshold'],
            calibration_images=r['calibration']['calibration_distribution']['n'],calibration_unique_scores=r['calibration']['unique_calibration_scores'],
            calibration_fpr=r['calibration']['observed_calibration_fpr'],development_exceedance=r['calibration']['development_exceedance_fraction'],
            model_p50_ms=r['latency']['model']['p50_ms'],model_p95_ms=r['latency']['model']['p95_ms'],
            end_to_end_p50_ms=r['latency']['end_to_end']['p50_ms'],end_to_end_p95_ms=r['latency']['end_to_end']['p95_ms'],
            decode_inclusive_p50_ms=r['latency']['decode_inclusive_p50_ms'],decode_inclusive_p95_ms=r['latency']['decode_inclusive_p95_ms'],
            train_peak_vram_gib=r['training']['resources']['peak_gpu_allocated_bytes']/2**30,
            train_peak_reserved_vram_gib=r['training']['resources']['peak_gpu_reserved_bytes']/2**30,
            inference_peak_vram_gib=r['latency']['peak_inference_allocated_bytes']/2**30,
            inference_peak_reserved_vram_gib=r['latency']['peak_inference_reserved_bytes']/2**30,
            cpu_peak_rss_gib=r['resources']['cpu_peak_rss_bytes']/2**30,
            fn_with_pixel_overlap=int(((images.outcome=='FN')&(images.pixel_detected_defect_pixels>0)).sum()))
        runs.append(row)
        with np.load(local/'bootstrap_vectors.npz') as data:boot={k:data[k].copy() for k in METRICS}
        for k in METRICS: vectors[k].append(boot[k])
        for k in bias:bias[k].append(r['bootstrap_binned_pixel_point_estimates'][k]-r[k])
        for name,folder in BASELINES.items():
            identifier=baseline_tables[name].loc[(category,seed),'experiment_id']
            b=json.loads((ROOT/f'reports/{folder}/evaluation/{identifier}-result.json').read_text())
            baseline_evaluation_seconds[name]+=b.get('evaluation_seconds',b.get('resources',{}).get('seconds',0.))
            baseline_inference_vram[name].append(b['latency']['peak_inference_allocated_bytes']/2**30)
            other=pd.read_csv(ROOT/f'reports/{folder}/evaluation/{identifier}-images.csv',float_precision='round_trip')
            assert images.path.tolist()==other.path.tolist() and images.label.tolist()==other.label.tolist()
            with np.load((ROOT/b['local_maps']).parent/'bootstrap_vectors.npz') as data:
                for k in IMAGE:
                    assert boot[k].shape==data[k].shape
                    paired[name][k].append(boot[k]-data[k])
            y=images.label.to_numpy().astype(bool);current=images.predicted.to_numpy();previous=other.predicted.to_numpy()
            disagreements.append(dict(category=category,seed=seed,comparison=name,
                efficientad_only_detected_defects=int((y&current&~previous).sum()),baseline_only_detected_defects=int((y&~current&previous).sum()),
                efficientad_only_false_alarms=int((~y&current&~previous).sum()),baseline_only_false_alarms=int((~y&~current&previous).sum())))
        artifacts.extend(r['artifacts']);artifacts.append(artifact(ROOT/item['result']))
    table=pd.DataFrame(runs);assert len(table)==45 and not table[['category','seed']].duplicated().any()
    table.to_csv(OUT/'all_runs.csv',index=False,lineterminator='\n')
    categories=table.groupby('category')[METRICS].agg(['mean','std']);categories.columns=['_'.join(c) for c in categories.columns]
    categories.reindex(CATEGORIES).to_csv(OUT/'all_category_means.csv',lineterminator='\n')
    seeds=table.groupby('seed')[METRICS].mean();seeds.to_csv(OUT/'macro_by_seed.csv',lineterminator='\n')
    table.groupby('kind')[METRICS].mean().to_csv(OUT/'texture_object.csv',lineterminator='\n')
    pd.DataFrame(defects).to_csv(OUT/'per_defect_results.csv',index=False,lineterminator='\n')
    pd.DataFrame(disagreements).to_csv(OUT/'paired_prediction_disagreements.csv',index=False,lineterminator='\n')
    images=pd.concat(all_images,ignore_index=True)
    images['size_bin']=pd.cut(images.defect_fraction,[-1,0,.001,.01,1],labels=['normal','tiny_le_0.1pct','small_le_1pct','larger'])
    size=images[images.label.eq(1)].groupby(['category','size_bin'],observed=True).agg(images=('label','size'),
        recall=('predicted','mean'),mean_defect_fraction=('defect_fraction','mean'),image_fn_with_pixel_overlap=('pixel_detected_defect_pixels',lambda v:int((v>0).sum())))
    # Above count is pixel-overlap incidence in the stratum; explicit FN-only table below avoids conflating TP incidence.
    size=size.rename(columns={'image_fn_with_pixel_overlap':'images_with_pixel_overlap'})
    size.to_csv(OUT/'small_defect_summary.csv',lineterminator='\n')
    failure=images[images.outcome.isin(['FN','FP'])].copy()
    failure.to_csv(OUT/'failure_images.csv',index=False,float_format='%.17g',lineterminator='\n')
    macro={k:dict(mean=float(seeds[k].mean()),seed_std=float(seeds[k].std()),
        image_bootstrap_95ci=np.quantile(np.stack(vectors[k]).mean(0),[.025,.975]).tolist(),
        bootstrap_target='exact image metric, fitted models/thresholds fixed' if k in IMAGE else '4096-bin signed arctangent pixel approximation') for k in METRICS}
    summary=dict(status='all final evaluations complete; publication validation separate',categories=15,seeds=frozen['seeds'],models=45,pilot_runs=1,development_runs=4,
        macro=macro,frozen_configuration_sha256=FREEZE,total_training_seconds=float(table.training_seconds.sum()),
        total_optimizer_steps=int(table.optimizer_steps.sum()),min_optimizer_steps_per_model=int(table.optimizer_steps.min()),max_optimizer_steps_per_model=int(table.optimizer_steps.max()),
        total_calibration_seconds=float(table.calibration_seconds.sum()),total_evaluation_seconds=float(table.evaluation_seconds.sum()),
        total_preparation_seconds=float(table.preparation_seconds.sum()),total_map_normalization_seconds=float(table.map_normalization_seconds.sum()),
        total_checkpoint_serialization_verification_seconds=float(table.checkpoint_serialization_verification_seconds.sum()),
        total_run_pipeline_seconds=float(table.wall_seconds.sum()),total_native_metrics_bootstrap_seconds=float(table.native_metrics_bootstrap_seconds.sum()),
        mean_model_p50_ms=float(table.model_p50_ms.mean()),mean_model_p95_ms=float(table.model_p95_ms.mean()),
        mean_end_to_end_p50_ms=float(table.end_to_end_p50_ms.mean()),mean_end_to_end_p95_ms=float(table.end_to_end_p95_ms.mean()),
        mean_decode_inclusive_p50_ms=float(table.decode_inclusive_p50_ms.mean()),mean_decode_inclusive_p95_ms=float(table.decode_inclusive_p95_ms.mean()),
        peak_training_gpu_allocated_gib=float(table.train_peak_vram_gib.max()),peak_training_gpu_reserved_gib=float(table.train_peak_reserved_vram_gib.max()),
        peak_inference_allocated_gib=float(table.inference_peak_vram_gib.max()),peak_inference_reserved_gib=float(table.inference_peak_reserved_vram_gib.max()),
        peak_evaluation_cpu_rss_gib=float(table.cpu_peak_rss_gib.max()),max_checkpoint_bytes=int(table.checkpoint_bytes.max()),
        pixel_bootstrap_quantization_bias={k:dict(mean_binned_minus_exact=float(np.mean(v)),maximum_absolute_run_difference=float(np.max(np.abs(v)))) for k,v in bias.items()},
        calibration_runs_with_ties=int((table.calibration_unique_scores<table.calibration_images).sum()),
        calibration_fpr_mean=float(table.calibration_fpr.mean()),test_fpr_macro=macro['fpr']['mean'],
        image_runs_with_no_alarms=int((table.tp+table.fp==0).sum()),fn_with_pixel_overlap=int(table.fn_with_pixel_overlap.sum()),
        configuration_changed_after_test=False,test_evaluations_per_category_seed=1,
        uncertainty_limit='Conditional image bootstrap uses same draws across seeds/models; does not include fitting/calibration uncertainty. Pixel intervals approximate, primary point metrics exact.',
        latency_scope='Mean of 45 per-run percentiles, not pooled population percentiles; full two-map eager GPU forward, selected score local only; main E2E excludes disk decode',
        limitations=['Reduced budget, no external-image penalty, third-party teacher provenance/license uncertainty','Local-only selected score excludes global map despite AE training','Fit-background proxy selection is optimistic; only two development categories and one seed','Normal calibration ties and small samples limit nominal FPR guarantees'])
    atomic_json(OUT/'summary.json',summary)
    # Exact category-matched point metrics and paired category-resampling CIs for all nine metrics.
    comparisons={};costs=[];paired_rows=[];category_paired=[]
    rng=np.random.default_rng(8606);draws=rng.integers(0,15,size=(10000,15))
    current=table.groupby('category')[METRICS].mean().reindex(CATEGORIES)
    common_backbone=json.loads((ROOT/'reports/patchcore/summary.json').read_text())['shared_backbone_storage']
    for name,folder in BASELINES.items():
        baseline=json.loads((ROOT/f'reports/{folder}/summary.json').read_text());bt=baseline_tables[name].reset_index()
        comparisons[name]={k:baseline['macro'][k]['mean'] for k in METRICS}
        assert all(np.isclose(bt[k].mean(),comparisons[name][k]) for k in METRICS)
        other=bt.groupby('category')[METRICS].mean().reindex(CATEGORIES)
        for key in METRICS:
            diff=(current[key]-other[key]).to_numpy();resampled=diff[draws].mean(1)
            category_paired.append(dict(comparison=name,metric=key,point_difference=float(diff.mean()),
                low=float(np.quantile(resampled,.025)),high=float(np.quantile(resampled,.975)),
                method='Paired 15-category resampling; seed means within category; 10000 draws; exploratory heterogeneity interval, no multiplicity correction'))
        for key in IMAGE:
            delta=np.stack(paired[name][key]).mean(0)
            paired_rows.append(dict(comparison=name,metric=key,point_difference=macro[key]['mean']-comparisons[name][key],
                low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),method='Paired fixed-category/seed image bootstrap; 2000 shared draws, fixed fitted models and thresholds'))
        categories.join(pd.read_csv(ROOT/f'reports/{folder}/all_category_means.csv').set_index('category'),lsuffix='_efficientad',rsuffix='_baseline').to_csv(OUT/f'{folder}_category_comparison.csv',lineterminator='\n')
        costs.append(dict(model=name,training_fitting_seconds=float(bt.training_seconds.sum()),evaluation_seconds=baseline_evaluation_seconds[name],
            model_p50_ms=float(bt.model_p50_ms.mean()),model_p95_ms=float(bt.model_p95_ms.mean()),
            end_to_end_p50_ms=float(bt.end_to_end_p50_ms.mean()),end_to_end_p95_ms=float(bt.end_to_end_p95_ms.mean()),
            peak_training_vram_gib=float(bt.train_peak_vram_gib.max()),peak_inference_vram_gib=max(baseline_inference_vram[name]),
            peak_cpu_rss_gib=float(bt.cpu_peak_rss_gib.max()) if 'cpu_peak_rss_gib' in bt else None,
            max_checkpoint_bytes=int(bt.checkpoint_bytes.max()),
            shared_backbone_bytes=int(baseline.get('shared_backbone_storage',common_backbone if folder=='padim' else {}).get('bytes',0)),
            single_category_deployment_bytes=int(bt.checkpoint_bytes.max())+int(baseline.get('shared_backbone_storage',common_backbone if folder=='padim' else {}).get('bytes',0)),
            recorded_feature_cache_bytes=int(baseline.get('deterministic_feature_cache_bytes',0)),
            caveat='Recorded category checkpoints; pretrained shared-backbone storage/pretraining and cache extraction may be additional. Basic CNN/AE evaluation CPU RSS not recorded. Latency scopes/hardware states differ.'))
    comparisons[NAME]={k:v['mean'] for k,v in macro.items()}
    costs.append(dict(model=NAME,training_fitting_seconds=summary['total_training_seconds'],evaluation_seconds=summary['total_evaluation_seconds'],
        model_p50_ms=summary['mean_model_p50_ms'],model_p95_ms=summary['mean_model_p95_ms'],end_to_end_p50_ms=summary['mean_end_to_end_p50_ms'],
        end_to_end_p95_ms=summary['mean_end_to_end_p95_ms'],peak_training_vram_gib=summary['peak_training_gpu_allocated_gib'],
        peak_inference_vram_gib=summary['peak_inference_allocated_gib'],peak_cpu_rss_gib=summary['peak_evaluation_cpu_rss_gib'],
        max_checkpoint_bytes=summary['max_checkpoint_bytes'],caveat='Checkpoint includes teacher/student/AE; local score selected but full forward timed. Pretrained distillation effort not measured.'))
    costs[-1].update(shared_backbone_bytes=0,single_category_deployment_bytes=summary['max_checkpoint_bytes'],recorded_feature_cache_bytes=0)
    pd.DataFrame(comparisons).to_csv(OUT/'model_macro_comparison.csv',lineterminator='\n')
    pd.DataFrame(costs).to_csv(OUT/'model_cost_comparison.csv',index=False,lineterminator='\n')
    pd.DataFrame(paired_rows).to_csv(OUT/'paired_image_difference_intervals.csv',index=False,lineterminator='\n')
    pd.DataFrame(category_paired).to_csv(OUT/'paired_category_difference_intervals.csv',index=False,lineterminator='\n')
    table[['category','seed','image_threshold','calibration_images','calibration_unique_scores','calibration_fpr','development_exceedance','fpr']].to_csv(OUT/'calibration_summary.csv',index=False,lineterminator='\n')
    atomic_json(OUT/'artifact_manifest.json',artifacts)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
