"""Validate all 45 confirmations and compare retained predictions across families."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from anomaly_detection.data import CATEGORIES,TEXTURES,sha256
from anomaly_detection.padim_evaluation import verify_freeze
from anomaly_detection.padim import PaDiMExtractor

ROOT=Path(__file__).resolve().parents[1]
METRICS=['image_auroc','image_ap','pixel_auroc','pixel_ap','au_pro','recall','fpr','precision','f1']
IMAGE=['image_auroc','image_ap','recall','fpr','precision','f1']
BASELINES={'Basic CNN':'basic_cnn','Autoencoder':'autoencoder','PatchCore variant':'patchcore','SuperSimpleNet Lite':'supersimplenet'}


def main():
    frozen,identity=verify_freeze()
    out=ROOT/'reports/padim'
    rows=[];artifacts=[];defects=[];vectors={k:[] for k in METRICS};bias={k:[] for k in METRICS[2:5]}
    baseline_tables={name:pd.read_csv(ROOT/f'reports/{folder}/all_runs.csv').set_index(['category','seed']) for name,folder in BASELINES.items()}
    paired={name:{k:[] for k in IMAGE} for name in BASELINES};disagreements=[]
    for seed in frozen['seeds']:
        for category in CATEGORIES:
            identifier=f'05-test-{category}-s{seed}-v1'
            path=out/f'evaluation/{identifier}-result.json'
            r=json.loads(path.read_text());local=(ROOT/r['local_maps']).parent
            assert r['freeze_sha256']==identity and r['configuration']==dict(frozen['configuration'],seed=seed)
            assert sha256(ROOT/r['checkpoint'])==r['checkpoint_sha256']
            assert sha256(ROOT/r['local_maps'])==r['maps_sha256']
            images=pd.read_csv(out/f'evaluation/{identifier}-images.csv')
            assert len(images)==r['image_rows']
            assert np.array_equal(images.score.to_numpy()>r['calibration']['image_threshold'],images.predicted)
            assert int(((images.label==1)&images.predicted).sum())==r['tp']
            assert int(((images.label==0)&images.predicted).sum())==r['fp']
            row={k:r[k] for k in METRICS+['tn','fp','fn','tp','training_seconds','checkpoint_bytes']}
            row.update(category=category,seed=seed,experiment_id=identifier,kind='texture' if category in TEXTURES else 'object',
                image_threshold=r['calibration']['image_threshold'],pixel_threshold=r['calibration']['pixel_threshold'],
                calibration_images=r['calibration']['images'],model_p50_ms=r['latency']['model']['p50_ms'],
                model_p95_ms=r['latency']['model']['p95_ms'],end_to_end_p50_ms=r['latency']['end_to_end']['p50_ms'],
                end_to_end_p95_ms=r['latency']['end_to_end']['p95_ms'],train_peak_vram_gib=r['fitting_peak_allocated_bytes']/2**30,
                train_peak_reserved_vram_gib=r['fitting_peak_reserved_bytes']/2**30,
                inference_peak_vram_gib=r['latency']['peak_inference_allocated_bytes']/2**30,
                inference_peak_reserved_vram_gib=r['latency']['peak_inference_reserved_bytes']/2**30,
                cpu_peak_rss_gib=r['resources']['cpu_peak_rss_bytes']/2**30,evaluation_seconds=r['resources']['seconds'])
            for key,value in r['pixel_operating_point'].items():row['pixel_'+key]=value
            rows.append(row);defects.extend(r['per_defect'])
            with np.load(local/'bootstrap_vectors.npz') as data:boot={key:data[key].copy() for key in METRICS}
            for key in METRICS:
                assert boot[key].shape==(2000,)
                vectors[key].append(boot[key])
            for key in bias:bias[key].append(r['bootstrap_binned_pixel_point_estimates'][key]-r[key])
            for name,folder in BASELINES.items():
                baseline_id=baseline_tables[name].loc[(category,seed),'experiment_id']
                baseline=json.loads((ROOT/f'reports/{folder}/evaluation/{baseline_id}-result.json').read_text())
                other=pd.read_csv(ROOT/f'reports/{folder}/evaluation/{baseline_id}-images.csv')
                assert images.path.tolist()==other.path.tolist() and images.label.tolist()==other.label.tolist()
                with np.load((ROOT/baseline['local_maps']).parent/'bootstrap_vectors.npz') as data:
                    for key in IMAGE:paired[name][key].append(boot[key]-data[key])
                truth=images.label.to_numpy().astype(bool);current=images.predicted.to_numpy();old=other.predicted.to_numpy()
                disagreements.append(dict(category=category,seed=seed,comparison=name,
                    padim_only_detected_defects=int((truth&current&~old).sum()),baseline_only_detected_defects=int((truth&~current&old).sum()),
                    padim_only_false_alarms=int((~truth&current&~old).sum()),baseline_only_false_alarms=int((~truth&~current&old).sum())))
            for artifact in (path,out/f'evaluation/{identifier}-calibration.json',out/f'evaluation/{identifier}-images.csv',
                    ROOT/r['checkpoint'],ROOT/r['local_maps'],local/'bootstrap_vectors.npz',local/'native_image_histograms.npz',local/'latency_ms.csv'):
                artifacts.append(dict(path=artifact.relative_to(ROOT).as_posix(),sha256=sha256(artifact),bytes=artifact.stat().st_size))
    table=pd.DataFrame(rows)
    assert len(table)==45 and not table[['category','seed']].duplicated().any()
    table.to_csv(out/'all_runs.csv',index=False,lineterminator='\n')
    category=table.groupby('category')[METRICS].agg(['mean','std']);category.columns=['_'.join(x) for x in category.columns]
    category.reindex(CATEGORIES).to_csv(out/'all_category_means.csv',lineterminator='\n')
    seeds=table.groupby('seed')[METRICS].mean();seeds.to_csv(out/'macro_by_seed.csv',lineterminator='\n')
    table.groupby('kind')[METRICS].mean().to_csv(out/'texture_object.csv',lineterminator='\n')
    pd.DataFrame(defects).to_csv(out/'per_defect_results.csv',index=False,lineterminator='\n')
    pd.DataFrame(disagreements).to_csv(out/'paired_prediction_disagreements.csv',index=False,lineterminator='\n')
    macro={k:dict(mean=float(seeds[k].mean()),seed_std=float(seeds[k].std()),
        image_bootstrap_95ci=np.quantile(np.stack(vectors[k]).mean(0),[.025,.975]).tolist(),
        bootstrap_target='exact image metric, fixed fitted models/thresholds' if k in IMAGE else '4096-bin calibration-scaled pixel approximation') for k in METRICS}
    dimension=frozen['configuration']['dimension']
    architecture=PaDiMExtractor(dimension,pretrained=False)
    summary=dict(categories=15,seeds=frozen['seeds'],models=45,development_runs=16,macro=macro,
        frozen_backbone_parameters=sum(p.numel() for p in architecture.parameters()),trainable_backbone_parameters=0,
        spatial_gaussians=4096,gaussian_dimension=dimension,
        independent_gaussian_mean_covariance_parameters=4096*(dimension+dimension*(dimension+1)//2),
        stored_gaussian_float32_values=4096*(dimension+dimension*dimension),
        frozen_configuration_sha256=identity,total_fit_wall_seconds=float(table.training_seconds.sum()),
        total_evaluation_seconds=float(table.evaluation_seconds.sum()),
        mean_model_p50_ms=float(table.model_p50_ms.mean()),mean_model_p95_ms=float(table.model_p95_ms.mean()),
        mean_end_to_end_p50_ms=float(table.end_to_end_p50_ms.mean()),mean_end_to_end_p95_ms=float(table.end_to_end_p95_ms.mean()),
        peak_training_gpu_allocated_gib=float(table.train_peak_vram_gib.max()),peak_inference_allocated_gib=float(table.inference_peak_vram_gib.max()),
        peak_training_gpu_reserved_gib=float(table.train_peak_reserved_vram_gib.max()),peak_inference_reserved_gib=float(table.inference_peak_reserved_vram_gib.max()),
        peak_evaluation_cpu_rss_gib=float(table.cpu_peak_rss_gib.max()),max_checkpoint_bytes=int(table.checkpoint_bytes.max()),
        pixel_bootstrap_quantization_bias={k:dict(mean_binned_minus_exact=float(np.mean(v)),maximum_absolute_run_difference=float(np.max(np.abs(v)))) for k,v in bias.items()},
        pixel_operating_macro=table[['pixel_recall','pixel_fpr','pixel_precision','pixel_f1']].mean().to_dict(),
        image_runs_with_no_alarms=int((table.tp+table.fp==0).sum()),precision_convention='shared zero_division=0',
        configuration_changed_after_test=False,test_evaluations_per_category_seed=1,
        uncertainty_limit='Conditional on fitted models and calibration thresholds. Seed spread and calibration uncertainty are distinct.',
        storage_scope='Category Gaussian checkpoints; shared cached ImageNet backbone; licensed maps and bootstrap vectors local and ignored')
    comparisons={};costs=[];paired_rows=[]
    for name,folder in BASELINES.items():
        baseline=json.loads((ROOT/f'reports/{folder}/summary.json').read_text())
        comparisons[name]={k:baseline['macro'][k]['mean'] for k in METRICS}
        for key in IMAGE:
            delta=np.stack(paired[name][key]).mean(0)
            paired_rows.append(dict(comparison=name,metric=key,point_difference=macro[key]['mean']-float(baseline_tables[name][key].mean()),
                low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),
                method='Paired category/seed/image bootstrap; 2000 same draws; conditional on fixed models/thresholds'))
        category.join(pd.read_csv(ROOT/f'reports/{folder}/all_category_means.csv').set_index('category'),lsuffix='_padim',rsuffix='_baseline').to_csv(out/f'{folder}_category_comparison.csv',lineterminator='\n')
        costs.append(dict(model=name,model_p50_ms=baseline['mean_model_p50_ms'],end_to_end_p50_ms=baseline['mean_end_to_end_p50_ms']))
    comparisons['PaDiM']={k:v['mean'] for k,v in macro.items()}
    costs.append(dict(model='PaDiM',model_p50_ms=summary['mean_model_p50_ms'],end_to_end_p50_ms=summary['mean_end_to_end_p50_ms']))
    pd.DataFrame(comparisons).to_csv(out/'model_macro_comparison.csv',lineterminator='\n')
    pd.DataFrame(costs).to_csv(out/'model_latency_comparison.csv',index=False,lineterminator='\n')
    pd.DataFrame(paired_rows).to_csv(out/'paired_image_difference_intervals.csv',index=False,lineterminator='\n')
    for name,value in [('summary.json',summary),('artifact_manifest.json',artifacts)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
