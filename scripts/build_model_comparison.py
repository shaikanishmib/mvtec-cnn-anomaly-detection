"""Audit and compare existing final records. No training/inference dependencies."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import hashlib
import json
import subprocess
import time
import numpy as np
import pandas as pd
from scipy.stats import beta
from anomaly_detection.model_comparison import (MODELS, NAMES, METRICS, validate_coverage,
    category_macro, paired_statistics, threshold_diagnostics)
from anomaly_detection.metrics import binary_metrics
from anomaly_detection.tracking import config_hash

OUT = ROOT / 'reports/final_comparison'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv(frame, name):
    frame.to_csv(OUT / name, index=False, float_format='%.17g', lineterminator='\n')


def build():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    provenance = {}
    cache_path = ROOT/'outputs/final_comparison/hash_cache.json'
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    cache = load(cache_path) if cache_path.exists() else {}

    def audit(path, expected=None, role='source'):
        path = Path(str(path).replace('\\', '/'))
        if not path.is_absolute():
            path = ROOT / path
        relative = path.relative_to(ROOT).as_posix()
        if relative not in provenance:
            assert path.is_file(), relative
            stat=path.stat()
            key=f'{stat.st_size}:{stat.st_mtime_ns}'
            old=cache.get(relative,{})
            hashed=old['sha256'] if old.get('stat')==key else digest(path)
            provenance[relative] = dict(path=relative, bytes=stat.st_size,
                                        sha256=hashed, role=role)
            cache[relative]=dict(stat=key,sha256=hashed)
            if len(provenance)%50==0:
                write_json(cache_path,cache)
        if expected:
            assert provenance[relative]['sha256'] == expected, f'Identity mismatch: {relative}'
        return provenance[relative]['sha256']

    # Snapshot historical tracked files before analysis; excludes only authorized updates.
    protected_path = OUT / 'historical_protection.json'
    if not protected_path.exists():
        tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
        excluded = {'README.md', 'docs/milestone_status.md', '.gitattributes'}
        protected = {p: digest(ROOT / p) for p in tracked if p not in excluded
                     and not p.startswith(('reports/final_comparison/', 'notebooks/07_',
                                           'scripts/build_model_comparison', 'scripts/validate_model_comparison',
                                           'src/anomaly_detection/model_comparison', 'tests/test_model_comparison'))}
        write_json(protected_path, protected)
    partitions = {}
    for part in ('fit', 'development', 'calibration', 'test'):
        p = ROOT / f'reports/partitions/seed2026-{part}.csv'
        partitions[part] = pd.read_csv(p).fillna('')
        audit(p, role='partition')
    sets = [set(p.path) for p in partitions.values()]
    assert all(not sets[i] & sets[j] for i in range(4) for j in range(i))
    assert all('/train/good/' in p for key in ('fit','development','calibration') for p in partitions[key].path)
    rows, reconciliation, defects, failures, distributions = [], [], [], [], []
    for model in MODELS:
        base = ROOT / 'reports' / model
        source = pd.read_csv(base / 'all_runs.csv', float_precision='round_trip')
        audit(base / 'all_runs.csv')
        audit(base / 'artifact_manifest.json')
        # Verify original complete evaluation inventories, including local maps/checkpoints.
        for entry in load(base / 'artifact_manifest.json'):
            audit(entry['path'], entry['sha256'], 'original_manifest_artifact')
        frozen = ROOT / 'configs' / (f'{model}_frozen.json' if model != 'efficientad' else 'efficientad_reduced_frozen.json')
        frozen_sha = audit(frozen, role='frozen_policy')
        policy=load(frozen)
        for path,expected in policy.get('source_sha256',{}).items():
            audit(path,expected,'frozen_source')
        for _, original in source.iterrows():
            eid, category, seed = original.experiment_id, original.category, int(original.seed)
            result_path = base / ('final/runs' if model == 'efficientad' else 'evaluation') / f'{eid}-result.json'
            r = load(result_path)
            result_sha = audit(result_path)
            registry=ROOT/'experiments'/f'{eid}-result.json'
            event=load(registry)
            assert event['status']=='completed',registry
            audit(registry,role='append_only_registry')
            assert (r['category'], r['seed'], r['freeze_sha256']) == (category, seed, frozen_sha)
            assert all(abs(float(original[k]) - r[k]) < 1e-12 for k in METRICS)
            train = r
            if 'training_experiment' in r:
                train_path = base / 'runs' / (r['training_experiment'] + '-result.json')
                train = load(train_path)
                audit(train_path)
            checkpoint = r['checkpoint'] if 'checkpoint' in r else train['checkpoint']
            if isinstance(checkpoint, dict):
                audit(checkpoint['path'], checkpoint['sha256'], 'checkpoint')
            else:
                audit(checkpoint, r.get('checkpoint_sha256', train.get('checkpoint_sha256')), 'checkpoint')
            maps = r['local_maps']
            if isinstance(maps, dict):
                audit(maps['path'], maps.get('sha256'), 'maps')
            else:
                audit(maps, r.get('maps_sha256'), 'maps')
            pred_path = result_path.with_name(eid + '-images.csv')
            pred_sha = audit(pred_path)
            predictions = pd.read_csv(pred_path, float_precision='round_trip')
            expected_test = partitions['test'].query('category == @category')
            assert set(predictions.path) == set(expected_test.path)
            assert len(predictions) == len(expected_test)
            labels = predictions.path.str.contains('/test/good/').map({True:0,False:1}).to_numpy()
            assert np.array_equal(labels, predictions.label.to_numpy())
            cal = r['calibration']
            cal_path = result_path.with_name(eid + '-calibration.json')
            audit(cal_path)
            expected_cal = partitions['calibration'].query('category == @category')
            identity_columns=['path','sha256','partition'] if model=='padim' else ['path','sha256']
            source_hash = config_hash(expected_cal[identity_columns].to_dict('records'))
            assert cal.get('source_paths_sha256', cal.get('source_sha256')) == source_hash
            scores = cal.get('calibration_scores', cal.get('scores'))
            threshold = cal.get('image_threshold', cal.get('threshold'))
            diagnostic = threshold_diagnostics(scores, threshold)
            for population,values in [('calibration_normal',np.asarray(scores)),
                    ('test_normal',predictions.loc[predictions.label.eq(0),'score'].to_numpy()),
                    ('test_defect',predictions.loc[predictions.label.eq(1),'score'].to_numpy())]:
                distributions.append(dict(model=model,category=category,seed=seed,population=population,
                    n=len(values),minimum=values.min(),q25=np.quantile(values,.25),median=np.median(values),
                    q75=np.quantile(values,.75),q95=np.quantile(values,.95),maximum=values.max(),
                    frozen_threshold=threshold))
            assert diagnostic['calibration_n'] == len(expected_cal)
            assert np.quantile(scores, .95, method='higher') == threshold
            # Historical float32 CSV exports need float32 restoration before promotion.
            checks = []
            for strategy, values in [('round_trip', predictions.score.to_numpy()),
                                     ('restore_float32', predictions.score.to_numpy().astype(np.float32).astype(float))]:
                got = binary_metrics(labels, values, threshold)
                errors = {k: abs(got[{'image_auroc':'auroc','image_ap':'ap'}.get(k,k)]-r[k])
                          for k in ('image_auroc','image_ap','recall','fpr','precision','f1')}
                flags_ok = np.array_equal(values > threshold, predictions.predicted.to_numpy())
                checks.append((max(errors.values()), not flags_ok, strategy, errors))
            error, flags_bad, strategy, errors = min(checks, key=lambda x:(x[0],x[1]))
            reconciliation.append(dict(model=model,category=category,seed=seed,strategy=strategy,
                                       max_metric_error=error,prediction_flags_match=not flags_bad))
            assert error < 1e-8 and not flags_bad, (eid, checks)
            # Exact binomial intervals describe one operating point conditional on model.
            fp, tn, tp, fn = (int(r[k]) for k in ('fp','tn','tp','fn'))
            n = fp+tn
            fpr_lo = 0. if fp == 0 else beta.ppf(.025, fp, n-fp+1)
            fpr_hi = 1. if fp == n else beta.ppf(.975, fp+1, n-fp)
            row = dict(model=model,model_name=NAMES[model],category=category,seed=seed,
                       experiment_id=eid,source_result=result_path.relative_to(ROOT).as_posix(),
                       source_result_sha256=result_sha,predictions_path=pred_path.relative_to(ROOT).as_posix(),
                       predictions_sha256=pred_sha,frozen_sha256=frozen_sha,
                       calibration_source_sha256=source_hash,score_direction='higher_is_anomaly',
                       nominal_fpr=.05,image_threshold=threshold,**diagnostic,
                       test_normal_n=n,test_defect_n=tp+fn,test_fpr_ci_low=fpr_lo,test_fpr_ci_high=fpr_hi,
                       **{k:r[k] for k in METRICS},tn=tn,fp=fp,fn=fn,tp=tp,
                       training_seconds=r['training_seconds'],evaluation_seconds=r.get('evaluation_seconds',original.get('evaluation_seconds',np.nan)),
                       checkpoint_bytes=original.checkpoint_bytes,
                       model_p50_ms=r['latency']['model']['p50_ms'],model_p95_ms=r['latency']['model']['p95_ms'],
                       end_to_end_p50_ms=r['latency']['end_to_end']['p50_ms'],end_to_end_p95_ms=r['latency']['end_to_end']['p95_ms'],
                       inference_peak_vram_gib=r['latency']['peak_inference_allocated_bytes']/2**30,
                       train_peak_vram_gib=original.train_peak_vram_gib,
                       cpu_peak_rss_gib=original.get('cpu_peak_rss_gib',np.nan),
                       latency_iterations=r['latency']['iterations'],latency_warmup=r['latency']['warmup'],
                       latency_scope=r['latency']['scope'],
                       original_extraction_seconds=r.get('original_extraction_seconds',np.nan))
            rows.append(row)
            for entry in r.get('per_defect', []):
                if isinstance(entry, dict):
                    defects.append(dict(entry,model=model,category=category,seed=seed))
            failure_path = base / 'failure_examples.csv'
            if failure_path.exists() and not any(x.get('source') == str(failure_path.relative_to(ROOT)).replace('\\','/') for x in failures):
                audit(failure_path)
                f = pd.read_csv(failure_path)
                failures.append(dict(model=model,source=failure_path.relative_to(ROOT).as_posix(),rows=len(f),categories=int(f.category.nunique())))
        print(f'{model}: 45 source evaluations and original artifact identities verified', flush=True)
        write_json(cache_path,cache)
    frame = pd.DataFrame(rows)
    validate_coverage(frame)
    csv(frame, 'all_runs.csv')
    csv(pd.DataFrame(reconciliation), 'prediction_reconciliation.csv')
    category = frame.groupby(['model','category'])[list(METRICS)].mean().reset_index()
    csv(category, 'category_summary.csv')
    csv(category_macro(frame).reset_index(), 'model_macro.csv')
    csv(frame.groupby(['model','seed'])[list(METRICS)].mean().reset_index(), 'seed_summary.csv')
    csv(frame.groupby(['model','category'])[list(METRICS)].std().reset_index(), 'within_category_seed_sd.csv')
    csv(paired_statistics(frame), 'paired_comparisons.csv')
    cal_columns = ['model','category','seed','nominal_fpr','calibration_n','calibration_fpr','calibration_unique','threshold_ties',
                   'image_threshold','calibration_std','calibration_min','calibration_median','calibration_max',
                   'fpr','recall','precision','f1','test_normal_n','test_fpr_ci_low','test_fpr_ci_high']
    csv(frame[cal_columns], 'calibration_runs.csv')
    cal_summary = frame.groupby('model')[['calibration_n','calibration_fpr','fpr','recall','precision','f1']].mean()
    cal_summary['runs_with_threshold_ties'] = frame.assign(ties=frame.threshold_ties>1).groupby('model').ties.sum()
    csv(cal_summary.reset_index(), 'calibration_summary.csv')
    # Cross-seed thresholds are not comparable across model score scales.
    threshold = frame.groupby(['model','category']).image_threshold.agg(['min','mean','max','std']).reset_index()
    threshold['relative_range'] = (threshold['max']-threshold['min'])/threshold['mean'].abs().replace(0,np.nan)
    csv(threshold, 'threshold_seed_variability.csv')
    cost = frame.groupby('model').agg(training_seconds=('training_seconds','sum'),evaluation_seconds=('evaluation_seconds','sum'),
        model_p50_ms=('model_p50_ms','median'),model_p95_ms=('model_p95_ms','median'),
        end_to_end_p50_ms=('end_to_end_p50_ms','median'),end_to_end_p95_ms=('end_to_end_p95_ms','median'),
        checkpoint_bytes=('checkpoint_bytes','max'),train_peak_vram_gib=('train_peak_vram_gib','max'),
        inference_peak_vram_gib=('inference_peak_vram_gib','max'),cpu_peak_rss_gib=('cpu_peak_rss_gib','max'))
    cost['training_hours'] = cost.training_seconds/3600
    cost['evaluation_hours'] = cost.evaluation_seconds/3600
    cost['category_checkpoint_mib'] = cost.checkpoint_bytes/2**20
    cost['shared_trunk_mib'] = [11179957/2**20 if m in ('patchcore','supersimplenet','padim') else 0 for m in cost.index]
    cost['deployment_mib_one_category'] = cost.category_checkpoint_mib+cost.shared_trunk_mib
    csv(cost.reset_index(), 'compute_summary.csv')
    csv(pd.DataFrame(failures), 'failure_source_inventory.csv')
    csv(pd.DataFrame(distributions),'score_distribution_summary.csv')
    defect_frame=pd.DataFrame(defects)
    if 'confusion_matrix' in defect_frame:
        defect_frame['confusion_matrix']=defect_frame.confusion_matrix.map(json.dumps)
    csv(defect_frame,'per_defect_runs.csv')
    # Per-defect schema is heterogeneous. Preserve original saved aggregate tables as provenance.
    for model in MODELS:
        for name in ('small_defect_summary.csv','defect_type_size_summary.csv','per_defect_results.csv','summary.json'):
            p = ROOT/'reports'/model/name
            if p.exists():
                audit(p)
    write_json(OUT/'source_provenance.json', sorted(provenance.values(),key=lambda x:x['path']))
    write_json(OUT/'audit_summary.json', dict(status='passed',records=len(frame),models=6,categories=15,seeds=[2026,2027,2028],
        source_files=len(provenance),source_bytes=sum(x['bytes'] for x in provenance.values()),
        current_pass_seconds=time.perf_counter()-start,hash_cache='Previously SHA256-verified unchanged size/mtime entries reused on recovery',partition_counts={k:len(v) for k,v in partitions.items()},
        prediction_max_metric_error=max(x['max_metric_error'] for x in reconciliation),
        inference_or_training_executed=False,method='source identities plus saved-prediction reconciliation; primary pixel metrics preserved'))
    print('Canonical comparison complete', flush=True)


if __name__ == '__main__':
    build()
