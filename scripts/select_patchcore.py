"""Choose exactly one permitted development candidate and freeze before testing."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,copy
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT,read_partition
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.tracking import record
output=ROOT/'reports/patchcore'
rows=[];results={}
for budget in [256,1024,2048]:
    for category in ['bottle','cable','carpet','tile']:
        identifier=f'03-dev-b{budget}-{category}-s2026-v1'
        result=json.loads((output/f'runs/{identifier}-result.json').read_text())
        results[(budget,category)]=result
        for aggregation,metrics in result['metrics'].items():
            rows.append(dict(budget=budget,category=category,aggregation=aggregation,
                coreset_seconds=result['coreset_seconds'],wall_seconds=result['wall_seconds'],
                bank_bytes=result['bank_vector_bytes'],search_p50_ms=result['development']['search_p50_ms'],
                search_p95_ms=result['development']['search_p95_ms'],
                gpu_peak_bytes=result['bank_resources']['peak_gpu_allocated_bytes'],
                cpu_peak_bytes=result['bank_resources']['cpu_peak_rss_bytes'],
                **{k:v for k,v in metrics.items() if k!='confusion_matrix'}))
table=pd.DataFrame(rows)
macro=table.groupby(['budget','aggregation']).mean(numeric_only=True).reset_index().sort_values(
    ['objective','search_p50_ms','budget'],ascending=[False,True,True])
table.to_csv(output/'development_comparison.csv',index=False,lineterminator='\n')
macro.to_csv(output/'development_macro.csv',index=False,lineterminator='\n')
chosen=macro.iloc[0];budget=int(chosen.budget)
config=dict(results[(budget,'bottle')]['configuration'],aggregation=chosen.aggregation)
source_paths=['src/anomaly_detection/patchcore.py','src/anomaly_detection/patchcore_research.py',
    'src/anomaly_detection/patchcore_runs.py','src/anomaly_detection/patchcore_evaluation.py',
    'src/anomaly_detection/cnn_evaluation.py','src/anomaly_detection/basic_cnn.py']
pilot=json.loads((output/'03-pilot-resnet18-bottle-s2026-v1-result.json').read_text())
frozen=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),configuration=config,
    seeds=[2026,2027,2028],categories=list(CATEGORIES),selected_variant=f'b{budget}',
    weight_sha256=pilot['weight_sha256'],weight_url=pilot['weight_url'],
    source_sha256={p:sha256(ROOT/p) for p in source_paths},
    evaluation_source_sha256=sha256(ROOT/'src/anomaly_detection/patchcore_evaluation.py'),
    pixel_metric_source_sha256=sha256(ROOT/'src/anomaly_detection/pixel_metrics.py'),
    image_calibration='higher reserved-normal q95, strict >; secondary q99 diagnostic',
    pixel_calibration='exact higher pooled reserved-normal 256-map q99.5, strict >',
    uncertainty='2000 whole-image bootstrap; exact image vectors; pixel 4096 bins of d/(d+calibration median patch distance); raw native point metrics exact',
    simplifications=['ResNet18','direct multiscale concatenation','20000 random candidates before projected coreset',
        'no paper image-score neighborhood reweighting','no Gaussian smoothing','whole-image 256 geometry override'])
with (ROOT/'configs/patchcore_frozen.json').open('x') as stream:json.dump(frozen,stream,indent=2)
with (output/'freeze_identity.json').open('x') as stream:json.dump(dict(sha256=sha256(ROOT/'configs/patchcore_frozen.json')),stream,indent=2)
selected=[results[(budget,c)] for c in ['bottle','cable','carpet','tile']]
budget_report=dict(pilot=pilot,selected_configuration=config,models=45,
    expected_coreset_seconds=45*sum(r['coreset_seconds'] for r in selected)/4,
    expected_fitting_extraction_seconds_once=pilot['projected_three_seed_final_extraction_seconds'],
    expected_all_partition_extraction_seconds=pilot['extraction_seconds']/24*5354,
    expected_final_bank_vectors_bytes_total=45*budget*384*4,
    max_fitting_embedding_bytes=275*1024*384*4,
    max_measured_development_gpu_allocated_bytes=max(r['bank_resources']['peak_gpu_allocated_bytes'] for r in selected),
    max_measured_development_cpu_rss_bytes=max(r['bank_resources']['cpu_peak_rss_bytes'] for r in selected),
    measured_development_search_p50_ms=float(chosen.search_p50_ms),
    expected_native_evaluation_minutes=75,expected_bank_and_cache_minutes=10,
    decision='Feasible: run all 15 categories with three coreset seeds; extraction shared; no official test tuning')
with (output/'compute_budget.json').open('x') as stream:json.dump(budget_report,stream,indent=2)
selection=json.loads((ROOT/'experiments/03-dev-b256-bottle-s2026-v1-start.json').read_text())
selection.pop('config_sha256');selection.pop('recorded_at')
selection.update(experiment_id='03-development-selection-v1',categories=list(CATEGORIES),
    hyperparameters=frozen,metrics=dict(candidates=macro.to_dict('records'),compute_budget=budget_report),
    hypothesis='Choose budget/aggregation from the fixed four-category synthetic-proxy objective within measured resource budgets.',
    status='completed',decision=f'freeze b{budget} / {chosen.aggregation}',
    interpretation='Only proxy evidence informs this selection. Larger banks did not consistently improve textures. No evidence justifies expanding backbone/normalization/smoothing searches in this bounded variant.')
record(ROOT/'experiments',selection)
for (tested_budget,category),result in results.items():
    identifier=result['experiment_id']
    event=json.loads((ROOT/f'experiments/{identifier}-result.json').read_text())
    event.pop('config_sha256');event.pop('recorded_at')
    candidate=macro[macro.budget.eq(tested_budget)].iloc[0]
    event.update(experiment_id=identifier+'-decision',decision='keep' if tested_budget==budget else 'reject',
        interpretation=f"Budget {tested_budget}: best four-category objective {candidate.objective:.6f}; chosen b{budget}/{chosen.aggregation} objective {chosen.objective:.6f}. All categories and both aggregations retained; proxies are not real defects.")
    record(ROOT/'experiments',event)
print(macro.to_string(index=False));print(json.dumps({k:v for k,v in budget_report.items() if k!='pilot'},indent=2))
