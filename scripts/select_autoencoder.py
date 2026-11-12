"""Selection uses only registered development proxies; requires complete coverage."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT,read_partition
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.tracking import record

output=ROOT/'reports/autoencoder'
rows=[];results={}
for variant in ['base','deeper','narrow','wide','mse']:
    for category in ['bottle','cable','carpet','tile']:
        identifier=f'02-dev-{variant}-{category}-s2026-v1'
        result=json.loads((output/f'runs/{identifier}-result.json').read_text())
        results[(variant,category)]=result
        for kind,metrics in result['metrics'].items():
            rows.append(dict(variant=variant,category=category,map_kind=kind,
                objective=(metrics['auroc']+metrics['pixel_ap'])/2,
                parameters=result['parameters'],training_seconds=result['training_seconds_including_loading_validation'],
                peak_allocated_bytes=result['peak_allocated_bytes'],**{k:v for k,v in metrics.items() if k!='confusion_matrix'}))
table=pd.DataFrame(rows)
table.to_csv(output/'development_comparison.csv',index=False)
macro=table.groupby(['variant','map_kind']).mean(numeric_only=True).reset_index()
macro=macro.sort_values(['objective','recall','parameters'],ascending=[False,False,True])
macro.to_csv(output/'development_macro.csv',index=False)
chosen=macro.iloc[0]
config=results[(chosen.variant,'bottle')]['configuration'].copy()
config['map_kind']=chosen.map_kind
freeze=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),selected_variant=chosen.variant,
    configuration=config,categories=list(CATEGORIES),seeds=[2026,2027,2028],
    scoring='channel-mean residual, fixed top ceil(1% HW) mean at model resolution; native bilinear pixel maps',
    calibration='image higher q95, strict >; pixel upper bin edge pooled normal q99.5 at model resolution, 65536 bins',
    evaluation_source_sha256=sha256(ROOT/'src/anomaly_detection/ae_evaluation.py'),
    pixel_metric_source_sha256=sha256(ROOT/'src/anomaly_detection/pixel_metrics.py'),
    source_sha256={p:sha256(ROOT/p) for p in ['src/anomaly_detection/autoencoder.py','src/anomaly_detection/ae_research.py']},
    selection='four-category mean (proxy image AUROC + pixel AP)/2; no official test evidence',
    target_image_fpr=.05)
with (ROOT/'configs/autoencoder_frozen.json').open('x') as stream:json.dump(freeze,stream,indent=2)
identity=sha256(ROOT/'configs/autoencoder_frozen.json')
with (output/'freeze_identity.json').open('x') as stream:json.dump(dict(sha256=identity),stream,indent=2)
# Estimate the final selected architecture using measured epoch times per fitting image.
selected=[results[(chosen.variant,c)] for c in ['bottle','cable','carpet','tile']]
per_image=sum(sum(h['train_seconds'] for h in r['history']) for r in selected)/sum(r['gradient_normal_images']*config['epochs'] for r in selected)
images=sum(len(read_partition('fit',c)) for c in CATEGORIES)
budget=dict(pilot='02-pilot-shallow-bottle-s2026-v1',configuration=config,seeds=freeze['seeds'],
    models=45,fitting_images_per_seed=images,epochs=config['epochs'],
    measured_selected_seconds_per_training_image_epoch=per_image,
    expected_training_seconds=per_image*images*config['epochs']*3,
    observed_selected_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in selected),
    caveat='Estimate excludes initial decoding, validation, calibration, native metrics and latency benchmark; category sizes/hardware load vary.',
    planned_native_evaluation_minutes=60,final_training_budget_minutes=45)
with (output/'compute_budget.json').open('x') as stream:json.dump(budget,stream,indent=2)
event=json.loads((ROOT/'experiments/02-dev-base-bottle-s2026-v1-start.json').read_text())
for key in ['recorded_at','config_sha256']:event.pop(key)
event.update(experiment_id='02-development-selection-v1',categories=freeze['categories'],status='completed',
    hypothesis='Choose a single reconstruction policy using registered proxy evidence',hyperparameters=freeze,
    metrics=dict(candidates=macro.to_dict('records'),budget=budget),decision='freeze '+chosen.variant+' / '+chosen.map_kind,
    interpretation='Proxy superiority does not imply real-defect superiority. No combined loss/smoothing/resolution trial is justified by this bounded comparison.')
record(ROOT/'experiments',event)
print(macro.to_string(index=False));print(json.dumps(budget,indent=2))
