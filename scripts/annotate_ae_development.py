"""Append explicit ablation interpretations without rewriting original events."""
from pathlib import Path
import json,sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT
from anomaly_detection.tracking import record

output=ROOT/'reports/autoencoder'
macro=pd.read_csv(output/'development_macro.csv')
selected=macro.iloc[0]
hypotheses={
    'base':'A shallow, capacity-limited reconstruction model detects fitting-derived synthetic edits through residuals.',
    'deeper':'An additional downsampling stage improves context and prevents copying edited detail.',
    'narrow':'Reducing latent channels from 8 to 2 suppresses anomaly reconstruction but may harm normal reconstruction.',
    'wide':'Increasing latent channels from 8 to 32 improves normal reconstruction but may also copy synthetic edits.',
    'mse':'MSE emphasizes large residuals and may improve residual ranking relative to L1.'}
for variant,hypothesis in hypotheses.items():
    for category in ['bottle','cable','carpet','tile']:
        identifier=f'02-dev-{variant}-{category}-s2026-v1'
        event=json.loads((ROOT/f'experiments/{identifier}-result.json').read_text())
        event.pop('recorded_at');event.pop('config_sha256')
        event.update(experiment_id=identifier+'-decision',hypothesis=hypothesis,
            training=dict(result_event=identifier+'-result',selection_event='02-development-selection-v1'),
            decision='keep' if variant==selected.variant else 'reject',
            interpretation=f"Variant {variant}: best four-category variant/map objective "
                f"{macro[macro.variant.eq(variant)].objective.max():.6f}; selected {selected.variant}/{selected.map_kind} "
                f"objective {selected.objective:.6f}. Decisions use all fixed development categories, not this category alone. "
                'Both residual rules and every epoch remain recorded; proxy results do not establish real-defect quality.')
        record(ROOT/'experiments',event)
print('Appended 20 decision records')
