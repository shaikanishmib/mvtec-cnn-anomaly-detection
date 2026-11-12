"""Append a measured native-evaluation workload update without changing policy."""
from pathlib import Path
from datetime import datetime,timezone
import json
import pandas as pd
root=Path(__file__).resolve().parents[1]
output=root/'reports/autoencoder'
results=[json.loads(p.read_text()) for p in (output/'evaluation').glob('*-result.json')]
inventory=pd.read_csv(root/'outputs/eda/inventory.csv')
test=inventory[inventory.split.eq('test')]
all_pixels=int((test.height*test.width).sum())*3
observed_pixels=sum(r['positive_pixels']+r['negative_pixels'] for r in results)
seconds=sum(r['evaluation_seconds'] for r in results)
event=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),completed_models=len(results),
    measured_seconds=seconds,measured_native_pixels=observed_pixels,
    planned_total_native_pixels=all_pixels,
    pixel_count_scaled_expected_total_minutes=seconds/observed_pixels*all_pixels/60,
    policy_changed=False,interpretation='Measured evaluation cost update only; native shape, score uniqueness, cache and bootstrap costs vary. All 45 frozen evaluations remain in scope.')
with (output/f'evaluation_workload_after_{len(results)}_models.json').open('x') as stream:json.dump(event,stream,indent=2)
print(json.dumps(event,indent=2))
