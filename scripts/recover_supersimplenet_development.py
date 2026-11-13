"""Preserve externally interrupted trial; rerun only the unavailable state."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.tracking import record
from anomaly_detection.data import sha256
identifier='04-dev-focal_only-cable-s2026-v1'
history=ROOT/f'reports/supersimplenet/runs/{identifier}-history.csv'
assert history.exists() and not (ROOT/f'reports/supersimplenet/runs/{identifier}-result.json').exists()
entry=json.loads((ROOT/f'experiments/{identifier}-start.json').read_text())
for key in ['recorded_at','config_sha256']:entry.pop(key)
entry.update(experiment_id=identifier+'-interrupted',status='interrupted',
    metrics=dict(completed_epochs=2,history_sha256=sha256(history)),
    decision='Preserve two completed epochs and restart only this incomplete trial as v2; no checkpoint was available',
    interpretation='External turn interruption terminated Python. No learning-based rejection and no test access.',
    paths=dict(partial_history=history.relative_to(ROOT).as_posix()))
record(ROOT/'experiments',entry)
with (ROOT/'reports/supersimplenet/recovery.json').open('x') as stream:
    json.dump({'focal_only/cable':'04-dev-focal_only-cable-s2026-v2'},stream,indent=2)
print(entry['decision'])
