"""Read-only PaDiM freeze, registry, notebook and artifact validation."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from anomaly_detection.data import sha256,CATEGORIES
from anomaly_detection.tracking import config_hash
from anomaly_detection.padim_evaluation import verify_freeze

ROOT=Path(__file__).resolve().parents[1]


def validate(require_notebook=True):
    frozen,identity=verify_freeze();report=ROOT/'reports/padim'
    summary=json.loads((report/'summary.json').read_text())
    assert summary['frozen_configuration_sha256']==identity
    assert summary['models']==45 and summary['categories']==15
    runs=pd.read_csv(report/'all_runs.csv')
    assert set(zip(runs.category,runs.seed))=={(c,s) for c in CATEGORIES for s in frozen['seeds']}
    artifacts=json.loads((report/'artifact_manifest.json').read_text())
    for item in artifacts:
        path=ROOT/item['path']
        assert path.stat().st_size==item['bytes'] and sha256(path)==item['sha256'],str(path)
    events=list((ROOT/'experiments').glob('05-*.json'))
    for path in events:
        event=json.loads(path.read_text())
        expected=event.pop('config_sha256');event.pop('recorded_at')
        assert config_hash(event)==expected,path.name
    assert len(list((report/'evaluation').glob('*-result.json')))==45
    assert len(list((report/'runs').glob('05-dev-*-result.json')))==16
    assert len(events)==2*(1+16+45)
    for row in runs.itertuples():
        result=json.loads((report/f'evaluation/{row.experiment_id}-result.json').read_text())
        table=pd.read_csv(report/f'evaluation/{row.experiment_id}-images.csv')
        assert not table.path.duplicated().any()
        assert np.isfinite(table.score).all()
        assert result['calibration']['strict_greater']
        assert result['freeze_sha256']==identity
        assert result['latency']['iterations']==200 and result['latency']['warmup']==50
        assert result['fitting_peak_allocated_bytes']<6*1024**3
    notebook=None
    if require_notebook:
        sys.path.insert(0,str(ROOT/'scripts'))
        from validate_notebook import validate as validate_notebook
        notebook=validate_notebook(ROOT/'notebooks/05_padim.ipynb',require_executed=True)
    return dict(result='passed',categories=15,seeds=frozen['seeds'],evaluations=45,development_runs=16,
        registry_events=len(events),verified_artifacts=len(artifacts),freeze_sha256=identity,notebook=notebook)


if __name__=='__main__':print(json.dumps(validate('--without-notebook' not in sys.argv),indent=2))
