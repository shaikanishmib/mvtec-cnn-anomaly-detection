"""Deep pilot-stage validation; never claims final category/seed coverage."""
import json
from pathlib import Path
import pandas as pd
from anomaly_detection.data import sha256
from anomaly_detection.tracking import config_hash
from anomaly_detection.efficientad import load_checkpoint
from validate_notebook import validate
ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'reports/efficientad/pilot.json').read_text())
    assert r['status']=='passed' and len(r['history'])==32
    assert r['after']['hard']<r['before']['hard']
    assert r['after']['autoencoder']<r['before']['autoencoder']
    assert r['feature_shape']==[1,384,64,64] and r['map_shape']==[1,1,256,256]
    p=ROOT/r['checkpoint']['path']
    assert sha256(p)==r['checkpoint']['sha256'] and p.stat().st_size==r['checkpoint']['bytes']
    assert sha256(ROOT/'checkpoints/efficientad/assets/teacher_small.pth')==r['teacher']['sha256']
    _,meta=load_checkpoint(p)
    assert meta['teacher_sha256']==r['teacher']['sha256'] and meta['pilot']
    fit=ROOT/'reports/partitions/seed2026-fit.csv'
    assert sha256(fit)==r['partition_manifest_sha256']
    table=pd.read_csv(fit)
    expected=table[table.category=='bottle'].sort_values('path').head(8).path.tolist()
    assert r['input_paths']==expected
    for row in table[table.path.isin(expected)].itertuples():
        assert sha256(ROOT/'data/mvtec_ad'/row.path)==row.sha256
    events=list((ROOT/'experiments').glob('06-*.json'))
    assert len(events)==2
    for p in events:
        event=json.loads(p.read_text()); identity=event.pop('config_sha256'); event.pop('recorded_at')
        assert config_hash(event)==identity
    return dict(status='pilot_stage_passed',final_evaluations=0,registry_events=len(events),notebook=validate(ROOT/'notebooks/06_efficientad.ipynb',require_executed=True))


if __name__=='__main__':
    result=main()
    (ROOT/'reports/efficientad/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
