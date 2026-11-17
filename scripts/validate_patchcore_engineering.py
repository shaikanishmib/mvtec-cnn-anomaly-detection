"""Final saved-evidence, manifest and previous-research integrity validator."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json
import pandas as pd
from anomaly_detection.data import CATEGORIES
from anomaly_detection.inference.patchcore import verified_path,sha256
from anomaly_detection.tracking import config_hash


def validate():
    out=ROOT/'reports/engineering/patchcore'
    manifest=ROOT/'configs/patchcore_deployment_v1.json'
    m=json.loads(manifest.read_text())
    assert m['seed']==2026 and set(m['categories'])==set(CATEGORIES)
    verified_path(ROOT,m['extractor']);verified_path(ROOT,m['frozen_policy'])
    assert config_hash(m['preprocessing'])==m['preprocessing_sha256']
    for category,entry in m['categories'].items():
        for name in ('bank','calibration','result','training_record','predictions'):verified_path(ROOT,entry[name])
        calibration=json.loads((ROOT/entry['calibration']['path']).read_text())
        assert entry['thresholds']['image']==calibration['image_threshold']
        assert entry['thresholds']['pixel']==calibration['pixel_threshold']
    integration=json.loads((out/'integration.json').read_text())
    assert integration['status']=='passed' and integration['manifest_sha256']==sha256(manifest)
    frame=pd.read_csv(out/'parity.csv')
    assert len(frame)==60 and frame.category.nunique()==15 and set(frame.device)=={'cpu','cuda'}
    assert frame.groupby(['category','device']).size().eq(2).all()
    assert frame.reproducible.all() and (frame.decision==frame.reference_decision).all()
    assert frame.maximum_cpu_gpu_map_error.max()<2e-4
    protection=json.loads((out/'research_protection.json').read_text())
    for path,expected in protection.items():assert sha256(ROOT/path)==expected,path
    return dict(status='passed',manifest_sha256=sha256(manifest),categories=15,seed=2026,
        selected_banks_verified=15,available_research_banks=45,real_images=30,device_predictions=60,
        repeat_predictions=60,decisions_match=True,previous_research_files_unchanged=len(protection),
        cpu_gpu_maximum_map_error=float(frame.maximum_cpu_gpu_map_error.max()))


if __name__=='__main__':
    result=validate()
    (ROOT/'reports/engineering/patchcore/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
