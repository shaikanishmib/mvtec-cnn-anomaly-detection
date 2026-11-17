"""Register existing seed-2026 artifacts; never fit, calibrate or download."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import importlib.metadata
import json
import subprocess
import pandas as pd
from anomaly_detection.data import CATEGORIES
from anomaly_detection.inference.patchcore import sha256
from anomaly_detection.tracking import config_hash

OUT=ROOT/'reports/engineering/patchcore'
MANIFEST=ROOT/'configs/patchcore_deployment_v1.json'


def artifact(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    if not path.is_file():raise FileNotFoundError(path.name)
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha256(path))


def build():
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():
        raise FileExistsError('Deployment v1 already registered; validate and reuse it')
    old=json.loads((ROOT/'reports/patchcore/artifact_manifest.json').read_text())
    expected={x['path']:x['sha256'] for x in old}
    runs=pd.read_csv(ROOT/'reports/patchcore/all_runs.csv')
    assert len(runs)==45 and set(runs.category)==set(CATEGORIES)
    policy=ROOT/'configs/patchcore_frozen.json'
    frozen=json.loads(policy.read_text())
    shared=json.loads((ROOT/'reports/patchcore/summary.json').read_text())['shared_backbone_storage']
    extractor=artifact(shared['path']);assert extractor['sha256']==shared['sha256']
    extractor['original_weight_sha256']=shared['original_weight_sha256']
    preprocessing=dict(version='PIL-RGB-bilinear-v1',resolution=[256,256],resize='PIL bilinear',
        orientation='no EXIF rotation',range='float32 RGB /255',normalization='torchvision ImageNet V1',
        mean=[.485,.456,.406],std=[.229,.224,.225],layers=['layer2','layer3'],
        pooling='3x3 average stride1 padding1',feature_alignment='bilinear align_corners=False',
        map_resize='32 -> 256 -> native bilinear align_corners=False',aggregation='top ceil(1% of 1024) patch distances mean',
        distances='raw unnormalized float32 L2; no Gaussian smoothing')
    categories={};availability=[]
    for row in runs.itertuples():
        result_path=ROOT/'reports/patchcore/evaluation'/f'{row.experiment_id}-result.json'
        result=json.loads(result_path.read_text());assert sha256(result_path)==expected[result_path.relative_to(ROOT).as_posix()]
        training_path=ROOT/'reports/patchcore/runs'/f"{result['training_experiment']}-result.json"
        training=json.loads(training_path.read_text())
        bank=artifact(training['checkpoint']);assert bank['sha256']==training['checkpoint_sha256']==result['checkpoint_sha256']
        availability.append(dict(category=row.category,seed=int(row.seed),bank_sha256=bank['sha256'],available=True))
        if row.seed!=2026:continue
        calibration_path=result_path.with_name(row.experiment_id+'-calibration.json')
        cal=json.loads(calibration_path.read_text())
        assert result['calibration']==cal
        categories[row.category]=dict(bank=bank,calibration=artifact(calibration_path),
            thresholds=dict(image=cal['image_threshold'],pixel=cal['pixel_threshold'],strict_greater=True),
            calibration_normal_count=cal['images'],result=artifact(result_path),
            training_record=artifact(training_path),predictions=artifact(result_path.with_name(row.experiment_id+'-images.csv')))
    manifest=dict(schema_version=1,release='patchcore-inference-v1',model_family='Simplified PatchCore',
        baseline_commit='c8fd9fb6bf6ef9e0bfa5b63c9060350412df70ea',seed=2026,
        seed_policy='Lowest preregistered seed 2026 for every category, independent of test metrics',
        configuration=frozen['configuration'],frozen_policy=artifact(policy),extractor=extractor,
        preprocessing=preprocessing,preprocessing_sha256=config_hash(preprocessing),categories=categories,
        software={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','pillow']},
        limitations=['Frozen research thresholds have substantial category-dependent false alarms',
            'ResNet18 variant; not paper-equivalent PatchCore','Code licensing does not establish pretrained weight/data commercial rights'])
    MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(availability).to_csv(OUT/'artifact_availability.csv',index=False,lineterminator='\n')
    # Prior research bytes are protected, excluding the authorized rolling overview.
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    protected={p:sha256(ROOT/p) for p in tracked if p not in ('README.md','docs/milestone_status.md','.gitattributes')}
    (OUT/'research_protection.json').write_text(json.dumps(protected,indent=2)+'\n')
    print(dict(categories=len(categories),available_banks=len(availability),manifest_sha256=sha256(MANIFEST)))


if __name__=='__main__':build()
