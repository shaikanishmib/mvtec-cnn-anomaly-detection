"""Seal compact Milestone 10 code/evidence; model artifacts remain excluded."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.inference.patchcore import sha256

D=ROOT/'reports/engineering/patchcore'
TARGET=D/'publication_manifest.json'


def paths():
    result=[p for p in D.rglob('*') if p.is_file() and p!=TARGET]
    result += list((ROOT/'src/anomaly_detection/inference').glob('*.py'))
    result += [ROOT/p for p in ['.gitattributes','README.md','docs/milestone_status.md',
        'docs/patchcore_inference.md','configs/patchcore_deployment_v1.json',
        'tests/test_patchcore_inference.py','scripts/build_patchcore_deployment.py',
        'scripts/validate_patchcore_engineering.py','scripts/validate_patchcore_inference.py',
        'scripts/seal_patchcore_engineering.py']]
    return sorted(set(result))


if __name__=='__main__':
    if '--check' in sys.argv:
        m=json.loads(TARGET.read_text())
        for entry in m['files']:
            p=ROOT/entry['path']
            assert p.stat().st_size==entry['bytes'] and sha256(p)==entry['sha256'],entry['path']
        print(dict(status='passed',sealed_files=len(m['files'])))
    else:
        assert not TARGET.exists(),'Already sealed; use --check'
        validation=json.loads((D/'validation.json').read_text())
        tests=json.loads((D/'test_summary.json').read_text())
        assert validation['status']=='passed' and validation['categories']==15
        assert tests['tests']==161 and tests['errors']==tests['failures']==tests['skipped']==0
        entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in paths()]
        TARGET.write_text(json.dumps(dict(schema_version=1,baseline_commit='c8fd9fb6bf6ef9e0bfa5b63c9060350412df70ea',
            scope='Compact engineering package; no model weights, images or caches',files=entries),indent=2)+'\n')
        print(dict(status='sealed',files=len(entries)))
