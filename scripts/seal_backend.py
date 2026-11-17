"""Seal/check the compact API publication; excludes all uploads/weights/caches."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/backend'
TARGET=OUT/'publication_manifest.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def paths():
    files=[p for p in OUT.rglob('*') if p.is_file() and p!=TARGET]
    files+=list((ROOT/'src/anomaly_detection/backend').glob('*.py'))
    files+=[ROOT/p for p in ['.gitattributes','README.md','docs/milestone_status.md','docs/backend_api.md',
        'requirements-backend.txt','tests/test_backend.py','tests/test_backend_real.py']]
    files+=[ROOT/'scripts'/n for n in ['backend_server_process.py','benchmark_backend.py',
        'finalize_backend_reports.py','finalize_backend_validation.py','smoke_backend_server.py',
        'validate_backend.py','validate_backend_defects.py','validate_backend_real.py','seal_backend.py']]
    return sorted(set(files))


if __name__=='__main__':
    if '--check' in sys.argv:
        manifest=json.loads(TARGET.read_text())
        for entry in manifest['files']:
            p=ROOT/entry['path'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],entry['path']
        print(dict(status='passed',sealed_files=len(manifest['files'])))
    else:
        assert not TARGET.exists(),'Already sealed; use --check'
        complete=json.loads((OUT/'completion_validation.json').read_text())
        assert complete['status']=='passed' and complete['full_suite_tests']==203
        entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths()]
        TARGET.write_text(json.dumps(dict(schema_version=1,baseline='3a4eac2237ca3e1a4c35b9c18974ba81d663c21e',
            scope='Milestone 11 compact backend code/documentation/evidence only',files=entries),indent=2)+'\n')
        print(dict(status='sealed',files=len(entries)))
