"""Seal/check the compact publication package, without revisiting model artifacts."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'reports/final_comparison'
MANIFEST=D/'publication_manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_paths():
    paths=[p for p in D.rglob('*') if p.is_file() and p!=MANIFEST]
    paths += [ROOT/p for p in (
        '.gitattributes','README.md','docs/milestone_status.md',
        'notebooks/07_model_comparison.ipynb',
        'src/anomaly_detection/model_comparison.py','tests/test_model_comparison.py')]
    paths += list((ROOT/'scripts').glob('*model_comparison.py'))
    return sorted(set(paths))


def seal():
    validation=json.loads((D/'validation.json').read_text())
    tests=json.loads((D/'test_summary.json').read_text())
    assert validation['status']=='passed' and validation['records']==270
    assert validation['notebook']['executed_code_cells']==20
    assert tests['failures']==tests['errors']==tests['skipped']==0
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
             for p in publication_paths()]
    value=dict(baseline='815a14a81f30ccac4eb1b0678b3d1d2242da1ceb',
               scope='Compact Milestone 9 publication only; source artifacts unchanged',
               files=entries)
    assert not MANIFEST.exists(),'Use --check for an already sealed publication'
    MANIFEST.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return dict(files=len(entries),bytes=sum(x['bytes'] for x in entries),status='sealed')


def check():
    entries=json.loads(MANIFEST.read_text())['files']
    assert {x['path'] for x in entries}=={p.relative_to(ROOT).as_posix() for p in publication_paths()}
    for entry in entries:
        path=ROOT/entry['path']
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],entry['path']
    return dict(files=len(entries),status='publication identities verified')


if __name__=='__main__':
    print(check() if '--check' in sys.argv else seal())
