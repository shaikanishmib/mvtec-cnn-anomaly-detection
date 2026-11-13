"""Inspect the exact Git index for publication scope, pixels, sizes and identities."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
import nbformat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.tracking import config_hash
root=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
paths=git('diff','--cached','--name-only').decode().splitlines()
assert paths,'No staged files'
for path in paths:
    assert not path.startswith(('data/','outputs/','checkpoints/','.aws/','.env'))
    assert not path.endswith(('.pt','.pth','.npz','.npy','.dat','.png','.jpg','.zip'))
    blob=git('show',':'+path)
    assert len(blob)<5*2**20,(path,len(blob))
    assert not re.search(rb'AKIA[0-9A-Z]{16}',blob),path
    assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n',blob),path
    if path.endswith('.ipynb'):
        notebook=nbformat.reads(blob.decode('utf-8'),as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.metadata.get('local_dataset_pixels'):assert not cell.outputs
    if path.startswith('experiments/'):
        event=json.loads(blob)
        signature=event.pop('config_sha256');event.pop('recorded_at')
        assert config_hash(event)==signature,path
frozen=json.loads((root/'configs/patchcore_frozen.json').read_text())
pinned=dict(frozen['source_sha256'])
pinned['src/anomaly_detection/patchcore_evaluation.py']=frozen['evaluation_source_sha256']
pinned['configs/patchcore_frozen.json']=hashlib.sha256((root/'configs/patchcore_frozen.json').read_bytes()).hexdigest()
for item in json.loads((root/'reports/patchcore/artifact_manifest.json').read_text()):
    pinned[item['path']]=item['sha256']
checked=0
for path,expected in pinned.items():
    if path not in paths:continue
    # Git stores normalized text; verify the declared checkout byte identity.
    checkout=git('cat-file','--filters','--path='+path,':'+path)
    assert hashlib.sha256(checkout).hexdigest()==expected,path
    checked+=1
print(json.dumps(dict(staged_files=len(paths),pinned_checkout_identities_checked=checked,
    checks='scope, file sizes, credential patterns, private notebook output stripping, registry signatures, frozen/artifact byte identities'),indent=2))
