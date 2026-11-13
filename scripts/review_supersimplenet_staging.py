"""Review exact staged publication content and declared checkout byte hashes."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.tracking import config_hash
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
paths=git('diff','--cached','--name-only').decode().splitlines();assert paths
for path in paths:
    assert not path.startswith(('data/','outputs/','checkpoints/','.aws/','.env'))
    assert not path.endswith(('.pt','.pth','.npy','.npz','.dat','.png','.jpg','.zip'))
    blob=git('show',':'+path);assert len(blob)<5*2**20,(path,len(blob))
    assert not re.search(rb'AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',blob)
    if path.endswith('.ipynb'):
        notebook=nbformat.reads(blob.decode(),as_version=4);nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.metadata.get('local_dataset_pixels'):assert not cell.outputs
    if path.startswith('experiments/'):
        record=json.loads(blob);expected=record.pop('config_sha256');record.pop('recorded_at')
        assert config_hash(record)==expected,path
frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text());pinned=dict(frozen['source_sha256'])
pinned['configs/supersimplenet_frozen.json']=hashlib.sha256((ROOT/'configs/supersimplenet_frozen.json').read_bytes()).hexdigest()
for item in json.loads((ROOT/'reports/supersimplenet/artifact_manifest.json').read_text()):pinned[item['path']]=item['sha256']
checked=0
for path,value in pinned.items():
    if path in paths:
        assert hashlib.sha256(git('cat-file','--filters','--path='+path,':'+path)).hexdigest()==value,path
        checked+=1
print(json.dumps(dict(staged_paths=len(paths),pinned_checkout_bytes_verified=checked,checks='scope, sizes, secrets patterns, raw-image stripping, registry signatures, frozen byte identities'),indent=2))
