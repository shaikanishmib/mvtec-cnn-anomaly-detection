"""Validate completed research artifacts without opening official test images."""
import json,sys,subprocess
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.tracking import config_hash
from anomaly_detection.data import sha256

def main():
    events=list((ROOT/'experiments').glob('04-*.json'))
    for path in events:
        data=json.loads(path.read_text());expected=data.pop('config_sha256');data.pop('recorded_at')
        assert config_hash(data)==expected,path
    results=list((ROOT/'reports/supersimplenet/runs').glob('*-result.json'))
    for path in results:
        data=json.loads(path.read_text())
        assert sha256(ROOT/data['checkpoint'])==data['checkpoint_sha256'],path
        assert sha256(ROOT/data['history_path'])==data['history_sha256'],path
        assert data['checkpoint_replay_exact']
    notebook=nbformat.read(ROOT/'notebooks/04_supersimplenet.ipynb',as_version=4);nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type=='error']
    previous=['notebooks/00_dataset_eda.ipynb','notebooks/01_basic_cnn.ipynb','notebooks/02_convolutional_autoencoder.ipynb','notebooks/03_patchcore.ipynb',
        'reports/basic_cnn','reports/autoencoder','reports/patchcore','reports/partitions']
    assert not subprocess.check_output(['git','diff','7edf24aa38e115b72d41754493a1539d1845136b','--name-only','--',*previous],cwd=ROOT,text=True).strip()
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    assert not any(p.startswith(('data/','outputs/','checkpoints/')) or p.endswith(('.pt','.pth','.npy','.npz','.png')) for p in staged)
    print(json.dumps(dict(registry_records_verified=len(events),completed_training_artifacts_verified=len(results),
        executed_notebook_code_cells=len(code),notebook_bytes=(ROOT/'notebooks/04_supersimplenet.ipynb').stat().st_size,
        prior_notebooks_reports_partitions_unchanged=True,staged_paths=len(staged)),indent=2))

if __name__=='__main__':main()
