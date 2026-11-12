"""Reconcile AE research without retraining or repeating official inference."""
from pathlib import Path
import json,sys
import nbformat
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT,read_partition
from anomaly_detection.autoencoder import image_scores
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.tracking import config_hash
from anomaly_detection.metrics import binary_metrics


def validate():
    torch.set_num_threads(4)
    output=ROOT/'reports/autoencoder'
    frozen=json.loads((ROOT/'configs/autoencoder_frozen.json').read_text())
    identity=sha256(ROOT/'configs/autoencoder_frozen.json')
    assert identity==json.loads((output/'freeze_identity.json').read_text())['sha256']
    for path,expected in frozen['source_sha256'].items():assert sha256(ROOT/path)==expected
    assert sha256(ROOT/'src/anomaly_detection/ae_evaluation.py')==frozen['evaluation_source_sha256']
    events=[]
    for path in (ROOT/'experiments').glob('02-*.json'):
        event=json.loads(path.read_text());signature=event.pop('config_sha256');event.pop('recorded_at')
        assert config_hash(event)==signature,path
        events.append(json.loads(path.read_text()))
    trained=[e['recorded_at'] for e in events if e['experiment_id'].startswith('02-final-') and e['status']=='completed']
    tested=[e['recorded_at'] for e in events if e['experiment_id'].startswith('02-test-') and e['status']=='started']
    assert len(trained)==len(tested)==45 and max(trained)<min(tested)
    assert frozen['frozen_at_utc']<min(trained)
    table=pd.read_csv(output/'all_runs.csv')
    assert len(table)==45 and set(table.category)==set(CATEGORIES)
    assert table.groupby('category').seed.apply(set).apply(lambda s:s=={2026,2027,2028}).all()
    for row in table.itertuples():
        identifier=row.experiment_id
        r=json.loads((output/f'evaluation/{identifier}-result.json').read_text())
        run=json.loads((output/f"runs/{r['training_experiment']}-result.json").read_text())
        assert run['category']==row.category and run['configuration']['seed']==row.seed
        assert run['configuration']==dict(frozen['configuration'],seed=row.seed)
        assert r['freeze_sha256']==identity
        assert sha256(ROOT/run['checkpoint'])==run['checkpoint_sha256']==r['checkpoint_sha256']
        saved=torch.load(ROOT/run['checkpoint'],map_location='cpu',weights_only=True)
        assert saved['category']==row.category and saved['config']==run['configuration']
        assert saved['training_phase']=='final'
        assert run['gradient_normal_images']==len(read_partition('fit',row.category))
        assert run['probe_normal_sources']==0
        assert sha256(ROOT/r['local_maps'])==r['maps_sha256']
        assert json.loads((output/f'evaluation/{identifier}-calibration.json').read_text())==r['calibration']
        images=pd.read_csv(output/f'evaluation/{identifier}-images.csv')
        assert images.path.tolist()==read_partition('test',row.category).path.tolist()
        maps=np.load(ROOT/r['local_maps'],mmap_mode='r')
        recomputed=[]
        for start in range(0,len(maps),16):
            recomputed.extend(image_scores(torch.from_numpy(np.array(maps[start:start+16]))).numpy())
        np.testing.assert_allclose(recomputed,images.score,rtol=1e-6,atol=1e-7)
        measured=binary_metrics(images.label,images.score,r['calibration']['image_threshold'])
        for key,target in [('auroc','image_auroc'),('ap','image_ap'),('recall','recall'),('fpr','fpr'),('f1','f1')]:
            np.testing.assert_allclose(measured[key],r[target],rtol=1e-12,atol=1e-12)
        assert measured['confusion_matrix']==r['confusion_matrix']
        h=pd.read_csv(output/f"runs/{run['experiment_id']}-history.csv")
        assert len(h)==20
        np.testing.assert_allclose(h.train_loss,[v['train_loss'] for v in run['history']])
    public=nbformat.read(ROOT/'notebooks/02_convolutional_autoencoder.ipynb',as_version=4)
    local=nbformat.read(ROOT/'outputs/autoencoder/02_convolutional_autoencoder.complete.executed.ipynb',as_version=4)
    nbformat.validate(public);nbformat.validate(local)
    assert len(public.cells)==len(local.cells)
    for i,cell in enumerate(public.cells):
        if cell.cell_type!='code':continue
        compile(cell.source,f'Notebook02:{i}','exec')
        assert cell.execution_count is not None
        assert not any(o.output_type=='error' for o in cell.outputs)
        if cell.metadata.get('local_dataset_pixels'):assert not cell.outputs and local.cells[i].outputs
    result=dict(models=45,categories=15,seeds=[2026,2027,2028],registry_events=len(events),
        freeze_sha256=identity,public_notebook_sha256=sha256(ROOT/'notebooks/02_convolutional_autoencoder.ipynb'),
        local_notebook_sha256=sha256(ROOT/'outputs/autoencoder/02_convolutional_autoencoder.complete.executed.ipynb'),
        notebook_execution='Live pilot followed by separately executed artifact-analysis cells; no training replay claim',
        checks='registry hashes, frozen code, category/seed coverage, checkpoint/map/history identities, image metrics and notebook execution')
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(validate(),indent=2))
