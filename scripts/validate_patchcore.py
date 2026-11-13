"""Audit retained evidence; no model fitting or official feature extraction."""
from pathlib import Path
import json,sys,re
import nbformat
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT,read_partition
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.tracking import config_hash
from anomaly_detection.patchcore import image_scores,pixel_maps
from anomaly_detection.metrics import binary_metrics


def validate():
    torch.set_num_threads(4)
    report=ROOT/'reports/patchcore';freeze_path=ROOT/'configs/patchcore_frozen.json'
    frozen=json.loads(freeze_path.read_text());identity=sha256(freeze_path)
    assert identity==json.loads((report/'freeze_identity.json').read_text())['sha256']
    for path,expected in frozen['source_sha256'].items():assert sha256(ROOT/path)==expected,path
    events=[]
    for path in (ROOT/'experiments').glob('03-*.json'):
        event=json.loads(path.read_text());signature=event.pop('config_sha256');event.pop('recorded_at')
        assert config_hash(event)==signature,path
        events.append(json.loads(path.read_text()))
    fits=[e['recorded_at'] for e in events if e['experiment_id'].startswith('03-final-') and e['status']=='completed']
    tests=[e['recorded_at'] for e in events if e['experiment_id'].startswith('03-test-') and e['status']=='started']
    assert len(fits)==len(tests)==45 and frozen['frozen_at_utc']<min(fits) and max(fits)<min(tests)
    table=pd.read_csv(report/'all_runs.csv')
    storage=json.loads((report/'backbone_storage.json').read_text())
    assert sha256(ROOT/storage['path'])==storage['sha256']
    assert storage['original_weight_sha256']==frozen['weight_sha256']
    assert len(table)==45 and set(table.category)==set(CATEGORIES)
    assert table.groupby('category').seed.apply(set).apply(lambda s:s=={2026,2027,2028}).all()
    caches={};cache_misses=0
    for row in table.itertuples():
        identifier=row.experiment_id
        result=json.loads((report/f'evaluation/{identifier}-result.json').read_text())
        run=json.loads((report/f"runs/{result['training_experiment']}-result.json").read_text())
        saved=torch.load(ROOT/run['checkpoint'],weights_only=True,map_location='cpu')
        assert saved['category']==row.category and saved['config']==dict(frozen['configuration'],seed=row.seed)
        assert saved['provenance']['phase']=='final' and saved['provenance']['partition']=='fit'
        assert run['probe_sources']==0
        fit_rows=read_partition('fit',row.category)
        assert run['fitting_normals']==len(fit_rows)
        assert saved['provenance']['source_rows_sha256']==config_hash(fit_rows[['path','sha256']].to_dict('records'))
        assert saved['bank'].shape==(2048,384) and saved['indices'].unique().numel()==2048
        assert saved['indices'].min()>=0 and saved['indices'].max()<len(fit_rows)*1024
        assert sha256(ROOT/run['checkpoint'])==run['checkpoint_sha256']==result['checkpoint_sha256']
        assert result['freeze_sha256']==identity
        assert json.loads((report/f'evaluation/{identifier}-calibration.json').read_text())==result['calibration']
        images=pd.read_csv(report/f'evaluation/{identifier}-images.csv')
        assert images.path.tolist()==read_partition('test',row.category).path.tolist()
        maps_path=ROOT/result['local_maps'];patches=np.load(maps_path.parent/'patch_distances.npy',mmap_mode='r')
        maps=np.load(maps_path,mmap_mode='r')
        assert sha256(maps_path)==result['maps_sha256']
        for start in range(0,len(patches),16):
            p=torch.from_numpy(np.array(patches[start:start+16]))
            np.testing.assert_allclose(image_scores(p,frozen['configuration']['aggregation']).numpy(),images.score.iloc[start:start+16],atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(pixel_maps(p).numpy(),maps[start:start+16],atol=1e-6,rtol=1e-6)
        measured=binary_metrics(images.label,images.score,result['calibration']['image_threshold'])
        for key,target in [('auroc','image_auroc'),('ap','image_ap'),('recall','recall'),('fpr','fpr'),('f1','f1')]:
            np.testing.assert_allclose(measured[key],result[target],atol=1e-12)
        assert measured['confusion_matrix']==result['confusion_matrix']
        for key in ['recall','fpr','precision','f1','tn','fp','fn','tp']:
            np.testing.assert_allclose(getattr(row,'pixel_'+key),result['pixel_operating_point'][key],atol=1e-12)
        for cache in [run['feature_cache'],result['calibration']['feature_cache'],result['test_cache']]:caches[cache['path']]=cache
        cache_misses+=not run['feature_cache']['cache_hit']
    assert cache_misses==15 and len(caches)==45
    for path,cache in caches.items():assert sha256(ROOT/path)==cache['sha256'],path
    public=nbformat.read(ROOT/'notebooks/03_patchcore.ipynb',as_version=4)
    local=nbformat.read(ROOT/'outputs/patchcore/03_patchcore.complete.executed.ipynb',as_version=4)
    nbformat.validate(public);nbformat.validate(local);assert len(public.cells)==len(local.cells)
    headings=[];count=0;private=0
    for i,cell in enumerate(public.cells):
        if cell.cell_type=='markdown':headings.extend(int(x) for x in re.findall(r'^## (\d+)\.',cell.source,re.M));continue
        compile(cell.source,f'Notebook03:{i}','exec')
        assert cell.execution_count is not None
        assert not any(o.output_type=='error' for o in cell.outputs)
        count+=1
        if cell.metadata.get('local_dataset_pixels'):
            private+=1;assert not cell.outputs and local.cells[i].outputs
    assert set(range(1,26)).issubset(headings) and private==3
    for item in json.loads((report/'artifact_manifest.json').read_text()):assert sha256(ROOT/item['path'])==item['sha256'],item['path']
    result=dict(models=45,categories=15,seeds=[2026,2027,2028],registry_events=len(events),
        freeze_sha256=identity,executed_code_cells=count,private_panel_cells=private,
        compatible_final_feature_caches=len(caches),fitting_cache_misses=cache_misses,
        public_notebook_sha256=sha256(ROOT/'notebooks/03_patchcore.ipynb'),
        local_notebook_sha256=sha256(ROOT/'outputs/patchcore/03_patchcore.complete.executed.ipynb'),
        scope='Source/weight policy, registry hashes/order, category-specific fitting, cache identities/reuse, banks, retained-map scores, shared metrics, notebook sources/real execution/private output stripping; no refit/inference')
    (report/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(validate(),indent=2))
