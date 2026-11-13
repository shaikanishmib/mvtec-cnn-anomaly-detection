"""Validate completed research artifacts without opening official test images."""
import json,sys,subprocess,itertools
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
    final_summary=ROOT/'reports/supersimplenet/summary.json'
    if final_summary.exists():
        import numpy as np,pandas as pd,torch
        from anomaly_detection.cnn_research import read_partition
        from anomaly_detection.metrics import binary_metrics,calibrate_normal
        frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text());summary=json.loads(final_summary.read_text())
        freeze_sha=sha256(ROOT/'configs/supersimplenet_frozen.json')
        assert freeze_sha==summary['frozen_configuration_sha256']==json.loads((ROOT/'reports/supersimplenet/freeze_identity.json').read_text())['sha256']
        for path,expected in frozen['source_sha256'].items():assert sha256(ROOT/path)==expected,path
        table=pd.read_csv(ROOT/'reports/supersimplenet/all_runs.csv')
        assert set(zip(table.category,table.seed))==set(itertools.product(frozen['categories'],frozen['seeds']))
        assert len(table)==summary['models']==15*len(frozen['seeds'])
        for key,value in summary['macro'].items():assert abs(table[key].mean()-value['mean'])<1e-12,key
        final_times=[];test_times=[];cache_images={}
        for row in table.itertuples():
            identifier=row.experiment_id;r=json.loads((ROOT/f'reports/supersimplenet/evaluation/{identifier}-result.json').read_text())
            train=json.loads((ROOT/f"reports/supersimplenet/runs/{r['training_experiment']}-result.json").read_text())
            assert train['phase']=='final' and train['fit_normals']==len(read_partition('fit',row.category))
            expected=dict(frozen['configuration'],seed=row.seed);assert train['configuration']==expected
            saved=torch.load(ROOT/train['checkpoint'],weights_only=True,map_location='cpu')
            assert saved['category']==row.category and saved['phase']=='final' and saved['configuration']==expected
            assert saved['weight_sha256']==frozen['weight_sha256']
            assert saved['fit_rows_sha256']==config_hash(read_partition('fit',row.category)[['path','sha256']].to_dict('records'))
            calibration=r['calibration'];reserved=read_partition('calibration',row.category)
            assert calibration['source_paths_sha256']==config_hash(reserved[['path','sha256']].to_dict('records'))
            assert calibration['images']==len(reserved) and calibration['image_threshold']==calibrate_normal(calibration['calibration_scores'])
            images=pd.read_csv(ROOT/f'reports/supersimplenet/evaluation/{identifier}-images.csv');test=read_partition('test',row.category)
            assert images.path.tolist()==test.path.tolist() and images.label.tolist()==test['mask'].notna().astype(int).tolist()
            checked=binary_metrics(images.label,images.score,calibration['image_threshold'])
            for key,value in checked.items():
                if key=='confusion_matrix':assert value==r[key]
                else:assert abs(value-r[{'auroc':'image_auroc','ap':'image_ap'}.get(key,key)])<1e-12
            local=(ROOT/r['local_maps']).parent;maps=np.load(ROOT/r['local_maps'],mmap_mode='r');det=np.load(local/'raw_detection.npy')
            if expected['aggregation']=='head':scores=det
            elif expected['aggregation']=='max':scores=maps.reshape(len(maps),-1).max(1)
            else:
                flat=maps.reshape(len(maps),-1);k=int(np.ceil(flat.shape[1]*.01));scores=np.partition(flat,-k,axis=1)[:,-k:].mean(1)
            np.testing.assert_allclose(scores,images.score,rtol=1e-6,atol=1e-6)
            final_times.append(json.loads((ROOT/f"experiments/{r['training_experiment']}-result.json").read_text())['recorded_at'])
            test_times.append(json.loads((ROOT/f'experiments/{identifier}-start.json').read_text())['recorded_at'])
            for cache in [train['feature_caches']['fit'],train['feature_caches']['normal_development'],calibration['feature_cache'],r['test_cache']]:
                for image in cache['images']:cache_images[image['path']]=image
        assert max(final_times)<min(test_times),'Test accessed before all fits completed'
        for artifact in json.loads((ROOT/'reports/supersimplenet/artifact_manifest.json').read_text()):
            assert sha256(ROOT/artifact['path'])==artifact['sha256'],artifact['path']
        for image in cache_images.values():
            assert image['profile']['weight_sha256']==frozen['weight_sha256']
            assert image['profile']['source_sha256']==frozen['source_sha256']['src/anomaly_detection/supersimplenet.py']
            assert image['shape']==[384,64,64] and sha256(ROOT/image['path'])==image['sha256']
        assert len(cache_images)==summary['deterministic_feature_images']==5354
        full=nbformat.read(ROOT/'outputs/supersimplenet/04_supersimplenet.full.executed.ipynb',as_version=4)
        assert len(full.cells)==len(notebook.cells)
        for public,local in zip(notebook.cells,full.cells):
            assert public.source==local.source
            if public.cell_type=='code':
                assert public.execution_count==local.execution_count
                assert not public.outputs if public.metadata.get('local_dataset_pixels') else public.outputs==local.outputs
        print('Final frozen coverage, native metrics, fitting/calibration isolation, score replay,5354 caches and notebook session provenance verified')
    print(json.dumps(dict(registry_records_verified=len(events),completed_training_artifacts_verified=len(results),
        executed_notebook_code_cells=len(code),notebook_bytes=(ROOT/'notebooks/04_supersimplenet.ipynb').stat().st_size,
        prior_notebooks_reports_partitions_unchanged=True,staged_paths=len(staged)),indent=2))

if __name__=='__main__':main()
