"""One deep pass over all final identities; no repeated native metric sorting."""
from pathlib import Path
import json
import subprocess
import time
import hashlib
import nbformat
import numpy as np
import pandas as pd
import torch
from anomaly_detection.data import sha256,CATEGORIES
from anomaly_detection.efficientad import load_checkpoint
from anomaly_detection.efficientad_final import ROOT,REPORT,FREEZE,METRICS,freeze,identity_for,verify_complete,artifact,atomic_json,read_rows,source_identity
from anomaly_detection.metrics import calibrate_normal
from anomaly_detection.tracking import config_hash
from validate_notebook import validate as validate_notebook


def validate(require_notebook=True):
    start=time.perf_counter();torch.set_num_threads(4);frozen=freeze()
    assert sha256(ROOT/'requirements.lock.txt')==frozen['package_lock_sha256']
    baseline=json.loads((REPORT/'baseline_preservation.json').read_text())
    normalized_baselines=0
    for p,h in baseline['files'].items():
        if sha256(ROOT/p)!=h:
            expected=baseline.get('canonical_lf_sha256',{}).get(p)
            assert expected and hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,'Protected baseline changed: '+p
            normalized_baselines+=1
    manifest=json.loads((REPORT/'benchmark_manifest.json').read_text())
    assert len(manifest['models'])==45 and all(m['status']=='complete' for m in manifest['models'])
    assert {(m['category'],m['seed']) for m in manifest['models']}=={(c,s) for c in CATEGORIES for s in frozen['seeds']}
    table=pd.read_csv(ROOT/'reports/efficientad/all_runs.csv',float_precision='round_trip')
    assert len(table)==45 and not table[['category','seed']].duplicated().any()
    summary=json.loads((ROOT/'reports/efficientad/summary.json').read_text())
    assert summary['models']==45 and summary['frozen_configuration_sha256']==FREEZE
    for key in METRICS: assert np.isclose(table[key].mean(),summary['macro'][key]['mean'],atol=1e-12)
    teacher=torch.load(ROOT/frozen['teacher']['path'],map_location='cpu',weights_only=True)
    artifact_count=0;checkpoints=0;negative_maps=0;seen=set();moments={}
    for item in manifest['models']:
        r=verify_complete(ROOT/item['result'],identity_for(item['category'],item['seed']),deep=True)
        artifact_count+=len(r['artifacts'])
        model,meta=load_checkpoint(ROOT/r['checkpoint']['path'])
        assert meta['identity']==r['identity'] and meta['final'] and meta['trained_epochs']==9
        assert not any(p.requires_grad for p in model.teacher.parameters())
        assert all(torch.equal(value,teacher[key]) for key,value in model.teacher.state_dict().items())
        assert np.array_equal(model.map_quantiles.numpy(),np.array(r['training']['map_quantiles'],dtype=np.float32))
        moment_hash=hashlib.sha256(model.mean.numpy().tobytes()+model.std.numpy().tobytes()).hexdigest()
        if item['category'] in moments:assert moments[item['category']]==moment_hash,'Seed-independent teacher normalization changed'
        else:moments[item['category']]=moment_hash
        progress=Path(ROOT/r['checkpoint']['path']).with_name('progress.pt')
        state=torch.load(progress,map_location='cpu',weights_only=True)
        assert state['identity']==r['identity'] and len(state['history'])==9
        assert all(torch.equal(v,state['state_dict'][k]) for k,v in model.state_dict().items() if k!='map_quantiles')
        assert state['torch_rng'].dtype==torch.uint8 and state['cuda_rng'].dtype==torch.uint8 and state['optimizer']['state']
        del state,model;checkpoints+=1
        history=pd.read_csv(REPORT/'runs'/f"{r['experiment_id']}-history.csv",float_precision='round_trip')
        assert history.epoch.tolist()==list(range(1,10))
        assert history.steps.iloc[-1]==9*len(read_rows('fit',item['category']))
        assert np.isfinite(history.to_numpy()).all() and np.isclose(history.seconds.sum(),r['training_seconds'])
        assert history.penalty.eq(0).all()
        assert r['training']['fit_source_sha256']==source_identity(read_rows('fit',item['category']))
        assert r['training']['development_source_sha256']==source_identity(read_rows('development',item['category']))
        cal=r['calibration'];cal_rows=read_rows('calibration',item['category'])
        assert cal['source_sha256']==source_identity(cal_rows) and cal['paths']==cal_rows.path.tolist()
        assert cal['threshold']==calibrate_normal(cal['scores'],frozen['threshold_policy']['target_fpr'])
        assert cal['strict_greater'] and cal['quantile_method']=='higher'
        if cal['unique_calibration_scores']<len(cal['scores']):assert cal['future_fpr_order_statistic_95'] is None
        cal_table=pd.read_csv(REPORT/'runs'/f"{r['experiment_id']}-calibration-scores.csv",float_precision='round_trip')
        assert np.array_equal(cal_table.score.to_numpy(),np.asarray(cal['scores']))
        images=pd.read_csv(ROOT/r['images_csv'],float_precision='round_trip')
        raw=np.load(ROOT/r['raw_branch_maps'],mmap_mode='r');maps=np.load(ROOT/r['local_maps'],mmap_mode='r')
        assert raw.shape==(len(images),2,256,256) and maps.shape==(len(images),256,256)
        q=torch.tensor(r['training']['map_quantiles'])
        expected_scores=[]
        for i in range(0,len(images),16):
            x=torch.from_numpy(np.array(raw[i:i+16,0]))
            expected=(.1*(x-q[0,0])/(q[0,1]-q[0,0])).numpy()
            assert np.array_equal(expected,maps[i:i+16]),'Selected maps altered or clipped'
            assert np.isfinite(np.array(raw[i:i+16])).all()
            negative_maps+=int((expected<0).any())
            expected_scores.extend(expected.reshape(len(expected),-1).max(1).astype(np.float64))
        assert np.array_equal(np.array(expected_scores),images.score.to_numpy())
        del raw,maps
        point=table[table.experiment_id.eq(r['experiment_id'])].iloc[0]
        assert all(np.isclose(point[k],r[k],atol=1e-12) for k in METRICS)
        assert r['latency']['iterations']==200 and r['latency']['warmup']==50
        local=(ROOT/r['local_maps']).parent
        with np.load(local/'bootstrap_vectors.npz') as vectors:
            assert all(np.isfinite(vectors[k]).all() and vectors[k].shape==(2000,) for k in METRICS)
        event=json.loads((ROOT/f"experiments/{r['experiment_id']}-result.json").read_text())
        assert event['status']=='completed' and event['paths']['result_sha256']==sha256(ROOT/item['result'])
        assert event['hyperparameters']==r['identity']
        seen.add((item['category'],item['seed']))
        print('deep verified',r['experiment_id'],flush=True)
    events=list((ROOT/'experiments').glob('06-*.json'))
    for path in events:
        event=json.loads(path.read_text());identity=event.pop('config_sha256');event.pop('recorded_at')
        assert config_hash(event)==identity,path.name
    assert negative_maps>0
    comparison=pd.read_csv(ROOT/'reports/efficientad/model_macro_comparison.csv',index_col=0)
    for name,folder in [('Basic CNN','basic_cnn'),('Convolutional Autoencoder','autoencoder'),('PatchCore variant','patchcore'),('SuperSimpleNet Lite','supersimplenet'),('PaDiM','padim')]:
        previous=json.loads((ROOT/f'reports/{folder}/summary.json').read_text())
        assert all(np.isclose(comparison.loc[k,name],previous['macro'][k]['mean'],atol=1e-12) for k in METRICS)
    assert len(pd.read_csv(ROOT/'reports/efficientad/paired_category_difference_intervals.csv'))==45
    assert len(pd.read_csv(ROOT/'reports/efficientad/paired_image_difference_intervals.csv'))==30
    examples=pd.read_csv(ROOT/'reports/efficientad/failure_examples.csv',float_precision='round_trip')
    assert set(examples.category)==set(CATEGORIES)
    for example in examples.itertuples():
        assert sha256(ROOT/example.panel)==example.panel_sha256
        if isinstance(example.zoom_panel,str):assert sha256(ROOT/example.zoom_panel)==example.zoom_sha256
        run=table[table.category.eq(example.category)&table.seed.eq(example.seed)].iloc[0]
        saved=json.loads((REPORT/f'runs/{run.experiment_id}-result.json').read_text())
        source=pd.read_csv(ROOT/saved['images_csv'],float_precision='round_trip')
        image=source[source.path.eq(example.path)].iloc[0]
        assert image.outcome==example.outcome and image.score==example.score
    notebook=None
    if require_notebook:
        notebook=validate_notebook(ROOT/'notebooks/06_efficientad.ipynb',require_executed=True)
        previous=nbformat.reads(subprocess.check_output(['git','show','664e106dbdd25104ea4e3c3be192fc8be7a0b772:notebooks/06_efficientad.ipynb'],cwd=ROOT).decode(),4)
        current=nbformat.read(ROOT/'notebooks/06_efficientad.ipynb',4);by_id={c.id:c for c in current.cells}
        for cell in previous.cells:
            assert by_id[cell.id].source==cell.source
            if cell.cell_type=='code':assert by_id[cell.id].outputs==cell.outputs
        assert current.metadata['efficientad_final_replay']['models']==45
    result=dict(status='passed',categories=15,seeds=frozen['seeds'],final_models=45,checkpoints_verified=checkpoints,
        verified_run_artifacts=artifact_count,teacher_normalization_categories=len(moments),protected_baseline_files=len(baseline['files']),baseline_files_git_line_endings_only=normalized_baselines,registry_events=len(events),
        freeze_sha256=FREEZE,evaluator_sha256=sha256(REPORT/'evaluator_identity.json'),notebook=notebook,seconds=time.perf_counter()-start)
    return result


if __name__=='__main__':
    import sys
    result=validate('--without-notebook' not in sys.argv)
    print(json.dumps(result,indent=2))
    if '--without-notebook' not in sys.argv:atomic_json(REPORT/'validation.json',result)
