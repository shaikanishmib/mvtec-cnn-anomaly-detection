"""Deep development identities, selection/calibration order and coverage audit."""
from pathlib import Path
import json
import subprocess
import sys
import time
import nbformat
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from anomaly_detection.data import sha256
from anomaly_detection.tracking import config_hash
from anomaly_detection.efficientad import load_checkpoint
from anomaly_detection.efficientad_research import select_configuration,calibration_diagnostics
from anomaly_detection.efficientad_calibration import qualified_calibration,read_scores
from validate_notebook import validate as validate_notebook

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/efficientad/development'


def validate(require_notebook=True):
    started=time.perf_counter()
    assert Path(sys.executable).resolve()==ROOT/'.venv/Scripts/python.exe'
    plan=json.loads((ROOT/'configs/efficientad_development.json').read_text())
    plan_hash=sha256(ROOT/'configs/efficientad_development.json')
    events=[]
    for path in sorted((ROOT/'experiments').glob('06-*.json')):
        event=json.loads(path.read_text()); expected=event.pop('config_sha256'); event.pop('recorded_at')
        assert config_hash(event)==expected,path.name
        events.append(json.loads(path.read_text()))
    results=[json.loads(p.read_text()) for p in sorted(OUT.glob('06-dev-*-v1.json'))]
    assert len(results)==plan['fresh_fits']==4
    assert {(r['category'],r['epochs'],r['seed']) for r in results}=={(c,e,2026) for c in plan['categories'] for e in plan['epoch_budgets']}
    assert sum(r['steps'] for r in results)==plan['maximum_optimizer_steps']==4116
    partitions={name:pd.read_csv(ROOT/f'reports/partitions/seed2026-{name}.csv') for name in ('fit','development','calibration','test')}
    for part,table in partitions.items():
        assert table.partition.eq(part).all()
        if part!='test': assert table.split.eq('train').all() and table.path.str.contains('/train/good/').all()
        for other,rows in partitions.items():
            if part!=other: assert not set(table.path)&set(rows.path)
    teacher_hash=sha256(ROOT/'checkpoints/efficientad/assets/teacher_small.pth')
    assert teacher_hash==plan['teacher_sha256']
    teacher_state=torch.load(ROOT/'checkpoints/efficientad/assets/teacher_small.pth',weights_only=True,map_location='cpu')
    artifact_count=0
    for r in results:
        assert r['plan_sha256']==plan_hash and r['finite'] and r['steps']==r['epochs']*r['fit_n']
        assert len(r['curves'])==r['epochs']
        curves=pd.read_csv(OUT/f"{r['experiment_id']}-epochs.csv")
        assert np.allclose(curves.seconds.sum(),r['training_seconds'])
        assert np.isfinite(curves.to_numpy()).all()
        for artifact in r['artifacts']+[r['checkpoint']]:
            p=ROOT/artifact['path']; assert sha256(p)==artifact['sha256'] and p.stat().st_size==artifact['bytes']
            artifact_count+=1
        model,meta=load_checkpoint(ROOT/r['checkpoint']['path'])
        assert meta['teacher_sha256']==teacher_hash and meta['plan_sha256']==plan_hash
        assert meta['epochs']==r['epochs'] and meta['category']==r['category'] and not meta['final']
        assert not any(p.requires_grad for p in model.teacher.parameters())
        assert all(torch.equal(value,teacher_state[key]) for key,value in model.teacher.state_dict().items())
        assert np.allclose(model.map_quantiles.numpy(),r['map_quantiles'])
        del model
        category=r['category']
        dev=pd.read_csv(OUT/f"{r['experiment_id']}-development.csv")
        assert set(dev.path)==set(partitions['development'].query('category == @category').path)
        proxy=pd.read_csv(OUT/f"{r['experiment_id']}-proxies.csv")
        assert set(proxy.path).issubset(set(partitions['fit'].query('category == @category').path))
        assert proxy.groupby('edit').size().to_dict()==dict(normal=16,scratch=16,block=16,color_patch=16)
        for score,metrics in r['proxy'].items():
            assert np.isfinite(proxy[score]).all() and np.isfinite(dev[score]).all()
            assert np.isclose(roc_auc_score(proxy.label,proxy[score]),metrics['auroc'])
            assert np.isclose(average_precision_score(proxy.label,proxy[score]),metrics['ap'])
        event=next(e for e in events if e['experiment_id']==r['experiment_id']+'-result')
        assert event['status']=='completed' and event['paths']['checkpoint']['sha256']==r['checkpoint']['sha256']
        for source,identity in event['training']['source_sha256'].items(): assert sha256(ROOT/source)==identity
    selected=json.loads((OUT/'selection.json').read_text())
    recomputed=select_configuration(results,plan)
    assert selected['selected']==recomputed['selected'] and selected['candidates']==recomputed['candidates']
    assert selected['eligible']==recomputed['eligible'] and selected['threshold_policy']==plan['threshold_policy']
    selection_hash=sha256(OUT/'selection.json')
    for category in plan['categories']:
        diagnosis=json.loads((OUT/f'calibration-{category}.json').read_text())
        table_path=OUT/f'calibration-{category}-scores.csv'; table=pd.read_csv(table_path)
        assert sha256(table_path)==diagnosis['scores_sha256'] and diagnosis['selection_sha256']==selection_hash
        assert set(table.path)==set(partitions['calibration'].query('category == @category').path)
        run=next(r for r in results if r['category']==category and r['epochs']==selected['selected']['epochs'])
        assert diagnosis['checkpoint_sha256']==run['checkpoint']['sha256']
        dev=pd.read_csv(OUT/f"{run['experiment_id']}-development.csv")[selected['selected']['score']]
        computed=calibration_diagnostics(table.score.to_numpy(),dev.to_numpy(),plan['threshold_policy'])
        assert np.isclose(computed['threshold'],diagnosis['threshold'])
        assert computed['exceedances']==diagnosis['exceedances']
        start=next(e for e in events if e['experiment_id']==f'06-calibration-{category}-v1-start')
        assert start['hyperparameters']['selection_sha256']==selection_hash
        assert start['training']['plan_sha256']==plan_hash
        assert max(e['recorded_at'] for e in events if e['experiment_id'].startswith('06-dev-') and e['status']=='completed')<start['recorded_at']
        corrected=json.loads((OUT/f'calibration-{category}-corrected.json').read_text())
        exact=read_scores(OUT/f'{category}-calibration-exact-scores.csv')
        exact_dev=read_scores(OUT/f'{category}-development-exact-scores.csv')
        computed=qualified_calibration(exact.score.to_numpy(),exact_dev.score.to_numpy(),plan['threshold_policy'])
        assert corrected['threshold']==computed['threshold']==diagnosis['threshold']
        assert corrected['development_exceedance_fraction']==computed['development_exceedance_fraction']
        assert corrected['unique_calibration_scores']==computed['unique_calibration_scores']
        assert corrected['future_fpr_order_statistic_95']==computed['future_fpr_order_statistic_95']
        assert set(exact.path)==set(table.path)
    latency=json.loads((OUT/'latency.json').read_text())
    assert len(latency['samples'])==300 and len(latency['end_to_end_samples_ms'])==100
    for key,values in [('resident_wall',[x['wall_ms'] for x in latency['samples']]),('resident_gpu_event',[x['gpu_ms'] for x in latency['samples']])]:
        assert np.isclose(latency[key]['p50_ms'],np.quantile(values,.5))
        assert np.isclose(latency[key]['p95_ms'],np.quantile(values,.95))
    gate=json.loads((OUT/'feasibility.json').read_text())
    assert gate['models']==45 and gate['categories']==15 and gate['seeds']==[2026,2027,2028]
    assert not gate['final_benchmark_started']
    total=sum(x['conservative_three_seed_seconds'] for x in gate['workload'])/3600
    assert np.isclose(total,gate['estimated_conservative_total_elapsed_hours'])
    assert gate['recommendation']==('GO' if selected['eligible'] and total<=12 else 'NO-GO')
    assert len(gate['workload'])==15
    frozen_path=ROOT/'configs/efficientad_reduced_frozen.json'
    frozen=json.loads(frozen_path.read_text())
    assert frozen['epochs']==selected['selected']['epochs'] and frozen['score']==selected['selected']['score']
    assert frozen['threshold_policy']==plan['threshold_policy']
    assert frozen['seeds']==gate['seeds'] and len(frozen['categories'])==15
    assert frozen['inference_execution']=='eager; graph diagnostic not adopted'
    reconciliation=json.loads((OUT/'reconciliation.json').read_text())
    assert reconciliation['frozen_sha256']==sha256(frozen_path) and reconciliation['final_models']==0
    manifest=json.loads((OUT/'artifact_manifest.json').read_text())
    for item in manifest:
        artifact=ROOT/item['path']
        assert sha256(artifact)==item['sha256'] and artifact.stat().st_size==item['bytes']
    control=json.loads((OUT/'launch-control.json').read_text())
    if control['graph_status']=='measured':
        assert max(control['graph_max_abs_error'])<=1e-6
        assert len(control['graph']['samples'])==300
    assert len(control['eager']['samples'])==300
    notebook=None
    if require_notebook:
        path=ROOT/'notebooks/06_efficientad.ipynb'
        notebook=validate_notebook(path,require_executed=True)
        # Each original pilot cell is retained with identical source and outputs.
        baseline=nbformat.reads(subprocess.check_output(['git','show','ee19ec85e914bd0b8cda8c6fb1c0c8768c1557a1:notebooks/06_efficientad.ipynb'],cwd=ROOT).decode(),4)
        current=nbformat.read(path,4); current_by_id={c.id:c for c in current.cells}
        for cell in baseline.cells:
            assert cell.id in current_by_id
            other=current_by_id[cell.id]
            assert cell.source==other.source
            if cell.cell_type=='code': assert cell.outputs==other.outputs and cell.execution_count==other.execution_count
    return dict(status='passed',development_fits=4,optimizer_steps=4116,final_models=0,verified_fit_artifacts=artifact_count,
        registry_events=len(events),verified_manifest_artifacts=len(manifest),plan_sha256=plan_hash,selection_sha256=selection_hash,gate=gate['recommendation'],
        notebook=notebook,validator_seconds=time.perf_counter()-started)


if __name__=='__main__':
    result=validate('--without-notebook' not in sys.argv)
    print(json.dumps(result,indent=2))
    if '--without-notebook' not in sys.argv:
        (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
