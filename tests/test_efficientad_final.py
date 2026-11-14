"""Final evaluation fixtures; no official test pixels or model fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from anomaly_detection import efficientad_final as ef
from anomaly_detection.data import sha256


def test_signed_histogram_monotonic_and_finite():
    x=np.array([-1e20,-100.,-1.,0.,1.,100.,1e20])
    bins=ef.histogram_indices(x,0.,1.)
    assert (np.diff(bins)>=0).all() and bins.min()>=0 and bins.max()<4096
    assert bins[2]<bins[3]<bins[4]


def test_csv_preserves_ties_and_strict_threshold(tmp_path):
    value=np.float32(.5066908597946167)
    table=pd.DataFrame(dict(score=np.array([value,value],np.float32),predicted=[False,False]))
    restored=ef.csv_exact(tmp_path/'scores.csv',table)
    assert np.array_equal(restored.score.to_numpy(),np.array([value,value],np.float64))
    assert not (restored.score>float(value)).any()


def test_artifact_corruption_detected(tmp_path,monkeypatch):
    monkeypatch.setattr(ef,'ROOT',tmp_path)
    p=tmp_path/'artifact.json';ef.atomic_json(p,dict(state='valid'))
    identity=ef.artifact(p);ef.check_artifact(identity)
    p.write_text('corrupt')
    with pytest.raises(AssertionError): ef.check_artifact(identity)


def test_native_metrics_signed_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(ef,'ROOT',tmp_path)
    local=tmp_path/'local';local.mkdir()
    (tmp_path/'outputs/eda').mkdir(parents=True)
    mask_path=tmp_path/'data/mvtec_ad/bottle/ground_truth/defect/mask.png';mask_path.parent.mkdir(parents=True)
    truth=np.zeros((8,8),np.uint8);truth[2:4,2:4]=255;Image.fromarray(truth).save(mask_path)
    paths=['bottle/test/good/normal.png','bottle/test/defect/anomaly.png']
    pd.DataFrame(dict(path=paths,height=[8,8],width=[8,8])).to_csv(tmp_path/'outputs/eda/inventory.csv',index=False)
    rows=pd.DataFrame(dict(path=paths,mask=[np.nan,'bottle/ground_truth/defect/mask.png'],mask_sha256=[np.nan,sha256(mask_path)]))
    maps=np.full((2,8,8),-.5,np.float32);maps[1,2:4,2:4]=.7
    def bootstrap(*args):return {},{'fixture':np.array([1.])}
    monkeypatch.setattr(ef,'bootstrap_evaluation',bootstrap)
    result,details=ef.native_evaluation(rows,maps,np.array([-.5,.7]),dict(threshold=0.,pixel_threshold=0.,histogram_center=0.,histogram_scale=1.),'bottle',local)
    assert result['pixel_auroc']==1 and result['pixel_ap']==1 and result['au_pro']==1
    assert result['positive_pixels']==4 and details[1]['pixel_detected_defect_pixels']==4
    assert not list(local.glob('native-*.dat'))


def test_frozen_identity_unchanged():
    f=ef.freeze()
    assert f['epochs']==9 and f['score']=='local_max' and f['seeds']==[2026,2027,2028]


def recovery_module():
    import importlib.util
    path=ef.ROOT/'scripts/resume_efficientad_final.py'
    spec=importlib.util.spec_from_file_location('efficientad_recovery_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_live_manager_prevents_duplicate_launch(tmp_path,monkeypatch):
    module=recovery_module();monkeypatch.setattr(module,'REPORT',tmp_path)
    monkeypatch.setattr(module,'active',lambda:[dict(pid=123,worker=False,manager=True)])
    def forbidden(*args,**kwargs):raise AssertionError('Duplicate launch')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    module.main()


def test_orphan_worker_finishes_before_new_manager(tmp_path,monkeypatch):
    module=recovery_module();monkeypatch.setattr(module,'REPORT',tmp_path)
    states=iter([[dict(pid=123,worker=True,manager=False)],[]]);calls=[];waited=[]
    monkeypatch.setattr(module,'active',lambda:next(states))
    monkeypatch.setattr(module.time,'sleep',lambda seconds:waited.append(seconds))
    monkeypatch.setattr(module.subprocess,'run',lambda command,**kwargs:calls.append(command))
    module.main()
    assert waited==[30] and len(calls)==2
    assert calls[0][-1]=='--all' and 'reconcile_efficientad_registry.py' in calls[1][1]
