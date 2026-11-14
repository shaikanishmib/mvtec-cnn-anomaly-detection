"""Focused core tests; random teacher is a shape fixture, never a trained model."""
import pytest
import torch
from anomaly_detection.efficientad import EfficientAD,hard_feature_loss,save_checkpoint,load_checkpoint


def test_hard_feature_loss():
    d=torch.tensor([1.,2.,3.,4.],requires_grad=True)
    assert hard_feature_loss(d,.5)==3.5
    assert hard_feature_loss(d,0)==2.5
    hard_feature_loss(d).backward()
    assert torch.isfinite(d.grad).all()
    with pytest.raises(ValueError): hard_feature_loss(d,2)


def test_shapes_normalization_gradients_maps_checkpoint(tmp_path):
    torch.set_num_threads(2); torch.manual_seed(10)
    m=EfficientAD().eval(); x=torch.rand(1,3,256,256)
    m.fit_feature_normalization([x])
    with torch.no_grad():
        t=m.teacher_features(x)
        assert t.shape==(1,384,64,64)
        assert m.student(x).shape==(1,768,64,64)
        assert m.autoencoder(x).shape==t.shape
        assert torch.allclose(t.mean((0,2,3)),torch.zeros(384),atol=1e-4)
        assert torch.allclose(t.var((0,2,3),unbiased=False),torch.ones(384),atol=1e-3)
    losses=m.losses(x,x,external=x)
    losses['total'].backward()
    assert all(p.grad is None for p in m.teacher.parameters())
    for branch in (m.student,m.autoencoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    m.fit_map_normalization([x,x*.9])
    with torch.no_grad():
        maps,score=m(x)
        assert maps.shape==(1,1,256,256)
        assert torch.equal(score,maps.flatten(1).max(1).values)
        assert torch.equal(maps,m(x)[0])
        local,global_map=m.raw_maps(x)
        expected=sum(.1*(a-q[0])/(q[1]-q[0]) for a,q in zip((local,global_map),m.map_quantiles))*.5
        assert torch.allclose(maps,expected)
    p=tmp_path/'pilot.pt'; save_checkpoint(p,m,{'fixture':True})
    other,metadata=load_checkpoint(p)
    assert metadata['fixture']
    with torch.no_grad(): assert torch.equal(maps,other(x)[0])
    m.train()
    with pytest.raises(ValueError): m.fit_map_normalization([x])


def test_pilot_split_isolation():
    from pathlib import Path
    import pandas as pd
    root=Path(__file__).resolve().parents[1]
    tables={name:pd.read_csv(root/f'reports/partitions/seed2026-{name}.csv') for name in ('fit','development','calibration','test')}
    for a,table in tables.items():
        assert table.partition.eq(a).all()
        for b,other in tables.items():
            if a!=b: assert not set(table.path)&set(other.path)
    assert tables['calibration'].split.eq('train').all()
