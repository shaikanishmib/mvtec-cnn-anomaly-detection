from pathlib import Path
import numpy as np
import pytest
import torch
from anomaly_detection.patchcore import (PatchExtractor,align_features,patch_embeddings,projected_coreset,
    nearest_neighbors,pixel_maps,image_scores,save_bank,load_bank)


def test_feature_shapes_freezing_and_determinism():
    torch.set_num_threads(4)
    model=PatchExtractor(pretrained=False)
    model.train()
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    x=torch.rand(1,3,256,256)
    a,b,c=model.feature_maps(x)
    assert [tuple(t.shape) for t in (a,b,c)]==[(1,64,64,64),(1,128,32,32),(1,256,16,16)]
    first=model(x);second=model(x)
    assert first.shape==(1,384,32,32) and torch.equal(first,second)
    assert patch_embeddings(first).shape==(1024,384)


def test_alignment_and_maps():
    a=torch.ones(2,2,8,8);b=torch.ones(2,3,4,4)*2
    aligned=align_features(a,b)
    assert aligned.shape==(2,5,8,8)
    assert torch.allclose(aligned[:,:2,3,3],torch.ones(2,2))
    assert torch.allclose(aligned[:,2:,3,3],torch.ones(2,3)*2)
    maps=pixel_maps(torch.ones(2,4,4)*3,(31,45))
    assert maps.shape==(2,31,45) and torch.allclose(maps,torch.ones_like(maps)*3)


def test_coreset_unique_reproducible_and_checkpoint(tmp_path):
    x=torch.randn(100,16,generator=torch.Generator().manual_seed(1))
    first,_=projected_coreset(x,12,2026,candidate_limit=80)
    second,_=projected_coreset(x,12,2026,candidate_limit=80)
    assert torch.equal(first,second) and first.unique().numel()==12
    # Identical embeddings must still yield unique source indices.
    identical,_=projected_coreset(torch.zeros(20,4),10)
    assert identical.unique().numel()==10
    path=tmp_path/'bank.pt'
    save_bank(path,x[first],first,'bottle',{'seed':2026},{'partition':'fit'})
    bank,_=load_bank(path,'bottle')
    assert torch.equal(bank,x[first])
    with pytest.raises(ValueError):load_bank(path,'cable')
    with pytest.raises(FileExistsError):save_bank(path,bank,first,'bottle',{}, {})


@pytest.mark.parametrize('chunks',[(1,1),(7,11),(100,100)])
def test_nearest_distances_against_float64_reference(chunks):
    g=torch.Generator().manual_seed(3)
    bank=torch.randn(37,16,generator=g)
    queries=torch.cat([torch.randn(13,16,generator=g),bank[:3]])
    actual,indices=nearest_neighbors(queries,bank,*chunks)
    expected=torch.cdist(queries.double(),bank.double(),compute_mode='donot_use_mm_for_euclid_dist')
    torch.testing.assert_close(actual.double(),expected.min(1).values,rtol=1e-6,atol=1e-6)
    assert torch.equal(indices,expected.argmin(1))
    assert actual[-3:].eq(0).all()


def test_aggregation_and_calibration_isolation():
    from anomaly_detection.metrics import calibrate_normal,binary_metrics
    maps=torch.zeros(2,10,10);maps[0,0,0]=5;maps[1,:2,:2]=3
    assert image_scores(maps).tolist()==[5,3]
    assert image_scores(maps,'top1pct').tolist()==[5,3]
    threshold=calibrate_normal(np.arange(20))
    result=binary_metrics([0,0,1,1],[18,19,20,21],threshold)
    assert result['confusion_matrix']==[[2,0],[0,2]]
