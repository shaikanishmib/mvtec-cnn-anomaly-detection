import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from anomaly_detection.padim import SpatialGaussian, PaDiMExtractor, aligned_embedding, anomaly_maps


def test_spatial_covariance_and_distances_against_float64_reference(tmp_path):
    x = torch.randn(17, 4, 2, 3, generator=torch.Generator().manual_seed(19))
    q = torch.randn(3, 4, 2, 3, generator=torch.Generator().manual_seed(20))
    model = SpatialGaussian(.01, block=2).fit(x)
    actual = model.score(q)
    for i in range(2):
        for j in range(3):
            values = x[:, :, i, j].double()
            covariance = torch.cov(values.T) + .01*torch.eye(4, dtype=torch.float64)
            delta = q[:, :, i, j].double() - values.mean(0)
            expected = (delta * torch.linalg.solve(covariance, delta.T).T).sum(1).sqrt()
            torch.testing.assert_close(actual[:, i, j].double(), expected, atol=2e-5, rtol=2e-5)
    path = tmp_path/'gaussian.pt'
    model.save(path, 'bottle', {'seed': 19}, {'partition': 'fit'})
    restored, _ = SpatialGaussian.load(path, 'bottle')
    torch.testing.assert_close(actual, restored.score(q))
    with pytest.raises(ValueError): SpatialGaussian.load(path, 'cable')
    with pytest.raises(FileExistsError): model.save(path, 'bottle', {}, {})


def test_singular_and_invalid_inputs():
    x = torch.ones(3, 5, 2, 2)
    model = SpatialGaussian().fit(x)
    assert torch.equal(model.score(x), torch.zeros(3, 2, 2))
    torch.testing.assert_close(model.score(x+1), torch.full((3, 2, 2), np.sqrt(500), dtype=torch.float32))
    with pytest.raises(ValueError): SpatialGaussian().fit(x[:1])
    with pytest.raises(ValueError): model.score(x[:, :2])
    with pytest.raises(ValueError): model.score(x*float('nan'))
    with pytest.raises(RuntimeError): SpatialGaussian().score(x)


def test_alignment_selection_and_frozen_backbone():
    maps = (torch.zeros(1, 64, 8, 8), torch.ones(1, 128, 4, 4), torch.full((1, 256, 2, 2), 2.))
    aligned = aligned_embedding(maps)
    assert aligned.shape == (1, 448, 8, 8)
    assert (aligned[:, 64:192] == 1).all() and (aligned[:, 192:] == 2).all()
    a = PaDiMExtractor(100, 2026, pretrained=False)
    b = PaDiMExtractor(100, 2026, pretrained=False)
    assert torch.equal(a.indices, b.indices) and len(a.indices.unique()) == 100
    a.train()
    assert not a.training and not any(p.requires_grad for p in a.parameters())
    assert not a.layer1[0].bn1.training


def test_gaussian_map_matches_independent_filter():
    distances = torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(2)).abs()
    raw = anomaly_maps(distances, 0).numpy()
    expected = np.stack([gaussian_filter(x, 4., mode='mirror') for x in raw])
    np.testing.assert_allclose(anomaly_maps(distances).numpy(), expected, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_matches_cpu_with_multiple_rhs_and_spatial_blocks():
    x=torch.randn(13,7,3,4,generator=torch.Generator().manual_seed(8))
    q=torch.randn(5,7,3,4,generator=torch.Generator().manual_seed(9))
    cpu=SpatialGaussian(block=3).fit(x).score(q)
    gpu=SpatialGaussian(block=5).fit(x,'cuda').score(q.cuda()).cpu()
    torch.testing.assert_close(cpu,gpu,atol=2e-5,rtol=2e-5)


def test_native_evaluator_on_small_independent_fixture(tmp_path,monkeypatch):
    import pandas as pd
    from PIL import Image
    from sklearn.metrics import roc_auc_score,average_precision_score
    import anomaly_detection.padim_evaluation as evaluation
    from anomaly_detection.data import sha256
    from anomaly_detection.metrics import au_pro
    root=tmp_path
    (root/'outputs/eda').mkdir(parents=True)
    local=root/'outputs/padim/fixture';local.mkdir(parents=True)
    mask=np.zeros((5,7),np.uint8);mask[1:3,2:4]=1
    Image.fromarray(mask*255).save(root/'mask.png')
    rows=pd.DataFrame(dict(path=['normal','defect'],mask=[None,'mask.png'],mask_sha256=[None,sha256(root/'mask.png')]))
    pd.DataFrame(dict(path=rows.path,height=[5,5],width=[7,7])).to_csv(root/'outputs/eda/inventory.csv',index=False)
    maps=np.random.default_rng(2).random((2,256,256),dtype=np.float32)
    monkeypatch.setattr(evaluation,'ROOT',root)
    monkeypatch.setattr(evaluation,'autodiscover_data_root',lambda _:root)
    result,_=evaluation.native_evaluation(rows,maps,np.array([.2,.8]),dict(histogram_scale=.5,pixel_threshold=.7,image_threshold=.5),'bottle',local)
    native=torch.nn.functional.interpolate(torch.from_numpy(maps[:,None]),size=(5,7),mode='bilinear',align_corners=False).numpy()[:,0]
    truth=np.stack([np.zeros_like(mask),mask])
    assert result['pixel_auroc']==pytest.approx(roc_auc_score(truth.ravel(),native.ravel()))
    assert result['pixel_ap']==pytest.approx(average_precision_score(truth.ravel(),native.ravel()))
    assert result['au_pro']==pytest.approx(au_pro(truth,native))
    assert (local/'bootstrap_vectors.npz').is_file()
    assert not list(local.glob('native-*.dat'))
