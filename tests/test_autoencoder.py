import io
import numpy as np
import pytest
import torch
from anomaly_detection.autoencoder import ConvAutoencoder, reconstruction_loss, anomaly_map, image_scores
from anomaly_detection.basic_cnn import seed_everything
from anomaly_detection.cnn_research import read_partition, require_fitting
from anomaly_detection.metrics import calibrate_normal, binary_metrics


@pytest.mark.parametrize('depth', [3, 4])
def test_shape_backward_and_checkpoint(depth):
    model = ConvAutoencoder(width=4, depth=depth, latent_channels=2)
    x = torch.rand(2,3,32,48)
    y = model(x)
    assert y.shape == x.shape
    assert ((y >= 0) & (y <= 1)).all()
    reconstruction_loss(y,x).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
    saved = io.BytesIO()
    torch.save(model.state_dict(),saved)
    saved.seek(0)
    restored = ConvAutoencoder(width=4,depth=depth,latent_channels=2)
    restored.load_state_dict(torch.load(saved,weights_only=True))
    assert torch.equal(y,restored(x))


def test_loss_maps_and_score():
    x = torch.zeros(1,3,8,8)
    y = x.clone()
    y[0,:,0,0] = torch.tensor([1.,.5,0.])
    assert reconstruction_loss(y,x,'l1').item() == pytest.approx(1.5/192)
    assert reconstruction_loss(y,x,'mse').item() == pytest.approx(1.25/192)
    a = anomaly_map(x,y,'absolute')
    s = anomaly_map(x,y,'squared')
    assert a[0,0,0] == .5
    assert s[0,0,0].item() == pytest.approx(1.25/3)
    assert image_scores(a,.02).item() == .25  # ceil(64*.02)=2
    assert not anomaly_map(x,x).any()
    with pytest.raises(ValueError): reconstruction_loss(y,x,'combined')
    with pytest.raises(ValueError): ConvAutoencoder()(torch.zeros(1,3,31,32))


def test_normal_only_and_category_isolation():
    a = read_partition('fit','bottle')
    b = read_partition('fit','carpet')
    require_fitting(a)
    assert set(a.path).isdisjoint(b.path)
    assert a.category.eq('bottle').all()
    assert set(a.path).isdisjoint(read_partition('development','bottle').path)
    with pytest.raises(ValueError): require_fitting(read_partition('calibration','bottle'))
    bad = a.copy()
    bad.loc[0,'split'] = 'test'
    with pytest.raises(ValueError): require_fitting(bad)


def test_reproducibility_and_shared_calibration_metrics():
    seed_everything(2026)
    first = ConvAutoencoder(width=4)(torch.zeros(1,3,32,32))
    seed_everything(2026)
    second = ConvAutoencoder(width=4)(torch.zeros(1,3,32,32))
    assert torch.equal(first,second)
    threshold = calibrate_normal(np.arange(20), .05)
    assert threshold == 19
    result = binary_metrics([0,0,1,1], [18,19,20,21], threshold)
    assert result['confusion_matrix'] == [[2,0],[0,2]]
    assert result['auroc'] == result['ap'] == result['f1'] == 1.


def test_calibration_uses_only_normal_partition_and_raw_residual(monkeypatch):
    import pandas as pd
    from anomaly_detection import ae_evaluation as evaluation
    requests=[]
    def partition(name,category):
        requests.append((name,category))
        assert name=='calibration'
        return pd.DataFrame(dict(path=['bottle/train/good/a.png','bottle/train/good/b.png'],sha256=['a','b']))
    monkeypatch.setattr(evaluation,'read_partition',partition)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    images=torch.stack([torch.full((3,8,8),51,dtype=torch.uint8),torch.full((3,8,8),102,dtype=torch.uint8)])
    monkeypatch.setattr(evaluation,'load_resized',lambda rows,resolution:images)
    monkeypatch.setattr(evaluation,'reconstruct',lambda model,images:torch.zeros_like(images,dtype=torch.float32))
    model=ConvAutoencoder(width=4)
    model.map_kind='squared'
    result=evaluation.calibrate_model(model,'bottle',8)
    assert requests==[('calibration','bottle')]
    assert result['images']==2
    assert result['image_threshold']==pytest.approx(.16)
    assert result['pixel_threshold']==pytest.approx(np.ceil(.16*65536)/65536)
    assert result['calibration_scores']==pytest.approx([.04,.16])


def test_training_step_reproducibility():
    def step(seed):
        seed_everything(seed)
        model=ConvAutoencoder(width=4)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        batch=torch.rand(2,3,32,32)
        loss=reconstruction_loss(model(batch),batch,'mse')
        loss.backward();optimizer.step()
        return loss.detach(),{k:v.clone() for k,v in model.state_dict().items()}
    a,weights_a=step(2026)
    b,weights_b=step(2026)
    assert torch.equal(a,b)
    assert all(torch.equal(weights_a[k],weights_b[k]) for k in weights_a)
