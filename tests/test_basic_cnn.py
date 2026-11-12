from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
from anomaly_detection.basic_cnn import (BasicCNN, seed_everything, synthetic_batch,
                                        segmentation_loss, image_scores)
from anomaly_detection.cnn_research import require_fitting, inner_fit_split
from anomaly_detection.metrics import calibrate_normal
from anomaly_detection.tracking import config_hash


@pytest.mark.parametrize("strategy", ["patch", "perturb", "mixed"])
def test_exact_synthetic_masks_and_unchanged_borders(strategy):
    images = torch.rand(8, 3, 64, 64, generator=torch.Generator().manual_seed(21))
    changed, mask = synthetic_batch(images, torch.Generator().manual_seed(11), strategy, 0.)
    assert torch.equal(mask.bool(), changed.ne(images).any(1, keepdim=True))
    assert mask.sum() > 0 and mask.sum() < mask.numel()*.2
    assert changed.min() >= 0 and changed.max() <= 1
    assert torch.equal(images[:, :, :2], changed[:, :, :2])
    assert torch.equal(images[:, :, -2:], changed[:, :, -2:])
    assert torch.equal(images[:, :, :, :2], changed[:, :, :, :2])
    assert torch.equal(images[:, :, :, -2:], changed[:, :, :, -2:])
    repeated, repeated_mask = synthetic_batch(images, torch.Generator().manual_seed(11), strategy, 0.)
    assert torch.equal(changed, repeated) and torch.equal(mask, repeated_mask)


def test_normal_generator_and_grayscale_no_shortcut():
    images = torch.rand(4, 1, 32, 32).repeat(1, 3, 1, 1)
    normal, mask = synthetic_batch(images, torch.Generator().manual_seed(9), normal_probability=1.)
    assert torch.equal(normal, images) and mask.sum() == 0
    changed, _ = synthetic_batch(images, torch.Generator().manual_seed(9), "perturb", 0.)
    assert torch.equal(changed[:, 0], changed[:, 1]) and torch.equal(changed[:, 0], changed[:, 2])


@pytest.mark.parametrize("width,size", [(8,32), (16,64), (8,96)])
def test_cnn_shapes_gradients_and_scores(width, size):
    model = BasicCNN(width)
    images = torch.rand(2, 3, size, size, requires_grad=True)
    logits = model(images)
    assert logits.shape == (2, 1, size, size)
    mask = torch.zeros_like(logits); mask[0, :, 4:12, 6:14] = 1
    loss = segmentation_loss(logits, mask, 8., 1.)
    loss.backward()
    assert torch.isfinite(images.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(image_scores(logits), logits.sigmoid().flatten(1).amax(1))


def test_deterministic_initialization_and_checkpoint_roundtrip(tmp_path):
    seed_everything(2026); model = BasicCNN(8)
    seed_everything(2026); other = BasicCNN(8)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(), other.parameters()))
    target = tmp_path / "fixture.pt"
    torch.save({"state_dict": model.state_dict()}, target)
    other.load_state_dict(torch.load(target, weights_only=True)["state_dict"])
    inputs = torch.rand(1,3,32,32)
    torch.testing.assert_close(model(inputs), other(inputs), rtol=0, atol=0)


def test_inner_fit_isolation_and_no_annotation_sources():
    rows = pd.DataFrame({"path": [f"bottle/train/good/{i}.png" for i in range(20)],
                         "category": "bottle", "partition": "fit", "split": "train", "mask": np.nan})
    training, probe = inner_fit_split(rows)
    assert len(training) == 16 and len(probe) == 4
    assert not set(training.path) & set(probe.path)
    again, _ = inner_fit_split(rows)
    pd.testing.assert_frame_equal(training, again)
    for partition in ("test", "development", "calibration"):
        bad = rows.assign(partition=partition)
        with pytest.raises(ValueError): require_fitting(bad)
    with pytest.raises(ValueError): require_fitting(rows.assign(mask="real_mask.png"))


def test_calibration_interface_accepts_only_scores_and_config_identity():
    # No labels or test masks enter this API; exact high-order statistic is fixed.
    assert calibrate_normal([.1, .2, .3, .4]) == .4
    with pytest.raises(TypeError): calibrate_normal([.1, .2], test_labels=[0, 1])
    a = {"width":8, "seed":2026}
    assert config_hash(a) == config_hash(dict(reversed(list(a.items()))))
    assert config_hash(a) != config_hash(dict(a, seed=2027))
