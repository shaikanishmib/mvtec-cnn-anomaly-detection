import numpy as np
import pytest
from anomaly_detection.metrics import binary_metrics, calibrate_normal, au_pro


def test_known_rank_and_confusion():
    result = binary_metrics([0, 0, 1, 1], [.1, .4, .35, .8], .5)
    assert result["auroc"] == .75
    assert result["ap"] == pytest.approx(5/6)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["precision"] == 1
    assert result["recall"] == .5
    assert result["f1"] == pytest.approx(2/3)
    assert result["fpr"] == 0


def test_ties_and_calibration():
    result = binary_metrics([0, 1], [.5, .5], .5)
    assert result["auroc"] == .5
    assert result["ap"] == .5
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]
    assert result["precision"] == 0
    assert calibrate_normal([1, 2, 3, 4]) == 4
    assert calibrate_normal([1, 2, 3, 4], .5) == 3


def test_aupro_perfect_reversed_and_tied():
    mask = np.array([[1, 0], [0, 0]])
    assert au_pro([mask], [mask.astype(float)]) == pytest.approx(1)
    assert au_pro([mask], [1-mask.astype(float)]) == pytest.approx(0)
    assert au_pro([mask], [np.zeros((2, 2))]) == pytest.approx(.15)


def test_aupro_equal_region_weight_not_pixel_weight():
    mask = np.array([[1, 0, 1, 1], [0, 0, 1, 1]])
    score = np.zeros((2, 4)); score[0, 0] = 1
    # One of two regions found at FPR=0; remaining region tied with background.
    assert au_pro([mask], [score]) == pytest.approx(.575)


def test_aupro_normal_images_contribute_background():
    mask = np.array([[1, 0]])
    assert au_pro([mask, np.zeros((1, 2), int)],
                  [np.array([[.8, .1]]), np.array([[.9, .1]])]) == pytest.approx(0)


@pytest.mark.parametrize("labels,scores", [([0, 0], [1, 2]), ([0, 2], [1, 2]), ([0, 1], [1, np.nan])])
def test_invalid_metrics(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores, .5)


def test_invalid_calibration_and_region_masks():
    with pytest.raises(ValueError): calibrate_normal([])
    with pytest.raises(ValueError): calibrate_normal([1], 1)
    with pytest.raises(ValueError): au_pro([np.zeros((2, 2))], [np.zeros((2, 2))])
