"""Score orientation: larger means more anomalous. No automatic threshold tuning."""
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)


def binary_metrics(labels, scores, threshold):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not np.isfinite(s).all():
        raise ValueError("Expected matching finite vectors")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("Both binary classes required")
    tn, fp, fn, tp = confusion_matrix(y, s > threshold, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return dict(auroc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)),
                recall=float(recall), fpr=float(fp/(fp+tn)), precision=float(precision),
                f1=float(2*precision*recall/(precision+recall)) if precision+recall else 0.0,
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def calibrate_normal(scores, target_fpr=.05):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite normal calibration vector required")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")
    return float(np.quantile(scores, 1-target_fpr, method="higher"))


def au_pro(masks, maps, max_fpr=.3):
    """Exact unique-score sweep, 8-connected regions, equal weight per region.

    Pixel FPR uses all background pixels including normal images. Linear
    interpolation clips the curve at max_fpr; area is normalized by max_fpr.
    For small fixtures only: pixel-sized vectors are costly on full datasets.
    A scalable evaluator must be cross-checked against this reference first.
    """
    masks, maps = list(masks), list(maps)
    if not masks or len(masks) != len(maps) or not 0 < max_fpr <= 1:
        raise ValueError("Matching nonempty lists and valid max_fpr required")
    truth, scores, weights, region_count = [], [], [], 0
    for mask, score in zip(masks, maps):
        mask, score = np.asarray(mask), np.asarray(score, dtype=float)
        if mask.ndim != 2 or mask.shape != score.shape or not np.isfinite(score).all():
            raise ValueError("Matching 2D mask and finite map required")
        if not set(np.unique(mask)).issubset({0, 1}):
            raise ValueError("Masks must be binary 0/1")
        components, count = label(mask, structure=np.ones((3, 3)))
        w = np.ones(mask.shape, dtype=float)
        for component in range(1, count+1):
            region = components == component
            w[region] = 1 / region.sum()
        region_count += count
        truth.extend(mask.ravel()); scores.extend(score.ravel()); weights.extend(w.ravel())
    if not region_count or np.all(truth):
        raise ValueError("At least one region and background pixel required")
    fpr, pro, _ = roc_curve(truth, scores, sample_weight=weights, drop_intermediate=False)
    below = fpr < max_fpr
    x = np.append(fpr[below], max_fpr)
    y = np.append(pro[below], np.interp(max_fpr, fpr, pro))
    return float(np.trapezoid(y, x) / max_fpr)
