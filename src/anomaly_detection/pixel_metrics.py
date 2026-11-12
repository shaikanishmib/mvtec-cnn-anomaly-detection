"""Exact, memory-bounded pixel ranking metrics with one sort per category.

Disk-backed native-resolution vectors are accepted. Sorted indices occupy O(N)
RAM; cumulative curves are processed in chunks, with exact cross-chunk ties.
No pixel downsampling or score histogram is used in primary point estimates.
"""
import numpy as np


def exact_pixel_metrics(scores, labels, region_weights, region_count, chunk_size=1_000_000, max_fpr=.3):
    scores, labels, region_weights = (np.asarray(v) for v in (scores, labels, region_weights))
    if scores.ndim != 1 or labels.shape != scores.shape or region_weights.shape != scores.shape:
        raise ValueError("Matching flat score, binary truth and region-weight vectors required")
    positives = int(labels.sum(dtype=np.int64))
    negatives = len(labels)-positives
    if not positives or not negatives or not region_count:
        raise ValueError("Both pixel classes and at least one defect region required")
    if not 0 < max_fpr <= 1 or chunk_size < 1:
        raise ValueError("Invalid integration limit/chunk size")
    # Finite values and binary truth checked in bounded chunks before sorting.
    for start in range(0, len(scores), chunk_size):
        end = start+chunk_size
        if not np.isfinite(scores[start:end]).all() or not np.isin(labels[start:end], [0,1]).all():
            raise ValueError("Invalid pixel scores/labels")
    order = np.argsort(scores, kind="stable")[::-1]
    state = dict(tp=0, fp=0, regions=0., auroc=0., ap=0., aupro=0.)

    def consume(pos, neg, region):
        if not len(pos): return
        tp = state["tp"] + np.cumsum(pos, dtype=np.int64)
        fp = state["fp"] + np.cumsum(neg, dtype=np.int64)
        rg = state["regions"] + np.cumsum(region, dtype=np.float64)
        x = np.r_[state["fp"]/negatives, fp/negatives]
        y = np.r_[state["tp"]/positives, tp/positives]
        state["auroc"] += float(np.trapezoid(y, x))
        state["ap"] += float(np.sum(pos/positives * tp/(tp+fp)))
        pro = np.r_[state["regions"]/region_count, rg/region_count]
        if x[0] < max_fpr:
            keep = x < max_fpr
            clipped_x, clipped_pro = x[keep], pro[keep]
            if x[-1] >= max_fpr:
                clipped_x = np.r_[clipped_x, max_fpr]
                clipped_pro = np.r_[clipped_pro, np.interp(max_fpr, x, pro)]
            state["aupro"] += float(np.trapezoid(clipped_pro, clipped_x))/max_fpr
        state.update(tp=int(tp[-1]), fp=int(fp[-1]), regions=float(rg[-1]))

    pending = None
    for start in range(0, len(order), chunk_size):
        indices = order[start:start+chunk_size]
        values = scores[indices]
        boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
        pos = np.add.reduceat(labels[indices], boundaries, dtype=np.int64)
        count = np.diff(np.r_[boundaries, len(indices)])
        neg = count-pos
        region = np.add.reduceat(region_weights[indices], boundaries, dtype=np.float64)
        if pending is not None:
            if pending[0] == values[0]:
                pos[0] += pending[1]; neg[0] += pending[2]; region[0] += pending[3]
            else:
                consume(np.array([pending[1]]), np.array([pending[2]]), np.array([pending[3]]))
        consume(pos[:-1], neg[:-1], region[:-1])
        pending = (values[boundaries[-1]], int(pos[-1]), int(neg[-1]), float(region[-1]))
    if pending is not None:
        consume(np.array([pending[1]]), np.array([pending[2]]), np.array([pending[3]]))
    return {"pixel_auroc": state["auroc"], "pixel_ap": state["ap"], "au_pro": state["aupro"],
            "positive_pixels": positives, "negative_pixels": negatives, "regions": int(region_count),
            "method": "exact native-resolution score sweep, 8-connected equal-weight regions"}


def histogram_metrics(positive, negative, region, region_count):
    """Binned image-bootstrap estimates only; primary point estimates are exact."""
    p, n, r = positive[::-1], negative[::-1], region[::-1]
    tp, fp, pro = np.cumsum(p), np.cumsum(n), np.cumsum(r)/region_count
    recall, fpr = tp/tp[-1], fp/fp[-1]
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)>0)
    roc = float(np.trapezoid(np.r_[0, recall], np.r_[0, fpr]))
    ap = float(np.sum(p/tp[-1]*precision))
    x, y = np.r_[0,fpr], np.r_[0,pro]
    keep = x < .3
    pro_area = float(np.trapezoid(np.r_[y[keep],np.interp(.3,x,y)], np.r_[x[keep],.3])/.3)
    return roc, ap, pro_area
