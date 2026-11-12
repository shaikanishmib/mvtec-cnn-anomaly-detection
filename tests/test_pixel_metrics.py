import numpy as np
import pytest
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, average_precision_score
from anomaly_detection.metrics import au_pro
from anomaly_detection.pixel_metrics import exact_pixel_metrics, histogram_metrics


@pytest.mark.parametrize("chunk", [1,3,7,1000])
def test_exact_native_ranking_matches_references_with_ties(chunk):
    rng=np.random.default_rng(11)
    masks=[(rng.random((12,13))>.8).astype(np.uint8), np.zeros((12,13),np.uint8)]
    maps=[np.round(rng.random(m.shape),1) for m in masks]
    weights=[]; regions=0
    for mask in masks:
        labels,count=label(mask,structure=np.ones((3,3)))
        sizes=np.bincount(labels.ravel()); sizes[0]=1
        weights.append(np.where(mask,1/sizes[labels],0.))
        regions+=count
    truth=np.concatenate([m.ravel() for m in masks])
    scores=np.concatenate([m.ravel() for m in maps])
    result=exact_pixel_metrics(scores,truth,np.concatenate([w.ravel() for w in weights]),regions,chunk)
    assert result['pixel_auroc']==pytest.approx(roc_auc_score(truth,scores),abs=1e-12)
    assert result['pixel_ap']==pytest.approx(average_precision_score(truth,scores),abs=1e-12)
    assert result['au_pro']==pytest.approx(au_pro(masks,maps),abs=1e-12)


def test_constant_and_perfect_native_maps():
    labels=np.array([1,0,0,0],np.uint8)
    weights=labels.astype(float)
    constant=exact_pixel_metrics(np.zeros(4),labels,weights,1,2)
    assert constant['pixel_auroc']==.5 and constant['pixel_ap']==.25
    assert constant['au_pro']==pytest.approx(.15)
    perfect=exact_pixel_metrics(labels.astype(float),labels,weights,1,2)
    assert perfect['pixel_auroc']==perfect['pixel_ap']==perfect['au_pro']==1


def test_histogram_reference_for_exactly_quantized_scores():
    p=np.array([0,2,1,4]); n=np.array([5,3,2,0]); r=p/7
    a,b,c=histogram_metrics(p,n,r,1)
    labels=np.concatenate([np.r_[np.ones(p[i]),np.zeros(n[i])] for i in range(4)]).astype(np.uint8)
    scores=np.concatenate([np.repeat(i,p[i]+n[i]) for i in range(4)])
    exact=exact_pixel_metrics(scores,labels,labels/7,1,3)
    assert (a,b,c)==pytest.approx((exact['pixel_auroc'],exact['pixel_ap'],exact['au_pro']))
