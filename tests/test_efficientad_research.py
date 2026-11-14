"""Bounded study rules, scoring and threshold feasibility without GPU fitting."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from anomaly_detection.efficientad_research import (proxy_edits,normalized,score_maps,
    calibration_diagnostics,select_configuration,read_normals)


def test_proxy_edits_deterministic_and_local():
    x=torch.full((1,3,256,256),.4)
    a=proxy_edits(x,0); b=proxy_edits(x,0)
    assert set(a)=={'scratch','block','color_patch'}
    assert torch.equal(x,torch.full_like(x,.4))
    for key in a:
        assert torch.equal(a[key],b[key]) and not torch.equal(x,a[key])
        assert (a[key]>=0).all() and (a[key]<=1).all()
    with pytest.raises(ValueError): proxy_edits(torch.zeros(3,32,32),0)


def test_score_orientation_and_ae_ablation():
    low=torch.zeros(2,1,32,32); high=low.clone(); high[1,0,16,16]=2
    q=torch.tensor([[0.,1.],[0.,1.]])
    scores=score_maps((high,low),q)
    assert scores['combined_max'][1]>scores['combined_max'][0]
    assert scores['local_max'][1]==2*scores['combined_max'][1]
    assert scores['global_max'].sum()==0


def test_normal_calibration_frozen_policy_uncertainty():
    policy=dict(target_fpr=.05,bootstrap_replicates=100,bootstrap_seed=6806)
    a=np.arange(31,dtype=float)
    result=calibration_diagnostics(a,np.array([0.,30.,31.]),policy)
    assert result['threshold']==29 and result['observed_calibration_fpr']==1/31
    assert result['strict_greater'] and result['quantile_method']=='higher'
    assert result['calibration_distribution']['n']==31
    assert result['future_fpr_order_statistic_95'][0]<.05<result['future_fpr_order_statistic_95'][1]
    assert result==calibration_diagnostics(a,np.array([0.,30.,31.]),policy)


def test_selection_rules_and_coverage():
    plan=dict(epoch_budgets=[3,9],categories=['bottle','carpet'],candidate_scores=['combined_max','local_max'])
    results=[]
    for epoch in (3,9):
        for cat in plan['categories']:
            results.append(dict(category=cat,epochs=epoch,finite=True,probe_before=dict(hard=10,autoencoder=2),
                probe_after=dict(hard=5,autoencoder=1),proxy={k:dict(auroc=.8 if epoch==3 else .81) for k in plan['candidate_scores']}))
    chosen=select_configuration(results,plan)
    assert chosen['eligible'] and chosen['selected']['epochs']==3 and chosen['selected']['score']=='combined_max'
    bad=deepcopy(results)
    for r in bad: r['proxy']['combined_max']['auroc']=.6; r['proxy']['local_max']['auroc']=.6
    assert not select_configuration(bad,plan)['eligible']
    with pytest.raises(ValueError): select_configuration(results[:-1],plan)
    longer=deepcopy(results)
    for r in longer:
        r['proxy']['combined_max']['auroc']=.7 if r['epochs']==3 else .895
        r['proxy']['local_max']['auroc']=.7 if r['epochs']==3 else .91
    chosen=select_configuration(longer,plan)
    assert chosen['selected']['epochs']==9 and chosen['selected']['score']=='combined_max'
    for r in longer:
        if r['epochs']==9: r['proxy']['combined_max']['auroc']=.85
    assert select_configuration(longer,plan)['selected']['score']=='local_max'


def test_reject_official_test_loading(tmp_path):
    with pytest.raises(ValueError): read_normals(tmp_path,'bottle','test')


def test_signed_scores_compatible_with_project_metrics():
    from anomaly_detection.metrics import binary_metrics
    from anomaly_detection.pixel_metrics import exact_pixel_metrics
    labels=np.array([0,0,1,1],np.uint8)
    scores=np.array([-.3,-.2,.1,.5],np.float32)
    assert binary_metrics(labels,scores,threshold=0.)['auroc']==1.
    result=exact_pixel_metrics(scores,labels,np.array([0.,0.,.5,.5]),region_count=1)
    assert result['pixel_auroc']==1. and result['pixel_ap']==1. and result['au_pro']==1.


def test_exact_score_serialization_and_tie_uncertainty(tmp_path):
    from anomaly_detection.efficientad_calibration import write_scores,read_scores,qualified_calibration
    score=np.float32(.5066908597946167)
    a=np.repeat(score,31)
    path=tmp_path/'scores.csv'
    write_scores(path,[f'normal-{i}' for i in range(31)],a)
    restored=read_scores(path).score.to_numpy()
    assert np.array_equal(restored,a.astype(np.float64))
    policy=dict(target_fpr=.05,bootstrap_replicates=100,bootstrap_seed=6806)
    diagnosis=qualified_calibration(restored,restored,policy)
    assert diagnosis['exceedances']==0 and diagnosis['development_exceedance_fraction']==0.
    assert diagnosis['largest_tie_count']==31 and diagnosis['unique_calibration_scores']==1
    assert diagnosis['future_fpr_order_statistic_95'] is None
    assert diagnosis['continuous_order_statistic_applicable'] is False
