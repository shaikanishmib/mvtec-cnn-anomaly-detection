"""Comparison logic tests use recorded results; never instantiate a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from anomaly_detection.model_comparison import (MODELS,METRICS,validate_coverage,category_macro,
    paired_statistics,holm,threshold_diagnostics,normalize_units)

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'reports/final_comparison'


@pytest.fixture(scope='module')
def frame():
    return pd.read_csv(D/'all_runs.csv',float_precision='round_trip')


def test_exact_six_model_coverage(frame):
    assert validate_coverage(frame)
    assert set(frame.model)==set(MODELS)
    assert frame.groupby('model').size().eq(45).all()


def test_missing_duplicate_or_wrong_seed_rejected(frame):
    for broken in [frame.iloc[:-1],pd.concat([frame.iloc[:-1],frame.iloc[:1]]),frame.assign(seed=2026)]:
        with pytest.raises(ValueError):validate_coverage(broken)


def test_category_macro_and_seed_pairing(frame):
    assert frame.groupby(['model','category']).seed.nunique().eq(3).all()
    expected=pd.read_csv(D/'model_macro.csv',index_col='model',float_precision='round_trip')
    assert np.allclose(category_macro(frame),expected,atol=1e-14)
    extra=pd.concat([frame,frame.query("category == 'bottle'")])
    assert np.allclose(category_macro(extra),category_macro(frame))


def test_original_precision_and_provenance(frame):
    import hashlib
    for row in frame.itertuples():
        p=ROOT/row.source_result
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row.source_result_sha256
        original=json.loads(p.read_text())
        for k in METRICS:assert getattr(row,k)==original[k]


def test_strict_threshold_ties_and_orientation():
    result=threshold_diagnostics([1,2,2,3],2)
    assert result['calibration_fpr']==.25
    assert result['threshold_ties']==2
    assert result['calibration_unique']==3
    with pytest.raises(ValueError):threshold_diagnostics([np.nan],2)


def test_units():
    assert normalize_units(3600,2**30)==dict(hours=1,mib=1024,gib=1)


def test_holm_known_example():
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])


def test_pairing_cluster_uncertainty_and_direction(frame):
    results=paired_statistics(frame,draws=1000)
    assert len(results)==135
    assert results.categories.eq(15).all()
    assert results.p_holm_135.ge(results.p_sign_randomization-1e-15).all()
    fpr=results.query("metric == 'fpr'")
    for row in fpr.itertuples():
        a=frame.query('model == @row.model_a').groupby('category').fpr.mean()
        b=frame.query('model == @row.model_b').groupby('category').fpr.mean()
        assert row.favorable_categories==int(((a-b)<-1e-12).sum())
    assert np.allclose(results.difference,paired_statistics(frame,draws=1000).difference)


def test_category_bootstrap_constant_difference_and_seed_dependence(frame):
    # Add opposing seed effects shared across models. They must cancel within
    # each category, rather than artificially narrow a 45-independent-row CI.
    controlled=frame.copy()
    for i,model in enumerate(MODELS):
        selected=controlled.model.eq(model)
        controlled.loc[selected,list(METRICS)]=.2+i*.1
        for seed,offset in [(2026,-.05),(2027,0),(2028,.05)]:
            controlled.loc[selected & controlled.seed.eq(seed),list(METRICS)]+=offset
    result=paired_statistics(controlled,draws=1000)
    first=result.query("model_a == 'basic_cnn' and model_b == 'autoencoder' and metric == 'image_auroc'").iloc[0]
    assert np.allclose([first.difference,first.ci_low,first.ci_high],[-.1,-.1,-.1])
    assert first.p_sign_randomization==2/32768
    duplicate=pd.concat([controlled,controlled])
    repeated=paired_statistics(duplicate,draws=1000)
    assert np.allclose(result[['ci_low','ci_high']],repeated[['ci_low','ci_high']])


def test_metric_range_roundoff_and_latency_units(frame):
    valid=frame.copy();valid.loc[0,'image_ap']=1+2e-16
    assert validate_coverage(valid)
    valid.loc[0,'image_ap']=1.01
    with pytest.raises(ValueError):validate_coverage(valid)
    assert (frame.model_p95_ms>=frame.model_p50_ms).all()
    assert (frame.end_to_end_p95_ms>=frame.end_to_end_p50_ms).all()


def test_saved_prediction_reconciliation_and_normal_calibration(frame):
    reconciliation=pd.read_csv(D/'prediction_reconciliation.csv')
    assert reconciliation.prediction_flags_match.all()
    assert reconciliation.max_metric_error.max()<1e-8
    assert frame.calibration_n.between(9,58).all()
    assert frame.score_direction.eq('higher_is_anomaly').all()
    assert (frame.calibration_fpr<=.05+1e-12).all()
    assert (frame.latency_iterations==200).all()


def test_notebook_saved_analysis_only():
    import nbformat
    nb=nbformat.read(ROOT/'notebooks/07_model_comparison.ipynb',as_version=4)
    codes='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    assert 'paired_statistics(runs)' in codes
    for forbidden in ('torch.load','backward(','.fit(','.predict(','run_efficientad'):
        assert forbidden not in codes
    assert any(c.metadata.get('contains_dataset_images') for c in nb.cells)
