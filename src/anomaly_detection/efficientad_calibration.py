"""Exact score serialization and explicit tied-score calibration qualifications."""
import numpy as np
import pandas as pd
from anomaly_detection.efficientad_research import calibration_diagnostics


def write_scores(path,paths,scores):
    """Promote exact float32 values before decimal formatting, never after reading."""
    a=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError('Finite scores required')
    pd.DataFrame(dict(path=paths,score=a)).to_csv(path,index=False,float_format='%.17g',lineterminator='\n')
    restored=read_scores(path)
    assert np.array_equal(restored.score.to_numpy(),a),'Score round-trip changed bits'


def read_scores(path):
    return pd.read_csv(path,float_precision='round_trip')


def qualified_calibration(scores,development,policy):
    a=np.asarray(scores,dtype=np.float64)
    result=calibration_diagnostics(a,np.asarray(development,dtype=np.float64),policy)
    unique,counts=np.unique(a,return_counts=True)
    result.update(unique_calibration_scores=len(unique),largest_tie_count=int(counts.max()),
        ties_at_threshold=int((a==result['threshold']).sum()),continuous_order_statistic_applicable=len(unique)==len(a))
    if len(unique)!=len(a):
        result['continuous_iid_hypothetical']=dict(mean=result.pop('future_fpr_order_statistic_mean'),interval=result.pop('future_fpr_order_statistic_95'))
        result['future_fpr_order_statistic_mean']=None
        result['future_fpr_order_statistic_95']=None
        result['uncertainty_note']='Tied calibration scores violate the continuous-score assumption: Beta future-FPR interval is not asserted. The retained hypothetical interval is reference only; degenerate bootstrap tails do not establish certainty. CP calibration interval is descriptive because threshold was fitted on those samples.'
    return result
