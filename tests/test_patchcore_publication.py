from pathlib import Path
import json
import numpy as np
import pandas as pd
from anomaly_detection.tracking import config_hash
from anomaly_detection.data import CATEGORIES
ROOT=Path(__file__).resolve().parents[1]


def test_patchcore_registry_and_frozen_cohort_order():
    events=[]
    for path in (ROOT/'experiments').glob('03-*.json'):
        entry=json.loads(path.read_text());signature=entry.pop('config_sha256');timestamp=entry.pop('recorded_at')
        assert config_hash(entry)==signature
        events.append(dict(**entry,recorded_at=timestamp))
    fits=[e['recorded_at'] for e in events if e['experiment_id'].startswith('03-final-') and e['status']=='completed']
    tests=[e['recorded_at'] for e in events if e['experiment_id'].startswith('03-test-') and e['status']=='started']
    assert len(fits)==len(tests)==45 and max(fits)<min(tests)
    dev=pd.read_csv(ROOT/'reports/patchcore/development_comparison.csv')
    assert len(dev)==24 and set(dev.budget)=={256,1024,2048} and set(dev.aggregation)=={'max','top1pct'}


def test_patchcore_metric_reconciliation_and_exact_comparisons():
    output=ROOT/'reports/patchcore'
    table=pd.read_csv(output/'all_runs.csv')
    assert len(table)==45 and set(table.category)==set(CATEGORIES)
    assert table.groupby('category').seed.apply(set).apply(lambda s:s=={2026,2027,2028}).all()
    summary=json.loads((output/'summary.json').read_text())
    for key,value in summary['macro'].items():np.testing.assert_allclose(table.groupby('category')[key].mean().mean(),value['mean'])
    comparison=pd.read_csv(output/'model_macro_comparison.csv',index_col=0)
    for label,folder in [('Basic CNN','basic_cnn'),('Autoencoder','autoencoder')]:
        baseline=json.loads((ROOT/f'reports/{folder}/summary.json').read_text())
        for key,value in baseline['macro'].items():np.testing.assert_allclose(comparison.loc[key,label],value['mean'])
    paired=pd.read_csv(output/'paired_image_difference_intervals.csv')
    assert len(paired)==12 and (paired.low<=paired.high).all()
    for key,value in summary['pixel_operating_macro'].items():np.testing.assert_allclose(table[key].mean(),value)
