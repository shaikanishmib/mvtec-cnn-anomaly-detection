"""Audit registered research artifacts without repeating official inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.tracking import config_hash

ROOT=Path(__file__).resolve().parents[1]


def test_ae_registry_identities_and_complete_development():
    paths=list((ROOT/'experiments').glob('02-*.json'))
    assert paths
    for path in paths:
        event=json.loads(path.read_text())
        signature=event.pop('config_sha256');event.pop('recorded_at')
        assert config_hash(event)==signature,path
    comparison=pd.read_csv(ROOT/'reports/autoencoder/development_comparison.csv')
    assert len(comparison)==40
    assert comparison.groupby(['variant','map_kind']).size().eq(4).all()
    assert set(comparison.category)=={'bottle','cable','carpet','tile'}
    ranked=pd.read_csv(ROOT/'reports/autoencoder/development_macro.csv')
    frozen=json.loads((ROOT/'configs/autoencoder_frozen.json').read_text())
    assert frozen['selected_variant']==ranked.iloc[0].variant
    assert frozen['configuration']['map_kind']==ranked.iloc[0].map_kind
    assert sha256(ROOT/'configs/autoencoder_frozen.json')==json.loads((ROOT/'reports/autoencoder/freeze_identity.json').read_text())['sha256']


def test_ae_final_results_reconcile():
    table=pd.read_csv(ROOT/'reports/autoencoder/all_runs.csv')
    assert len(table)==45 and set(table.category)==set(CATEGORIES)
    assert table.groupby('category').seed.apply(set).apply(lambda s:s=={2026,2027,2028}).all()
    summary=json.loads((ROOT/'reports/autoencoder/summary.json').read_text())
    for metric,values in summary['macro'].items():
        np.testing.assert_allclose(table.groupby('category')[metric].mean().mean(),values['mean'])
    comparison=pd.read_csv(ROOT/'reports/autoencoder/cnn_macro_comparison.csv',index_col=0)
    baseline=json.loads((ROOT/'reports/basic_cnn/summary.json').read_text())
    for metric,values in baseline['macro'].items():
        np.testing.assert_allclose(comparison.loc[metric,'Basic CNN'],values['mean'])
