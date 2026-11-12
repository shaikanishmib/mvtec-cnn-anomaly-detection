"""Recorded real results are validated after the full milestone exists."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
from anomaly_detection.tracking import config_hash
from anomaly_detection.data import CATEGORIES,sha256

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from validate_basic_cnn import validate_notebook


def test_basic_cnn_notebook_full_story_and_execution():
    if not (ROOT/"notebooks/01_basic_cnn.ipynb").exists():pytest.skip("Notebook not yet produced")
    status=validate_notebook()
    assert status['sections']==27 and status['executed_code_cells']==27


def test_basic_cnn_registry_identities_are_preserved():
    events=list((ROOT/"experiments").glob('01-*.json'))
    assert events
    for path in events:
        entry=json.loads(path.read_text())
        recorded=entry.pop('config_sha256');entry.pop('recorded_at')
        assert config_hash(entry)==recorded,path.name
    for path in (ROOT/"experiments").glob('01-dev-*-decision.json'):
        decision=json.loads(path.read_text())
        assert decision['decision'] in {'keep','reject'}
        assert path.with_name(path.name.replace('-decision.json','-result.json')).exists()


def test_frozen_category_coverage_and_confirmation_order():
    summary=ROOT/'reports/basic_cnn/summary.json'
    if not summary.exists():pytest.skip("Frozen confirmation pending")
    table=pd.read_csv(ROOT/'reports/basic_cnn/all_runs.csv')
    assert len(table)==45 and set(table.category)==set(CATEGORIES)
    assert table.groupby('category').seed.apply(set).apply(lambda v:v=={2026,2027,2028}).all()
    frozen=json.loads((ROOT/'configs/basic_cnn_frozen.json').read_text())
    train_times=[json.loads(p.read_text())['recorded_at'] for p in (ROOT/'experiments').glob('01-final-*-result.json')]
    test_times=[json.loads(p.read_text())['recorded_at'] for p in (ROOT/'experiments').glob('01-test-*-start.json')]
    assert len(train_times)==len(test_times)==45
    assert max(train_times)<min(test_times) and frozen['frozen_at_utc']<min(test_times)
    assert np.isfinite(table[['image_auroc','pixel_auroc','pixel_ap','au_pro']]).all().all()
    assert sha256(ROOT/'configs/basic_cnn_frozen.json')==json.loads((ROOT/'reports/basic_cnn/freeze_identity.json').read_text())['sha256']
