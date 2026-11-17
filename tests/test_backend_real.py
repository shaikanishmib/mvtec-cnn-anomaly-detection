"""Real-model HTTP integration, separate from lightweight mocked contract tests."""
from io import BytesIO
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient
from anomaly_detection.backend.app import create_app
from anomaly_detection.backend.contracts import Settings
from anomaly_detection.inference import PatchCorePredictor
from anomaly_detection.data import CATEGORIES,autodiscover_data_root

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def real():
    previous=torch.get_num_threads();torch.set_num_threads(4)
    app=create_app()
    with TestClient(app) as c:yield c,autodiscover_data_root(ROOT)
    torch.set_num_threads(previous)


@pytest.mark.parametrize('category',CATEGORIES)
def test_real_cpu_all_category_paths(real,category):
    c,data=real
    table=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
    row=table.query('category == @category and device == "cpu"').iloc[0]
    source=data/row.path
    response=c.post('/api/v1/predict',files={'image':('inspection.png',source.read_bytes(),'image/png')},
                    data={'category':category,'visualize':'false'})
    assert response.status_code==200,response.text
    result=response.json()
    assert abs(result['anomaly_score']-row.score)<2e-4
    assert result['is_anomaly']==bool(row.decision)
    assert result['provenance']['seed']==2026
    assert len(c.app.state.manager.cache)<=2


@pytest.mark.parametrize('category',['bottle','carpet'])
def test_real_cuda_direct_parity_and_native_visualizations(category):
    if not torch.cuda.is_available():pytest.skip('CUDA hardware unavailable')
    previous=torch.get_num_threads();torch.set_num_threads(4)
    settings=replace(Settings(),device='cuda')
    source_rows=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
    source=autodiscover_data_root(ROOT)/source_rows.query('category == @category').iloc[0].path
    direct=PatchCorePredictor(settings.manifest,ROOT,category,'cuda').predict(source,visualize=True)
    try:
        with TestClient(create_app(settings)) as c:
            r=c.post('/api/v1/predict',files={'image':('inspection.png',source.read_bytes(),'image/png')},data={'category':category,'visualize':'true'})
            assert r.status_code==200,r.text
            result=r.json()
            assert abs(result['anomaly_score']-direct.score)<1e-6
            assert result['threshold']==direct.threshold and result['is_anomaly']==direct.is_anomaly
            assert [result['input_height'],result['input_width']]==list(direct.anomaly_map.shape)
            for name,expected in [('heatmap',direct.heatmap),('overlay',direct.overlay)]:
                content=c.get(result['visualizations'][name+'_url']).content
                actual=np.asarray(Image.open(BytesIO(content)))
                assert np.array_equal(actual,expected)
    finally:torch.set_num_threads(previous)
