"""Additional genuine object/texture defect parity, without model selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataclasses import replace
import logging
import pandas as pd
import torch
from fastapi.testclient import TestClient
from anomaly_detection.backend.app import create_app
from anomaly_detection.backend.contracts import Settings
from anomaly_detection.data import autodiscover_data_root
from anomaly_detection.inference import PatchCorePredictor


def run():
    target=ROOT/'reports/backend/defect_parity.csv'
    assert not target.exists(),'Reuse completed defect parity evidence'
    logging.getLogger('anomaly.backend').setLevel(logging.WARNING)
    torch.set_num_threads(4);rows=[]
    saved=pd.read_csv(ROOT/'reports/engineering/patchcore/parity.csv')
    data_root=autodiscover_data_root(ROOT)
    for device in ('cpu','cuda'):
        with TestClient(create_app(replace(Settings(),device=device))) as client:
            for category in ('bottle','carpet'):
                selected=saved.query('category == @category and device == "cpu"')
                path=selected[~selected.path.str.contains('/test/good/')].iloc[0].path
                image=(data_root/path).read_bytes()
                reference=PatchCorePredictor(ROOT/'configs/patchcore_deployment_v1.json',ROOT,category,device).predict(image)
                response=client.post('/api/v1/predict',files={'image':('inspection.png',image,'image/png')},data={'category':category})
                assert response.status_code==200,response.text
                value=response.json()
                assert abs(value['anomaly_score']-reference.score)<1e-6
                assert value['threshold']==reference.threshold and value['is_anomaly']==reference.is_anomaly
                rows.append(dict(category=category,device=device,path=path,api_score=value['anomaly_score'],
                    direct_score=reference.score,absolute_error=abs(value['anomaly_score']-reference.score),
                    threshold=reference.threshold,decision=reference.is_anomaly))
    pd.DataFrame(rows).to_csv(target,index=False,float_format='%.17g',lineterminator='\n')
    print('Four genuine object/texture defect CPU/CUDA HTTP parity checks passed')


if __name__=='__main__':run()
