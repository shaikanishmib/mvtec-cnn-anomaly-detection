"""Post-freeze normal-development sensitivity; never changes calibration."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch,numpy as np,pandas as pd
from anomaly_detection.supersimplenet import SSNExtractor,anomaly_maps,image_scores
from anomaly_detection.ssn_research import feature_cache,predict
from anomaly_detection.ssn_evaluation import load_heads
from anomaly_detection.cnn_research import read_partition,load_resized
from anomaly_detection.data import sha256
from anomaly_detection.tracking import record
from anomaly_detection.basic_cnn import seed_everything
torch.set_num_threads(4);seed_everything(2026);torch.backends.cuda.matmul.allow_tf32=False
out=ROOT/'reports/supersimplenet';frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text());config=dict(frozen['configuration'],seed=2026)
extractor=SSNExtractor(config['backbone']).cuda();rows_out=[]
with torch.inference_mode():
    for category in ['bottle','cable','carpet','tile']:
        r=json.loads((out/f'runs/04-final-{category}-s2026-v1-result.json').read_text())
        evaluation=json.loads((out/f'evaluation/04-test-{category}-s2026-v1-result.json').read_text())
        threshold=evaluation['calibration']['image_threshold'];heads,_=load_heads(ROOT/r['checkpoint'],category,config)
        rows=read_partition('development',category).iloc[:8];cache=feature_cache(extractor,rows,8)
        maps,det,_=predict(heads,cache,8);base=image_scores(maps,det,config['aggregation']).numpy()
        images=load_resized(rows,256).cuda().float()/255
        for factor in [.95,1.,1.05]:
            if factor==1:scores=base
            else:
                seg,det=heads(extractor((images*factor).clamp(0,1)))
                scores=image_scores(anomaly_maps(seg),det,config['aggregation']).cpu().numpy()
            for i,row in enumerate(rows.itertuples()):
                rows_out.append(dict(category=category,path=row.path,source_sha256=row.sha256,brightness_factor=factor,
                    score=float(scores[i]),base_score=float(base[i]),score_change=float(scores[i]-base[i]),
                    frozen_threshold=threshold,alarm=bool(scores[i]>threshold),base_alarm=bool(base[i]>threshold)))
table=pd.DataFrame(rows_out);table.to_csv(out/'normal_brightness_sensitivity.csv',index=False,lineterminator='\n')
record(ROOT/'experiments',dict(experiment_id='04-normal-brightness-diagnostic-v1',model_family='SuperSimpleNetLite',categories=['bottle','cable','carpet','tile'],
    hypothesis='Small brightness changes can alter clean-normal scores under the frozen inference policy',preprocessing='Normal development RGB256 multiplied by .95/1/1.05 and clamped; no fitting or calibration updates',
    hyperparameters=dict(configuration=config,brightness_factors=[.95,1,1.05],first_development_normals=8),seed=2026,
    environment=dict(source_sha256=sha256(Path(__file__)),freeze_sha256=sha256(ROOT/'configs/supersimplenet_frozen.json')),
    training=dict(phase='post-freeze diagnostic only'),metrics=table.to_dict('records'),calibration=dict(policy='Frozen thresholds unchanged; augmented views never enter calibration'),
    paths=dict(table='reports/supersimplenet/normal_brightness_sensitivity.csv'),interpretation='Normal sensitivity does not prove the cause of test false positives',decision='Report without retuning',status='completed'))
print(table.groupby(['category','brightness_factor'])[['score_change','alarm']].mean().to_string())
