import numpy as np
import pytest
from sklearn.metrics import roc_auc_score,average_precision_score
from anomaly_detection.cnn_evaluation import image_bootstrap
import json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from anomaly_detection.basic_cnn import BasicCNN
from anomaly_detection.data import sha256


def test_image_bootstrap_exact_ties_and_operating_point():
    labels=np.array([0,0,1,1]);scores=np.array([.1,.4,.4,.8])
    weights=np.array([[1,1,1,1],[1,2,1,0],[0,0,2,2]])
    vectors,valid=image_bootstrap(labels,scores,.5,weights)
    assert valid.tolist()==[True,True,False]
    for j,row in enumerate(weights[:2]):
        repeated_labels=np.repeat(labels,row); repeated_scores=np.repeat(scores,row)
        assert vectors['image_auroc'][j]==pytest.approx(roc_auc_score(repeated_labels,repeated_scores))
        assert vectors['image_ap'][j]==pytest.approx(average_precision_score(repeated_labels,repeated_scores))
    assert vectors['recall'][0]==.5 and vectors['fpr'][0]==0
    assert vectors['f1'][0]==pytest.approx(2/3)


def test_frozen_evaluation_pipeline_on_synthetic_files_only(dataset,tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip("CUDA pipeline check")
    import anomaly_detection.cnn_evaluation as evaluation
    project=tmp_path/'research';(project/'reports').mkdir(parents=True)
    (project/'configs').mkdir()
    real_project=Path(__file__).resolve().parents[1]
    (project/'reports/environment.json').write_bytes((real_project/'reports/environment.json').read_bytes())
    config={'width':8,'resolution':32,'batch_size':2,'seed':2026}
    frozen={'configuration':config,'categories':['bottle'],'seeds':[2026],
            'evaluation_source_sha256':sha256(Path(evaluation.__file__)),
            'pixel_metric_source_sha256':sha256(Path(evaluation.__file__).with_name('pixel_metrics.py'))}
    freeze=project/'configs/basic_cnn_frozen.json';freeze.write_text(json.dumps(frozen))
    checkpoint=project/'fixture.pt'
    torch.save({'state_dict':BasicCNN(8).state_dict(),'config':config,'category':'bottle','training_phase':'final'},checkpoint)
    def frame(part):
        files=(sorted((dataset/'bottle/train/good').glob('*.png'))[:4] if part=='calibration'
               else sorted((dataset/'bottle/test').rglob('*.png')))
        rows=[]
        for image in files:
            mask=dataset/'bottle/ground_truth/scratch/000_mask.png' if image.parent.name=='scratch' else None
            rows.append({'path':image.relative_to(dataset).as_posix(),'category':'bottle',
                         'sha256':sha256(image),
                         'mask':mask.relative_to(dataset).as_posix() if mask else np.nan,
                         'mask_sha256':sha256(mask) if mask else np.nan})
        return pd.DataFrame(rows)
    def resized(rows,resolution,allow_test=False):
        values=[]
        for p in rows.path:
            with Image.open(dataset/p) as image:
                values.append(np.asarray(image.convert('RGB').resize((resolution,resolution),Image.Resampling.BILINEAR)).transpose(2,0,1).copy())
        return torch.from_numpy(np.stack(values))
    inventory=project/'outputs/eda';inventory.mkdir(parents=True)
    frame('test').assign(height=8,width=10).to_csv(inventory/'inventory.csv',index=False)
    monkeypatch.setattr(evaluation,'ROOT',project)
    monkeypatch.setattr(evaluation,'read_partition',lambda part,category:frame(part))
    monkeypatch.setattr(evaluation,'load_resized',resized)
    monkeypatch.setattr(evaluation,'autodiscover_data_root',lambda _:dataset)
    training={'category':'bottle','configuration':config,'checkpoint':'fixture.pt',
              'checkpoint_sha256':sha256(checkpoint),'experiment_id':'fixture-training',
              'training_seconds_including_loading_validation':0.,'peak_allocated_bytes':0,
              'checkpoint_bytes':checkpoint.stat().st_size,'parameters':sum(p.numel() for p in BasicCNN(8).parameters())}
    result=evaluation.evaluate_frozen('fixture-evaluation',training,sha256(freeze))
    assert result['image_rows']==2 and result['positive_pixels']==6
    assert result['negative_pixels']==154 and result['regions']==1
    assert result['calibration']['images']==4
    assert sum(np.asarray(result['confusion_matrix']).ravel())==2
    assert not list((project/'outputs/basic_cnn/fixture-evaluation').glob('native-*.dat'))
    with pytest.raises(FileExistsError):evaluation.evaluate_frozen('fixture-evaluation',training,sha256(freeze))
