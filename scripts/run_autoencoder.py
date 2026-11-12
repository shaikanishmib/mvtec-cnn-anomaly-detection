from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from anomaly_detection.ae_research import ROOT,fit

torch.set_num_threads(4)
base=dict(width=16,depth=3,latent_channels=8,loss='l1',resolution=256,batch_size=16,learning_rate=.001,epochs=20,seed=2026)
variants={'base':{},'deeper':dict(depth=4),'narrow':dict(latent_channels=2),'wide':dict(latent_channels=32),'mse':dict(loss='mse')}
if __name__=='__main__':
    if sys.argv[1]=='development':
        for name,change in variants.items():
            for category in ['bottle','cable','carpet','tile']:
                identifier=f'02-dev-{name}-{category}-s2026-v1'
                if (ROOT/f'reports/autoencoder/runs/{identifier}-result.json').exists():continue
                fit(identifier,category,dict(base,**change))
    elif sys.argv[1] in ['final-train','final-evaluate']:
        from anomaly_detection.data import sha256
        frozen_path=ROOT/'configs/autoencoder_frozen.json'
        frozen=json.loads(frozen_path.read_text())
        assert sha256(frozen_path)==json.loads((ROOT/'reports/autoencoder/freeze_identity.json').read_text())['sha256']
        for seed in frozen['seeds']:
            for category in frozen['categories']:
                identifier=f"02-final-{frozen['selected_variant']}-{category}-s{seed}-v1"
                path=ROOT/f'reports/autoencoder/runs/{identifier}-result.json'
                if sys.argv[1]=='final-train':
                    if path.exists():continue
                    fit(identifier,category,dict(frozen['configuration'],seed=seed),final=True)
                else:
                    from anomaly_detection.ae_evaluation import evaluate_frozen
                    result=json.loads(path.read_text())
                    eval_id=f"02-test-{frozen['selected_variant']}-{category}-s{seed}-v1"
                    if (ROOT/f'reports/autoencoder/evaluation/{eval_id}-result.json').exists():continue
                    measured=evaluate_frozen(eval_id,result,sha256(frozen_path))
                    print(json.dumps({k:measured[k] for k in ['category','seed','image_auroc','pixel_auroc','evaluation_seconds']}),flush=True)
    else:raise ValueError('Unknown phase')
