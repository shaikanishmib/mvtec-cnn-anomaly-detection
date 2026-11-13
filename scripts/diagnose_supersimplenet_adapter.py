"""CPU-only post-freeze adapter diagnostics on held-out fitting normals."""
import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_supersimplenet import candidates,development_identifier
from anomaly_detection.ssn_research import model_for
from anomaly_detection.cnn_research import inner_fit_split,read_partition
from anomaly_detection.tracking import config_hash,record
from anomaly_detection.data import sha256
torch.set_num_threads(2)
out=ROOT/'reports/supersimplenet';frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text());rows=[]
started=time.perf_counter()
for name,config in candidates().items():
    for category in ['bottle','cable','carpet','tile']:
        identifier=development_identifier(name,category);r=json.loads((out/f'runs/{identifier}-result.json').read_text())
        assert sha256(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        saved=torch.load(ROOT/r['checkpoint'],weights_only=True,map_location='cpu')
        torch.manual_seed(config['seed']);initial=model_for(config).eval().requires_grad_(False)
        model=model_for(config).eval().requires_grad_(False);model.load_state_dict(saved['state_dict'])
        drift=float((model.adapter.weight-initial.adapter.weight).norm()/initial.adapter.weight.norm())
        assert abs(drift-r['adapter_relative_weight_drift'])<1e-7
        _,probes=inner_fit_split(read_partition('fit',category));paths=probes.path.iloc[:8].tolist()
        lookup={entry['source_image']:entry for entry in r['feature_caches']['probes']['images']}
        statistics=[]
        with torch.no_grad():
            for start in range(0,len(paths),2):
                selected=paths[start:start+2]
                for p in selected:assert sha256(ROOT/lookup[p]['path'])==lookup[p]['sha256']
                x=torch.from_numpy(np.stack([np.load(ROOT/lookup[p]['path']) for p in selected]))
                a=model.adapter(x);original=initial.adapter(x)
                statistics.append(dict(images=len(selected),raw_spatial_std=float(x.flatten(2).std(2).mean()),
                    initial_adapter_spatial_std=float(original.flatten(2).std(2).mean()),trained_adapter_spatial_std=float(a.flatten(2).std(2).mean()),
                    raw_rms=float(x.square().mean().sqrt()),trained_adapter_rms=float(a.square().mean().sqrt())))
        values={key:sum(s[key]*s['images'] for s in statistics)/len(paths) for key in statistics[0] if key!='images'}
        rows.append(dict(candidate=name,category=category,seed=config['seed'],adapter_trainable=config['adapter_trainable'],
            training_noise_std=config['noise_std'],relative_weight_drift=drift,
            trained_to_initial_weight_norm_ratio=float(model.adapter.weight.norm()/initial.adapter.weight.norm()),
            images=len(paths),probe_paths_sha256=config_hash(paths),**values))
table=pd.DataFrame(rows);table['trained_to_initial_spatial_std_ratio']=table.trained_adapter_spatial_std/table.initial_adapter_spatial_std
table.to_csv(out/'adapter_diagnostics.csv',index=False,lineterminator='\n')
record(ROOT/'experiments',dict(experiment_id='04-adapter-diagnostic-v1',model_family='SuperSimpleNetLite',categories=['bottle','cable','carpet','tile'],
    hypothesis='The adapter changes clean feature variation; quantify drift and spatial variation without inferring real-defect quality',
    preprocessing='Original held-out fitting-normal cached features; no pretrained extraction repeated',
    hyperparameters=dict(candidates=list(candidates()),first_probe_images=8),seed=2026,
    environment=dict(device='cpu',threads=2,source_sha256=sha256(Path(__file__)),freeze_sha256=sha256(ROOT/'configs/supersimplenet_frozen.json')),
    training=dict(phase='Post-freeze descriptive analysis only'),metrics=dict(rows=table.to_dict('records'),seconds=time.perf_counter()-started),
    calibration={},paths=dict(table='reports/supersimplenet/adapter_diagnostics.csv'),
    interpretation='Feature statistics and weight drift are descriptive, not causal evidence for test performance',decision='Report without changing the frozen policy',status='completed'))
print(table.to_string(index=False))
