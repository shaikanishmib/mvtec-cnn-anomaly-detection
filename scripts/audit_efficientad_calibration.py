"""Repair evaluation serialization without changing fitted models or selection."""
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
from anomaly_detection.data import sha256
from anomaly_detection.tracking import record
from anomaly_detection.efficientad import load_checkpoint
from anomaly_detection.efficientad_research import configure,read_normals,collect_raw_maps,score_maps
from anomaly_detection.efficientad_calibration import write_scores,read_scores,qualified_calibration

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/efficientad/development'


def main():
    configure(ROOT)
    assert not (OUT/'calibration-precision-audit.json').exists(),'Reuse completed audit'
    plan=json.loads((ROOT/'configs/efficientad_development.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    event=json.loads((ROOT/'configs/experiment_template.json').read_text())
    event.update(experiment_id='06-calibration-precision-v1-start',model_family=plan['model_name'],categories=plan['categories'],
        hypothesis='Exact score round-trip corrects false exceedances from decimal float32 CSV parsing; diagnose clean-score ties without selecting a new model',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda),
        training=dict(source_sha256=sha256(Path(__file__)),selection_sha256=sha256(OUT/'selection.json')),
        hyperparameters=dict(retraining=False,selection_changed=False,threshold_policy=plan['threshold_policy']),status='started')
    record(ROOT/'experiments',event)
    begin=time.perf_counter(); reports=[]
    for category in plan['categories']:
        run=json.loads((OUT/f"06-dev-{category}-e{selected['epochs']}-seed2026-v1.json").read_text())
        model,_=load_checkpoint(ROOT/run['checkpoint']['path'],'cuda')
        q=model.map_quantiles.cpu(); parts={}; floor={}
        for part in ('development','calibration'):
            rows,images=read_normals(ROOT,category,part)
            pairs=collect_raw_maps(model,images)
            scores=score_maps(pairs,q)[selected['score']]
            path=OUT/f'{category}-{part}-exact-scores.csv'
            write_scores(path,rows.path,scores)
            parts[part]=read_scores(path).score.to_numpy()
            flat=pairs[0].flatten(1)
            positions=flat.argmax(1).numpy()
            y=positions//256; x=positions%256
            floor[part]=dict(unique_image_scores=len(np.unique(scores)),largest_tie_count=int(np.unique(scores,return_counts=True)[1].max()),
                peak_positions_yx=np.stack((y,x),1).tolist(),
                fraction_peaks_within_16px_border=float(((y<16)|(y>=240)|(x<16)|(x>=240)).mean()),
                mean_raw_map_spatial_std=float(pairs[0].flatten(1).std(1).mean()),
                mean_raw_map_across_image_std=float(pairs[0].std(0).mean()),
                raw_interior_max_std=float(pairs[0][:,:,16:-16,16:-16].flatten(1).amax(1).std()),
                interpretation='Constant max can be a response floor even when spatial/inter-image maps vary; interior statistics are diagnosis only, no crop or alternate score selected.')
        diagnosis=qualified_calibration(parts['calibration'],parts['development'],plan['threshold_policy'])
        old=json.loads((OUT/f'calibration-{category}.json').read_text())
        assert diagnosis['threshold']==old['threshold'],'Threshold policy/model changed unexpectedly'
        diagnosis.update(category=category,selection_sha256=sha256(OUT/'selection.json'),checkpoint_sha256=run['checkpoint']['sha256'],
            supersedes=f'calibration-{category}.json',correction='Promote exact float32 scores to float64 before 17-digit CSV; round_trip reader; no retraining, new threshold or model selection',
            original_development_exceedance_fraction=old['development_exceedance_fraction'],floor_diagnostics=floor)
        (OUT/f'calibration-{category}-corrected.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
        reports.append(diagnosis)
        del model
    audit=dict(status='corrected',elapsed_seconds=time.perf_counter()-begin,selection_changed=False,retraining=False,
        original_reports_preserved=True,corrected_reports=[f"calibration-{r['category']}-corrected.json" for r in reports],
        findings=[dict(category=r['category'],original_dev_exceedance=r['original_development_exceedance_fraction'],corrected_dev_exceedance=r['development_exceedance_fraction'],unique_calibration_scores=r['unique_calibration_scores']) for r in reports])
    (OUT/'calibration-precision-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    event.update(experiment_id='06-calibration-precision-v1-result',status='completed',metrics=audit,
        calibration=plan['threshold_policy'],interpretation='Score serialization repaired. Strong ties qualify uncertainty, not evidence of feature collapse or reliable future FPR.',
        decision='Use corrected calibration reports; retain selected configuration and original threshold policy')
    record(ROOT/'experiments',event)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
