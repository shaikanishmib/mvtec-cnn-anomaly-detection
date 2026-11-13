"""Cheap status read: never loads a model, cache or official image."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'reports/supersimplenet'
finished=list((folder/'runs').glob('04-final-*-result.json'));evaluated=list((folder/'evaluation').glob('04-test-*-result.json'))
histories=list((folder/'runs').glob('04-final-*-history.csv'));latest=max(histories,key=lambda p:p.stat().st_mtime) if histories else None
result=dict(completed_fits=len(finished),completed_evaluations=len(evaluated),planned_models=45)
if latest and len(finished)<45:
    try:
        with latest.open() as stream:rows=list(csv.DictReader(stream))
        last=rows[-1];result.update(current=latest.stem,epoch=int(last['epoch']),train_loss=float(last['train_loss']),epoch_seconds=float(last['train_epoch_seconds']))
    except (ValueError,KeyError,IndexError):result['current']='history write in progress'
if len(finished)==45:
    completed={p.name.removesuffix('-result.json') for p in evaluated}
    active=[p for p in (ROOT/'experiments').glob('04-test-*-start.json') if p.name.removesuffix('-start.json') not in completed]
    if active:result['current_evaluation']=max(active,key=lambda p:p.stat().st_mtime).name.removesuffix('-start.json')
print(json.dumps(result,indent=2))
