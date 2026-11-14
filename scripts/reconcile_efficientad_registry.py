"""Recover the atomic-result/registry publication gap without refitting."""
from pathlib import Path
import json
from anomaly_detection.efficientad_final import ROOT,REPORT,verify_complete,identity_for,registry,freeze
from anomaly_detection.efficientad_research import configure


def main():
    configure(ROOT);freeze();manifest=json.loads((REPORT/'benchmark_manifest.json').read_text());recovered=[]
    for item in manifest['models']:
        if item['status']!='complete':continue
        result=json.loads((ROOT/item['result']).read_text());identifier=result['experiment_id']
        path=ROOT/f'experiments/{identifier}-result.json'
        if path.exists():continue
        result=verify_complete(ROOT/item['result'],identity_for(item['category'],item['seed']),deep=True)
        from anomaly_detection.data import sha256
        from anomaly_detection.efficientad_final import METRICS
        registry(identifier,'result',result['identity'],'completed',metrics={k:result[k] for k in METRICS},calibration=result['calibration'],
            paths=dict(result=item['result'],result_sha256=sha256(ROOT/item['result']),checkpoint=result['checkpoint']),
            interpretation='Recovered result published atomically before interruption; all required artifacts reverified, no fitting/evaluation repeated')
        recovered.append(identifier)
    print(json.dumps(dict(recovered_completion_events=recovered,count=len(recovered)),indent=2))


if __name__=='__main__':main()
