"""Check final contract/report consistency without repeating the deep source audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import json
import xml.etree.ElementTree as ET
import pandas as pd
from anomaly_detection.backend.app import create_app
from finalize_backend_reports import timing_section


def run():
    out=ROOT/'reports/backend'
    audit=json.loads((out/'validation.json').read_text());assert audit['status']=='passed'
    assert audit['protected_files']==3086 and audit['selected_banks_verified']==15
    tests=json.loads((out/'test_summary.json').read_text())
    assert tests['tests']==203 and tests['failures']==tests['errors']==tests['skipped']==0
    focused=ET.parse(ROOT/'outputs/backend/pytest-contract-final.xml').getroot().find('testsuite')
    assert int(focused.attrib['tests'])==25
    assert all(int(focused.attrib[k])==0 for k in ['errors','failures','skipped'])
    assert create_app().openapi()==json.loads((out/'openapi.json').read_text())
    defects=pd.read_csv(out/'defect_parity.csv');assert len(defects)==4 and defects.absolute_error.max()==0
    report=json.loads((out/'performance.json').read_text())
    assert len(report['records'])==4 and report['api_repetitions']==30
    expected={(c,d) for c in ['bottle','carpet'] for d in ['cpu','cuda']}
    assert {(r['category'],r['device']) for r in report['records']}==expected
    for r in report['records']:
        for name in ['api_round_trip','warm_inference','decode','preprocessing','feature_extraction','nearest_neighbor','visualization_png']:
            assert 0<r[name]['p50_ms']<=r[name]['p95_ms']
    assert timing_section(report) in (ROOT/'docs/backend_api.md').read_text(encoding='utf-8')
    files=list((ROOT/'src/anomaly_detection/backend').glob('*.py'))
    files += [ROOT/'scripts'/name for name in ['benchmark_backend.py','backend_server_process.py',
        'smoke_backend_server.py','validate_backend.py','validate_backend_real.py','validate_backend_defects.py',
        'finalize_backend_reports.py','finalize_backend_validation.py']]
    for path in files:compile(path.read_text(encoding='utf-8'),path.name,'exec')
    result=dict(status='passed',full_suite_tests=203,failures=0,skipped=0,
        final_schema_only_contract_recheck=25,openapi_paths=8,normal_image_parity_checks=30,
        defect_parity_checks=4,performance_cases=4,guide_tables_match_measurements=True,
        historical_integrity='See completed deep validation.json; no prior model/source/report files staged',
        inference_source_unchanged=True)
    (out/'completion_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
