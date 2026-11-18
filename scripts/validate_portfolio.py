"""Reconcile portfolio checks while preserving all prior research/release bytes."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_portfolio import check


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def junit(path):
    root=ET.parse(ROOT/path).getroot()
    return {key:sum(int(s.attrib.get(key,0)) for s in root.iter('testsuite')) for key in ['tests','failures','errors','skipped']}


def validate():
    protected=read('reports/portfolio/preservation.json')
    for name,expected in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    host=junit('outputs/portfolio/public-tests.xml')
    fresh=junit('outputs/portfolio/verified-results/pytest.xml')
    for entry in [host,fresh]:
        assert entry['tests']==90 and not any(entry[k] for k in ['failures','errors','skipped'])
    source=read('outputs/portfolio/verified-results/checks.json')
    assert source['private_assets_absent'] and not source['model_inference_executed']
    frontend=read('outputs/portfolio/frontend-tests.json')
    assert frontend['success'] and frontend['numPassedTests']==39 and frontend['numFailedTests']==0
    security=read('reports/portfolio/security_check.json');assert security['status']=='lookup_completed'
    assert security['previous_runtime_records_still_reported']==100
    assert security['npm']['production']['total']==0
    result=dict(status='passed',baseline_commit='d6bc59753fa72ced09b027cf8cd89d0102c8df6e',protected_files=len(protected),
        source_contracts=check(),host_public_tests=host,fresh_cpu_container_tests=fresh,
        frontend_tests=dict(passed=frontend['numPassedTests'],failed=frontend['numFailedTests'],skipped=frontend['numPendingTests']),
        frontend_typecheck_lint_build='passed; lockfile install also passed',
        fresh_scope='Public-source copy, read-only workspace; no dataset/extractor/banks/GPU mounted; CPU runtime plus isolated pinned pytest 9.0.3 source-check dependencies; pip check passed',
        retained_artifact_preflight=dict(status='passed',categories=15,files=33,scope='Documented Compose checksum-only preflight; no model inference'),
        not_rerun=['270 research evaluations','training/refitting','real-image inference','threshold calibration','browser real-model integration','deployment performance benchmarks'],
        ci='Workflow parsed and equivalent Linux CPU checks passed locally; remote run must be checked for the pushed commit, not inferred',
        security=dict(runtime_records_still_reported=100,npm_dev_affected_packages=security['npm']['all']['total'],npm_production=0,
            image_scan=read('reports/portfolio/image_scan.json')['status']),
        licensing='No project-wide license added. Source visibility, upstream code, MVTec data and weight permissions documented separately.',
        metadata='Read only; description/topics observed empty and exact manual recommendations supplied',
        recovered_existing_change='Pre-existing local up.json timing change backed up under ignored outputs/portfolio before restoring the immutable published record')
    (ROOT/'reports/portfolio/validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':validate()
