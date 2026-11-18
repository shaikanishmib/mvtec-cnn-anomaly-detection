"""Final read-only identity checks and compact release evidence reconciliation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from deployment.artifacts import validate as validate_artifacts
from validate_backend import validate as validate_backend
from validate_model_comparison import validate_comparison
from release_deployment import source_hash, docker_json, compose_command


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def validate():
    protected=read('reports/deployment/research_protection.json')
    for name,expected in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    validate_artifacts(ROOT);validate_artifacts(ROOT/'outputs/deployment/artifacts')
    backend=validate_backend()
    research=validate_comparison()
    root=ET.parse(ROOT/'outputs/deployment/pytest.xml').getroot()
    tests={k:sum(int(s.attrib.get(k,0)) for s in root.iter('testsuite')) for k in ['tests','errors','failures','skipped']}
    assert tests['tests']>=219 and not any(tests[k] for k in ['errors','failures','skipped'])
    unit=read('outputs/deployment/frontend-tests.json')
    assert unit['success'] and unit['numPassedTests']==39 and unit['numFailedTests']==0
    browser=read('outputs/deployment/browser-tests.json')['stats']
    assert browser['expected']==3 and not any(browser[k] for k in ['unexpected','flaky','skipped'])
    builds=read('reports/deployment/builds.json');images={}
    for key,entry in builds.items():
        image=docker_json('image','inspect',f'fieldwork-{key}:local')[0]
        assert image['Id']==entry['image_id']
        service='backend' if key.startswith('backend') else 'frontend'
        assert source_hash(service,key.endswith('-cuda'))==entry['source_sha256'],key
        assert image['Config']['User'] in ['10001:10001','101:101']
        images[key]=dict(image_id=image['Id'],bytes=image['Size'],user=image['Config']['User'])
    config=json.loads(subprocess.check_output([*compose_command(),'config','--format','json'],cwd=ROOT))
    cpu=config['services']['backend'];front=config['services']['frontend']
    assert 'ports' not in cpu and cpu['read_only'] and front['read_only']
    assert front['ports'][0]['host_ip']=='127.0.0.1'
    assert cpu['environment']['ANOMALY_DEVICE']=='cpu'
    assert cpu['volumes'][0]['read_only']
    gpu_config=json.loads(subprocess.check_output([*compose_command(True),'config','--format','json'],cwd=ROOT))
    assert gpu_config['services']['backend']['environment']['ANOMALY_DEVICE']=='cuda'
    measurements={}
    for device in ['cpu','cuda']:
        path=ROOT/f'reports/deployment/container-{device}.json'
        if not path.exists():continue
        report=json.loads(path.read_text());assert report['status']=='passed' and len(report['parity'])==15
        assert report['successful_request_errors']==0
        assert all(p['decision_matches'] for p in report['parity'])
        assert report['reliability']['concurrent_4']['bounded']
        for entry in report['records']:
            assert entry['http']['n']==30 and entry['model_only']['model_tensor_wall']['n']==30
        reliability=read(f'reports/deployment/reliability-{device}.json');assert reliability['status']=='passed'
        startup=read(f'reports/deployment/startup-{device}.json');assert startup['status']=='passed'
        measurements[device]=dict(categories=15,maximum_score_error=max(x['absolute_error'] for x in report['parity']),
            warm_http_samples=60,model_tensor_samples=60,visual_http_samples=10,png_get_samples=20,reliability='passed',startup='passed')
    assert 'cpu' in measurements
    audit=read('reports/deployment/dependency_audit.json');assert audit['status']=='completed'
    assert audit['npm']['production']['counts']['total']==0
    # Runtime software provenance is distinct from the historical fitting manifest.
    inventories={}
    code="import json,importlib.metadata as m,torch; print(json.dumps(dict(python=__import__('platform').python_version(),torch=torch.__version__,pillow=m.version('pillow'),setuptools=m.version('setuptools'),cuda_build=torch.version.cuda)))"
    for key in ['backend','backend-cuda']:
        if key in images:
            inventories[key]=json.loads(subprocess.check_output(['docker','run','--rm','--network','none','--entrypoint','python',f'fieldwork-{key}:local','-c',code],cwd=ROOT))
            assert inventories[key]['pillow']=='11.3.0' and inventories[key]['setuptools']=='78.1.1'
    browser_cases=[read(f'outputs/deployment/browser-{category}.json') for category in ['bottle','carpet']]
    result=dict(status='passed',protected_files=len(protected),backend_validation=backend,research_comparison=research,
        python_tests=tests,frontend_tests=dict(passed=unit['numPassedTests'],failed=unit['numFailedTests'],skipped=unit['numPendingTests']),
        browser_tests=browser,browser_cases=browser_cases,images=images,runtime_software=inventories,measurements=measurements,
        schema_and_compose='CPU and CUDA configuration validation passed; loopback-only frontend and private read-only backend verified',
        dependency_audit='Completed with unresolved findings; not a security approval',
        image_vulnerability_scan='Docker Scout 1.27.0 blocked: Docker Hub authentication required. No credentials requested or embedded; image CVE scan remains unverified.',
        external_publication='No image registry push or public tagged release authorized or performed')
    (ROOT/'reports/deployment/validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','protected_files','python_tests','frontend_tests','measurements']},indent=2))
    return result


if __name__=='__main__':validate()
