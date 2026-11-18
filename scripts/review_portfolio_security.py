"""Refresh public advisory/metadata lookups without rewriting release evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import httpx

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/portfolio';out.mkdir(exist_ok=True,parents=True)
    local=ROOT/'outputs/portfolio';local.mkdir(exist_ok=True,parents=True)
    runtime={}
    for name in ['deployment/requirements-runtime.txt','ci/requirements.txt']:
        for line in (ROOT/name).read_text().splitlines():
            if line and not line.startswith('#'):
                package,version=line.split('==');runtime[package]=version
    runtime.update(torch='2.11.0',torchvision='0.26.0')
    pins=sorted(runtime.items())
    with httpx.Client(timeout=90,follow_redirects=True) as client:
        result=client.post('https://api.osv.dev/v1/querybatch',json={'queries':[{'package':{'name':name,'ecosystem':'PyPI'},'version':version} for name,version in pins]})
        result.raise_for_status();findings=[]
        for (name,version),value in zip(pins,result.json()['results'],strict=True):
            for item in value.get('vulns',[]):findings.append(dict(package=name,version=version,id=item['id']))
        # The connector omits description/topics; public REST provides those read-only fields.
        response=client.get('https://api.github.com/repos/shaikanishmib/mvtec-cnn-anomaly-detection')
        response.raise_for_status();repo=response.json()
        metadata={key:repo.get(key) for key in ['full_name','description','topics','visibility','default_branch','license']}
    (out/'repository_metadata.json').write_text(json.dumps(dict(observed=metadata,changes_applied=False,
        recommended_description='Six-model industrial anomaly detection research on MVTec AD (270 evaluations), with a frozen PatchCore inspection app using PyTorch, FastAPI, React and Docker.',
        recommended_topics=['anomaly-detection','industrial-inspection','computer-vision','mvtec-ad','pytorch','patchcore','fastapi','react','typescript','docker','reproducible-research','machine-learning'],
        name_recommendation='Keep the existing repository name and URL; use Fieldwork as the README/application title.'),indent=2)+'\n',encoding='utf-8')
    npm={};command=shutil.which('npm.cmd') or shutil.which('npm');assert command
    for name,options in [('all',[]),('production',['--omit=dev'])]:
        result=subprocess.run([command,'audit','--json',*options],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8')
        (local/f'npm-audit-{name}.json').write_text(result.stdout,encoding='utf-8')
        data=json.loads(result.stdout);assert 'metadata' in data
        npm[name]=data['metadata']['vulnerabilities']
    old=json.loads((ROOT/'reports/deployment/dependency_audit.json').read_text())
    old_ids={(row['package'],row['id']) for row in old['python']['findings']}
    current_ids={(row['package'],row['id']) for row in findings}
    report=dict(status='lookup_completed',checked_utc=datetime.now(timezone.utc).isoformat(),python_packages=len(pins),
        python_advisory_records=findings,previous_runtime_records_still_reported=len(old_ids & current_ids),
        previous_runtime_records_not_returned=len(old_ids-current_ids),npm=npm,
        interpretation='OSV IDs may share aliases; absence from one lookup is not proof of remediation. Includes selected runtime and public-check pins, not complete inherited GPU/OS dependencies. No dependency versions or historical reports were changed.')
    (out/'security_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(packages=len(pins),advisory_records=len(findings),runtime_still_reported=report['previous_runtime_records_still_reported'],npm=npm,metadata=metadata),indent=2))


if __name__=='__main__':main()
