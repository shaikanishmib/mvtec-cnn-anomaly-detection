"""Audit release Python pins via OSV and frontend lock via npm; no installations."""
from pathlib import Path
import json
import subprocess
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    local=ROOT/'outputs/deployment'; local.mkdir(exist_ok=True,parents=True)
    out=ROOT/'reports/deployment'; out.mkdir(exist_ok=True,parents=True)
    pins=[]
    for line in (ROOT/'deployment/requirements-runtime.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name,version=line.split('=='); pins.append((name,version))
    pins += [('torch','2.11.0'),('torchvision','0.26.0')]
    with httpx.Client(timeout=90,follow_redirects=True) as client:
        response=client.post('https://api.osv.dev/v1/querybatch',json={'queries':[{'package':{'name':name,'ecosystem':'PyPI'},'version':version} for name,version in pins]})
        response.raise_for_status(); batch=response.json()['results']
        findings=[]
        for (name,version),result in zip(pins,batch,strict=True):
            for vuln in result.get('vulns',[]):
                info=client.get('https://api.osv.dev/v1/vulns/'+vuln['id']); info.raise_for_status(); info=info.json()
                fixed=sorted({e['fixed'] for a in info.get('affected',[]) if a.get('package',{}).get('name','').lower().replace('_','-')==name.lower().replace('_','-') for r in a.get('ranges',[]) for e in r.get('events',[]) if 'fixed' in e})
                findings.append(dict(package=name,version=version,id=vuln['id'],aliases=info.get('aliases',[]),summary=info.get('summary',''),fixed_versions=fixed))
    npm={}
    for key,arguments in [('all',[]),('production',['--omit=dev'])]:
        result=subprocess.run(['npm.cmd','audit','--json',*arguments],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8')
        (local/f'npm-audit-{key}.json').write_text(result.stdout,encoding='utf-8')
        data=json.loads(result.stdout)
        if 'metadata' not in data:raise RuntimeError('npm advisory lookup failed; inspect local output')
        npm[key]=dict(counts=data['metadata']['vulnerabilities'],packages=[dict(name=n,severity=v['severity'],fix_available=v['fixAvailable'],advisories=[i for i in v['via'] if isinstance(i,dict)]) for n,v in data.get('vulnerabilities',{}).items()])
    report=dict(status='completed',timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),python=dict(source='OSV querybatch, exact Linux runtime pins plus upstream torch/vision versions; does not audit OS libraries or unknown vulnerabilities',packages=len(pins),findings=findings),npm=npm)
    (out/'dependency_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
