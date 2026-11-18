"""Asset-free documentation, source-schema and CI contracts; never run inference."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
from urllib.parse import unquote
import yaml

ROOT=Path(__file__).resolve().parents[1]
DOCUMENTS=['README.md','docs/index.md','docs/project_overview.md','docs/reproducibility.md',
    'docs/licensing_and_attribution.md','SECURITY.md','docs/deployment_guide.md',
    'docs/backend_api.md','docs/frontend_guide.md','docs/patchcore_inference.md']


def headings(text):
    return {re.sub(r'[^\w\- ]','',s.lower()).strip().replace(' ','-') for s in re.findall(r'^#{1,6} (.+)$',text,re.M)}


def check_links(root,documents):
    count=0
    for name in documents:
        path=root/name;text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if re.match(r'^[a-z]+://',target) or target.startswith('mailto:'):continue
            filename,_,anchor=target.partition('#')
            resolved=(path.parent/unquote(filename)).resolve() if filename else path.resolve()
            assert resolved.is_relative_to(root.resolve()) and resolved.exists(),(name,target)
            if anchor and resolved.suffix=='.md':assert unquote(anchor) in headings(resolved.read_text(encoding='utf-8')),(name,target)
            count+=1
    return count


def check_workflow(workflow):
    assert workflow['permissions']=={'contents':'read'}
    assert set(workflow['on'])=={'push','pull_request','workflow_dispatch'}
    assert set(workflow['jobs'])=={'python-public','frontend'}
    for job in workflow['jobs'].values():
        assert job['runs-on']=='ubuntu-24.04' and int(job['timeout-minutes'])<=20
        for step in job['steps']:
            if 'uses' in step:assert re.fullmatch(r'actions/[\w-]+@[0-9a-f]{40}',step['uses'])
            if step.get('uses','').startswith('actions/checkout@'):assert step['with']['persist-credentials']=='false'
    text=json.dumps(workflow)
    assert 'secrets.' not in text and 'pull_request_target' not in text


def check(root=ROOT,require_asset_free=False):
    links=check_links(root,DOCUMENTS)
    readme=(root/'README.md').read_text(encoding='utf-8')
    names={'basic_cnn':'Basic CNN','autoencoder':'Convolutional AE','patchcore':'Simplified PatchCore',
        'supersimplenet':'SuperSimpleNet Lite','padim':'PaDiM','efficientad':'EfficientAD-S reduced-budget'}
    with (root/'reports/final_comparison/model_macro.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.reader(stream));assert len(rows)==7
        for row in rows[1:]:
            expected='| '+names[row[0]]+' | '+' | '.join(f'{float(value):.4f}' for value in row[1:])+' |'
            assert expected in readme,row[0]
    with (root/'reports/final_comparison/all_runs.csv').open(encoding='utf-8',newline='') as stream:records=list(csv.DictReader(stream))
    assert len(records)==270 and len({(r['model'],r['category'],r['seed']) for r in records})==270
    assert set(r['model'] for r in records)==set(names)
    manifest_path=root/'configs/patchcore_deployment_v1.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='58be2eafb16ac012ac18e0feb69f11b2f510e7417464eaba5bd9797b423272fa'
    manifest=json.loads(manifest_path.read_text());assert len(manifest['categories'])==15 and manifest['seed']==2026
    assert {(r['model'],r['category'],int(r['seed'])) for r in records}=={
        (model,category,seed) for model in names for category in manifest['categories'] for seed in [2026,2027,2028]}
    if require_asset_free:
        assert not (root/manifest['extractor']['path']).exists()
        assert not any((root/item['bank']['path']).exists() for item in manifest['categories'].values())
    code_cells=0
    notebooks=sorted((root/'notebooks').glob('*.ipynb'));assert len(notebooks)==8
    for notebook in notebooks:
        data=json.loads(notebook.read_text(encoding='utf-8'));assert data['nbformat']==4
        for cell in data['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),notebook.name,'exec');code_cells+=1
                assert cell['execution_count'] is not None
                assert not any(item['output_type']=='error' for item in cell['outputs'])
    schema=json.loads((root/'reports/backend/openapi.json').read_text())
    types=(root/'frontend/src/api/types.ts').read_text()
    fields=re.findall(r"'([^']+)'",re.search(r'PREDICTION_FIELDS = \[(.*?)\]',types).group(1))
    assert set(fields)==set(schema['components']['schemas']['PredictionResponse']['properties'])
    workflow=yaml.load((root/'.github/workflows/source-quality.yml').read_text(),Loader=yaml.BaseLoader)
    check_workflow(workflow)
    for name in ['ci/requirements.txt','deployment/requirements-runtime.txt']:
        assert all(re.fullmatch(r'[\w-]+==\d+\.\d+(?:\.\d+)?',s) for s in (root/name).read_text().splitlines() if s and not s.startswith('#'))
    return dict(status='passed',relative_links=links,canonical_records=270,macro_rows=6,notebooks=8,saved_code_cells=code_cells,
        api_fields=len(fields),workflow='parsed, least privilege, immutable action refs, no secrets/private assets',
        private_assets_absent=True if require_asset_free else None,project_license_declared=(root/'LICENSE').is_file(),model_inference_executed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-asset-free',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=check(require_asset_free=args.require_asset_free)
    if args.output:
        args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
