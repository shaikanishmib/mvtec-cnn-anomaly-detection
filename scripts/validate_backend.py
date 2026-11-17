"""Saved API evidence, artifact identities and prior research/notebook protection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import importlib.metadata
import json
import pandas as pd
import nbformat
from anomaly_detection.backend.app import create_app
from anomaly_detection.backend.contracts import Settings,PIN
from anomaly_detection.data import CATEGORIES
from anomaly_detection.inference.patchcore import sha256,verified_path
from validate_notebook import validate as validate_notebook


def notebook_evidence():
    result={}
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        # Historical notebooks have different section counts. Do not impose the
        # later 20-section template on their already validated immutable sources.
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        cells=[c for c in nb.cells if c.cell_type=='code']
        for i,cell in enumerate(cells):
            compile(cell.source,f'{path.name}:cell-{i}','exec')
            assert cell.execution_count is not None
            assert not any(o.output_type=='error' for o in cell.outputs)
            if cell.metadata.get('contains_dataset_images'):assert not cell.outputs
        result[path.name]=dict(code_cells=len(cells),executed_code_cells=len(cells),
            scope='nbformat, syntax, saved execution/error/image-output checks plus protected historical byte identity')
    return result


def validate():
    out=ROOT/'reports/backend'
    protected=json.loads((out/'research_protection.json').read_text())
    for path,expected in protected.items():assert sha256(ROOT/path)==expected,path
    manifest=ROOT/'configs/patchcore_deployment_v1.json'
    assert sha256(manifest)==PIN
    m=json.loads(manifest.read_text());assert set(m['categories'])==set(CATEGORIES) and m['seed']==2026
    verified_path(ROOT,m['extractor']);verified_path(ROOT,m['frozen_policy'])
    for category,entry in m['categories'].items():
        verified_path(ROOT,entry['bank']);cal=json.loads(verified_path(ROOT,entry['calibration']).read_text())
        assert entry['thresholds']['image']==cal['image_threshold']
        assert entry['thresholds']['pixel']==cal['pixel_threshold']
    integration=json.loads((out/'integration.json').read_text());assert integration['status']=='passed'
    frame=pd.read_csv(out/'parity.csv');assert len(frame)==30
    assert set(frame.device)=={'cpu','cuda'} and set(frame.category)==set(CATEGORIES)
    assert frame.groupby(['category','device']).size().eq(1).all()
    assert frame.absolute_score_error.max()<1e-6 and (frame.decision==frame.direct_decision).all()
    assert frame.heatmap_overlay_equal.all()
    assert json.loads((out/'server_smoke.json').read_text())['status']=='passed'
    schema=create_app().openapi()
    expected=['/health/live','/health/ready','/api/v1/models','/api/v1/categories','/api/v1/model-info',
              '/api/v1/predict','/api/v1/predictions/{prediction_id}/heatmap','/api/v1/predictions/{prediction_id}/overlay']
    assert set(schema['paths'])==set(expected)
    assert 'image/png' in schema['paths'][expected[-1]]['get']['responses']['200']['content']
    schema_path=out/'openapi.json'
    if schema_path.exists():assert json.loads(schema_path.read_text())==schema
    else:schema_path.write_text(json.dumps(schema,indent=2)+'\n')
    notebooks=notebook_evidence()
    for line in (ROOT/'requirements-backend.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name,version=line.split('==');assert importlib.metadata.version(name)==version,name
    return dict(status='passed',protected_files=len(protected),notebooks=notebooks,selected_banks_verified=15,
        frozen_manifest_sha256=PIN,api_category_device_checks=30,maximum_api_direct_error=float(frame.absolute_score_error.max()),
        thresholds_decisions_and_png_arrays_match=True,openapi_paths=len(schema['paths']),dependencies='all pins match installed versions',
        inference_and_research_source_changed=False)


if __name__=='__main__':
    result=validate()
    (ROOT/'reports/backend/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
