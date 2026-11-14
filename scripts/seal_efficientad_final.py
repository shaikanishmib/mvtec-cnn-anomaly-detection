"""Seal verified publication identities; no training or metric recomputation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import nbformat
from anomaly_detection.efficientad_final import ROOT,REPORT,FREEZE,atomic_json,artifact,utc


def main():
    target=REPORT/'publication_manifest.json';assert not target.exists(),'Preserve existing publication seal'
    validation=json.loads((REPORT/'validation.json').read_text());assert validation['status']=='passed' and validation['final_models']==45
    summary=json.loads((ROOT/'reports/efficientad/summary.json').read_text());assert summary['models']==45
    suites=list(ET.parse(REPORT/'pytest.xml').getroot().iter('testsuite'))
    tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert tests['tests']==121 and tests['failures']==tests['errors']==tests['skipped']==0
    # Stabilize new text publication bytes to the explicit Git LF attributes.
    for relative in ('notebooks/06_efficientad.ipynb','docs/efficientad_final_protocol.md','docs/efficientad_final_recovery.md',
                     'docs/efficientad_final_results.md','docs/efficientad_visual_review.md','reports/efficientad/final/pytest.xml'):
        p=ROOT/relative;p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    full_path=ROOT/'outputs/efficientad/final/06_efficientad.full.executed.ipynb'
    full=nbformat.read(full_path,4);nbformat.validate(full)
    assert all(c.execution_count is not None for c in full.cells if c.cell_type=='code')
    for c in full.cells:
        if c.cell_type=='code':compile(c.source,'efficientad-full-notebook','exec')
    manifest=json.loads((REPORT/'benchmark_manifest.json').read_text())
    manifest.update(reporting_complete=True,final_validation='passed',message='All 45 runs, reporting, tests and deep notebook/artifact validation complete; publication package sealed.',updated_at_utc=utc())
    atomic_json(REPORT/'benchmark_manifest.json',manifest)
    verification=dict(status='passed',freeze_sha256=FREEZE,pilot_runs=1,development_runs=4,final_models=45,
        pytest=tests,pytest_seconds=sum(float(s.attrib.get('time',0)) for s in suites),deep_validation=validation,
        full_local_notebook=dict(path=full_path.relative_to(ROOT).as_posix(),cells=len(full.cells),validated=True,dataset_outputs_retained_locally=True),
        final_pipeline_hours=summary['total_run_pipeline_seconds']/3600,training_hours=summary['total_training_seconds']/3600,
        final_resource_budget_hours=12,configuration_changed_after_test=False,next_milestone='Notebook 07 — Final Model Comparison (not started)')
    atomic_json(REPORT/'verification_summary.json',verification)
    entries=json.loads((ROOT/'reports/efficientad/artifact_manifest.json').read_text())
    seen={v['path'] for v in entries};baseline=json.loads((REPORT/'baseline_preservation.json').read_text())['files']
    candidates=list((ROOT/'reports/efficientad').rglob('*'))
    candidates+=list((ROOT/'scripts').glob('*efficientad*.py'))
    candidates+=[ROOT/'src/anomaly_detection/efficientad_final.py',ROOT/'tests/test_efficientad_final.py',ROOT/'notebooks/06_efficientad.ipynb',full_path]
    candidates+=list((ROOT/'docs').glob('efficientad_final*.md'))+[ROOT/'docs/efficientad_visual_review.md']
    for p in candidates:
        if not p.is_file() or p==target:continue
        relative=p.relative_to(ROOT).as_posix()
        if relative in baseline or relative in seen:continue
        if p.suffix not in ('.json','.csv','.xml','.py','.md','.ipynb'):continue
        entries.append(artifact(p));seen.add(relative)
    atomic_json(target,dict(status='sealed',freeze_sha256=FREEZE,created_at_utc=utc(),artifacts=entries,
        scope='All completed run artifacts plus new reports/source/notebooks; weights/maps/full image-output notebook remain local and ignored. Prior pilot/development artifacts are independently protected by baseline hashes.'))
    print(json.dumps(dict(status='sealed',identities=len(entries),tests=tests,notebook_bytes=(ROOT/'notebooks/06_efficientad.ipynb').stat().st_size),indent=2))


if __name__=='__main__':main()
