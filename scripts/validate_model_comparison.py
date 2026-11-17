"""Validate compact analysis plus previously completed deep identity audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import json
import numpy as np
import pandas as pd
from anomaly_detection.model_comparison import validate_coverage,category_macro,paired_statistics,METRICS,MODELS,NAMES
from build_model_comparison import digest
from validate_notebook import validate


def validate_comparison(require_notebook=True):
    d=ROOT/'reports/final_comparison'
    f=pd.read_csv(d/'all_runs.csv',float_precision='round_trip')
    validate_coverage(f)
    macro=pd.read_csv(d/'model_macro.csv',index_col='model',float_precision='round_trip')
    assert np.allclose(category_macro(f),macro,atol=1e-14,rtol=0)
    # Reconcile derived summaries and the literal published nine-metric table.
    category=pd.read_csv(d/'category_summary.csv',float_precision='round_trip').set_index(['model','category'])
    expected_category=f.groupby(['model','category'])[list(METRICS)].mean()
    assert np.allclose(category[list(METRICS)],expected_category,atol=1e-14,rtol=0)
    seed=pd.read_csv(d/'seed_summary.csv',float_precision='round_trip').set_index(['model','seed'])
    assert np.allclose(seed[list(METRICS)],f.groupby(['model','seed'])[list(METRICS)].mean(),atol=1e-14,rtol=0)
    table=(d/'nine_metric_table.md').read_text(encoding='utf-8')
    narrative=(d/'report.md').read_text(encoding='utf-8')
    assert table.strip() in narrative
    for model in MODELS:
        row='| '+NAMES[model]+' | '+' | '.join(f'{category_macro(f).loc[model,k]:.4f}' for k in METRICS)+' |'
        assert row in table
    cost=pd.read_csv(d/'compute_summary.csv',float_precision='round_trip').set_index('model')
    for model in MODELS:
        records=f[f.model.eq(model)]
        assert np.isclose(cost.loc[model,'training_seconds'],records.training_seconds.sum(),atol=1e-9)
        assert np.isclose(cost.loc[model,'training_hours'],records.training_seconds.sum()/3600,atol=1e-14)
        for scope in ('model','end_to_end'):
            for percentile in ('p50','p95'):
                key=f'{scope}_{percentile}_ms'
                assert np.isclose(cost.loc[model,key],records[key].median(),atol=1e-12)
    stats=paired_statistics(f)
    saved=pd.read_csv(d/'paired_comparisons.csv',float_precision='round_trip')
    for col in ['difference','ci_low','ci_high','p_sign_randomization','p_holm_135']:
        assert np.allclose(stats[col],saved[col],atol=1e-14,rtol=0)
    for _,row in f.iterrows():
        source=ROOT/row.source_result
        assert digest(source)==row.source_result_sha256
        original=json.loads(source.read_text())
        assert all(row[k]==original[k] for k in METRICS)
        assert digest(ROOT/row.predictions_path)==row.predictions_sha256
    sources=json.loads((d/'source_provenance.json').read_text())
    # Large files were deeply hashed once by the builder; check retained size/stat
    # against that local audit cache to avoid duplicate multi-GB verification.
    cache=json.loads((ROOT/'outputs/final_comparison/hash_cache.json').read_text())
    for entry in sources:
        p=ROOT/entry['path'];st=p.stat()
        assert st.st_size==entry['bytes']
        if st.st_size<1_000_000:
            assert digest(p)==entry['sha256'],entry['path']
        else:
            assert cache[entry['path']]==dict(stat=f'{st.st_size}:{st.st_mtime_ns}',sha256=entry['sha256'])
    protection=json.loads((d/'historical_protection.json').read_text())
    for evidence in json.loads((d/'visual_evidence_provenance.json').read_text()):
        assert digest(ROOT/evidence['path'])==evidence['sha256']
    for asset in json.loads((d/'dependency_audit.json').read_text())['assets']:
        assert digest(ROOT/asset['path'])==asset['sha256']
    assert all(digest(ROOT/p)==h for p,h in protection.items()),'Historical file changed'
    report=dict(status='passed',records=270,original_sources=len(sources),protected_historical_files=len(protection),
                statistics=135,large_artifact_method='deep SHA256 builder audit plus unchanged size/mtime cache; small files rehashed',
                no_training_or_inference=True,report_and_derived_tables='source-consistent',dependency_assets=3)
    if require_notebook:
        report['notebook']=validate(ROOT/'notebooks/07_model_comparison.ipynb',require_executed=True)
    return report


if __name__=='__main__':
    result=validate_comparison()
    (ROOT/'reports/final_comparison/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
