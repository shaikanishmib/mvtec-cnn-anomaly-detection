"""Reuse the established aggregation and uncertainty conventions for Notebook 02."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'scripts/summarize_basic_cnn.py').read_text()
source=source.replace('basic_cnn_frozen.json','autoencoder_frozen.json').replace('reports/basic_cnn','reports/autoencoder')
source=source.replace('01-test-','02-test-')
source=source.replace('"parameters","checkpoint_bytes"]','"parameters","checkpoint_bytes","evaluation_seconds"]')
source=source.replace('train_peak_vram_gib=result["training_peak_allocated_bytes"]/2**30,',
    'train_peak_vram_gib=result["training_peak_allocated_bytes"]/2**30,\n                       train_peak_reserved_vram_gib=run["peak_reserved_bytes"]/2**30,')
source=source.replace('"total_training_seconds":float(table.training_seconds.sum()),',
    '"total_training_seconds":float(table.training_seconds.sum()),\n             "total_evaluation_seconds":float(table.evaluation_seconds.sum()),\n             "peak_training_reserved_vram_gib":float(table.train_peak_reserved_vram_gib.max()),')
source=source.replace('            assert sha256(ROOT/run["history"])==run["history_sha256"]',
    '            history_values=run["history"]\n            run["history"]=f"reports/autoencoder/runs/{run[\'experiment_id\']}-history.csv"\n            history=pd.read_csv(ROOT/run["history"])\n            assert len(history)==freeze["configuration"]["epochs"]\n            np.testing.assert_allclose(history.train_loss,[h["train_loss"] for h in history_values])')
source=source.replace('    print(json.dumps(summary,indent=2))', '''    baseline=json.loads((ROOT/'reports/basic_cnn/summary.json').read_text())
    comparison=pd.DataFrame({key:{'Basic CNN':baseline['macro'][key]['mean'],
        'Autoencoder':summary['macro'][key]['mean'],
        'AE minus CNN':summary['macro'][key]['mean']-baseline['macro'][key]['mean']} for key in METRICS}).T
    comparison.to_csv(output/'cnn_macro_comparison.csv')
    cnn=pd.read_csv(ROOT/'reports/basic_cnn/all_category_means.csv').set_index('category')
    ae=pd.read_csv(output/'all_category_means.csv').set_index('category')
    paired=ae[[k+'_mean' for k in METRICS]].join(cnn[[k+'_mean' for k in METRICS]],lsuffix='_ae',rsuffix='_cnn')
    paired.to_csv(output/'cnn_category_comparison.csv')
    costs=['model_p50_ms','model_p95_ms','end_to_end_p50_ms','end_to_end_p95_ms',
           'train_peak_vram_gib','checkpoint_bytes','training_seconds']
    cnn_runs=pd.read_csv(ROOT/'reports/basic_cnn/all_runs.csv')
    cost_comparison=table.groupby('category')[costs].mean().join(
        cnn_runs.groupby('category')[costs].mean(),lsuffix='_ae',rsuffix='_cnn')
    cost_comparison.to_csv(output/'cnn_category_cost_comparison.csv')
    print(json.dumps(summary,indent=2))''')
(root/'scripts/summarize_autoencoder.py').write_text(source)
