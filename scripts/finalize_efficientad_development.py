"""Reconcile completed development artifacts into a locked reduced protocol."""
from pathlib import Path
import json
import subprocess
from datetime import datetime,timezone
from anomaly_detection.data import sha256,CATEGORIES
from anomaly_detection.tracking import record

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/efficientad/development'


def main():
    import sys
    assert Path(sys.executable).resolve()==ROOT/'.venv/Scripts/python.exe'
    plan=json.loads((ROOT/'configs/efficientad_development.json').read_text())
    gate=json.loads((OUT/'feasibility.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    launch=json.loads((OUT/'launch-control.json').read_text())
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('06-dev-*-v1.json'))]
    path=ROOT/'configs/efficientad_reduced_frozen.json'
    assert not path.exists(),'Preserve existing reduced protocol'
    frozen=dict(schema=1,name='EfficientAD-S reduced-budget variant',
        development_gate=gate['recommendation'],categories=list(CATEGORIES),seeds=[2026,2027,2028],
        epochs=selection['selected']['epochs'],score=selection['selected']['score'],batch_size=1,size=256,
        teacher=dict(path='checkpoints/efficientad/assets/teacher_small.pth',source='nelson1425/EfficientAD',revision='fcab5146f84ae17597044ad5ddf1656ccf805401',sha256=plan['teacher_sha256'],license='repository Apache-2.0; separate weight license unconfirmed'),
        teacher_normalization='all fitting normals only; channel population moments',
        external_penalty=False,hard_quantile=.999,optimizer=plan['optimizer'],lr_policy=plan['lr_policy'],
        augmentation=plan['augmentation'],precision=plan['precision'],map_policy=plan['map_policy'],
        threshold_policy=plan['threshold_policy'],inference_execution='eager; graph diagnostic not adopted',
        score_serialization='exact float32 scores promoted to float64 before 17-digit CSV; round_trip reader',
        plan_sha256=sha256(ROOT/'configs/efficientad_development.json'),selection_sha256=sha256(OUT/'selection.json'),
        partition_sha256={part:sha256(ROOT/f'reports/partitions/seed2026-{part}.csv') for part in ('fit','development','calibration','test')},
        source_sha256={p:sha256(ROOT/p) for p in ('src/anomaly_detection/efficientad.py','src/anomaly_detection/efficientad_research.py','src/anomaly_detection/efficientad_calibration.py','scripts/develop_efficientad.py')},
        package_lock_sha256=sha256(ROOT/'requirements.lock.txt'),
        final_models_started=0,final_evaluator_status='not executed; must preserve native metric conventions and validate before final reporting')
    path.write_text(json.dumps(frozen,indent=2)+'\n')
    identity=sha256(path)
    manifest=[]
    for artifact in sorted(OUT.iterdir()):
        if artifact.is_file() and artifact.suffix in ('.json','.csv') and artifact.name not in ('validation.json','artifact_manifest.json','reconciliation.json'):
            manifest.append(dict(path=artifact.relative_to(ROOT).as_posix(),sha256=sha256(artifact),bytes=artifact.stat().st_size))
    for run in runs: manifest.append(run['checkpoint'])
    manifest.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=identity,bytes=path.stat().st_size))
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    calibration_seconds=sum(json.loads((OUT/f'calibration-{c}.json').read_text())['elapsed_seconds'] for c in plan['categories'])
    probe_seconds=json.loads((OUT/'metric-cost-probe.json').read_text())['seconds']
    accounting=dict(training_seconds=sum(r['training_seconds'] for r in runs),
        fit_pipeline_seconds=sum(r['elapsed_seconds'] for r in runs),
        latency_audit_seconds=json.loads((OUT/'latency.json').read_text())['elapsed_seconds'],
        exploratory_launch_control_seconds=launch['elapsed_seconds'],
        calibration_seconds=calibration_seconds,metric_probe_seconds=probe_seconds,
        initial_data_load_seconds=sum(gate['data_load_seconds'].values()),
        calibration_precision_audit_seconds=json.loads((OUT/'calibration-precision-audit.json').read_text())['elapsed_seconds'],
        caveat='Training is synchronized CUDA-active wall time, not energy or exclusive GPU busy-time. Fit pipeline includes training/preparation/diagnostics/checkpoint checks. Other fields are disjoint additional work; global phase wall also includes orchestration/documentation/tests.')
    accounting['measured_compute_pipeline_total_seconds']=sum(accounting[k] for k in ('fit_pipeline_seconds','latency_audit_seconds','exploratory_launch_control_seconds','calibration_seconds','metric_probe_seconds','initial_data_load_seconds','calibration_precision_audit_seconds'))
    reconciliation=dict(status='development complete',frozen_sha256=identity,gate=gate['recommendation'],
        development_fits=4,final_models=0,compute_accounting=accounting,utc=datetime.now(timezone.utc).isoformat())
    (OUT/'reconciliation.json').write_text(json.dumps(reconciliation,indent=2)+'\n')
    event=json.loads((ROOT/'configs/experiment_template.json').read_text())
    event.update(experiment_id='06-development-selection-v1-result',model_family=frozen['name'],categories=plan['categories'],
        hypothesis='A predefined reduced epoch budget is eligible and fits a conservative 12-hour 45-model estimate',
        hyperparameters=frozen,environment=dict(git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True),
        training=accounting,metrics=dict(candidates=selection['candidates'],estimated_total_hours=gate['estimated_conservative_total_elapsed_hours']),
        calibration=plan['threshold_policy'],paths=dict(frozen_configuration=path.relative_to(ROOT).as_posix(),frozen_sha256=identity),
        interpretation='Selection used development/fitting proxies only; reserved-normal calibration was subsequent and did not change the rule. Synthetic results do not imply real anomaly quality.',
        decision=gate['recommendation']+' for reduced-budget protocol only; no final model launched',status='completed')
    record(ROOT/'experiments',event)
    lines=['# EfficientAD bounded development: measured results','',
        'Four predefined fresh fits; no official test pixels/masks or final model evaluations. All results refer to the EfficientAD-S reduced-budget variant. The pilot is unchanged.','',
        '| Category | Epochs | Steps | Train s | Hard probe before / after | AE probe before / after | Combined proxy AUROC | Local proxy AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        lines.append(f"| {run['category']} | {run['epochs']} | {run['steps']} | {run['training_seconds']:.2f} | {run['probe_before']['hard']:.3f} / {run['probe_after']['hard']:.3f} | {run['probe_before']['autoencoder']:.4f} / {run['probe_after']['autoencoder']:.4f} | {run['proxy']['combined_max']['auroc']:.4f} | {run['proxy']['local_max']['auroc']:.4f} |")
    selected=selection['selected']
    lines.extend(['',f"Selected **{selected['epochs']} epochs, {selected['score']}**, macro fit-proxy AUROC {selected['macro_proxy_auroc']:.4f}, eligible={selection['eligible']}. Selection followed the declared .65 category floor and .02 shorter-budget/combined-map preference. Calibration was opened only after the selection file was locked.",'',
        'The three-epoch bottle combined score is weak on block edits (AUROC .3359), despite decreasing losses. Nine-epoch bottle combined proxy AUROC rises to .9987. These are actual synthetic-edit observations, not real defect findings. Proxy AP uses 48 edited and 16 clean images (random ranking reference .75); all edits share fitting backgrounds. AE contribution is inference-only ablation, not a separate no-AE training experiment.','',
        '## Latency diagnosis',''])
    latency=json.loads((OUT/'latency.json').read_text())
    lines.extend([f"Unchanged pilot model, deterministic FP32, 50 warmups per repeat, 3 x 100 resident batch-one measurements: wall **{latency['resident_wall']['p50_ms']:.2f}/{latency['resident_wall']['p95_ms']:.2f} ms p50/p95**, CUDA event **{latency['resident_gpu_event']['p50_ms']:.2f}/{latency['resident_gpu_event']['p95_ms']:.2f} ms**. H2D **{latency['host_to_device']['p50_ms']:.3f}/{latency['host_to_device']['p95_ms']:.3f} ms**. Decode-through-CPU-map/score **{latency['end_to_end']['p50_ms']:.2f}/{latency['end_to_end']['p95_ms']:.2f} ms** (100). Both synchronization and event streams are retained in latency.json.",'',
        'Sampled GPU P0/clocks were stable and CPU process utilization was near one core. Burst execution did not materially lower latency. CUDA events include gaps between launches and do not isolate pure kernel computation; OS scheduling is a possible explanation, not established causation. No clock/power settings changed.',''])
    if launch['graph_status']=='measured':
        lines.extend([f"Exploratory paired selected-checkpoint control (300 measurements each): eager wall **{launch['eager']['wall']['p50_ms']:.2f}/{launch['eager']['wall']['p95_ms']:.2f} ms**; fixed-shape CUDA graph **{launch['graph']['wall']['p50_ms']:.2f}/{launch['graph']['wall']['p95_ms']:.2f} ms**. Capture/transfer excluded; all output equality checks passed. This supports a dispatch/scheduling contribution if graph replay is faster; graph execution is not adopted in the frozen configuration or conservative estimate.",''])
    else: lines.extend(['CUDA graph control unavailable; recorded failure retained in launch-control.json. Eager timing remains authoritative.',''])
    lines.extend(['## Normal-only threshold feasibility','',
        '| Category | Calibration n | Higher 95% threshold | Nominal FPR | Observed calibration FPR | IID future FPR 95% interval | Bootstrap threshold 95% range |',
        '|---|---:|---:|---:|---:|---|---|'])
    for category in plan['categories']:
        cal=json.loads((OUT/f'calibration-{category}-corrected.json').read_text())
        f=cal['future_fpr_order_statistic_95']; b=cal['bootstrap_threshold_p025_p50_p975']
        interval=f'[{f[0]:.4f}, {f[1]:.4f}]' if f is not None else 'not applicable: tied scores'
        lines.append(f"| {category} | {cal['calibration_distribution']['n']} | {cal['threshold']:.6f} | .05 | {cal['observed_calibration_fpr']:.4f} | {interval} | [{b[0]:.6f}, {b[2]:.6f}] |")
    lines.extend(['','A serialization audit found decimal float32 CSV values compared as float64 could inflate tied-score exceedances. Original reports are retained; corrected reports promote exact float32 values before 17-digit export and use a round-trip reader. No weights, thresholds or selection changed. Bottle calibration has one unique score across 31 images; the max response floor is not evidence that all feature maps collapsed. Its zero-width bootstrap range does not imply certainty, and the continuous Beta interval is marked inapplicable. Exact-score normal maps and peak positions are diagnosed in the corrected report.',''])
    lines.extend(['','Policy stays higher quantile with strict score > threshold, per category and freshly fitted model. Calibration exceedance intervals are descriptive because the same normals fit the threshold. Order-statistic intervals assume IID continuous scores; image dependence or discrete ties weaken that model. Small tails, especially toothbrush n=9 in a later benchmark, cannot guarantee nominal 5% FPR. No configuration or policy was changed from these calibration observations.','',
        '## Compute accounting and final feasibility','',
        f"Synchronized training total: **{accounting['training_seconds']:.2f} s**. Measured compute pipeline total including full fit diagnostics, latency controls, calibration, decode and synthetic native-metric probe: **{accounting['measured_compute_pipeline_total_seconds']:.2f} s**. Training is CUDA-active elapsed wall time, not an exclusive GPU utilization/energy integral. Notebook/tests/documentation add CPU wall time outside this counter.",'',
        f"Peak process working set: **{max(r['process_peak_working_set_bytes'] for r in runs)/1024**3:.3f} GiB** (OS cumulative process peak); peak CUDA allocated **{max(r['peak_gpu_allocated_bytes'] for r in runs)/1024**2:.2f} MiB**, reserved **{max(r['peak_gpu_reserved_bytes'] for r in runs)/1024**2:.2f} MiB** (training through diagnostics and reload). Checkpoint bytes: {runs[0]['checkpoint']['bytes']:,}.",'',
        f"Requested coverage stays **15 categories x 3 seeds = 45 fresh final models**. Projected raw training **{gate['estimated_training_hours_no_margin']:.2f} h**; conservative complete sequential workload **{gate['estimated_conservative_total_elapsed_hours']:.2f} h**, against 12 h. The entire estimate receives a factor-two margin and includes CPU work, so it is deliberately stricter than GPU-only training time. It is not a measured final runtime.",'',
        '| Category | Train/model s | Prepare/model s | Map scale/model s | Calibrate/model s | Infer/model s | Native metrics/model s | 3-seed total with margin h |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for row in gate['workload']:
        c=row['per_model_seconds']
        lines.append(f"| {row['category']} | {c['training']:.1f} | {c['preparation']:.1f} | {c['map_scaling']:.1f} | {c['calibration']:.1f} | {c['inference']:.1f} | {c['native_metrics']:.1f} | {row['conservative_three_seed_seconds']/3600:.3f} |")
    lines.extend(['','Add measured checkpoint serialization/reload and 60 s/model fixed overhead before applying the factor-two margin. The fixed allowance covers further latency sweeps, bootstrap/artifact handling and validation; actual final native-metric and sustained thermal cost remain uncertain. Native-metric cost was measured on synthetic truth/scores, then scaled by native pixel count and N log N; no official masks were loaded.','',
        f"Recommendation: **{gate['recommendation']}** for the locked reduced protocol only. Frozen identity: `{identity}`. This phase launches **zero** final models; no seed/category reductions are hidden. If actual resumed final costs exceed the target, stop and report before changing protocol.",'',
        '## Limitations and remaining work','',
        'Teacher is the unofficial nelson1425/EfficientAD PDN-S at revision fcab5146f84ae17597044ad5ddf1656ccf805401, SHA256 b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337. Repository Apache-2.0; separate weight licensing and original training history remain unconfirmed. Full audit: docs/efficientad_fidelity_audit.md. No external-image penalty or paper-length optimization; one development seed and two categories. All final nine metrics, real-defect failure analysis and six-model comparison are unavailable. Notebook 07 is not started. Tests and validator evidence are recorded separately in the milestone status and development validation report.',''])
    (ROOT/'docs/efficientad_development_results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(reconciliation,indent=2))


if __name__=='__main__': main()
