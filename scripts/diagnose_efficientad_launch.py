"""Bounded exploratory host-launch control; no training/configuration selection."""
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
from anomaly_detection.efficientad import load_checkpoint
from anomaly_detection.efficientad_research import configure,read_normals,normalized,percentiles,gpu_snapshot
from anomaly_detection.tracking import record
from anomaly_detection.data import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/efficientad/development'


def main():
    configure(ROOT)
    path=OUT/'launch-control.json'
    assert not path.exists(),'Reuse completed launch control'
    event=json.loads((ROOT/'configs/experiment_template.json').read_text())
    event.update(experiment_id='06-launch-control-v1-start',model_family='EfficientAD-S reduced-budget variant',
        categories=['bottle'],hypothesis='Test host dispatch contribution with equivalent fixed-shape CUDA graph replay',
        hyperparameters=dict(warmup=50,repeats=3,iterations=100),status='started',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda),
        training=dict(source_sha256=sha256(Path(__file__)),selection_sha256=sha256(OUT/'selection.json')))
    record(ROOT/'experiments',event)
    selection=json.loads((OUT/'selection.json').read_text())['selected']
    model,meta=load_checkpoint(ROOT/f"checkpoints/efficientad/development/06-dev-bottle-e{selection['epochs']}-seed2026-v1.pt",'cuda')
    _,images=read_normals(ROOT,'bottle','development')
    x=normalized(images[0]).cuda(); telemetry=[gpu_snapshot()]
    result=dict(purpose='Exploratory execution-only control after selection, no architecture/weight/score change. Graph replay is not silently adopted for final timing or feasibility.',
                selected_checkpoint=meta,shape=list(x.shape),precision='deterministic FP32',warmup=50,repeats=3,iterations_per_repeat=100)
    start=time.perf_counter()
    with torch.inference_mode():
        for _ in range(50): model(x)
        torch.cuda.synchronize(); eager_outputs=tuple(y.clone() for y in model(x))
        def measure(fn):
            rows=[]
            for repeat in range(3):
                for i in range(100):
                    torch.cuda.synchronize(); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    t=time.perf_counter(); a.record(); fn(); b.record(); b.synchronize()
                    rows.append(dict(repeat=repeat,index=i,wall_ms=(time.perf_counter()-t)*1000,gpu_ms=a.elapsed_time(b)))
            return dict(wall=percentiles([r['wall_ms'] for r in rows]),gpu_event=percentiles([r['gpu_ms'] for r in rows]),samples=rows)
        result['eager']=measure(lambda:model(x))
        try:
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10): model(x)
            torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): captured=model(x)
            graph.replay(); torch.cuda.synchronize()
            result['graph_max_abs_error']=[float((a-b).abs().max()) for a,b in zip(eager_outputs,captured)]
            result['graph_bitwise_equal']=all(torch.equal(a,b) for a,b in zip(eager_outputs,captured))
            assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(eager_outputs,captured))
            result['graph']=measure(graph.replay)
            errors=[]
            for image in images[1:4]:
                x.copy_(normalized(image).cuda())
                reference=tuple(y.clone() for y in model(x))
                graph.replay(); torch.cuda.synchronize()
                errors.extend(float((a-b).abs().max()) for a,b in zip(reference,captured))
                assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(reference,captured))
            result['additional_input_max_abs_errors']=errors
            result['graph_status']='measured'
        except Exception as exc:
            result['graph_status']='unavailable'; result['graph_failure']=repr(exc)
        # Probe CPU dispatch overhead through operator profiling. CUDA activity may be unavailable on Windows.
        try:
            activities=[torch.profiler.ProfilerActivity.CPU]
            if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                for _ in range(5): model(x)
                torch.cuda.synchronize()
            result['profiler_table']=prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=15)
            result['profiler_activities']=[str(a) for a in activities]
        except Exception as exc: result['profiler_failure']=repr(exc)
    result['telemetry']=telemetry+[gpu_snapshot()]
    result['elapsed_seconds']=time.perf_counter()-start
    result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
    path.write_text(json.dumps(result,indent=2)+'\n')
    event.update(experiment_id='06-launch-control-v1-result',model_family='EfficientAD-S reduced-budget variant',
        categories=['bottle'],hypothesis='Test host dispatch contribution with equivalent fixed-shape CUDA graph replay',
        hyperparameters=dict(warmup=50,repeats=3,iterations=100),status='completed',
        metrics=dict(eager_wall=result['eager']['wall'],graph_status=result['graph_status']),paths=dict(results=path.relative_to(ROOT).as_posix()),
        interpretation='Exploratory after-selection control; equality checked before graph measurement; never used for model selection.',decision='Keep eager final-cost estimate conservative')
    record(ROOT/'experiments',event)
    print(json.dumps({k:v for k,v in result.items() if k not in ('eager','graph')},indent=2))
    print({k:v['wall'] for k,v in result.items() if k in ('eager','graph')})


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
