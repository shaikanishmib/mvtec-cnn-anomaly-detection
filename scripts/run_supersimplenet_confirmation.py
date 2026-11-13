"""Wait for the known fitter, then run gated frozen confirmation and audits.

Never fits/tunes a model, changes thresholds, retries failed evaluations or pushes.
"""
import sys,time,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/supersimplenet'
index=1+len(list(out.glob('confirmation-*.log')))
log_path=out/f'confirmation-{index:03d}.log';events=[]
pid=int(sys.argv[1])
def event(phase,status,**kwargs):
    item=dict(utc=datetime.now(timezone.utc).isoformat(),phase=phase,status=status,**kwargs);events.append(item)
    (out/f'confirmation-{index:03d}-status.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    print(json.dumps(item),flush=True)
try:
    try:process=psutil.Process(pid)
    except psutil.NoSuchProcess:process=None
    event('waiting-for-all-fits','started',observed_pid=pid)
    while process is not None and process.is_running():time.sleep(5)
    fitted=list((ROOT/'reports/supersimplenet/runs').glob('04-final-*-result.json'))
    assert len(fitted)==45,'Fit process ended without all45 completed results; stop for technical diagnosis'
    event('all-fits','verified',models=len(fitted))
    stages=[('frozen-evaluation',['scripts/benchmark_supersimplenet.py','evaluate']),
        ('reconcile',['scripts/summarize_supersimplenet.py']),('native-failure-analysis',['scripts/analyze_supersimplenet.py']),
        ('normal-brightness-diagnostic',['scripts/diagnose_supersimplenet_brightness.py']),
        ('measured-findings',['scripts/write_supersimplenet_findings.py']),
        ('notebook-execution',['scripts/publish_supersimplenet_notebook.py']),
        ('full-tests',['-m','pytest','-q','-p','no:cacheprovider','--basetemp=outputs/supersimplenet/tests-final-v1']),
        ('artifact-validation',['scripts/validate_supersimplenet.py']),
        ('readme',['scripts/finalize_supersimplenet_readme.py'])]
    with log_path.open('x',encoding='utf-8') as log:
        for phase,args in stages:
            event(phase,'started');subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            log.flush();event(phase,'completed')
    event('publication','ready-for-human-and-agent-content-review',log=log_path.relative_to(ROOT).as_posix())
except BaseException as exc:
    event('workflow','stopped',reason=f'{type(exc).__name__}: {exc}',log=log_path.relative_to(ROOT).as_posix());raise
