"""Observe a known trainer PID; device-wide samples are not allocator peaks."""
import sys,time,json,csv,subprocess,statistics
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[1]
pid=int(sys.argv[1]);process=psutil.Process(pid);interval=5.
folder=ROOT/'outputs/supersimplenet';folder.mkdir(parents=True,exist_ok=True)
index=1+len(list(folder.glob('training-monitor-*.csv')))
path=folder/f'training-monitor-{index:03d}.csv';records=[];started=datetime.now(timezone.utc).isoformat()
fields=['utc','pid','process_rss_bytes','gpu_util_percent','device_memory_used_mib','gpu_temperature_c','device_power_w']
with path.open('x',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    while process.is_running():
        try:
            row=dict(utc=datetime.now(timezone.utc).isoformat(),pid=pid,process_rss_bytes=process.memory_info().rss)
            values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()[0].split(',')
            for key,value in zip(fields[3:],values):
                try:row[key]=float(value.strip())
                except ValueError:row[key]=None
            writer.writerow(row);stream.flush();records.append(row);time.sleep(interval)
        except psutil.NoSuchProcess:break
result=dict(pid=pid,start_utc=started,end_utc=datetime.now(timezone.utc).isoformat(),sample_interval_seconds=interval,samples=len(records),
    path=path.relative_to(ROOT).as_posix(),scope='Observed segment only, starting after fitting began; NVIDIA fields are whole-device, including other applications; primary per-fit CUDA allocator peaks are recorded separately',
    maximum_process_rss_bytes=max((r['process_rss_bytes'] for r in records),default=0))
for key in fields[3:]:
    values=[r[key] for r in records if r[key] is not None]
    result[key]=dict(mean=statistics.mean(values),maximum=max(values)) if values else None
with (ROOT/f'reports/supersimplenet/training_monitor_{index:03d}.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))
