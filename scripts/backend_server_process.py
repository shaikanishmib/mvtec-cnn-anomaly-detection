"""Stop only a subprocess created by our validation/benchmark caller."""
import socket
import time
import psutil


def stop_owned_server(process,port=8000):
    try:
        parent=psutil.Process(process.pid)
        owned=parent.children(recursive=True)+[parent]
    except psutil.NoSuchProcess:owned=[]
    for child in owned:
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=10)
    for child in alive:
        try:child.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=10)
    process.wait(timeout=10)
    # Windows can release the listener just after the launcher exits.
    deadline=time.monotonic()+10
    while True:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))!=0:return
        if time.monotonic()>deadline:raise RuntimeError('Listener remains occupied after owned-server shutdown; no unrelated process was stopped')
        time.sleep(.1)
