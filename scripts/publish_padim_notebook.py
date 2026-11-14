"""Execute Notebook 05 honestly; retain full local output and compact publication."""
import asyncio,os,sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'outputs/padim'
os.environ['IPYTHONDIR']=str(folder/'ipython')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
path=ROOT/'notebooks/05_padim.ipynb'
notebook=nbf.read(path,as_version=4)
name=sys.argv[1] if len(sys.argv)>1 else '05_padim.full.executed.ipynb'
if Path(name).name!=name or not name.endswith('.ipynb'):raise ValueError('Execution filename only')
executed=folder/name
if executed.exists():raise FileExistsError('Preserve previous execution; use a named recovery')
try:
    NotebookClient(notebook,timeout=900,kernel_name='mvtec-research',resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    executed.write_bytes(nbf.writes(notebook).encode('utf-8'))
for cell in notebook.cells:
    if cell.metadata.get('contains_dataset_images'):cell.outputs=[]
path.write_bytes(nbf.writes(notebook).encode('utf-8'))
print('Notebook 05 executed and published with actual compact outputs')
