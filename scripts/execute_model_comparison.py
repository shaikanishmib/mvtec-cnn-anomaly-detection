"""Execute saved-data analysis; retain licensed image outputs only locally."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
for key,leaf in [('IPYTHONDIR','ipython-comparison'),('JUPYTER_RUNTIME_DIR','jupyter-comparison'),('MPLCONFIGDIR','matplotlib-comparison')]:
    path=ROOT/'.cache'/leaf
    path.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(path)
import nbformat
from nbclient import NotebookClient
sys.path.insert(0,str(ROOT/'scripts'))
from validate_notebook import validate
path=ROOT/'notebooks/07_model_comparison.ipynb'
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=600,kernel_name='mvtec-research',resources={'metadata':{'path':str(ROOT)}}).execute()
local=ROOT/'outputs/final_comparison/07_model_comparison.full.executed.ipynb'
local.parent.mkdir(parents=True,exist_ok=True)
nbformat.write(nb,local)
for cell in nb.cells:
    if cell.metadata.get('contains_dataset_images'):
        cell.outputs=[]
with path.open('w',encoding='utf-8',newline='\n') as stream:
    nbformat.write(nb,stream)
print(validate(path,require_executed=True))
