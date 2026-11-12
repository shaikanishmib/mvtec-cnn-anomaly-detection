"""Replay the complete measured research in the registered CUDA kernel."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from validate_basic_cnn import validate_notebook

ROOT=Path(__file__).resolve().parents[1]


def main():
    for name,child in (("IPYTHONDIR","ipython"),("JUPYTER_RUNTIME_DIR","jupyter"),("MPLCONFIGDIR","matplotlib")):
        folder=ROOT/"outputs"/child;folder.mkdir(parents=True,exist_ok=True);os.environ[name]=str(folder)
    print(validate_notebook(require_executed=False),flush=True)
    path=ROOT/"notebooks/01_basic_cnn.ipynb"
    notebook=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name="mvtec-research")
    assert Path(manager.kernel_spec.argv[0]).resolve()==Path(sys.executable).resolve()
    client=NotebookClient(notebook,timeout=1200,km=manager,allow_errors=False,
           resources={"metadata":{"path":str(ROOT)}},
           on_cell_start=lambda cell,cell_index,**kwargs:print(f"Notebook cell {cell_index}: {cell.source.splitlines()[0]}",flush=True))
    local=ROOT/"outputs/basic_cnn/01_basic_cnn.executed.ipynb"
    try:client.execute()
    finally:
        nbformat.write(notebook,local)
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    for cell in notebook.cells:
        if cell.cell_type=="code" and cell.metadata.get("contains_dataset_images"):
            cell.outputs=[]
            cell.metadata["publication_note"]="Full licensed image panels retained only in local executed notebook"
    # Canonical LF matches .gitattributes and preserves a stable published hash.
    path.write_text(nbformat.writes(notebook),encoding="utf-8",newline="")
    result=validate_notebook(require_executed=True)
    panels=json.loads((ROOT/"outputs/basic_cnn/notebook/panel_index.json").read_text())
    missing=sum(p.get("missing_local_artifact",False) for p in panels)
    result.update(executed_at_utc=datetime.now(timezone.utc).isoformat(),
                  kernel="mvtec-research",mode="replay of immutable actual training/evaluation artifacts",
                  source_and_artifact_checks=True,missing_local_prediction_panels=missing,
                  full_visual_replay=missing==0,local_full_notebook=local.relative_to(ROOT).as_posix())
    (ROOT/"reports/basic_cnn/notebook_execution.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
