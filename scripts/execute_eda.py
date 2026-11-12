"""Run EDA locally, retain executed notebook only under ignored outputs/."""
from pathlib import Path
import json
import os
import sys
from datetime import datetime, timezone
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_notebook import validate


def main():
    # Keep Jupyter/IPython runtime files within the project, including on Windows.
    for key, child in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter")):
        directory = ROOT / "outputs" / child
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["MPLCONFIGDIR"] = str(ROOT / "outputs/matplotlib")
    print(validate())
    notebook = nbformat.read(ROOT / "notebooks/00_dataset_eda.ipynb", as_version=4)
    # Use the registered repository kernel, and verify its interpreter explicitly.
    from jupyter_client import KernelManager
    manager = KernelManager(kernel_name="mvtec-research")
    assert Path(manager.kernel_spec.argv[0]).resolve() == Path(sys.executable).resolve()
    client = NotebookClient(notebook, timeout=1200, km=manager,
                            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
                            on_cell_start=lambda cell, cell_index, **kwargs: print(
                                f"Executing notebook cell {cell_index}: {cell.source.splitlines()[0]}", flush=True))
    try:
        client.execute()
    finally:
        output = ROOT / "outputs/eda"
        output.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output / "00_dataset_eda.executed.ipynb")
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    status = json.loads((output / "status.json").read_text())
    report = dict(status, code_cells_executed=sum(c.cell_type == "code" and
                  c.execution_count is not None for c in notebook.cells),
                  execution_interpreter="project isolated .venv; registered mvtec-research kernel",
                  kernel_verified=True, executed_at_utc=datetime.now(timezone.utc).isoformat())
    (ROOT / "reports/eda_execution.json").write_text(json.dumps(report, indent=2) + "\n")
    # Retain genuine text/table and derived aggregate plots in Git. Never publish
    # panels containing dataset pixels; the full executed copy remains local.
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.metadata.get("contains_dataset_images"):
            cell.outputs = []
            cell.metadata["publication_note"] = "Image panel retained in local full executed notebook only"
    nbformat.write(notebook, ROOT / "notebooks/00_dataset_eda.ipynb")
    print(validate(require_executed=True))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
