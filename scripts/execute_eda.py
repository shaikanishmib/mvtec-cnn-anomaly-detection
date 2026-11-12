"""Run EDA locally, retain executed notebook only under ignored outputs/."""
from pathlib import Path
import json
import os
import sys
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
    print(validate())
    notebook = nbformat.read(ROOT / "notebooks/00_dataset_eda.ipynb", as_version=4)
    # Use this interpreter explicitly rather than a potentially unrelated kernel.
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec
    manager = KernelManager(kernel_name="python3")
    manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher",
                                          "-f", "{connection_file}"],
                                     display_name="MVTec execution interpreter", language="python")
    client = NotebookClient(notebook, timeout=600, km=manager,
                            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
    try:
        client.execute()
    finally:
        output = ROOT / "outputs/eda"
        output.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output / "00_dataset_eda.executed.ipynb")
    status = json.loads((output / "status.json").read_text())
    report = dict(status, code_cells_executed=sum(c.cell_type == "code" and
                  c.execution_count is not None for c in notebook.cells),
                  execution_interpreter="project .venv (inherits audited system packages)")
    (ROOT / "reports/eda_execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
