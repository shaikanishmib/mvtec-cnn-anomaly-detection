"""Validate the committed notebook's syntax, narrative and cleared-output policy."""
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def validate(path=ROOT / "notebooks/00_dataset_eda.ipynb"):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    headings = [int(n) for c in nb.cells if c.cell_type == "markdown"
                for n in re.findall(r"^## (\d+)\.", c.source, re.M)]
    assert headings == list(range(1, 21)), headings
    code_cells = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            code_cells += 1
            compile(cell.source, f"{path.name}:cell-{i}", "exec")
            assert nb.cells[i-1].cell_type == "markdown"
            assert i+1 < len(nb.cells) and nb.cells[i+1].cell_type == "markdown"
            assert len(nb.cells[i-1].source) > 80 and len(nb.cells[i+1].source) > 80
            assert cell.execution_count is None and not cell.outputs, "Commit cleared outputs"
    return {"cells": len(nb.cells), "code_cells": code_cells, "sections": len(headings),
            "committed_outputs": "cleared"}


if __name__ == "__main__":
    print(validate())
