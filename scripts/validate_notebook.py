"""Validate notebook syntax, narrative and compact publication output policy."""
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def validate(path=ROOT / "notebooks/00_dataset_eda.ipynb", require_executed=False):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    headings = [int(n) for c in nb.cells if c.cell_type == "markdown"
                for n in re.findall(r"^## (\d+)\.", c.source, re.M)]
    assert headings == list(range(1, 21)), headings
    code_cells = 0
    executed = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            code_cells += 1
            compile(cell.source, f"{path.name}:cell-{i}", "exec")
            assert nb.cells[i-1].cell_type == "markdown"
            assert i+1 < len(nb.cells) and nb.cells[i+1].cell_type == "markdown"
            assert len(nb.cells[i-1].source) > 80 and len(nb.cells[i+1].source) > 80
            if cell.execution_count is not None:
                executed += 1
            assert not any(o.output_type == "error" for o in cell.outputs)
            if cell.metadata.get("contains_dataset_images"):
                assert not cell.outputs, "Dataset image panels stay local"
            for output in cell.outputs:
                if output.output_type in ("display_data", "execute_result"):
                    assert "image/jpeg" not in output.get("data", {}), "No raw dataset JPEGs"
    if require_executed:
        assert executed == code_cells, "Not every code cell executed"
    assert Path(path).stat().st_size < 2_000_000, "Keep published notebook compact"
    return {"cells": len(nb.cells), "code_cells": code_cells, "sections": len(headings),
            "executed_code_cells": executed,
            "committed_outputs": "compact tables/aggregate plots; dataset panels local" if executed else "cleared"}


if __name__ == "__main__":
    print(validate())
