"""Explicit integration checks against locally available original dataset evidence."""
import json
from pathlib import Path
import pandas as pd
import pytest
import sys
from anomaly_detection.data import CATEGORIES, autodiscover_data_root, partitions, sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_real_data_frozen_partitions():
    evidence = ROOT / "reports/dataset_validation.json"
    inventory = ROOT / "outputs/eda/inventory.csv"
    if not evidence.exists() or not inventory.exists():
        pytest.skip("Run prepare_data.py and EDA for real-data integration evidence")
    report = json.loads(evidence.read_text())
    root = autodiscover_data_root(ROOT)
    assert root.relative_to(ROOT).as_posix() == report["dataset_root"]
    frame = pd.read_csv(inventory)
    assert set(frame.category) == set(CATEGORIES)
    assert len(frame) == report["images"]
    assert int(frame["mask"].notna().sum()) == report["mask_count"]
    assert sha256(inventory) == report["inventory_sha256"]
    expected = partitions(frame, report["seed"])
    actual_parts = []
    seen_paths, seen_content = set(), set()
    for part, entry in report["partitions"].items():
        path = ROOT / entry["path"]
        assert sha256(path) == entry["sha256"]
        selected = pd.read_csv(path)
        assert len(selected) == entry["images"]
        assert set(selected.category) == set(CATEGORIES)
        assert selected.partition.eq(part).all()
        assert not seen_paths.intersection(selected.path)
        assert not seen_content.intersection(selected.sha256)
        seen_paths.update(selected.path); seen_content.update(selected.sha256)
        if part != "test":
            assert selected.split.eq("train").all()
            assert selected["mask"].isna().all()
            assert selected.path.str.contains("/train/good/").all()
        else:
            assert selected.split.eq("test").all()
        assert all((root / p).is_file() for p in selected.path)
        actual_parts.append(selected[expected.columns])
    actual = pd.concat(actual_parts).sort_values("path").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)


def test_full_real_notebook_execution():
    report = ROOT / "reports/eda_execution.json"
    if not report.exists():
        pytest.skip("No EDA execution evidence")
    status = json.loads(report.read_text())
    if not status["dataset_available_and_validated"]:
        pytest.skip("Real-data EDA not yet executed")
    assert status["categories_discovered"] == list(CATEGORIES)
    assert not status["missing_categories"]
    assert status["kernel_verified"] and status["cuda_available"]
    assert status["code_cells_executed"] == 19
    from validate_notebook import validate
    assert validate(require_executed=True)["executed_code_cells"] == 19
