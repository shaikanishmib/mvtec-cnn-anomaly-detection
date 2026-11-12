import json
from pathlib import Path
import sys
import pytest
from anomaly_detection.tracking import record, config_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_notebook import validate


def test_notebook_narrative_structure():
    assert validate()["sections"] == 20


def test_registry_is_append_only(tmp_path):
    event = json.loads((ROOT / "configs/experiment_template.json").read_text())
    event["experiment_id"] = "fixture-start"
    entry = record(tmp_path, event)
    assert entry["config_sha256"] == config_hash(event)
    with pytest.raises(FileExistsError): record(tmp_path, event)
    with pytest.raises(ValueError): record(tmp_path, {})
    event["experiment_id"] = "../escape"
    with pytest.raises(ValueError): record(tmp_path, event)
