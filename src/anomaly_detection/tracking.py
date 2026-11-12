"""Append-only, one JSON file per experiment; reserve before running."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

REQUIRED = {"experiment_id", "model_family", "categories", "hypothesis", "preprocessing",
            "hyperparameters", "seed", "environment", "training", "metrics",
            "calibration", "paths", "interpretation", "decision", "status"}


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def record(directory, experiment):
    missing = REQUIRED - experiment.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    identifier = experiment["experiment_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,100}", identifier):
        raise ValueError("Invalid experiment ID")
    entry = dict(experiment, recorded_at=datetime.now(timezone.utc).isoformat(),
                 config_sha256=config_hash(experiment))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{identifier}.json").open("x", encoding="utf-8") as stream:
        json.dump(entry, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return entry
