"""Read-only validation of original data; freeze manifests outside data/."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly_detection.data import (CATEGORIES, autodiscover_data_root, inventory,
                                    partitions, sha256)

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    root = autodiscover_data_root(ROOT)
    output = ROOT / "outputs/eda"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Validating {root.relative_to(ROOT)}", flush=True)
    frame = inventory(root, progress=lambda c: print(f"Decoding and hashing {c}", flush=True))
    frame.to_csv(output / "inventory.csv", index=False, lineterminator="\n")
    duplicates = frame[frame.sha256.duplicated(keep=False)]
    if len(duplicates):
        duplicates.to_csv(output / "duplicate_images.csv", index=False)
        raise ValueError(f"Found {len(duplicates)} duplicate image records; investigate before freezing")
    manifest = partitions(frame, seed=2026)
    assert manifest.equals(partitions(frame.sample(frac=1, random_state=91), seed=2026))
    frozen = ROOT / "reports/partitions"
    frozen.mkdir(parents=True, exist_ok=True)
    identities = {}
    # All official test files are retained in a separate evaluation manifest.
    for split in ("fit", "development", "calibration", "test"):
        selected = manifest[manifest.partition == split].merge(
            frame[["path", "mask", "mask_sha256"]], on="path", validate="one_to_one")
        if split != "test":
            assert selected.split.eq("train").all() and selected["mask"].isna().all()
        text = selected.to_csv(index=False, lineterminator="\n")
        target = frozen / f"seed2026-{split}.csv"
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise ValueError(f"Frozen manifest differs: {target.name}; do not overwrite scientific history")
        target.write_text(text, encoding="utf-8", newline="")
        identities[split] = {"path": target.relative_to(ROOT).as_posix(),
                             "sha256": sha256(target), "images": len(selected)}
    metadata = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": root.relative_to(ROOT).as_posix(), "seed": 2026,
        "categories": list(CATEGORIES), "images": len(frame),
        "mask_count": int(frame["mask"].notna().sum()),
        "inventory_sha256": sha256(output / "inventory.csv"),
        "partitions": identities,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "integrity": "all images and masks verified/decoded; dimensions, binary masks and pairing passed",
        "exact_duplicate_images": len(duplicates),
        "provenance": {"placement": "user-provided complete dataset; not downloaded by agent",
                       "official_source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
                       "archive_checksum": None, "original_download_timestamp": None,
                       "limitation": "No original archive or publisher checksum supplied; content hashes establish local identity, not independent authenticity",
                       "metadata_sha256": {p.name: sha256(p) for p in
                           (root / "license.txt", root / "readme.txt") if p.is_file()}},
    }
    frame.groupby(["category", "split", "defect"]).size().rename("images").to_csv(
        ROOT / "reports/category_defect_counts.csv", lineterminator="\n")
    manifest.groupby(["category", "partition"]).size().unstack(fill_value=0).to_csv(
        ROOT / "reports/partition_counts.csv", lineterminator="\n")
    (ROOT / "reports/dataset_validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    prepare()
