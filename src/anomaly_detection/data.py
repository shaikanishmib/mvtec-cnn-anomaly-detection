"""Project-local discovery, full decoding, masks, hashes and normal partitions."""
from pathlib import Path
import hashlib
import random
import os

import numpy as np
import pandas as pd
from PIL import Image

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
TEXTURES = frozenset(("carpet", "grid", "leather", "tile", "wood"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COLUMNS = ["path", "category", "split", "defect", "anomalous", "width", "height",
           "channels", "mode", "format", "sha256", "mask", "mask_sha256", "mean", "std"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root):
    root = Path(root)
    return tuple(c for c in CATEGORIES if (root / c / "train" / "good").is_dir())


def autodiscover_data_root(project):
    """Find a unique category root anywhere under data/, without following links.

    Stop at category roots instead of traversing thousands of image directories.
    Multiple candidate roots require an explicit selection rather than guessing.
    """
    base = Path(project).resolve() / "data"
    if not base.is_dir() or base.is_symlink() or base.is_junction():
        raise ValueError("Missing or linked project data directory")
    candidates = []
    pending = [base]
    while pending:
        folder = pending.pop()
        children = sorted(folder.iterdir())
        directories = []
        for child in children:
            if child.is_symlink() or child.is_junction():
                raise ValueError("Linked paths are not permitted under project data/")
            if child.is_dir():
                directories.append(child)
        if any(child.name in CATEGORIES for child in directories):
            candidates.append(folder)
        else:
            pending.extend(directories)
    if len(candidates) != 1:
        raise ValueError(f"Expected one dataset root under data/, found {len(candidates)}")
    root = candidates[0]
    missing = sorted(set(CATEGORIES) - set(discover(root)))
    if missing:
        raise ValueError(f"Incomplete dataset root; missing categories: {missing}")
    return root


def project_data_root(project, relative=None):
    """Reject roots outside this repository, including junction/symlink escapes."""
    project = Path(project).resolve()
    if relative in (None, "auto"):
        return autodiscover_data_root(project)
    root = (project / relative).resolve()
    if not root.is_relative_to(project / "data"):
        raise ValueError("Dataset root must remain inside project data/")
    return root


def inventory(root, require_complete=True, progress=None):
    """Decode every image and mask; errors stop publication of partial statistics.

    Hashes detect byte duplicates, not all near duplicates. Mean/std are RGB [0,1]
    per image (population std), never normalization parameters fitted to test data.
    """
    root = Path(root).resolve()
    categories = discover(root)
    if require_complete and categories != CATEGORIES:
        raise ValueError(f"Missing categories: {sorted(set(CATEGORIES) - set(categories))}")
    records, expected_masks = [], set()
    for category in categories:
        if progress:
            progress(category)
        # Reject junctions/symlinks before traversing or opening their targets.
        for folder, directories, files in os.walk(root / category, followlinks=False):
            for name in directories + files:
                child = Path(folder) / name
                if child.is_symlink() or child.is_junction():
                    raise ValueError("Linked dataset entries are not permitted")
        if not (root / category / "ground_truth").is_dir():
            raise ValueError(f"Missing ground_truth: {category}")
        for split in ("train", "test"):
            split_dir = root / category / split
            if not split_dir.is_dir():
                raise ValueError(f"Missing split: {category}/{split}")
            paths = sorted(p for p in split_dir.rglob("*") if p.is_file())
            images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
            if not images or not any(p.parent.name == "good" for p in images):
                raise ValueError(f"Empty split or no normal images: {category}/{split}")
            if split == "test" and not any(p.parent.name != "good" for p in images):
                raise ValueError(f"No defective test images: {category}")
            for path in images:
                if not path.resolve().is_relative_to(root):
                    raise ValueError("Image escapes dataset root")
                if path.parent.parent != split_dir:
                    raise ValueError(f"Unexpected image nesting: {path.relative_to(root)}")
                defect = path.parent.name
                if split == "train" and defect != "good":
                    raise ValueError("Official training data must be normal only")
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    width, height = im.size
                    mode, fmt, channels = im.mode, im.format, len(im.getbands())
                    pixels = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
                mask = None
                if split == "test" and defect != "good":
                    target = root / category / "ground_truth" / defect / f"{path.stem}_mask.png"
                    if not target.is_file():
                        raise ValueError(f"Missing mask: {target.relative_to(root)}")
                    if not target.resolve().is_relative_to(root):
                        raise ValueError("Mask escapes dataset root")
                    with Image.open(target) as im:
                        im.verify()
                    with Image.open(target) as im:
                        im.load()
                        values = np.asarray(im)
                        if im.size != (width, height) or values.ndim != 2:
                            raise ValueError("Mask dimensions/channels mismatch")
                        if not set(np.unique(values)).issubset({0, 255}) or not values.any():
                            raise ValueError("Expected nonempty binary 0/255 defect mask")
                    mask = target.relative_to(root).as_posix()
                    expected_masks.add(mask)
                records.append(dict(path=path.relative_to(root).as_posix(), category=category,
                                    split=split, defect=defect, anomalous=defect != "good",
                                    width=width, height=height, channels=channels, mode=mode,
                                    format=fmt, sha256=sha256(path), mask=mask,
                                    mask_sha256=sha256(root / mask) if mask else None,
                                    mean=float(pixels.mean()), std=float(pixels.std())))
        actual = {p.relative_to(root).as_posix() for p in
                  (root / category / "ground_truth").rglob("*") if p.is_file()
                  and p.suffix.lower() in IMAGE_SUFFIXES}
        if actual != {p for p in expected_masks if p.startswith(category + "/")}:
            raise ValueError(f"Orphan or unexpected ground truth: {category}")
    return pd.DataFrame(records, columns=COLUMNS)


def partitions(frame, seed=2026):
    """70/15/15 by category: floor for development/calibration, rest fit.

    Byte-identical official images are rejected rather than silently assigned.
    Freeze the returned manifest before any model fitting.
    """
    if frame.path.duplicated().any() or frame.sha256.duplicated().any():
        raise ValueError("Duplicate paths/content: investigate leakage before partitioning")
    result = frame[["path", "category", "split", "sha256"]].copy()
    result["partition"] = "test"
    for category in sorted(frame.category.unique()):
        group = frame[(frame.category == category) & (frame.split == "train")]
        if group.anomalous.any():
            raise ValueError("Anomalous training image")
        paths = sorted(group.path)
        if len(paths) < 7:
            raise ValueError("Insufficient normal images for three nonempty partitions")
        random.Random(f"{seed}:{category}").shuffle(paths)
        n = int(len(paths) * .15)
        assignments = {p: "development" if i < n else "calibration" if i < 2*n else "fit"
                       for i, p in enumerate(paths)}
        result.loc[group.index, "partition"] = group.path.map(assignments)
    return result.sort_values("path").reset_index(drop=True)
