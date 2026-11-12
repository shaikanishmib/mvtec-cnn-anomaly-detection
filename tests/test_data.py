import shutil
import numpy as np
import pytest
from PIL import Image
from anomaly_detection.data import CATEGORIES, discover, inventory, partitions, project_data_root


def test_discovery_missing_and_complete(dataset, tmp_path):
    assert discover(dataset) == CATEGORIES
    assert discover(tmp_path / "missing") == ()
    with pytest.raises(ValueError, match="Missing categories"):
        inventory(tmp_path / "missing")


@pytest.mark.parametrize("category", CATEGORIES)
def test_all_categories_and_pairing(dataset, category):
    frame = inventory(dataset)
    subset = frame[frame.category == category]
    assert len(subset) == 12
    assert subset.width.eq(10).all() and subset.height.eq(8).all()
    assert subset.channels.eq(3).all() and subset.format.eq("PNG").all()
    assert subset[subset.anomalous]["mask"].notna().all()
    assert subset[subset.anomalous]["mask_sha256"].str.len().eq(64).all()
    assert subset[~subset.anomalous]["mask"].isna().all()


def test_deterministic_disjoint_partitions(dataset):
    frame = inventory(dataset)
    a = partitions(frame, 2026)
    b = partitions(frame.sample(frac=1, random_state=9), 2026)
    assert a.equals(b)
    assert not a.equals(partitions(frame, 2027))
    assert a.path.is_unique and len(a) == 15*12
    for _, group in a.groupby("category"):
        assert group.partition.value_counts().to_dict() == {"fit": 8, "test": 2, "development": 1, "calibration": 1}
    assert a.loc[a.split == "test", "partition"].eq("test").all()
    assert not a.loc[a.split == "train", "partition"].eq("test").any()


@pytest.mark.parametrize("damage", ["corrupt", "missing", "mismatch", "nonbinary", "orphan", "train_defect", "duplicate"])
def test_integrity_failures(dataset, damage):
    image = dataset / "bottle/test/scratch/000.png"
    mask = dataset / "bottle/ground_truth/scratch/000_mask.png"
    if damage == "corrupt":
        image.write_bytes(b"not an image")
    elif damage == "missing":
        mask.unlink()
    elif damage == "mismatch":
        Image.fromarray(np.full((4, 4), 255, np.uint8)).save(mask)
    elif damage == "nonbinary":
        Image.fromarray(np.full((8, 10), 128, np.uint8)).save(mask)
    elif damage == "orphan":
        shutil.copyfile(mask, mask.with_name("orphan_mask.png"))
    elif damage == "train_defect":
        target = dataset / "bottle/train/scratch"
        target.mkdir()
        shutil.copyfile(image, target / "000.png")
    elif damage == "duplicate":
        shutil.copyfile(dataset / "bottle/train/good/000.png", image)
    with pytest.raises((ValueError, OSError)):
        partitions(inventory(dataset))


def test_project_root_isolation(tmp_path):
    assert project_data_root(tmp_path) == tmp_path / "data/mvtec_ad"
    with pytest.raises(ValueError, match="inside project"):
        project_data_root(tmp_path, "../outside")


def test_missing_test_split(dataset):
    shutil.rmtree(dataset / "wood/test")
    with pytest.raises(ValueError, match="Missing split"):
        inventory(dataset)
