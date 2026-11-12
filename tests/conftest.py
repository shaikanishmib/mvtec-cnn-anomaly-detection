import numpy as np
import pytest
from PIL import Image
from anomaly_detection.data import CATEGORIES


@pytest.fixture
def dataset(tmp_path):
    """Tiny unique synthetic files; never masquerade as official MVTec data."""
    rng = np.random.default_rng(42)
    for category in CATEGORIES:
        for split, defect, count in (("train", "good", 10), ("test", "good", 1),
                                     ("test", "scratch", 1)):
            folder = tmp_path / category / split / defect
            folder.mkdir(parents=True)
            for i in range(count):
                Image.fromarray(rng.integers(0, 256, (8, 10, 3), dtype=np.uint8)).save(folder / f"{i:03}.png")
        masks = tmp_path / category / "ground_truth/scratch"
        masks.mkdir(parents=True)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:4, 3:6] = 255
        Image.fromarray(mask).save(masks / "000_mask.png")
    return tmp_path
