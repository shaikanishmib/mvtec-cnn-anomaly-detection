import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_cuda import verify_gpu


def test_cuda_plumbing():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware required; milestone verification additionally runs verify_cuda.py")
    result = verify_gpu()
    assert result["isolated"]
    assert result["matrix_multiply"] == "passed"
    assert result["peak_allocated_bytes"] > 0
