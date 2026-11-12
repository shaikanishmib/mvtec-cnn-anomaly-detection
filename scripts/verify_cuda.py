"""Synthetic CUDA plumbing checks only: no dataset, optimizer or training."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys


def verify_gpu():
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable: torch={torch.__version__}, build={torch.version.cuda}")
    device = torch.cuda.get_device_name(0)
    if "RTX 4060" not in device:
        raise RuntimeError(f"Expected RTX 4060, detected {device}")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cpu_a = torch.linspace(-1, 1, 256*256).reshape(256, 256)
    cpu_b = cpu_a.T.contiguous()
    product = cpu_a.cuda() @ cpu_b.cuda()
    torch.cuda.synchronize()
    error = float((product.cpu() - cpu_a @ cpu_b).abs().max())
    torch.testing.assert_close(product.cpu(), cpu_a @ cpu_b, rtol=1e-4, atol=1e-4)
    network = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                  torch.nn.Conv2d(8, 4, 3, padding=1)).cuda()
    inputs = torch.randn(2, 3, 64, 64, device="cuda", requires_grad=True)
    output = network(inputs)
    output.square().mean().backward()
    torch.cuda.synchronize()
    assert output.shape == (2, 4, 64, 64)
    assert torch.isfinite(inputs.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    # Also exercise a compiled torchvision CUDA operation to catch build mismatches.
    kept = torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device="cuda"),
                              torch.tensor([.9, .8], device="cuda"), .5)
    assert kept.numel() == 2
    return {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "interpreter": str(Path(sys.executable).relative_to(Path(__file__).resolve().parents[1])),
            "isolated": sys.prefix != sys.base_prefix and not any("site-packages" in p and
                        not Path(p).is_relative_to(Path(sys.prefix)) for p in sys.path),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_available": True, "device": device,
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "matrix_max_absolute_error": error, "matrix_multiply": "passed",
            "cnn_forward_backward": "passed; synthetic tensors, no optimizer or weight updates",
            "torchvision_cuda_nms": "passed",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "plumbing smoke check; not model feasibility or benchmark measurement"}


if __name__ == "__main__":
    result = verify_gpu()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
