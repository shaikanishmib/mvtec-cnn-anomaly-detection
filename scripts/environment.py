"""Read-only software/hardware inspection; safe on CPU-only installations."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


def inspect():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pillow", "pandas", "matplotlib",
                 "scikit-learn", "nbformat", "nbclient", "ipykernel", "jupyterlab", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "packages": packages}
    try:
        import torch
        import torchvision
        result.update(torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                      cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            result["cuda_smoke"] = float(torch.ones(4, device="cuda").sum().cpu())
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=15, check=True)
        result["nvidia_smi"] = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["nvidia_smi_error"] = str(exc)
    return result


if __name__ == "__main__":
    text = json.dumps(inspect(), indent=2)
    if len(sys.argv) > 1:
        from pathlib import Path
        Path(sys.argv[1]).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
