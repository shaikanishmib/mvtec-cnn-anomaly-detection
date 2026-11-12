"""Freeze installed versions plus SHA256 of wheels actually fetched by pip.

Reads the retained pip --report files; never queries an alternate package source.
The resulting artifact lock targets Windows x64 / CPython 3.12 specifically.
"""
from pathlib import Path
import importlib.metadata
import json
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly_detection.data import sha256

ROOT = Path(__file__).resolve().parents[1]


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def freeze():
    downloaded = {}
    for name in ("cuda-install.json", "research-install.json"):
        report = json.loads((ROOT / "outputs" / name).read_text(encoding="utf-8"))
        for item in report["install"]:
            downloaded[canonical(item["metadata"]["name"])] = item
    lines = ["# Windows x64 / CPython 3.12.10; exact downloaded-wheel hashes.",
             "# Bootstrap pip==25.0.1; install project itself separately with --no-deps.",
             "--require-hashes", "--only-binary=:all:", ""]
    distributions = sorted(importlib.metadata.distributions(), key=lambda d: canonical(d.metadata["Name"]))
    count = 0
    for dist in distributions:
        name = canonical(dist.metadata["Name"])
        if name in {"pip", "mvtec-cnn-anomaly-detection"}:
            continue
        item = downloaded[name]
        assert dist.version == item["metadata"]["version"], (name, dist.version)
        info = item["download_info"]
        digest = info["archive_info"]["hashes"]["sha256"]
        if name in {"torch", "torchvision"}:
            assert info["url"].startswith(("https://download.pytorch.org/", "https://download-r2.pytorch.org/"))
            requirement = f"{name} @ {info['url']}"
        else:
            requirement = f"{name}=={dist.version}"
        lines.append(f"{requirement} --hash=sha256:{digest}")
        count += 1
    target = ROOT / "requirements.lock.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    summary = {"platform": "Windows x64", "python": "3.12.10", "packages_with_artifact_hashes": count,
               "lock_sha256": sha256(target), "pip": importlib.metadata.version("pip"),
               "sources": ["official PyTorch CUDA 12.8 wheel index", "PyPI"],
               "scope": "installed and tested isolated environment; no second independent rebuild performed"}
    (ROOT / "reports/environment_lock.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    freeze()
