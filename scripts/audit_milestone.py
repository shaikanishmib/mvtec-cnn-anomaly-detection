"""Read-only staged-file and local Markdown-link audit before publication."""
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def audit():
    remote = git("remote", "get-url", "origin").decode().strip()
    assert remote == "https://github.com/shaikanishmib/mvtec-cnn-anomaly-detection.git", remote
    assert git("branch", "--show-current").decode().strip() == "main"
    files = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")
    files = [name for name in files if name]
    assert files, "Nothing staged"
    total = 0
    for name in files:
        path = Path(name)
        assert path.parts[0] not in {"data", "checkpoints", "outputs", ".aws", ".cache"}
        assert not any(p.startswith(".venv") for p in path.parts)
        assert path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".pem", ".key", ".zip", ".png",
                                           ".tar", ".gz", ".7z", ".safetensors", ".onnx"}
        assert not path.name.startswith((".env", "credentials"))
        contents = git("show", f":{name}")
        assert len(contents) < 5_000_000, f"Oversized: {name}"
        total += len(contents)
        # Detect references to the excluded sibling, without matching this repo.
        assert not re.search(rb"mvtec-cnn(?:[/\\\s\"']|$)", contents, re.I), name
        for pattern in (rb"ghp_[A-Za-z0-9]{36}", rb"AKIA[A-Z0-9]{16}",
                        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"):
            assert not re.search(pattern, contents), f"Potential credential in {name}"
    links = 0
    for source in [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), ROOT / "experiments/README.md"]:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#")[0]
            assert (source.parent / target).exists(), f"Broken link in {source.name}: {target}"
            links += 1
    return {"staged_files": len(files), "staged_bytes": total, "local_links_checked": links,
            "origin": remote, "result": "passed"}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
