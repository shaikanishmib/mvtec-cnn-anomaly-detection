"""Emit a small Linux runtime lock from the validated project distribution metadata.

Run only intentionally when refreshing deployment dependencies. Does not install
anything or alter the working CUDA environment. CPU/CUDA torch wheels are pinned
separately by Dockerfile arguments; their Linux CUDA extras use a separate lock.
"""
from pathlib import Path
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.markers import default_environment


def main():
    todo = ['numpy', 'pandas', 'pillow', 'psutil', 'fastapi', 'uvicorn', 'python-multipart', 'httpx', 'torch', 'torchvision']
    environment = dict(default_environment(), sys_platform='linux', platform_system='Linux', os_name='posix', extra='')
    versions = {}
    while todo:
        name = todo.pop().lower().replace('_','-')
        if name in versions:
            continue
        dist = metadata.distribution(name)
        versions[name] = dist.version
        for item in dist.requires or []:
            requirement = Requirement(item)
            if not requirement.marker or requirement.marker.evaluate(environment):
                todo.append(requirement.name)
    del versions['torch']; del versions['torchvision']
    # Container-only compatible security updates; never mutate the research venv.
    versions.update(pillow='11.3.0', setuptools='78.1.1')
    text = '# Linux inference-only transitive pins; torch/torchvision are pinned by Dockerfile.\n'
    text += '\n'.join(f'{name}=={version}' for name, version in sorted(versions.items()))+'\n'
    Path(__file__).with_name('requirements-runtime.txt').write_text(text,encoding='utf-8',newline='\n')
    print(f'{len(versions)} runtime dependencies pinned; research/Jupyter/testing dependencies excluded')


if __name__ == '__main__':
    main()
