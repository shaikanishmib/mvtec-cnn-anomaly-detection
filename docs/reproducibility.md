# Three reproducibility paths

A fresh clone contains source, notebooks and compact evidence. It does **not**
contain an existing `.venv`, MVTec images, pretrained weights, fitted banks or
large prediction caches. Choose the path that matches the assets you hold.

## A. Run the application with retained release artifacts

Requirements: Docker/Compose with Linux amd64 containers and legitimate,
byte-identical retained artifacts. Restore the 33 files at the relative paths
listed in [the deployment manifest](../configs/patchcore_deployment_v1.json)
under `outputs/deployment/artifacts/`. Include the manifest itself, original
frozen configuration, extractor, all 15 banks and calibration JSONs. Never
replace hashes with those of different files to make validation pass.

From the cloned repository root in PowerShell:

```powershell
docker compose -p fieldwork-public --env-file deployment/.env.example build
docker compose -p fieldwork-public --env-file deployment/.env.example run --rm --no-deps --entrypoint python backend -m deployment.artifacts --root /artifacts
docker compose -p fieldwork-public --env-file deployment/.env.example up -d --wait --wait-timeout 180
```

Open http://127.0.0.1:8080. Shut down only this named project:

```powershell
docker compose -p fieldwork-public --env-file deployment/.env.example down --timeout 30
```

The image build does not require model assets. Runtime validation/startup does.
The same Compose definitions and runtime were verified in Milestone 13; this
portfolio review validates configuration and artifact identities without
repeating real-image inference or replacing its timing evidence. See the
[deployment guide](deployment_guide.md) for the separately verified CUDA variant.

There is no public bank/trunk download promised by this repository. Regenerating
a fitted model does not guarantee identical checkpoint serialization or checksum.
If the original assets are unavailable, this exact frozen release cannot be
served until they are legitimately restored. An alternate trained model must
be a separately identified future experiment, not a substitute for these records.

If the original project environment/assets are already present, the previously
verified exporter is optional:

```powershell
.\.venv\Scripts\python.exe -m deployment.artifacts --root . --export outputs/deployment/artifacts
```

That command is not presented as a fresh-clone environment installer.

## B. Build source and run checks without model assets

Frontend prerequisites are Node 26.13.0/npm 11.20.0. From PowerShell:

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run typecheck
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
```

These checks need no backend, images, GPU, banks or Python environment. They
validate source/UI behavior with explicitly marked schema-derived fixtures;
they do not establish real-model performance. Install the selected Node version
before running them; no Node installer is silently executed here.

For the public Python contracts and documentation checks on Windows, Docker
can supply Python and CPU PyTorch without creating a host virtual environment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ci/check-public.ps1 -RunId verified
```

The script snapshots tracked public files (plus explicit portfolio additions
before their first commit), mounts that copy read-only, installs pinned test-only
dependencies into a temporary container directory and runs the public test list.
No data/checkpoint directory or GPU is mounted. Docker may anonymously download
base images and dependencies; no Docker Hub credentials or private secrets are
required. The CPU base is built if absent. A repeated snapshot is refused to
preserve evidence; choose a new `-RunId` for an independent repeat.
Results stay under ignored `outputs/portfolio/verified-results`. It does not stop a
user's running application or overwrite deployment reports.

On GitHub, [Source quality](../.github/workflows/source-quality.yml) creates an
isolated CPU environment on an ordinary Ubuntu runner and runs the same explicit
[public test scope](../ci/public-tests.txt). It also runs frontend checks. Actions
are pinned by full commit SHA; permissions are read-only, persisted Git credentials
are disabled, and there is no privileged pull_request_target workflow. It requires
network package downloads, not model assets, GPUs or Docker Hub authentication.
Remote CI status must be checked for the actual commit; a local pass alone is
not described as a remote workflow success.

The explicit Python install commands are in the workflow. They bypass project
dependency resolution via `PYTHONPATH=src:.`, because the historical `pyproject.toml`
and Windows CUDA wheel lock describe a different environment. Do not install that
CUDA lock as if it were a portable Linux CPU lock. Ordinary CI does not run the
real-model/real-dataset/GPU tests or deep cache validators.

## C. Reproduce research from legitimate sources

This is a separate, expensive scientific workflow, not an installation step.
Obtain the original MVTec AD dataset under its terms and appropriate pretrained
weights; use the original partition manifests, frozen configurations, seeds and
per-method protocols. Read [the eight notebooks](../notebooks),
[the final report](../reports/final_comparison/report.md),
[PatchCore reproduction](patchcore_reproduction.md), and the
[EfficientAD fidelity/provenance audit](efficientad_fidelity_audit.md).

The existing Windows research wheel lock is platform-specific. Dataset licensing,
source archive availability, teacher-weight provenance and historical software
must be reviewed before attempting a new environment. Some source downloads and
exact retained artifacts are not guaranteed to remain available. Scientific
reproduction may yield numerically close results while failing byte identity;
it must never overwrite published records or be advertised as exact restoration.
No research training, threshold calibration or 270-run reevaluation is performed
or triggered by public CI or this portfolio audit.
