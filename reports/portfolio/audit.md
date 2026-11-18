# Final portfolio-quality review

Baseline: `d6bc59753fa72ced09b027cf8cd89d0102c8df6e`. This review improves the
public entry points and source-quality checks. It is not a new experiment or
deployment benchmark. Original research/application/deployment evidence remains
byte-identical; [validation.json](validation.json) reconciles the checks.

## Assessment and changes

The project has a credible end-to-end research story: six methods, 270 frozen
evaluations, explicit uncertainty/calibration analysis, one selected inference
adapter, and a real full-stack application. Its strongest portfolio evidence is
traceability from saved experiments to a frozen engineering candidate. Its
limitations are equally visible: false alarms, private retained artifacts,
incomplete licensing and security clearance, and local rather than prospective
industrial validation.

The README now presents the actual safe screenshot, application scope, complete
nine-metric table, architecture, fresh-clone choices, notebooks, evidence and
rights in that order. It preserves scientific caveats and all recorded metric
values. A [documentation index](../../docs/index.md),
[concise overview](../../docs/project_overview.md),
[reproducibility guide](../../docs/reproducibility.md),
[rights review](../../docs/licensing_and_attribution.md), and
[security disclosure](../../SECURITY.md) provide clear next steps.
The original screenshot pixels and source provenance are unchanged. No generated
inspection example, new inference result or performance claim was added.

## GitHub About settings — manual recommendation

Public REST metadata reported an empty description and no topics. The connected
GitHub tool supports repository inspection; no settings mutation was requested
or performed. Keep the existing repository name and URL to preserve links and
use **Fieldwork** as the application/README title. Apply these exact values using
the repository About gear if the owner approves:

**Description**

```text
Six-model industrial anomaly detection research on MVTec AD (270 evaluations), with a frozen PatchCore inspection app using PyTorch, FastAPI, React and Docker.
```

**Topics**

```text
anomaly-detection, industrial-inspection, computer-vision, mvtec-ad, pytorch, patchcore, fastapi, react, typescript, docker, reproducible-research, machine-learning
```

Observed fields and recommendations are also in
[repository_metadata.json](repository_metadata.json). No rename, license setting,
public release tag or registry publication was made.

## Fresh-clone reproducibility

| Path | Current verification and boundary |
|---|---|
| A — serve retained artifacts | The documented Compose checksum-only command passed for 33 files/all 15 categories, without inference. CPU/CUDA serving remains the original Milestone 13 evidence; model runs were not repeated. Exact banks/extractor are not public downloads. |
| B — source and tests without assets | 90 public Python tests passed both in the existing host environment and a clean Linux CPU source copy. The latter mounted no dataset, extractor, banks or GPU and required no host Python environment. Dependency installation and pip check passed with pytest 9.0.3. |
| B — frontend | Pinned lockfile install, typecheck, lint, 39 unit tests and production build passed. No backend or model assets were required. |
| C — reproduce trained research | Not rerun. Legitimate MVTec/pretrained sources, historical software, splits and budgets are required; teacher permissions and some retained assets remain unresolved. New serialized checkpoints are not promised to match the recorded SHA-256 values. |

The two 90-test results are the same logical suite in different environments,
not 180 independent tests. Failed preliminary source checks identified a missing
test dependency and inherited production artifact-path environment settings;
both were corrected in the check setup. Their snapshots remain local under
ignored outputs. No backend/inference source was changed to make the tests pass.

The existing host `.venv` was not recreated or upgraded. The new CI test-only
pytest 9.0.3 pin avoids the older tmpdir advisory and was tested independently.
There is no broad application dependency upgrade. Full real-data/GPU suites,
real-image/browser inference and the 270 evaluations were intentionally not run
again. They are separate local integration evidence, not ordinary PR checks.

## Source-quality CI

[The workflow](../../.github/workflows/source-quality.yml) has two bounded jobs:

- Linux CPU public Python tests plus documentation, notebook syntax/saved-output,
  canonical-table and API schema checks.
- Frontend install, typecheck, lint, unit tests and production build.

Actions are pinned to full official release commit SHAs. Permissions are
`contents: read`; checkout does not persist credentials. There are no secrets,
private model mounts, NVIDIA requirements, Docker Hub credentials, privileged PR
triggers or publication steps. CI uses an isolated CPU environment rather than
the historical Windows/CUDA project resolver. Downloads still require network
availability; CI is not an offline or security-clearance guarantee.

YAML/contracts and equivalent Linux checks passed locally. This committed audit
does not invent a remote CI pass: after publication, inspect the actual run for
the final commit on the [Source quality page](https://github.com/shaikanishmib/mvtec-cnn-anomaly-detection/actions/workflows/source-quality.yml).
GitHub's run is authoritative; a local pass is not substituted for it.

## Licensing and attribution

No root project license is declared or added. Public visibility does not grant
general reuse rights. The owner recommendation is an explicit MIT decision for
original source only after ownership/upstream compatibility review, with retained
notices and data/weight exclusions; that recommendation has not been applied.

The [MVTec authors](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
state CC BY-NC-SA 4.0 for the dataset. Torchvision code is BSD-3-Clause, while
ImageNet/pretrained asset permissions require their own review. The unofficial
EfficientAD repository's Apache-2.0 notice is retained, but separate teacher-weight
licensing/training history is unconfirmed. The SuperSimpleNet MIT notice is
retained. Dataset-derived banks, raw images and teacher weights are not published.

Primary code-license pages were inspected; this is not an exhaustive transitive
legal bill of materials. The ImageNet access-terms page could not be fetched
independently during this review; permissions are not assumed resolved. See the
[rights inventory](../../docs/licensing_and_attribution.md) for exact distinctions
and source links. No commercial-use clearance is claimed.

## Security review

The fresh OSV lookup covered 46 selected runtime/source-check versions. All 100
prior runtime advisory records were still returned; aliases may overlap, so
these are not 100 independent exploitable vulnerabilities. New CI dependencies
added no remaining records in that lookup after the pytest pin was corrected.
The historical runtime versions and old security evidence were not rewritten.

npm reports zero production dependency findings and 12 affected development
packages from the picomatch advisory. The
[primary advisory](https://github.com/advisories/GHSA-fc5q-39xc-vjpf)
still lists no patched version for the affected range. Docker Scout was tried
again and still required Docker Hub authentication; image-CVE scanning remains
blocked. No credentials were requested and no clean-image claim is made.
See [security_check.json](security_check.json) and [image_scan.json](image_scan.json).

Existing controls remain non-root/read-only containers, a private single-worker
backend, loopback-only frontend exposure, restricted CORS, upload validation and
pinned artifact identity checks. Authentication/TLS/rate limits, complete image
security review and prospective industrial calibration remain prerequisites for
any public production proposal.

## Preservation and remaining scope

All 3,207 protected pre-existing files match the baseline: notebooks, metrics,
registries, configurations, source behavior, dependency locks, screenshots and
research/deployment reports. A pre-existing uncommitted `up.json` timing from a
later launch was backed up locally before restoring the immutable published
record. It was not turned into a new benchmark result or committed over evidence.

Current documentation/schema checks cover the README, notebook/report/image
navigation, guides, 270-record coverage, six exact macro rows, 15 categories,
20 API fields and eight notebooks/205 saved executed cells. Older environment
and CNN reproduction notes retain historical machine-path examples; the new
relative-path reproducibility guide is the fresh-clone entry point. Those
historical records were preserved rather than silently rewritten.

The code is ready for a technically transparent GitHub portfolio presentation.
It is not an unrestricted licensed release or a production-security approval.
Remaining owner/external decisions are repository About settings, project-wide
licensing, legitimate retained-asset availability, teacher/weight rights and
authenticated image scanning. Live remote CI status must be read for the
published commit. No unrelated research or engineering work was started.
