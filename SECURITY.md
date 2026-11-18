# Security scope and known limitations

This repository is a local research prototype, not an authenticated public
inspection service. Keep the default loopback-only frontend binding. The backend
has no published Compose host port. Do not expose it publicly without a new
threat review, authentication, TLS, rate limits and current dependency/image checks.

## Known findings

- The immutable [Milestone 13 review](reports/deployment/security_review.json)
  records an OSV audit of 35 selected Python runtime versions. Its unresolved
  records include duplicates sharing aliases; a record count is not a count of
  independent exploitable vulnerabilities. The GPU base includes additional
  inherited packages and an older Python patch version outside that audit scope.
- Container-only Pillow 11.3.0/setuptools 78.1.1 updates were parity-verified.
  Larger fixes requiring incompatible changes were deferred. This portfolio
  review does not change frozen runtime dependencies or claim their safety.
- The frontend production dependency audit was clear in Milestone 13. The full
  development tree had 12 affected packages from
  [GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf).
  The advisory still lists picomatch through 4.0.7 as affected with no patched
  version when reviewed for this audit. Do not treat an npm-suggested major
  downgrade as a verified fix. Avoid untrusted glob inputs and public dev servers.
- Docker Scout was installed but refused local-image scanning without Docker Hub
  authentication. No image-CVE clearance is claimed. Digest pinning is an identity
  control, not proof that an image is vulnerability-free.
- New public CI uses test-only pytest 9.0.3 rather than copying the historical
  pytest 8.3.5 pin affected by [GHSA-6w46-j5rx-g56g](https://github.com/advisories/GHSA-6w46-j5rx-g56g).
  Compatibility is checked in the isolated asset-free environment. The original
  research environment is preserved; this is not a broad runtime upgrade.

Fresh lookup results are recorded separately in the portfolio audit; original
deployment/security reports remain unchanged. CI checks source quality and
asset-free contracts. A green source-quality workflow is not a clean security
audit, real-model integration pass or license approval.

## Existing controls and their boundaries

Serving uses non-root users, read-only roots and artifact mounts, dropped
capabilities, bounded CPU/RAM/PID/cache settings and one guarded inference worker.
Upload validation checks size, format, dimensions and decoded pixel count.
Internal exception paths are not returned. nginx adds CSP/framing/content-type
headers; CORS is not broadened for the same-origin deployment.

Artifact SHA-256 verification assumes the pinned Git manifest is trusted. It does
not authenticate an arbitrary third-party download or make untrusted weights
safe. No arbitrary user-supplied model/checkpoint loading is offered by the API.
Temporary images expire and inspection history is not persisted. Client request
cancellation does not cancel work already running on the server.

There is no declared security-response SLA or invented reporting email. Do not
post secrets, sensitive images or exploit details in public issues; arrange an
appropriate private channel with the maintainer before disclosure.
