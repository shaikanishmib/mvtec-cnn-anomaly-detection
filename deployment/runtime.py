"""Fail-fast container entrypoint; exec preserves signals and clean shutdown."""
import os
import sys
from .artifacts import validate, ArtifactError


def main():
    try:
        validate(os.environ.get('ANOMALY_ARTIFACT_ROOT', '/artifacts'))
        import torch
        device = os.environ.get('ANOMALY_DEVICE', 'cpu')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
    except ArtifactError as exc:
        print('Deployment preflight failed: '+str(exc), file=sys.stderr)
        raise SystemExit(1)
    except Exception:
        print('Deployment preflight failed: verify read-only artifact mount, manifest checksum and requested device. No CPU fallback is performed.', file=sys.stderr)
        raise SystemExit(1)
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'anomaly_detection.backend.app:app',
        '--host', '0.0.0.0', '--port', '8000', '--workers', '1', '--no-access-log'])


if __name__ == '__main__':
    main()
