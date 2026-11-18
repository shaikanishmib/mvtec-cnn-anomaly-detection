"""Validate or export only the pinned deployment manifest and required artifacts."""
import argparse
import json
from pathlib import Path
import shutil
from anomaly_detection.backend.contracts import PIN
from anomaly_detection.inference.patchcore import ArtifactError, sha256, verified_path
from anomaly_detection.tracking import config_hash
from anomaly_detection.data import CATEGORIES

MANIFEST = Path('configs/patchcore_deployment_v1.json')


def validate(root):
    root = Path(root).resolve()
    path = root / MANIFEST
    if not path.is_file() or sha256(path) != PIN:
        raise ArtifactError('Missing or mismatched pinned deployment manifest')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['seed'] != 2026 or set(manifest['categories']) != set(CATEGORIES):
        raise ArtifactError('Unexpected category or seed coverage')
    if config_hash(manifest['preprocessing']) != manifest['preprocessing_sha256']:
        raise ArtifactError('Preprocessing checksum mismatch')
    policy = json.loads(verified_path(root, manifest['frozen_policy']).read_text(encoding='utf-8'))
    if policy['configuration'] != manifest['configuration']:
        raise ArtifactError('Frozen model policy mismatch')
    entries = [manifest['frozen_policy'], manifest['extractor']]
    for category in CATEGORIES:
        item = manifest['categories'][category]
        cal = json.loads(verified_path(root, item['calibration']).read_text(encoding='utf-8'))
        if item['thresholds'] != dict(image=cal['image_threshold'], pixel=cal['pixel_threshold'], strict_greater=True):
            raise ArtifactError('Original threshold identity mismatch: '+category)
        entries.extend([item['bank'], item['calibration']])
    for entry in entries:
        verified_path(root, entry)
    return manifest, entries


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.is_relative_to(destination):
        raise ArtifactError('Export destination must not contain the source workspace')
    _, entries = validate(source)
    entries = [dict(path=MANIFEST.as_posix(), sha256=PIN, bytes=(source/MANIFEST).stat().st_size), *entries]
    for entry in entries:
        target = (destination / entry['path']).resolve()
        if not target.is_relative_to(destination):
            raise ArtifactError('Export path escapes destination')
        if target.exists():
            verified_path(destination, entry)  # Never overwrite mismatched exports.
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry['path'], target)
            verified_path(destination, entry)
    validate(destination)
    return dict(status='passed', categories=15, files=len(entries), bytes=sum(e['bytes'] for e in entries), manifest_sha256=PIN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    try:
        if args.export:
            result = export(args.root, args.export)
        else:
            _, entries = validate(args.root)
            result = dict(status='passed', categories=15, files=len(entries)+1, manifest_sha256=PIN)
        print(json.dumps(result, indent=2))
    except ArtifactError as exc:
        parser.exit(1, 'Artifact validation failed: '+str(exc)+'. Restore the exact release artifact; no fitting is performed.\n')
    except (OSError, ValueError, KeyError):
        parser.exit(1, 'Artifact validation failed: check the pinned manifest and required bank/extractor/calibration files. No model fitting is performed.\n')


if __name__ == '__main__':
    main()
