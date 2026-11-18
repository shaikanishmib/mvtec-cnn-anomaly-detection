"""Read-only audit of the proposed M12 Git index; never stages or commits files."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def audit():
    paths = git('diff', '--cached', '--name-only').decode().splitlines()
    assert paths, 'Stage the verified M12 publication before auditing'
    explicit = {'.gitattributes', 'README.md', 'docs/milestone_status.md', 'docs/frontend_guide.md',
        'scripts/validate_frontend.py', 'scripts/verify_frontend_browser.py', 'scripts/audit_frontend_publication.py'}
    report = json.loads((ROOT/'reports/frontend/validation.json').read_text(encoding='utf-8'))
    for path in paths:
        assert path in explicit or path.startswith(('frontend/', 'reports/frontend/', 'docs/images/inspection-dashboard-')), path
        assert not any(part in {'node_modules', 'dist', 'outputs', '.vite', 'test-results', 'playwright-report', '__pycache__'} for part in Path(path).parts), path
        assert Path(path).suffix not in {'.pt','.pth','.npz','.npy','.bin','.tsbuildinfo'}, path
        assert not Path(path).name.startswith('.env') or Path(path).name == '.env.example', path
        content = git('cat-file', '--filters', '--path='+path, ':'+path)
        assert content == (ROOT/path).read_bytes(), 'Index/working byte mismatch: '+path
        assert len(content) < 1_000_000, path
        if Path(path).suffix == '.png':
            assert hashlib.sha256(content).hexdigest() == report['screenshot_sha256'][Path(path).name]
        else:
            text = content.decode('utf-8')
            denied = ['c:'+chr(92)+'users'+chr(92), 'c:'+'/users/', '-----begin '+'private key-----', 'gh'+'p_']
            assert not any(value in text.lower() for value in denied), path
    protected = json.loads((ROOT/'reports/frontend/research_protection.json').read_text(encoding='utf-8'))
    assert not set(paths).intersection(protected)
    assert not git('diff', '--name-only').strip(), 'Unstaged tracked changes remain'
    print(json.dumps(dict(status='passed',staged_files=len(paths),dataset_images_weights_and_caches_staged=False,
        protected_baseline_files_modified=False,public_images='Only two hash-verified authored-fixture screenshots',
        index_matches_worktree=True),indent=2))


if __name__ == '__main__':
    audit()
