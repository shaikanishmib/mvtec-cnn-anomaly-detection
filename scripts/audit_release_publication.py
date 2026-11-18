"""Audit staged release contents without staging, deleting or committing anything."""
from pathlib import Path
import subprocess
import json

ROOT=Path(__file__).resolve().parents[1]


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def audit():
    paths=git('diff','--cached','--name-only').decode().splitlines();assert paths
    explicit={'.dockerignore','.gitattributes','README.md','docs/milestone_status.md','docs/deployment_guide.md','compose.yaml','compose.cuda.yaml','tests/test_deployment.py'}
    prefixes=('deployment/','reports/deployment/')
    scripts={'release_deployment.py','benchmark_release.py','capture_release_environment.py','audit_release_dependencies.py','resolve_cuda_dependencies.py','verify_release_reliability.py','measure_release_startup.py','validate_release.py','write_release_report.py','audit_release_publication.py'}
    for path in paths:
        assert path in explicit or path.startswith(prefixes) or (path.startswith('scripts/') and Path(path).name in scripts),path
        assert not any(part in {'node_modules','outputs','__pycache__','.venv','dist','checkpoints','data','.git'} for part in Path(path).parts),path
        assert Path(path).suffix not in {'.png','.jpg','.jpeg','.pt','.pth','.npy','.npz','.bin','.tar','.zip'},path
        assert not Path(path).name.startswith('.env') or Path(path).name=='.env.example',path
        data=git('cat-file','--filters','--path='+path,':'+path)
        assert data==(ROOT/path).read_bytes(),'Unstaged byte difference: '+path
        assert len(data)<1_000_000,path
        text=data.decode('utf-8').lower()
        denied=['c:'+chr(92)+'users'+chr(92),'c:'+'/users/','-----begin '+'private key-----','gh'+'p_']
        assert not any(item in text for item in denied),path
    protected=json.loads((ROOT/'reports/deployment/research_protection.json').read_text(encoding='utf-8'))
    assert not set(paths).intersection(protected)
    assert not git('diff','--name-only').strip()
    print(json.dumps(dict(status='passed',files=len(paths),no_models_data_caches_or_secrets=True,index_matches_worktree=True,protected_files_unchanged=True),indent=2))


if __name__=='__main__':audit()
