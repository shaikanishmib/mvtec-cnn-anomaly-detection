"""Read-only portfolio Git index audit: no datasets, models, secrets or evidence edits."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    names=git('diff','--cached','--name-only').decode().splitlines();assert names
    allowed={'.gitattributes','README.md','SECURITY.md','docs/milestone_status.md','docs/index.md',
        'docs/project_overview.md','docs/licensing_and_attribution.md','docs/reproducibility.md',
        '.github/workflows/source-quality.yml','tests/test_portfolio.py'}
    scripts={'check_portfolio.py','run_public_tests.py','review_portfolio_security.py','validate_portfolio.py','audit_portfolio_publication.py'}
    for name in names:
        assert name in allowed or name.startswith(('ci/','reports/portfolio/')) or (name.startswith('scripts/') and Path(name).name in scripts),name
        assert not any(part in {'outputs','node_modules','data','checkpoints','__pycache__','.git','.venv'} for part in Path(name).parts),name
        assert Path(name).suffix not in {'.pt','.pth','.npy','.npz','.bin','.png','.jpg','.zip','.key','.pem'},name
        data=git('cat-file','--filters','--path='+name,':'+name)
        assert data==(ROOT/name).read_bytes(),name
        assert len(data)<750_000,name
        text=data.decode('utf-8').lower()
        denied=['c:'+chr(92)+'users'+chr(92),'c:'+'/users/','-----begin '+'private key-----','gh'+'p_']
        assert not any(item in text for item in denied),name
    protected=json.loads((ROOT/'reports/portfolio/preservation.json').read_text(encoding='utf-8'))
    assert not set(names).intersection(protected)
    assert not git('diff','--name-only').strip()
    print(json.dumps(dict(status='passed',files=len(names),immutable_evidence_unchanged=True,
        no_secrets_data_models_or_caches=True,index_matches_worktree=True),indent=2))


if __name__=='__main__':main()
