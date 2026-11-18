"""Run the explicit asset-free test scope identically locally and in CI."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def public_tests():
    return [line.strip() for line in (ROOT/'ci/public-tests.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junitxml',default='outputs/portfolio/public-tests.xml')
    parser.add_argument('--basetemp',default='.pytest-tmp-public')
    parser.add_argument('--cache-dir',default='.pytest_cache/public')
    args=parser.parse_args()
    Path(args.junitxml).parent.mkdir(exist_ok=True,parents=True)
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q',*public_tests(),
        '--basetemp='+args.basetemp,'-o','cache_dir='+args.cache_dir,'--junitxml='+args.junitxml],cwd=ROOT))


if __name__=='__main__':main()
