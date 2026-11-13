"""Final phases require exact frozen policy and complete fitting before test."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from anomaly_detection.supersimplenet import SSNExtractor
from anomaly_detection.ssn_final import fit_frozen
from anomaly_detection.ssn_evaluation import evaluate_frozen
from anomaly_detection.data import sha256
from anomaly_detection.basic_cnn import seed_everything

def main():
    torch.set_num_threads(4);seed_everything(2026);torch.backends.cuda.matmul.allow_tf32=False
    phase=sys.argv[1];assert phase in ['fit','evaluate']
    path=ROOT/'configs/supersimplenet_frozen.json';frozen=json.loads(path.read_text())
    assert sha256(path)==json.loads((ROOT/'reports/supersimplenet/freeze_identity.json').read_text())['sha256']
    for source,expected in frozen['source_sha256'].items():assert sha256(ROOT/source)==expected,source
    extractor=SSNExtractor(frozen['configuration']['backbone']).cuda()
    assert extractor.weight_sha256==frozen['weight_sha256']
    runs=[(seed,category,f'04-final-{category}-s{seed}-v1') for seed in frozen['seeds'] for category in frozen['categories']]
    if phase=='evaluate':
        assert all((ROOT/f'reports/supersimplenet/runs/{identifier}-result.json').is_file() for _,_,identifier in runs),'Fit all registered models before any test access'
    for seed,category,identifier in runs:
        result_path=ROOT/f'reports/supersimplenet/runs/{identifier}-result.json'
        if phase=='fit':
            if not result_path.exists():fit_frozen(identifier,category,dict(frozen['configuration'],seed=seed),extractor,sha256(path))
        else:
            evaluation=f'04-test-{category}-s{seed}-v1'
            if (ROOT/f'reports/supersimplenet/evaluation/{evaluation}-result.json').exists():continue
            result=evaluate_frozen(evaluation,json.loads(result_path.read_text()),sha256(path),extractor)
            print(json.dumps({k:result[k] for k in ['category','seed','image_auroc','pixel_auroc','evaluation_seconds']}),flush=True)

if __name__=='__main__':main()
