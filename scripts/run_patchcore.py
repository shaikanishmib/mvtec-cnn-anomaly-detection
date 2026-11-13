"""Explicit phases; cannot advance into all-category evaluation implicitly."""
from pathlib import Path
import sys,os,json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from anomaly_detection.basic_cnn import seed_everything
from anomaly_detection.patchcore import PatchExtractor
from anomaly_detection.patchcore_runs import ROOT,build_bank
from anomaly_detection.data import sha256

torch.set_num_threads(4);seed_everything(2026);torch.backends.cuda.matmul.allow_tf32=False
if __name__=='__main__':
    extractor=PatchExtractor().cuda()
    phase=sys.argv[1]
    base=dict(bank_budget=256,candidate_limit=20000,projection_dim=32,seed=2026,
        backbone='resnet18',weights='IMAGENET1K_V1',layers=['layer2','layer3'],resolution=256,
        normalize=False,query_chunk=512,bank_chunk=2048)
    if phase=='development':
        for budget in [256,1024,2048]:
            for category in ['bottle','cable','carpet','tile']:
                identifier=f'03-dev-b{budget}-{category}-s2026-v1'
                if (ROOT/f'reports/patchcore/runs/{identifier}-result.json').exists():continue
                build_bank(identifier,category,dict(base,bank_budget=budget),extractor)
    elif phase in ['final-fit','final-evaluate']:
        path=ROOT/'configs/patchcore_frozen.json';frozen=json.loads(path.read_text())
        assert sha256(path)==json.loads((ROOT/'reports/patchcore/freeze_identity.json').read_text())['sha256']
        for source,expected in frozen['source_sha256'].items():assert sha256(ROOT/source)==expected,source
        for seed in frozen['seeds']:
            for category in frozen['categories']:
                identifier=f"03-final-b{frozen['configuration']['bank_budget']}-{category}-s{seed}-v1"
                result_path=ROOT/f'reports/patchcore/runs/{identifier}-result.json'
                if phase=='final-fit':
                    if not result_path.exists():build_bank(identifier,category,dict(frozen['configuration'],seed=seed),extractor,final=True)
                else:
                    from anomaly_detection.patchcore_evaluation import evaluate_frozen
                    evaluation=f"03-test-b{frozen['configuration']['bank_budget']}-{category}-s{seed}-v1"
                    if (ROOT/f'reports/patchcore/evaluation/{evaluation}-result.json').exists():continue
                    result=evaluate_frozen(evaluation,json.loads(result_path.read_text()),sha256(path),extractor)
                    print(json.dumps({k:result[k] for k in ['category','seed','image_auroc','pixel_auroc','evaluation_seconds']}),flush=True)
    else:raise ValueError('Unknown phase')
