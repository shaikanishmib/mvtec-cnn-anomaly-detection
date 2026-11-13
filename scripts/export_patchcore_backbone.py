"""Retain one shared frozen feature-extractor artifact; no image extraction."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torchvision.models import ResNet18_Weights
from anomaly_detection.patchcore import PatchExtractor
from anomaly_detection.cnn_research import ROOT
from anomaly_detection.data import sha256
torch.set_num_threads(4)
model=PatchExtractor()
frozen=json.loads((ROOT/'configs/patchcore_frozen.json').read_text())
assert model.weight_sha256==frozen['weight_sha256']
path=ROOT/'checkpoints/patchcore/shared-resnet18-features.pt'
if not path.exists():torch.save(dict(state_dict=model.state_dict(),original_weight_sha256=model.weight_sha256),path)
saved=torch.load(path,weights_only=True,map_location='cpu')
for name,value in model.state_dict().items():assert torch.equal(value,saved['state_dict'][name])
original=Path(torch.hub.get_dir())/'checkpoints'/Path(ResNet18_Weights.IMAGENET1K_V1.url).name
metadata=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),bytes=path.stat().st_size,
    parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,
    original_weight_file_bytes=original.stat().st_size,original_weight_sha256=model.weight_sha256,
    scope='One shared frozen layer1-3 feature extractor, not repeated per category or seed')
with (ROOT/'reports/patchcore/backbone_storage.json').open('x') as stream:json.dump(metadata,stream,indent=2)
print(json.dumps(metadata,indent=2))
