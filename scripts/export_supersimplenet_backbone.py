"""Store the exact shared frozen inference extractor once, without extraction."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from anomaly_detection.supersimplenet import SSNExtractor
from anomaly_detection.data import sha256
frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())
torch.set_num_threads(4);model=SSNExtractor(frozen['configuration']['backbone'])
assert model.weight_sha256==frozen['weight_sha256']
path=ROOT/'checkpoints/supersimplenet/shared-resnet18-features.pt';path.parent.mkdir(parents=True,exist_ok=True)
if not path.exists():torch.save(dict(state_dict=model.state_dict(),original_weight_sha256=model.weight_sha256),path)
saved=torch.load(path,weights_only=True,map_location='cpu')
assert all(torch.equal(value,saved['state_dict'][key]) for key,value in model.state_dict().items())
metadata=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),bytes=path.stat().st_size,
    parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,original_weight_sha256=model.weight_sha256,
    scope='One shared frozen layer1-3 extractor; category-specific heads are stored separately; feature caches are not needed at inference')
with (ROOT/'reports/supersimplenet/backbone_storage.json').open('x') as stream:json.dump(metadata,stream,indent=2)
print(json.dumps(metadata,indent=2))
