"""Bounded ResNet18 PatchCore variant, not a paper-equivalent reproduction."""
import math
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PatchExtractor(nn.Module):
    """Frozen layer2/layer3 features; whole-image geometry preserves edge defects.

    Official V1 mean/std and torchvision weight transforms are used, with a
    documented crop-size override: inputs are already whole-image 256 squares,
    then resize/crop 256 (identity geometry). Default classification 224 center
    cropping is deliberately avoided. No learned test/normal statistics.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights=ResNet18_Weights.IMAGENET1K_V1
        if pretrained:
            path=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
            if not path.is_file():
                raise FileNotFoundError('Register a weight acquisition before downloading: '+str(path))
            self.weight_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            if not self.weight_sha256.startswith(path.stem.split('-')[-1]):
                raise ValueError('Cached ImageNet weight hash does not match torchvision identity')
        else:self.weight_sha256='untrained-test-only'
        network=resnet18(weights=weights if pretrained else None)
        self.stem=nn.Sequential(network.conv1,network.bn1,network.relu,network.maxpool)
        self.layer1=network.layer1
        self.layer2=network.layer2
        self.layer3=network.layer3
        self.transform=weights.transforms(crop_size=256,resize_size=256,antialias=True)
        self.requires_grad_(False)
        self.eval()

    def train(self,mode=True):
        # Even parent .train() must not update ImageNet BatchNorm statistics.
        return super().train(False)

    @torch.inference_mode()
    def feature_maps(self,images):
        if images.ndim!=4 or images.shape[1:]!=(3,256,256) or not images.is_floating_point():
            raise ValueError('Expected B,3,256,256 float RGB in [0,1]')
        x=self.transform(images)
        first=self.layer1(self.stem(x))
        second=self.layer2(first)
        third=self.layer3(second)
        return first,second,third

    @torch.inference_mode()
    def forward(self,images):
        _,second,third=self.feature_maps(images)
        return align_features(second,third)


def align_features(shallow,deep):
    """3x3 local average pooling, coarse bilinear alignment, channel concatenation."""
    if shallow.ndim!=4 or deep.ndim!=4 or shallow.shape[0]!=deep.shape[0]:
        raise ValueError('Matching B,C,H,W feature batches required')
    local=F.avg_pool2d(shallow,3,1,1)
    coarse=F.avg_pool2d(deep,3,1,1)
    coarse=F.interpolate(coarse,size=local.shape[-2:],mode='bilinear',align_corners=False)
    return torch.cat([local,coarse],dim=1)


def patch_embeddings(features):
    if features.ndim!=4:raise ValueError('Expected B,C,H,W features')
    return features.permute(0,2,3,1).reshape(-1,features.shape[1]).contiguous()


@torch.inference_mode()
def projected_coreset(embeddings,budget,seed=2026,candidate_limit=20000,projection_dim=32):
    """Random capped candidate pool -> projected farthest-first traversal.

    Returns unique indices into original embeddings plus measured projected
    coverage distances. Search uses original full-dimensional vectors, not the
    projection. The cap is a disclosed approximation to a full-data coreset.
    """
    if embeddings.ndim!=2 or not torch.isfinite(embeddings).all():raise ValueError('Finite N,D matrix required')
    n,d=embeddings.shape
    if not 1<=budget<=min(n,candidate_limit) or projection_dim<1:raise ValueError('Invalid coreset budget')
    generator=torch.Generator().manual_seed(seed)
    candidates=torch.randperm(n,generator=generator)[:candidate_limit].to(embeddings.device)
    projection=torch.randn(d,projection_dim,generator=generator).to(embeddings.device)/math.sqrt(projection_dim)
    reduced=embeddings[candidates].float()@projection
    distances=torch.full((len(candidates),),float('inf'),device=embeddings.device)
    available=torch.ones(len(candidates),dtype=torch.bool,device=embeddings.device)
    chosen=[]
    current=0  # randomized candidate order supplies a seeded initial center
    for _ in range(budget):
        chosen.append(current)
        available[current]=False
        delta=(reduced-reduced[current]).square().sum(1)
        distances=torch.minimum(distances,delta)
        current=int(distances.masked_fill(~available,-1).argmax())
    indices=candidates[torch.tensor(chosen,device=embeddings.device)]
    coverage=dict(candidate_count=len(candidates),projected_radius=float(distances.max().sqrt()),
                  projected_mean_distance=float(distances.sqrt().mean()),projection_dim=projection_dim)
    return indices,coverage


@torch.inference_mode()
def nearest_neighbors(queries,bank,query_chunk=512,bank_chunk=2048):
    """Exhaustive float32 L2 search with a bounded Qchunk x Mchunk matrix.

    GEMM finds the nearest index; direct subtraction refines the selected
    distance, avoiding cancellation in self/near-self distance values. As with
    float32 flat search, almost-tied index choices are numerical-precision limited.
    No Q x M x D tensor or complete image-batch distance matrix is constructed.
    """
    if queries.ndim!=2 or bank.ndim!=2 or queries.shape[1]!=bank.shape[1] or not len(bank):
        raise ValueError('Compatible query and nonempty bank matrices required')
    if query_chunk<1 or bank_chunk<1 or queries.device!=bank.device:raise ValueError('Invalid chunks/device')
    distances=[];indices=[]
    for q in queries.float().split(query_chunk):
        best=torch.full((len(q),),float('inf'),device=q.device)
        best_index=torch.zeros(len(q),dtype=torch.long,device=q.device)
        for start in range(0,len(bank),bank_chunk):
            b=bank[start:start+bank_chunk].float()
            squared=(q.square().sum(1,keepdim=True)+b.square().sum(1)[None]-2*(q@b.T)).clamp_min_(0)
            values,positions=squared.min(1)
            improved=values<best
            best=torch.minimum(best,values)
            best_index=torch.where(improved,positions+start,best_index)
        distances.append((q-bank[best_index]).square().sum(1).sqrt())
        indices.append(best_index)
    return torch.cat(distances),torch.cat(indices)


def pixel_maps(patch_distances,shape=(256,256)):
    if patch_distances.ndim!=3:raise ValueError('Expected B,Hpatch,Wpatch')
    return F.interpolate(patch_distances[:,None],size=shape,mode='bilinear',align_corners=False)[:,0]


def image_scores(patch_distances,aggregation='max'):
    flat=patch_distances.flatten(1)
    if aggregation=='max':return flat.amax(1)
    if aggregation=='top1pct':return flat.topk(max(1,math.ceil(flat.shape[1]*.01)),dim=1).values.mean(1)
    raise ValueError('Expected max or top1pct')


def save_bank(path,bank,indices,category,config,provenance):
    if path.exists():raise FileExistsError(path)
    torch.save(dict(bank=bank.cpu(),indices=indices.cpu(),category=category,config=config,provenance=provenance),path)


def load_bank(path,category,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['category']!=category:raise ValueError('Category-specific bank mismatch')
    return saved['bank'].to(device),saved
