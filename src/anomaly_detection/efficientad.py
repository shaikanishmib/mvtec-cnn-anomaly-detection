"""EfficientAD-S, paper Tables 6/8 (padded 64x64 features).

Independent implementation from Batzner et al., WACV 2024. Checkpoint-compatible
PDN layer numbering with nelson1425/EfficientAD (Apache-2.0); no source copied.
Inputs are ImageNet-normalized RGB, 256x256. Teacher weights must be supplied.
"""
import torch
from torch import nn
from torch.nn import functional as F


def pdn(channels=384):
    return nn.Sequential(nn.Conv2d(3,128,4,padding=3),nn.ReLU(),
        nn.AvgPool2d(2,2,padding=1),nn.Conv2d(128,256,4,padding=3),
        nn.ReLU(),nn.AvgPool2d(2,2,padding=1),
        nn.Conv2d(256,256,3,padding=1),nn.ReLU(),nn.Conv2d(256,channels,4))


def autoencoder():
    layers=[]
    for a,b in zip((3,32,32,64,64),(32,32,64,64,64)):
        layers.extend((nn.Conv2d(a,b,4,2,1),nn.ReLU()))
    layers.append(nn.Conv2d(64,64,8))
    for size in (3,8,15,32,63,127):
        layers.extend((nn.Upsample(size=size,mode='bilinear',align_corners=False),
            nn.Conv2d(64,64,4,padding=2),nn.ReLU(),nn.Dropout(.2)))
    layers.extend((nn.Upsample(size=64,mode='bilinear',align_corners=False),
        nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,384,3,padding=1)))
    return nn.Sequential(*layers)


def hard_feature_loss(distance, quantile=.999):
    if not 0 <= quantile <= 1:
        raise ValueError('quantile must be in [0,1]')
    cutoff=torch.quantile(distance.detach().flatten(),quantile)
    return distance[distance >= cutoff].mean()


class EfficientAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher=pdn().requires_grad_(False)
        self.student=pdn(768)
        self.autoencoder=autoencoder()
        self.register_buffer('mean',torch.zeros(1,384,1,1))
        self.register_buffer('std',torch.ones(1,384,1,1))
        self.register_buffer('map_quantiles',torch.tensor([[0.,1.],[0.,1.]]))

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def fit_feature_normalization(self,images):
        total=torch.zeros_like(self.mean,dtype=torch.float64)
        squares=total.clone(); n=0
        for x in images:
            y=self.teacher(x).double()
            total+=y.sum((0,2,3),keepdim=True)
            squares+=y.square().sum((0,2,3),keepdim=True)
            n+=y.shape[0]*y.shape[2]*y.shape[3]
        if not n: raise ValueError('empty fitting data')
        mean=total/n
        self.mean.copy_(mean)
        self.std.copy_((squares/n-mean.square()).clamp_min(1e-12).sqrt())

    def teacher_features(self,x):
        with torch.no_grad(): return (self.teacher(x)-self.mean)/self.std

    def losses(self,x,augmented,external=None,quantile=.999):
        target=self.teacher_features(x)
        local=self.student(x)[:,:384]
        hard=hard_feature_loss((target-local).square(),quantile)
        penalty=x.new_zeros(()) if external is None else self.student(external)[:,:384].square().mean()
        ae=self.autoencoder(augmented)
        # The reference joint objective propagates STAE into both trainable branches.
        ae_loss=(ae-self.teacher_features(augmented)).square().mean()
        stae=(self.student(augmented)[:,384:]-ae).square().mean()
        return dict(hard=hard,penalty=penalty,autoencoder=ae_loss,student_autoencoder=stae,
                    total=hard+penalty+ae_loss+stae)

    def raw_maps(self,x):
        s=self.student(x)
        local=(self.teacher_features(x)-s[:,:384]).square().mean(1,keepdim=True)
        global_map=(self.autoencoder(x)-s[:,384:]).square().mean(1,keepdim=True)
        return tuple(F.interpolate(m,size=x.shape[-2:],mode='bilinear',align_corners=False)
                     for m in (local,global_map))

    @torch.no_grad()
    def fit_map_normalization(self,images):
        if self.training: raise ValueError('map fitting requires eval mode')
        pairs=[self.raw_maps(x) for x in images]
        if not pairs: raise ValueError('empty development data')
        for i in range(2):
            values=torch.cat([p[i].flatten() for p in pairs])
            q=torch.quantile(values,values.new_tensor([.9,.995]))
            if q[1] <= q[0]: raise ValueError('degenerate map quantiles')
            self.map_quantiles[i].copy_(q)

    def forward(self,x):
        maps=self.raw_maps(x)
        normalized=[.1*(m-q[0])/(q[1]-q[0]).clamp_min(1e-12)
                    for m,q in zip(maps,self.map_quantiles)]
        combined=(normalized[0]+normalized[1])*.5
        return combined,combined.flatten(1).amax(1)


def save_checkpoint(path,model,metadata):
    torch.save(dict(schema=1,state_dict=model.state_dict(),metadata=metadata),path)


def load_checkpoint(path,device='cpu'):
    payload=torch.load(path,map_location=device,weights_only=True)
    if payload['schema'] != 1: raise ValueError('unsupported checkpoint')
    model=EfficientAD().to(device)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval(),payload['metadata']
