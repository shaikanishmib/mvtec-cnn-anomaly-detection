"""ICPR-style SuperSimpleNet components with explicit resource variants.

Architecture/loss design follows blaz-r/SuperSimpleNet (MIT, copyright 2024
Blaž Rolih); see docs/third_party/SuperSimpleNet_LICENSE.txt. Perlin gradient
noise is implemented independently with explicit seeded generators.
"""
import hashlib,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,wide_resnet50_2,ResNet18_Weights,Wide_ResNet50_2_Weights
from torchvision.transforms.functional import gaussian_blur


class SSNExtractor(nn.Module):
    """Frozen ImageNet V1 layers 2/3, upscale FIRST to 64, then pool 3x3.

    PatchCore's pooled 32-grid cache is NOT compatible with this computation.
    Uses official V1 normalization with whole-image 256 geometry (paper setting).
    """
    def __init__(self,backbone='resnet18',pretrained=True):
        super().__init__()
        if backbone not in ['resnet18','wide_resnet50_2']:raise ValueError('Unsupported backbone')
        self.backbone=backbone
        weights=ResNet18_Weights.IMAGENET1K_V1 if backbone=='resnet18' else Wide_ResNet50_2_Weights.IMAGENET1K_V1
        self.weight_url=weights.url
        self.channels=384 if backbone=='resnet18' else 1536
        network=(resnet18 if backbone=='resnet18' else wide_resnet50_2)(weights=None)
        if pretrained:
            name=Path(weights.url).name
            local=Path(__file__).resolve().parents[2]/'checkpoints/pretrained'/name
            path=local if local.exists() else Path(torch.hub.get_dir())/'checkpoints'/name
            if not path.is_file():raise FileNotFoundError('Acquire and register official weights: '+weights.url)
            self.weight_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            if not self.weight_sha256.startswith(path.stem.split('-')[-1]):raise ValueError('Weight identity mismatch')
            network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        else:self.weight_sha256='untrained-test-only'
        self.stem=nn.Sequential(network.conv1,network.bn1,network.relu,network.maxpool)
        self.layer1=network.layer1;self.layer2=network.layer2;self.layer3=network.layer3
        self.transform=weights.transforms(resize_size=256,crop_size=256,antialias=True)
        self.requires_grad_(False);self.eval()

    def train(self,mode=True):return super().train(False)

    @torch.no_grad()
    def feature_maps(self,images):
        if images.ndim!=4 or images.shape[1:]!=(3,256,256):raise ValueError('Expected B,3,256,256 RGB')
        first=self.layer1(self.stem(self.transform(images)))
        second=self.layer2(first);third=self.layer3(second)
        return second,third

    def forward(self,images):
        second,third=self.feature_maps(images)
        return aggregate_features(second,third)


def aggregate_features(second,third):
    size=(second.shape[-2]*2,second.shape[-1]*2)
    aligned=[F.interpolate(x,size=size,mode='bilinear',align_corners=True) for x in [second,third]]
    return F.avg_pool2d(torch.cat(aligned,dim=1),3,1,1)


def perlin_noise(height,width,res_y,res_x,generator):
    """Quintic-interpolated random gradient lattice, not resized white noise."""
    y=torch.arange(height,dtype=torch.float32)*res_y/height
    x=torch.arange(width,dtype=torch.float32)*res_x/width
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    iy=yy.long();ix=xx.long();dy=yy-iy;dx=xx-ix
    angles=2*math.pi*torch.rand(res_y+1,res_x+1,generator=generator)
    gy,gx=angles.sin(),angles.cos()
    def dot(oy,ox):return gy[iy+oy,ix+ox]*(dy-oy)+gx[iy+oy,ix+ox]*(dx-ox)
    fade=lambda t:6*t**5-15*t**4+10*t**3
    upper=torch.lerp(dot(0,0),dot(0,1),fade(dx))
    lower=torch.lerp(dot(1,0),dot(1,1),fade(dx))
    return math.sqrt(2)*torch.lerp(upper,lower,fade(dy))


def feature_anomalies(adapted,mask_generator,noise_generator,noise_std=.015,threshold=.2,normal_probability=.5):
    """Duplicate features; independent Perlin/no-anomaly decisions for BOTH copies.

    All source features are fitting normals, so real-anomaly masks are empty.
    Labels are derived from the actual binary perturbation support, never guessed.
    The input is not modified, allowing immutable clean calibration caches.
    """
    if adapted.ndim!=4 or noise_std<=0 or not 0<=normal_probability<=1:raise ValueError('Invalid synthetic configuration')
    duplicated=torch.cat([adapted,adapted],dim=0)
    b,_,h,w=duplicated.shape;masks=[]
    max_y=min(6,int(math.log2(h))+1);max_x=min(6,int(math.log2(w))+1)
    for _ in range(b):
        ry=2**int(torch.randint(0,max_y,(1,),generator=mask_generator))
        rx=2**int(torch.randint(0,max_x,(1,),generator=mask_generator))
        support=(perlin_noise(h,w,ry,rx,mask_generator)>threshold).float()
        if float(torch.rand((),generator=mask_generator))<normal_probability:support.zero_()
        masks.append(support)
    mask=torch.stack(masks)[:,None].to(adapted.device)
    noise=torch.randn(duplicated.shape,device=adapted.device,generator=noise_generator)*noise_std
    perturbed=duplicated+noise*mask
    labels=mask.flatten(1).any(1).float()
    return perturbed,mask,labels


class SuperSimpleHeads(nn.Module):
    """Reference: C=1536, seg=1024, detection=128; Lite: C=384,256,32.

    The original ICPR formulation uses adapted features for BOTH heads. Every
    classification route through the segmentation map is detached by default;
    classification can still update the adapter through its feature input.
    """
    def __init__(self,channels=384,seg_width=256,det_width=32,adapter_trainable=True,stop_grad=True):
        super().__init__()
        self.adapter=nn.Conv2d(channels,channels,1)
        self.seg=nn.Sequential(nn.Conv2d(channels,seg_width,1),nn.BatchNorm2d(seg_width),
            nn.LeakyReLU(.2),nn.Conv2d(seg_width,1,1,bias=False))
        self.det=nn.Sequential(nn.Conv2d(channels+1,det_width,5,padding=2),nn.BatchNorm2d(det_width),nn.ReLU())
        self.fc=nn.Linear(2*det_width+2,1)
        self.stop_grad=stop_grad
        for module in self.modules():
            if isinstance(module,(nn.Conv2d,nn.Linear)):nn.init.xavier_normal_(module.weight)
            if isinstance(module,nn.BatchNorm2d):nn.init.ones_(module.weight)
        self.adapter.requires_grad_(adapter_trainable)

    def discriminate(self,adapted):
        segmentation=self.seg(adapted)
        copied=segmentation.detach() if self.stop_grad else segmentation
        detection=self.det(torch.cat([adapted,copied],dim=1))
        pooled=torch.cat([detection.amax((2,3)),detection.mean((2,3)),
                          copied.amax((2,3)),copied.mean((2,3))],dim=1)
        return segmentation,self.fc(pooled)[:,0]

    def forward(self,features):return self.discriminate(self.adapter(features))

    def synthetic_forward(self,features,mask_generator,noise_generator,**kwargs):
        adapted=self.adapter(features)
        perturbed,mask,labels=feature_anomalies(adapted,mask_generator,noise_generator,**kwargs)
        segmentation,scores=self.discriminate(perturbed)
        return segmentation,scores,mask,labels


def focal_loss(logits,targets,gamma=4):
    ce=F.binary_cross_entropy_with_logits(logits.float(),targets.float(),reduction='none')
    p=logits.float().sigmoid();pt=p*targets+(1-p)*(1-targets)
    return (ce*(1-pt)**gamma).mean()


def training_loss(segmentation,scores,mask,labels,margin=.5,segmentation_loss='margin_focal'):
    if segmentation.shape!=mask.shape or scores.shape!=labels.shape:raise ValueError('Target/output shape mismatch')
    normal=segmentation[mask==0];anomaly=segmentation[mask>0]
    zero=segmentation.sum()*0
    hinge=(F.relu(normal+margin).mean() if len(normal) else zero)+(F.relu(margin-anomaly).mean() if len(anomaly) else zero)
    seg_focal=focal_loss(segmentation,mask)
    if segmentation_loss not in ['margin_focal','focal']:raise ValueError('Unknown loss')
    seg=seg_focal+(hinge if segmentation_loss=='margin_focal' else zero)
    detection=focal_loss(scores,labels)
    return seg+detection,dict(segmentation_focal=float(seg_focal.detach()),margin_loss=float(hinge.detach()),detection_focal=float(detection.detach()))


def anomaly_maps(segmentation,size=(256,256)):
    maps=F.interpolate(segmentation,size=size,mode='bilinear',align_corners=False)
    return gaussian_blur(maps,[25,25],[4.,4.])[:,0]


def image_scores(maps,detection,aggregation='head'):
    if aggregation=='head':return detection
    if aggregation=='max':return maps.flatten(1).amax(1)
    if aggregation=='top1pct':
        flat=maps.flatten(1);return flat.topk(math.ceil(flat.shape[1]*.01),dim=1).values.mean(1)
    raise ValueError('Unknown aggregation')


def optimizer_for(model):
    groups=[dict(params=model.seg.parameters(),lr=.0002,weight_decay=.00001),
        dict(params=list(model.det.parameters())+list(model.fc.parameters()),lr=.0002,weight_decay=.00001)]
    if any(p.requires_grad for p in model.adapter.parameters()):groups.append(dict(params=model.adapter.parameters(),lr=.0001,weight_decay=.01))
    return torch.optim.AdamW(groups)
