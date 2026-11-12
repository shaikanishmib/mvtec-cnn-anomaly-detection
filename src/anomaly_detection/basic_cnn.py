"""A small, scratch-trained segmentation CNN and fitting-only synthetic defects."""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class BasicCNN(nn.Module):
    """B,3,H,W -> B,1,H,W; H and W must be divisible by eight."""
    def __init__(self, width=16):
        super().__init__()
        if width % 4:
            raise ValueError("Width must be divisible by four for GroupNorm")
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(4, out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(4, out_channels), nn.ReLU())
        self.encoder1 = block(3, width)
        self.encoder2 = block(width, 2*width)
        self.encoder3 = block(2*width, 4*width)
        self.head = nn.Sequential(nn.Conv2d(5*width, width, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(width, 1, 1))
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or any(s % 8 for s in image.shape[-2:]):
            raise ValueError("Expected B,3,H,W with H,W divisible by eight")
        shallow = self.encoder1(2*image-1)
        middle = self.encoder2(shallow)
        coarse = self.encoder3(middle)
        enlarged = coarse.repeat_interleave(4, -2).repeat_interleave(4, -1)
        logits = self.head(torch.cat((shallow, enlarged), dim=1))
        return logits.repeat_interleave(2, -2).repeat_interleave(2, -1)


def synthetic_batch(images, generator, strategy="mixed", normal_probability=.5):
    """Edit a batch of fitting normals; output exact altered-pixel binary masks.

    Caller must enforce fitting provenance. Donors are shifted other images in
    this same batch. No official annotations, global image perturbation, padding
    or border crossing is used. Generator device must match images.device.
    """
    if strategy not in {"mixed", "patch", "perturb"} or not 0 <= normal_probability <= 1:
        raise ValueError("Unknown strategy or invalid normal probability")
    if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16:
        raise ValueError("Expected B,3,H,W RGB images at least 16 pixels wide/high")
    b, _, h, w = images.shape
    def rand(*shape):
        return torch.rand(shape, device=images.device, generator=generator)
    yy = torch.arange(h, device=images.device).view(1, h, 1) / (h-1)
    xx = torch.arange(w, device=images.device).view(1, 1, w) / (w-1)
    cx, cy = .25 + .5*rand(b, 1, 1), .25 + .5*rand(b, 1, 1)
    rx, ry = .04 + .15*rand(b, 1, 1), .04 + .15*rand(b, 1, 1)
    angle = 2*torch.pi*rand(b, 1, 1)
    dx, dy = xx-cx, yy-cy
    u = (dx*angle.cos() + dy*angle.sin()) / rx
    v = (-dx*angle.sin() + dy*angle.cos()) / ry
    radial = u.square() + v.square()
    # Slight lobes, soft finite support and an explicit interior margin.
    edge = 1 + .18*torch.sin(3*torch.atan2(v, u) + angle)
    alpha = ((edge-radial) / .35).clamp(0, 1)
    interior = (xx > .04) & (xx < .96) & (yy > .04) & (yy < .96)
    alpha = alpha * interior * (rand(b, 1, 1) >= normal_probability)
    alpha = alpha.unsqueeze(1) * (.6+.35*rand(b, 1, 1, 1))
    donor = torch.roll(images.roll(1, 0), shifts=(h//3, w//5), dims=(-2, -1))
    patch = images*(1-alpha) + donor*alpha
    color = (rand(b, 3, 1, 1)-.5)*.8
    grayscale = (images[:, :1].eq(images[:, 1:2]) &
                 images[:, :1].eq(images[:, 2:3])).flatten(1).all(1).view(b, 1, 1, 1)
    color = torch.where(grayscale, color.mean(1, keepdim=True), color)
    phase = 2*torch.pi*rand(b, 1, 1, 1)
    field = .15*torch.sin(17*xx.unsqueeze(1) + 13*yy.unsqueeze(1) + phase)
    perturb = (images + alpha*(color+field)).clamp(0, 1)
    choose_patch = rand(b, 1, 1, 1) < .5
    if strategy == "patch": choose_patch.fill_(True)
    if strategy == "perturb": choose_patch.fill_(False)
    edited = torch.where(choose_patch, patch, perturb)
    mask = edited.ne(images).any(dim=1, keepdim=True).to(images.dtype)
    return edited, mask


def segmentation_loss(logits, masks, positive_weight=8., dice_weight=0.):
    bce = F.binary_cross_entropy_with_logits(logits.float(), masks.float(),
                  pos_weight=torch.tensor(positive_weight, device=logits.device))
    probabilities = logits.float().sigmoid()
    intersection = (probabilities*masks).sum((1, 2, 3))
    dice = 1 - ((2*intersection+1) /
                (probabilities.sum((1, 2, 3)) + masks.sum((1, 2, 3))+1))
    return bce + dice_weight*dice.mean()


def image_scores(logits):
    return logits.float().sigmoid().flatten(1).amax(1)
