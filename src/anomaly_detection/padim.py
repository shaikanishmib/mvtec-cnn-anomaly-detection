"""PaDiM spatial Gaussians; frozen ImageNet features and bounded linear algebra."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .patchcore import PatchExtractor


def aligned_embedding(maps):
    """Nearest alignment preserves each coarse cell, then concatenates channels."""
    if len(maps) != 3 or any(x.ndim != 4 or x.shape[0] != maps[0].shape[0] for x in maps):
        raise ValueError('Three matching B,C,H,W batches required')
    return torch.cat([F.interpolate(x, size=maps[0].shape[-2:], mode='nearest') for x in maps], 1)


class PaDiMExtractor(PatchExtractor):
    """Reuse verified V1 weights/normalization; no PatchCore local pooling."""
    def __init__(self, dimension=100, seed=2026, pretrained=True):
        super().__init__(pretrained)
        if not 1 <= dimension <= 448:
            raise ValueError('Dimension must be in [1,448]')
        indices = torch.randperm(448, generator=torch.Generator().manual_seed(seed))[:dimension].sort().values
        self.register_buffer('indices', indices)

    @torch.inference_mode()
    def forward(self, images):
        return aligned_embedding(self.feature_maps(images)).index_select(1, self.indices)


class SpatialGaussian:
    """Unbiased covariance + epsilon I, per position, stored as Cholesky factors.

    Fit/score work in spatial blocks. Scoring uses multiple RHS rather than
    repeating the covariance tensor across images. No matrix inverse is formed.
    """
    def __init__(self, epsilon=.01, block=128):
        if epsilon <= 0 or block < 1:
            raise ValueError('Positive epsilon and block required')
        self.epsilon, self.block = float(epsilon), int(block)
        self.mean = self.cholesky = None

    @torch.inference_mode()
    def fit(self, features, device='cpu'):
        if features.ndim != 4 or len(features) < 2 or not torch.isfinite(features).all():
            raise ValueError('At least two finite N,C,H,W observations required')
        n, c, h, w = features.shape
        self.shape = (c, h, w)
        x = features.permute(2, 3, 1, 0).reshape(h*w, c, n)
        means, factors = [], []
        eye = torch.eye(c, device=device)
        for start in range(0, h*w, self.block):
            values = x[start:start+self.block].to(device=device, dtype=torch.float32)
            mean = values.mean(-1)
            centered = values - mean[..., None]
            covariance = centered @ centered.transpose(-1, -2) / (n-1) + self.epsilon*eye
            chol, info = torch.linalg.cholesky_ex(covariance)
            if info.any():
                raise RuntimeError('Covariance factorization failed; preserve run and inspect')
            means.append(mean); factors.append(chol)
        self.mean, self.cholesky = torch.cat(means), torch.cat(factors)
        return self

    @torch.inference_mode()
    def score(self, features):
        if self.mean is None:
            raise RuntimeError('Fit before scoring')
        if features.ndim != 4 or tuple(features.shape[1:]) != self.shape or not torch.isfinite(features).all():
            raise ValueError('Finite features matching fitted shape required')
        b, c, h, w = features.shape
        x = features.permute(2, 3, 1, 0).reshape(h*w, c, b)
        result = []
        for start in range(0, h*w, self.block):
            stop = start+self.block
            delta = x[start:stop].to(self.mean.device).float() - self.mean[start:stop, :, None]
            solved = torch.linalg.solve_triangular(self.cholesky[start:stop], delta, upper=False)
            result.append(solved.square().sum(1).clamp_min(0).sqrt())
        return torch.cat(result).T.reshape(b, h, w)

    def save(self, path, category, configuration, provenance):
        with Path(path).open('xb') as stream:
            torch.save(dict(mean=self.mean.cpu(), cholesky=self.cholesky.cpu(), shape=self.shape,
                            epsilon=self.epsilon, category=category, configuration=configuration,
                            provenance=provenance), stream)

    @classmethod
    def load(cls, path, category, device='cpu'):
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved['category'] != category:
            raise ValueError('Category mismatch')
        model = cls(saved['epsilon'])
        model.mean, model.cholesky, model.shape = saved['mean'], saved['cholesky'], tuple(saved['shape'])
        return model, saved


def anomaly_maps(distances, sigma=4.):
    """Bilinear 256-square maps, optional separable Gaussian with reflect edges."""
    maps = F.interpolate(distances[:, None], size=(256, 256), mode='bilinear', align_corners=False)
    if sigma:
        radius = int(4*sigma+.5)
        axis = torch.arange(-radius, radius+1, device=maps.device, dtype=maps.dtype)
        kernel = torch.exp(-axis.square()/(2*sigma*sigma)); kernel /= kernel.sum()
        maps = F.conv2d(F.pad(maps, (radius, radius, 0, 0), mode='reflect'), kernel[None,None,None,:])
        maps = F.conv2d(F.pad(maps, (0, 0, radius, radius), mode='reflect'), kernel[None,None,:,None])
    return maps[:, 0]


class PaDiMPipeline(nn.Module):
    def __init__(self, extractor, gaussian, sigma=4.):
        super().__init__()
        self.extractor, self.gaussian, self.sigma = extractor, gaussian, sigma

    @torch.inference_mode()
    def forward(self, images):
        return anomaly_maps(self.gaussian.score(self.extractor(images)), self.sigma)[:, None]
