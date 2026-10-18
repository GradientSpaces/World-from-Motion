"""SSIM implementations for efficient fitting and optional loss experiments.

The efficient preset caches the original SSIM window. Other loss variants remain
explicit experiments; their reduction-order changes require quality measurement.
"""
from __future__ import annotations

from functools import lru_cache
from math import exp

import torch
from torch import nn
import torch.nn.functional as F


@lru_cache(maxsize=32)
def gaussian_window(window_size, channels, device, dtype, separable=False):
    # Construct on CPU in exactly the same order as the reference helper.
    values = torch.tensor([exp(-((x - window_size // 2) ** 2) / (2 * 1.5**2))
                           for x in range(window_size)], dtype=torch.float32)
    values = values / values.sum()
    if separable:
        return values[None, None, None].expand(channels, 1, 1, -1).contiguous().to(
            device=device, dtype=dtype)
    window = values[:, None].mm(values[None]).float()[None, None]
    return window.expand(channels, 1, window_size, window_size).contiguous().to(
        device=device, dtype=dtype)


def _combine_moments(mu1, mu2, raw1, raw2, cross, size_average):
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1, sigma2, sigma12 = raw1 - mu1_sq, raw2 - mu2_sq, cross - mu12
    result = ((2 * mu12 + 0.01**2) * (2 * sigma12 + 0.03**2)) / (
        (mu1_sq + mu2_sq + 0.01**2) * (sigma1 + sigma2 + 0.03**2))
    return result.mean() if size_average else result.mean(1).mean(1).mean(1)


def cached_ssim(img1, img2, window_size=11, size_average=True):
    channels = img1.shape[1]
    window = gaussian_window(window_size, channels, img1.device, img1.dtype)
    def conv(x):
        return F.conv2d(x, window, padding=window_size // 2, groups=channels)
    return _combine_moments(conv(img1), conv(img2), conv(img1 * img1),
                            conv(img2 * img2), conv(img1 * img2), size_average)


def separable_ssim(img1, img2, window_size=11, size_average=True):
    channels = img1.shape[1]
    moments = torch.cat((img1, img2, img1 * img1, img2 * img2, img1 * img2), dim=1)
    window = gaussian_window(window_size, channels * 5, img1.device, img1.dtype,
                             separable=True)
    filtered = F.conv2d(moments, window, padding=(0, window_size // 2), groups=channels * 5)
    filtered = F.conv2d(filtered, window.transpose(-1, -2),
                        padding=(window_size // 2, 0), groups=channels * 5)
    return _combine_moments(*filtered.split(channels, dim=1), size_average)


def resize_reduction_weights(mask, input_size):
    """Adjoint of bilinear upsampling followed by the original masked mean.

    Only cache these weights when the mask is known to be immutable. Keeping
    full bilinear weights handles image boundaries and noninteger scale ratios.
    """
    with torch.enable_grad():
        source = torch.zeros((1, 1, *input_size), device=mask.device,
                             dtype=mask.dtype, requires_grad=True)
        resized = F.interpolate(source, size=mask.shape[-2:], mode="bilinear",
                                align_corners=False)
        value = (resized * mask).sum() / mask.sum().clamp_min(1e-6)
        return torch.autograd.grad(value, source)[0].detach()


class ReducedSpatialLPIPS(nn.Module):
    """Compute the spatial LPIPS masked mean without producing five full maps.

    The pretrained network is supplied by the caller. The target feature cache
    requires an explicit stable key; the default always computes fresh features.
    This class returns a scalar, not the reference model's spatial map.
    """
    def __init__(self, model, cache_bytes=0):
        super().__init__()
        if model.training or not model.spatial or not model.lpips or model.version != "0.1":
            raise ValueError("Requires the frozen spatial LPIPS v0.1 model")
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("LPIPS parameters must be frozen")
        from collections import OrderedDict
        self.model = model
        self.cache_bytes = cache_bytes
        self._targets = OrderedDict()
        self._target_bytes = 0
        self._weights = {}

    def forward(self, pred, target, mask, *, target_key=None, mask_key=None):
        from lpips import normalize_tensor
        model = self.model
        pred_features = model.net.forward(model.scaling_layer(pred))
        key = None if target_key is None else (
            target_key, mask_key, target.device, target.dtype, tuple(target.shape))
        cached = self._targets.get(key) if key is not None else None
        if cached is None:
            with torch.no_grad():
                target_features = model.net.forward(model.scaling_layer(target))
                cached = tuple(normalize_tensor(x) for x in target_features)
            size = sum(x.numel() * x.element_size() for x in cached)
            if key is not None and size <= self.cache_bytes:
                while self._targets and self._target_bytes + size > self.cache_bytes:
                    _, evicted = self._targets.popitem(last=False)
                    self._target_bytes -= sum(x.numel() * x.element_size() for x in evicted)
                self._targets[key] = cached
                self._target_bytes += size
        elif key is not None:
            self._targets.move_to_end(key)
        result = 0
        for index in range(model.L):
            difference = (normalize_tensor(pred_features[index]) - cached[index]) ** 2
            lowres = model.lins[index](difference)
            weight_key = None if mask_key is None else (
                mask_key, mask.device, mask.dtype, tuple(mask.shape), tuple(lowres.shape[-2:]))
            weights = self._weights.get(weight_key) if weight_key is not None else None
            if weights is None:
                weights = resize_reduction_weights(mask, lowres.shape[-2:])
                if weight_key is not None:
                    self._weights[weight_key] = weights
            result = result + (lowres * weights).sum()
        return result
