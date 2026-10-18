"""Cache immutable native rasterizer settings during reconstruction fitting.

The first use follows the original GPU arithmetic. Later uses reuse the exact
result. A bulk copy of the small intrinsic matrix supplies a value-based cache
key, so changed intrinsics cannot silently reuse old settings. Camera extrinsics,
Gaussian parameters, and their gradients are never cached here.
"""
from collections import OrderedDict
import math

import numpy as np
import torch


_geometry = OrderedDict()
_rasterizers = OrderedDict()
_LIMIT = 64


def _remember(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _LIMIT:
        cache.popitem(last=False)
    return value


def cached_rasterizer(K, height, width, device):
    """Return geometry plus a factory for a background-specific rasterizer."""
    import gauspl_renderer_native_add3 as original

    device = torch.device(device)
    key = (height, width, device, K.dtype, tuple(K.detach().cpu().reshape(-1).tolist()))
    geometry = _geometry.get(key)
    if geometry is None:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        centered = not (abs(height // 2 - cy) > 1.0 or abs(width // 2 - cx) > 1.0)
        # Crop decisions are constants, converted only when this calibration changes.
        left, right = cx, width - cx
        top, bottom = cy, height - cy
        new_width = width if centered else int(2 * max(left, right))
        new_height = height if centered else int(2 * max(top, bottom))
        fovx = original.focal2fov(fx, new_width)
        fovy = original.focal2fov(fy, new_height)
        view = torch.from_numpy(original.getWorld2View2(np.eye(3), np.zeros(3)).transpose(0, 1)).to(device)
        projection = original.getProjectionMatrix(
            znear=0.0001, zfar=1.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
        full = view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        center = view.inverse()[3, :3]
        geometry = _remember(_geometry, key, {
            "center_handling_flag": not centered,
            "left_w": float(left), "right_w": float(right),
            "top_h": float(top), "bottom_h": float(bottom),
            "image_height": new_height, "image_width": new_width,
            "tanfovx": math.tan(fovx * 0.5), "tanfovy": math.tan(fovy * 0.5),
            "viewmatrix": view, "projmatrix": full, "campos": center,
        })

    def make(background):
        background = list(background)
        if len(background) < 6:
            background = background + [0.0] * (6 - len(background))
        raster_key = (key, tuple(background))
        rasterizer = _rasterizers.get(raster_key)
        if rasterizer is None:
            settings = original.GaussianRasterizationSettings(
                **{k: geometry[k] for k in ("image_height", "image_width", "tanfovx",
                   "tanfovy", "viewmatrix", "projmatrix", "campos")},
                bg=torch.tensor(background, dtype=torch.float32, device=device),
                scale_modifier=1.0, sh_degree=0, prefiltered=False, debug=False)
            rasterizer = _remember(_rasterizers, raster_key,
                                  original.GaussianRasterizer(raster_settings=settings))
        return rasterizer

    return geometry, make
