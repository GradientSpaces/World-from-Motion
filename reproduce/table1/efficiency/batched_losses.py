"""Per-view RGB and spatial LPIPS losses with a shared neural-network batch."""
import torch
import torch.nn.functional as F

from efficiency.losses import cached_ssim


def rgb_losses(prediction, target, masks, ssim_lambda=0.1):
    """Return one original RGB loss per view; inputs are NCHW, NCHW, NHW."""
    target = target.detach()
    masks = masks.float()
    errors = (prediction - target).abs() * masks[:, None]
    values = errors.sum(dim=(1, 2, 3)) / masks.sum(dim=(1, 2))
    if ssim_lambda > 0:
        structural = cached_ssim(prediction * masks[:, None],
                                 target * masks[:, None], size_average=False)
        values = values + (1 - structural) * ssim_lambda
    return values


def lpips_losses(prediction, target, masks, model):
    """Return one masked spatial LPIPS loss per view, retaining the full maps."""
    masks = masks.float()
    pred_input = prediction * masks[:, None] * 2.0 - 1.0
    target_input = target.detach() * masks[:, None] * 2.0 - 1.0
    maps = model(pred_input, target_input)
    if maps.ndim != 4:
        raise ValueError("Expected spatial LPIPS output")
    weights = F.interpolate(masks[:, None], size=maps.shape[-2:], mode="nearest")
    return (maps * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1e-6)


def sequential_weighted_sum(values, weights):
    """Retain the fitting code's view order and scalar accumulation order."""
    total = torch.zeros((), device=values.device, dtype=values.dtype)
    for value, weight in zip(values.unbind(), weights):
        total = total + value * weight
    return total


def lpips_for_views(views, model, reference_loss):
    """Score ordered (CHW prediction, HWC target, HW mask) auxiliary views.

    Mixed image sizes retain the reference per-view implementation. The caller
    keeps the original camera weights and scalar accumulation order.
    """
    if not views:
        return ()
    if len(views) == 1 or any(view[0].shape != views[0][0].shape for view in views):
        return tuple(reference_loss(target, {"rgb": prediction}, mask, model)[0]
                     for prediction, target, mask in views)
    prediction = torch.stack([view[0] for view in views])
    target = torch.stack([view[1].permute(2, 0, 1) for view in views])
    masks = torch.stack([view[2] for view in views])
    return lpips_losses(prediction, target, masks, model).unbind()
