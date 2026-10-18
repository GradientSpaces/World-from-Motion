import torch
from torchmetrics.image import PeakSignalNoiseRatio

@torch.no_grad()
def video_psnr(video1: torch.Tensor, video2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two videos.

    Args:
        video1 (torch.Tensor): The first video tensor of shape [B, 3, n_frame, height, width].
        video2 (torch.Tensor): The second video tensor of shape [B, 3, n_frame, height, width].

    Returns:
        torch.Tensor: The PSNR value.
    """
    assert video1.shape == video2.shape, "Input videos must have the same shape"

    # Reshape to [B * n_frame, C, H, W]
    b, c, t, h, w = video1.shape
    video1_reshaped = video1.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    video2_reshaped = video2.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    # Compute PSNR
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(video1.device)
    return psnr_metric(video1_reshaped, video2_reshaped)


@torch.no_grad()
def video_psnr_masked(video1: torch.Tensor, video2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes PSNR only over masked (covisible) pixels.

    Args:
        video1: [B, C, T, H, W] in [0, 1].
        video2: [B, C, T, H, W] in [0, 1].
        mask:   [B, 1, T, H, W] binary mask; 1 = valid pixel.

    Returns:
        Scalar PSNR over valid pixels only.
    """
    assert video1.shape == video2.shape
    diff_sq = (video1 - video2) ** 2
    num_valid = mask.sum() * video1.shape[1]
    if num_valid == 0:
        return torch.tensor(float('inf'), device=video1.device)
    mse = (diff_sq * mask).sum() / num_valid
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
