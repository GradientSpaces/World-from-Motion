import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

@torch.no_grad()
def video_msssim(video1: torch.Tensor, video2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Multi-Scale Structural Similarity Index (MS-SSIM) between two videos.

    Args:
        video1 (torch.Tensor): The first video tensor of shape [B, 3, n_frame, height, width].
        video2 (torch.Tensor): The second video tensor of shape [B, 3, n_frame, height, width].

    Returns:
        torch.Tensor: The MS-SSIM value.
    """
    assert video1.shape == video2.shape, "Input videos must have the same shape"

    # Reshape to [B * n_frame, C, H, W]
    b, c, t, h, w = video1.shape
    video1_reshaped = video1.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    video2_reshaped = video2.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    # Compute MS-SSIM
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(video1.device)
    return ms_ssim_metric(video1_reshaped, video2_reshaped)
