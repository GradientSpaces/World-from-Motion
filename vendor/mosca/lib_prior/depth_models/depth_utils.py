import numpy as np
from matplotlib import cm
import imageio
import os, os.path as osp
from scipy.spatial.distance import pdist


def compute_max_baseline(extrinsics):
    """Compute max baseline from W2C extrinsics."""
    if extrinsics is None or len(extrinsics) < 2:
        return 0.0

    c2w = np.linalg.inv(extrinsics)
    positions = c2w[:, :3, 3]
    pairwise_distances = pdist(positions)
    return float(pairwise_distances.max()) if len(pairwise_distances) > 0 else 0.0


def check_trajectory_quality(extrinsics, min_baseline_ratio=0.001, min_rank_threshold=0.1):
    """Check if a trajectory has enough motion for stable DA3 scale alignment.

    Heuristic (matches SoM/Gen4DGS DA3 runner): reject nearly stationary or
    effectively 1D trajectories.
    """
    if extrinsics is None or len(extrinsics) < 2:
        return {
            "can_align": False,
            "reason": "Not enough frames (need at least 2)",
            "stats": {},
        }

    c2w = np.linalg.inv(extrinsics)
    positions = c2w[:, :3, 3]

    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    singular_values_normalized = singular_values / (singular_values[0] + 1e-10)
    effective_rank = int(np.sum(singular_values_normalized > min_rank_threshold))

    pairwise_distances = pdist(positions)
    max_baseline = pairwise_distances.max() if len(pairwise_distances) > 0 else 0.0
    mean_baseline = pairwise_distances.mean() if len(pairwise_distances) > 0 else 0.0
    scene_scale = np.linalg.norm(centered, axis=1).max()
    baseline_ratio = max_baseline / (scene_scale + 1e-10)

    stats = {
        "num_frames": len(extrinsics),
        "effective_rank": effective_rank,
        "singular_values": singular_values.tolist(),
        "singular_values_normalized": singular_values_normalized.tolist(),
        "max_baseline": float(max_baseline),
        "mean_baseline": float(mean_baseline),
        "scene_scale": float(scene_scale),
        "baseline_ratio": float(baseline_ratio),
    }

    reasons = []
    if effective_rank < 2:
        reasons.append(f"trajectory is nearly 1D (rank={effective_rank})")
    if max_baseline < 1e-6:
        reasons.append("camera is nearly stationary (no translation)")
    elif baseline_ratio < min_baseline_ratio:
        reasons.append(
            f"baseline too small (ratio={baseline_ratio:.4f} < {min_baseline_ratio})"
        )

    can_align = len(reasons) == 0
    if can_align:
        reason = (
            "trajectory suitable for alignment "
            f"(rank={effective_rank}, baseline_ratio={baseline_ratio:.4f})"
        )
    else:
        reason = "trajectory unsuitable: " + "; ".join(reasons)

    return {
        "can_align": can_align,
        "reason": reason,
        "stats": stats,
    }


def viz_depth_list(depths, save_fn, viz_quantile=3):

    dep_list = np.stack(depths, axis=0)

    # use disparity to viz, not depth
    dep_valid_mask = dep_list > 1e-6
    # use robust min and max to visualize
    dep_max = np.percentile(dep_list[dep_valid_mask], 100 - viz_quantile)
    dep_min = np.percentile(dep_list[dep_valid_mask], viz_quantile)
    dep_list = np.clip(dep_list, dep_min, dep_max)
    dep_list = (dep_list - dep_min) / (dep_max - dep_min)
    dep_list[~dep_valid_mask] = 0
    # h_dep = dep_list.reshape(-1)
    # plt.hist(h_dep, bins=300), plt.title("Depth Histogram")
    # plt.savefig(save_fn.replace(".mp4", ".jpg"))
    # plt.close()
    viz_list = []
    for dep in dep_list:
        viz = cm.viridis(dep)[:, :, :3]
        viz_list.append((viz * 255).astype(np.uint8))
    imageio.mimsave(save_fn, viz_list)
    return


def save_depth_list(dep_list, fn_list, dst, invalid_mask_list=None):
    assert len(dep_list) == len(fn_list)
    os.makedirs(dst, exist_ok=True)
    for i in range(len(dep_list)):
        if invalid_mask_list is not None:
            dep_list[i][invalid_mask_list[i] > 0] = 0
        save_fn = ".".join(fn_list[i].split(".")[:-1]) + ".npz"
        np.savez_compressed(osp.join(dst, save_fn), dep=dep_list[i])
    return
