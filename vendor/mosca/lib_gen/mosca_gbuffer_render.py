from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from lib_render.render_helper import GS_BACKEND, render
from lib_gen.mosca_render_types import CameraRenderSpec


def to_hw_float(arr: np.ndarray) -> np.ndarray:
    ret = np.asarray(arr, dtype=np.float32)
    while ret.ndim > 2:
        ret = ret[0]
    return ret


def alpha_to_u8(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    return np.repeat(alpha_u8[..., None], 3, axis=2)


def normal_to_u8(normal: np.ndarray) -> np.ndarray:
    vis = np.clip((normal.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def depth_to_grayscale_u8(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = depth > 0
    gray = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return gray
    depth_valid = depth[valid]
    depth_min = float(depth_valid.min())
    depth_max = float(depth_valid.max())
    depth_norm = np.zeros_like(depth, dtype=np.float32)
    if np.isfinite(depth_min) and np.isfinite(depth_max) and depth_max > depth_min + 1e-10:
        depth_norm[valid] = (depth_valid - depth_min) / (depth_max - depth_min)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
    gray_u8 = (depth_norm * 255.0).astype(np.uint8)
    gray = np.repeat(gray_u8[..., None], 3, axis=2)
    gray[~valid] = 0
    return gray


def compute_scene_flow_bounds(flow_paths: list[Path]) -> np.ndarray:
    if not flow_paths:
        raise ValueError("Cannot compute scene-flow bounds from an empty file list.")

    abs_chunks: list[np.ndarray] = []
    for flow_path in flow_paths:
        with np.load(flow_path) as data:
            if "scene_flow" not in data:
                raise KeyError(f"Missing 'scene_flow' key in {flow_path}")
            flow = data["scene_flow"]
        abs_chunks.append(np.abs(flow.reshape(-1, 3).astype(np.float32)))

    flat_abs = np.concatenate(abs_chunks, axis=0)
    hi = np.percentile(flat_abs, 99, axis=0)
    return np.maximum(hi, 1e-6)


def scene_flow_to_rgb_u8(flow: np.ndarray, hi: np.ndarray) -> np.ndarray:
    flow = flow.astype(np.float32)
    zero_mask = np.all(np.isclose(flow, 0.0, atol=1e-8), axis=-1)
    rgb = np.zeros(flow.shape, dtype=np.uint8)
    if zero_mask.all():
        return rgb

    normed = np.clip(flow / hi[None, None], -1.0, 1.0)
    rgb = ((normed * 0.5 + 0.5) * 255.0).astype(np.uint8)
    rgb[zero_mask] = 0
    return rgb


def cat_points_from_gs_list(gs_list) -> torch.Tensor:
    return torch.cat([gs[0] for gs in gs_list], dim=0).contiguous()


def render_payload_map(
    gs_list,
    payload_world: torch.Tensor,
    H: int,
    W: int,
    K: torch.Tensor,
    T_cw: torch.Tensor,
    bg_color: list[float],
) -> np.ndarray:
    if GS_BACKEND == "native_add3":
        render_dict = render(
            gs_list,
            H,
            W,
            K,
            T_cw,
            bg_color=bg_color,
            add_buffer=payload_world,
        )
        return (
            render_dict["buf"]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    render_dict = render(
        gs_list,
        H,
        W,
        K,
        T_cw,
        bg_color=[0.0, 0.0, 0.0],
        colors_precomp=payload_world,
    )
    return (
        render_dict["rgb"]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def transform_map(
    T_cw: torch.Tensor,
    xyz_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    flat = xyz_map.reshape(-1, 3)
    R = T_cw[:3, :3].detach().cpu().numpy()
    t = T_cw[:3, 3].detach().cpu().numpy()
    transformed = flat @ R.T + t
    transformed = transformed.reshape(xyz_map.shape).astype(np.float32)
    transformed[~valid_mask] = 0.0
    return transformed


def camera_center_world(T_cw: torch.Tensor) -> np.ndarray:
    T_wc = torch.linalg.inv(T_cw)
    return T_wc[:3, 3].detach().cpu().numpy().astype(np.float32)


def normals_from_xyz_map(
    xyz_map: np.ndarray,
    valid_mask: np.ndarray,
    cam_center: np.ndarray,
) -> np.ndarray:
    H, W, _ = xyz_map.shape
    normals = np.zeros((H, W, 3), dtype=np.float32)
    interior_valid = (
        valid_mask[1:-1, 1:-1]
        & valid_mask[:-2, 1:-1]
        & valid_mask[2:, 1:-1]
        & valid_mask[1:-1, :-2]
        & valid_mask[1:-1, 2:]
    )
    if not interior_valid.any():
        return normals

    dx = xyz_map[2:, 1:-1] - xyz_map[:-2, 1:-1]
    dy = xyz_map[1:-1, 2:] - xyz_map[1:-1, :-2]
    n = np.cross(dx, dy)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = interior_valid & (n_norm[..., 0] > 1e-8)
    n = n / np.clip(n_norm, 1e-8, None)

    pts = xyz_map[1:-1, 1:-1]
    view = pts - cam_center[None, None]
    flip = (np.sum(n * view, axis=-1) > 0.0) & valid
    n[flip] *= -1.0
    normals[1:-1, 1:-1][valid] = n[valid]
    return normals


def save_frame_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, img)


def save_frame_npz(data: np.ndarray, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: data})


def is_compatible_flow_source(gs_list, next_gs_list) -> bool:
    if next_gs_list is None or len(gs_list) != len(next_gs_list):
        return False
    for src_gs, dst_gs in zip(gs_list, next_gs_list):
        if src_gs[0].shape[0] != dst_gs[0].shape[0]:
            return False
    return True


def render_single_camera_frame(
    *,
    spec: CameraRenderSpec,
    gs_list,
    next_gs_list,
    outputs: set[str],
    scene_space: str,
    bg_color: list[float],
    alpha_threshold: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    H = int(spec.img_wh[1])
    W = int(spec.img_wh[0])
    K = torch.from_numpy(spec.K_raw).float().to(device)
    T_cw = torch.from_numpy(spec.T_cw).float().to(device)

    render_dict = render(gs_list, H, W, K, T_cw=T_cw, bg_color=bg_color)
    alpha = to_hw_float(render_dict["alpha"].detach().cpu().numpy())
    valid_mask = alpha > alpha_threshold

    frames: dict[str, np.ndarray] = {}
    tensors: dict[str, np.ndarray] = {}

    if "rgb" in outputs:
        rgb = np.clip(
            render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy(),
            0.0,
            1.0,
        )
        frames["rgb"] = (rgb * 255.0).astype(np.uint8)

    if "alpha" in outputs:
        frames["alpha"] = alpha_to_u8(alpha)

    if "depth" in outputs:
        depth = to_hw_float(render_dict["dep"].detach().cpu().numpy())
        depth[~valid_mask] = 0.0
        frames["depth"] = depth_to_grayscale_u8(depth)

    need_scene_flow = "scene_flow" in outputs or "scene_flow_rgb" in outputs
    world_coord_map = None
    if any(mod in outputs for mod in ("normal", "scene_coords")) or need_scene_flow:
        payload_world = cat_points_from_gs_list(gs_list)
        world_coord_map = render_payload_map(
            gs_list=gs_list,
            payload_world=payload_world,
            H=H,
            W=W,
            K=K,
            T_cw=T_cw,
            bg_color=bg_color,
        )
        world_coord_map[~valid_mask] = 0.0

    if "scene_coords" in outputs:
        if scene_space == "world":
            scene_coord = world_coord_map
        else:
            scene_coord = transform_map(T_cw, world_coord_map, valid_mask)
        tensors["scene_coords"] = scene_coord.astype(np.float16)

    if need_scene_flow:
        if not is_compatible_flow_source(gs_list, next_gs_list):
            scene_flow = np.zeros((H, W, 3), dtype=np.float32)
            scene_flow_valid = False
        else:
            dst_world = cat_points_from_gs_list(next_gs_list)
            dst_world_map = render_payload_map(
                gs_list=gs_list,
                payload_world=dst_world,
                H=H,
                W=W,
                K=K,
                T_cw=T_cw,
                bg_color=bg_color,
            )
            dst_world_map[~valid_mask] = 0.0
            if scene_space == "camera":
                src_coord = transform_map(T_cw, world_coord_map, valid_mask)
                dst_coord = transform_map(T_cw, dst_world_map, valid_mask)
                scene_flow = (dst_coord - src_coord).astype(np.float32)
            else:
                scene_flow = (dst_world_map - world_coord_map).astype(np.float32)
            scene_flow[~valid_mask] = 0.0
            scene_flow_valid = True
        if "scene_flow" in outputs:
            tensors["scene_flow"] = scene_flow.astype(np.float16)
    else:
        scene_flow_valid = None

    if "normal" in outputs:
        if "normal" in render_dict:
            normal = (
                render_dict["normal"]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            normal[~valid_mask] = 0.0
        else:
            normal = normals_from_xyz_map(
                world_coord_map,
                valid_mask,
                camera_center_world(T_cw),
            )
        frames["normal"] = normal_to_u8(normal)

    pose_meta = {
        "cam_id": spec.cam_id,
        "frame_name": spec.frame_name,
        "time_id": int(spec.time_id),
        "img_wh": [W, H],
        "K_raw": spec.K_raw.tolist(),
        "c2w": torch.linalg.inv(T_cw).detach().cpu().numpy().tolist(),
        "has_dataset_frame_at_time": bool(spec.has_dataset_frame_at_time),
        "camera_json_path": spec.camera_json_path,
        "render_pose_source_kind": spec.pose_assignment.source_kind,
        "render_pose_source_time_id": spec.pose_assignment.source_time_id,
        "render_pose_source_frame_name": spec.pose_assignment.source_frame_name,
        "render_pose_source_camera_json_path": spec.pose_assignment.source_camera_json_path,
        "render_pose_archive_path": spec.pose_assignment.pose_archive_path,
        "has_exact_tto_pose": bool(spec.pose_assignment.has_exact_tto_pose),
        "scene_flow_valid": scene_flow_valid,
    }
    return frames, tensors, pose_meta


__all__ = [
    "compute_scene_flow_bounds",
    "render_single_camera_frame",
    "save_frame_npz",
    "save_frame_png",
    "scene_flow_to_rgb_u8",
]
