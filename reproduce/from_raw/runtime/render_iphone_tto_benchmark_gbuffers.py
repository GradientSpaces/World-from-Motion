from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = str(Path(__file__).resolve().parents[2] / "table1" / "backends" / "wt")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from lib_render.render_helper import GS_BACKEND, render
from mosca_evaluate_ours import (
    _build_eval_gs_list,
    _freeze_eval_scene_models,
    _load_eval_assets,
    _load_eval_fachain_models,
)


IMAGE_MODALITIES = ("rgb", "depth", "normal", "alpha", "scene_flow_rgb", "test_dynamic")
TENSOR_MODALITIES = ("scene_coords", "scene_flow")
PAIR_TOKEN_SPLIT = re.compile(r"[-,:/]")
CAMERA_FILE_PATTERN = re.compile(r"(\d+)_(\d+)\.json$")
INTERVAL_DIR_PATTERN = re.compile(r"dst_L(\d+)_R(\d+)$")


@dataclass(frozen=True)
class FrameSpec:
    cam_id: str
    time_id: int
    frame_name: str
    camera_json_path: str
    image_path: str | None
    K_raw: np.ndarray
    img_wh: tuple[int, int]


@dataclass(frozen=True)
class PoseAssignment:
    T_cw: np.ndarray
    source_kind: str
    source_time_id: int | None
    source_frame_name: str | None
    source_camera_json_path: str | None
    pose_archive_path: str
    has_exact_tto_pose: bool


@dataclass(frozen=True)
class CameraRenderSpec:
    cam_id: str
    time_id: int
    frame_name: str
    camera_json_path: str | None
    image_path: str | None
    K_raw: np.ndarray
    img_wh: tuple[int, int]
    T_cw: np.ndarray
    has_dataset_frame_at_time: bool
    pose_assignment: PoseAssignment


@dataclass
class IntervalContext:
    node_name: str
    saved_dir: str
    left_bound: int
    right_bound: int
    cams: object
    s_model: object
    d_model: object
    fachain_d_model_list: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render SoM-style benchmark gbuffers from a WorldTree iPhone TTO result "
            "directory such as rendering_results_max_height_2."
        )
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help=(
            "Result directory that contains tto_test/ and tto_test_pose_*.npz, "
            "e.g. exp/.../rendering_results_max_height_2."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root for SoM-style benchmark artifacts. Defaults under result_dir.",
    )
    parser.add_argument(
        "--ws",
        type=str,
        default=None,
        help="Workspace override. If omitted, infer it from fit_commandline_args.txt.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Fit config override. If omitted, infer it from fit_commandline_args.txt.",
    )
    parser.add_argument(
        "--process_height",
        type=int,
        default=None,
        help="Tree height to render. If omitted, infer from result_dir name.",
    )
    parser.add_argument(
        "--reference_camera",
        type=str,
        default="0",
        help="Reference training camera id. Default: 0.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "Optional pair list like 0-1 0-2. Defaults to reference camera paired with "
            "all discovered test cameras."
        ),
    )
    parser.add_argument(
        "--reference_rgb_source",
        type=str,
        default="gt",
        choices=["gt", "render"],
        help=(
            "How to export rgb for the reference camera. "
            "'gt' copies the workspace image, 'render' keeps the rendered reference view."
        ),
    )
    parser.add_argument(
        "--test_dynamic_dirname",
        type=str,
        default="test_dynamic",
        help="Workspace directory that stores DyCheck test dynamic masks.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=[
            "rgb",
            "depth",
            "normal",
            "alpha",
            "scene_coords",
            "scene_flow",
            "scene_flow_rgb",
            "test_dynamic",
        ],
        choices=[
            "rgb",
            "depth",
            "normal",
            "alpha",
            "scene_coords",
            "scene_flow",
            "scene_flow_rgb",
            "test_dynamic",
        ],
    )
    parser.add_argument(
        "--scene_space",
        type=str,
        default="world",
        choices=["world", "camera"],
        help="Coordinate space for scene_coords and scene_flow tensors.",
    )
    parser.add_argument("--flow_step", type=int, default=1)
    parser.add_argument("--alpha_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--bg_color",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--short_pose_alignment",
        type=str,
        default="suffix",
        choices=["suffix", "prefix", "error"],
        help=(
            "How to align a short tto_test_pose_<i>.npz when it has fewer rows than "
            "the camera's dataset frames."
        ),
    )
    parser.add_argument(
        "--initial_pose_fill",
        type=str,
        default="first_available",
        choices=["first_available", "error"],
        help=(
            "How to fill canonical timesteps that occur before the first available TTO pose."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--save_gt",
        action="store_true",
        default=True,
        help="Copy GT RGB frames when available.",
    )
    parser.add_argument(
        "--no_save_gt",
        action="store_false",
        dest="save_gt",
        help="Do not copy GT RGB frames.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def camera_label(cam_id: str) -> str:
    return f"cam{int(cam_id)}" if cam_id.isdigit() else f"cam{cam_id}"


def normalize_camera_token(token: str) -> str:
    token = token.strip().lower()
    if token.startswith("cam"):
        token = token[3:]
    if token.isdigit():
        return str(int(token))
    return token


def parse_pair_token(token: str) -> tuple[str, str]:
    parts = [p for p in PAIR_TOKEN_SPLIT.split(token.replace(" ", "")) if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid pair token '{token}'.")
    a = normalize_camera_token(parts[0])
    b = normalize_camera_token(parts[1])
    if a == b:
        raise ValueError(f"Pair token '{token}' uses the same camera twice.")
    return a, b


def parse_pairs(pair_tokens: list[str] | None, reference_camera: str, test_camera_ids: list[str]) -> list[tuple[str, str]]:
    if pair_tokens is None:
        return [(reference_camera, cam_id) for cam_id in test_camera_ids]
    parsed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in pair_tokens:
        pair = parse_pair_token(token)
        if pair in seen:
            continue
        parsed.append(pair)
        seen.add(pair)
    return parsed


def resolve_repo_relative(path: str) -> str:
    if osp.isabs(path):
        return path
    return osp.normpath(osp.join(REPO_ROOT, path))


def read_commandline_tokens(path: str) -> list[str]:
    line = Path(path).read_text(encoding="utf-8").strip()
    if not line:
        raise ValueError(f"Command line file is empty: {path}")
    return shlex.split(line)


def get_flag_value(tokens: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def infer_run_dir(result_dir: str) -> str:
    result_path = Path(result_dir).resolve()
    if not result_path.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")
    return str(result_path.parent)


def find_fit_args_path(run_dir: str) -> str:
    run_root = Path(run_dir)
    exact = run_root / "fit_commandline_args.txt"
    if exact.is_file():
        return str(exact)

    reconstruct_candidates: list[Path] = []
    fallback_candidates: list[Path] = []
    for candidate in sorted(run_root.glob("fit_commandline_args*.txt")):
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        if "mosca_reconstruct.py" in text:
            reconstruct_candidates.append(candidate)
        else:
            fallback_candidates.append(candidate)

    if reconstruct_candidates:
        return str(reconstruct_candidates[0])
    if fallback_candidates:
        return str(fallback_candidates[0])

    raise FileNotFoundError(f"Could not find fit_commandline_args*.txt under {run_dir}")


def infer_cfg_path(run_dir: str, cfg_override: str | None) -> str:
    if cfg_override is not None:
        return resolve_repo_relative(cfg_override)
    fit_args_path = find_fit_args_path(run_dir)
    tokens = read_commandline_tokens(fit_args_path)
    cfg_path = get_flag_value(tokens, "--cfg")
    if cfg_path is None:
        raise ValueError(f"Could not infer --cfg from {fit_args_path}")
    resolved = resolve_repo_relative(cfg_path)
    return resolved if osp.exists(resolved) else cfg_path


def infer_workspace_path(run_dir: str, ws_override: str | None) -> str:
    if ws_override is not None:
        return osp.abspath(ws_override)
    fit_args_path = find_fit_args_path(run_dir)
    tokens = read_commandline_tokens(fit_args_path)
    ws_path = get_flag_value(tokens, "--ws")
    if ws_path is None:
        raise ValueError(f"Could not infer --ws from {fit_args_path}")
    resolved = resolve_repo_relative(ws_path)
    return osp.abspath(resolved if osp.exists(resolved) else ws_path)


def infer_process_height(result_dir: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    name = Path(result_dir).name
    for pattern in (
        re.compile(r"max_height_(\d+)"),
        re.compile(r"_h(\d+)(?:_|$)"),
    ):
        match = pattern.search(name)
        if match is not None:
            return int(match.group(1))
    raise ValueError(
        "Could not infer process height from result_dir name. Pass --process_height explicitly."
    )


def parse_interval_from_dir(name: str) -> tuple[int, int] | None:
    match = INTERVAL_DIR_PATTERN.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def find_root(node_names: list[str]) -> str:
    if not node_names:
        raise ValueError("Cannot find root from an empty node list.")
    return max(node_names, key=lambda n: parse_interval_from_dir(n)[1] - parse_interval_from_dir(n)[0])


def build_interval_tree(run_dir: str) -> list[str]:
    node_names = sorted(
        name for name in os.listdir(run_dir) if parse_interval_from_dir(name) is not None
    )
    if not node_names:
        raise RuntimeError(f"No interval checkpoint directories found under {run_dir}")

    segtree = ["None"] * 4096

    def recurse(names: list[str], node_idx: int) -> None:
        if not names:
            return
        root = find_root(names)
        segtree[node_idx] = root
        if len(names) == 1:
            return
        split_path = osp.join(run_dir, root, "split_ind.txt")
        with open(split_path, "r", encoding="utf-8") as f:
            split_ind = int(f.read().strip())
        left_names = []
        right_names = []
        for name in names:
            if name == root:
                continue
            left_bound, right_bound = parse_interval_from_dir(name)
            if right_bound < split_ind:
                left_names.append(name)
            if left_bound >= split_ind:
                right_names.append(name)
        recurse(left_names, node_idx * 2)
        recurse(right_names, node_idx * 2 + 1)

    recurse(node_names, 1)
    return segtree


def find_all_nodes_and_intervals(
    segtree: list[str],
    node_idx: int,
    max_tree_height: int,
    post_cut: int = -1,
) -> list[dict]:
    left_child = node_idx * 2
    right_child = node_idx * 2 + 1
    has_left = left_child < len(segtree) and segtree[left_child] != "None"
    has_right = right_child < len(segtree) and segtree[right_child] != "None"
    now_height = int(math.log2(node_idx))
    if now_height == max_tree_height:
        has_left = False
        has_right = False

    def keep_child(child_idx: int) -> bool:
        if child_idx >= len(segtree) or segtree[child_idx] == "None":
            return False
        if post_cut == -1:
            return True
        left_bound, right_bound = parse_interval_from_dir(segtree[child_idx])
        return (right_bound - left_bound + 1) >= post_cut

    if has_left and not keep_child(left_child):
        has_left = False
    if has_right and not keep_child(right_child):
        has_right = False

    if not has_left and not has_right:
        left_t, right_t = parse_interval_from_dir(segtree[node_idx])
        return [
            {
                "node_name": segtree[node_idx],
                "left_t": left_t,
                "right_t": right_t,
            }
        ]

    if has_left and not has_right:
        left_results = find_all_nodes_and_intervals(segtree, left_child, max_tree_height, post_cut)
        left_last = left_results[-1]
        _, node_right = parse_interval_from_dir(segtree[node_idx])
        return left_results + [
            {
                "node_name": segtree[node_idx],
                "left_t": left_last["right_t"] + 1,
                "right_t": node_right,
            }
        ]

    if not has_left and has_right:
        right_results = find_all_nodes_and_intervals(segtree, right_child, max_tree_height, post_cut)
        node_left, _ = parse_interval_from_dir(segtree[node_idx])
        return [
            {
                "node_name": segtree[node_idx],
                "left_t": node_left,
                "right_t": right_results[0]["left_t"] - 1,
            }
        ] + right_results

    return find_all_nodes_and_intervals(segtree, left_child, max_tree_height, post_cut) + find_all_nodes_and_intervals(
        segtree, right_child, max_tree_height, post_cut
    )


def load_camera_json(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        camera_data = json.load(f)

    focal = float(camera_data["focal_length"])
    principal_point = camera_data["principal_point"]
    cx = float(principal_point[0])
    cy = float(principal_point[1])
    K = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    image_size = camera_data.get("image_size")
    if image_size is not None and len(image_size) == 2:
        img_wh = (int(image_size[0]), int(image_size[1]))
    else:
        img_wh = (int(round(cx * 2.0)), int(round(cy * 2.0)))
    return K, img_wh


def resolve_frame_intrinsics(
    K_raw: np.ndarray,
    declared_img_wh: tuple[int, int],
    image_path: str | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if image_path is None or not osp.isfile(image_path):
        return K_raw, declared_img_wh

    with Image.open(image_path) as img:
        actual_img_wh = (int(img.width), int(img.height))

    if actual_img_wh == declared_img_wh:
        return K_raw, declared_img_wh

    declared_w, declared_h = declared_img_wh
    if declared_w <= 0 or declared_h <= 0:
        return K_raw, actual_img_wh

    scale_x = actual_img_wh[0] / float(declared_w)
    scale_y = actual_img_wh[1] / float(declared_h)
    K_scaled = K_raw.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    return K_scaled, actual_img_wh


def discover_camera_frames(camera_dir: str, image_dir: str) -> dict[str, list[FrameSpec]]:
    grouped: dict[str, list[FrameSpec]] = {}
    camera_root = Path(camera_dir)
    if not camera_root.is_dir():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
    for path in sorted(camera_root.glob("*.json")):
        match = CAMERA_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        cam_id = str(int(match.group(1)))
        time_id = int(match.group(2))
        frame_name = path.stem
        image_path = Path(image_dir) / f"{frame_name}.png"
        K_raw, img_wh = load_camera_json(str(path))
        image_path_str = str(image_path) if image_path.exists() else None
        K_raw, img_wh = resolve_frame_intrinsics(K_raw, img_wh, image_path_str)
        grouped.setdefault(cam_id, []).append(
            FrameSpec(
                cam_id=cam_id,
                time_id=time_id,
                frame_name=frame_name,
                camera_json_path=str(path),
                image_path=image_path_str,
                K_raw=K_raw,
                img_wh=img_wh,
            )
        )
    for entries in grouped.values():
        entries.sort(key=lambda spec: spec.time_id)
    return dict(sorted(grouped.items(), key=lambda kv: int(kv[0])))


def build_time_lookup(entries: list[FrameSpec]) -> dict[int, FrameSpec]:
    lookup: dict[int, FrameSpec] = {}
    for spec in entries:
        if spec.time_id not in lookup:
            lookup[spec.time_id] = spec
    return lookup


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
    """Match SoM's per-camera shared 99th-percentile scaling over the full video."""
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


def transform_map(T_cw: torch.Tensor, xyz_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
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


def normals_from_xyz_map(xyz_map: np.ndarray, valid_mask: np.ndarray, cam_center: np.ndarray) -> np.ndarray:
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


def copy_png(src_path: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def load_rgb_u8(path: str) -> np.ndarray:
    rgb = imageio.imread(path)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def save_frame_npz(data: np.ndarray, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: data})


def write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")


def write_camera_c2w_npz(c2w_list: list[np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(c2w_list, axis=0).astype(np.float32)
    np.savez_compressed(out_path, c2w=stacked)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def make_synthetic_frame_name(cam_id: str, time_id: int) -> str:
    return f"{cam_id}_{time_id:05d}"


def decode_frame_names(arr: np.ndarray) -> list[str]:
    decoded = []
    for value in arr.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def build_expected_test_specs(
    reference_entries: list[FrameSpec],
    test_entries: list[FrameSpec],
) -> list[FrameSpec]:
    reference_times = {spec.time_id for spec in reference_entries}
    return [spec for spec in test_entries if spec.time_id in reference_times]


def align_pose_archive_to_specs(
    archive_path: str,
    expected_specs: list[FrameSpec],
    short_pose_alignment: str,
) -> tuple[list[tuple[FrameSpec, np.ndarray]], str]:
    archive = np.load(archive_path)
    poses = archive["poses"].astype(np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected pose shape in {archive_path}: {poses.shape}")
    if len(poses) == 0:
        raise ValueError(f"No poses found in {archive_path}")
    if not expected_specs:
        raise ValueError(f"No expected dataset frames available to align poses from {archive_path}")

    if "frame_names" in archive.files:
        frame_names = decode_frame_names(archive["frame_names"])
        if len(frame_names) != len(poses):
            raise ValueError(
                f"Archive {archive_path} has {len(poses)} poses but {len(frame_names)} frame_names."
            )
        frame_lookup = {spec.frame_name: spec for spec in expected_specs}
        aligned: list[tuple[FrameSpec, np.ndarray]] = []
        for frame_name, pose in zip(frame_names, poses):
            if frame_name not in frame_lookup:
                raise KeyError(
                    f"Frame name '{frame_name}' from {archive_path} not found in expected test specs."
                )
            aligned.append((frame_lookup[frame_name], pose))
        return aligned, "frame_names"

    if len(poses) == len(expected_specs):
        return list(zip(expected_specs, poses)), "exact_length"

    if len(poses) > len(expected_specs):
        raise ValueError(
            f"Pose archive {archive_path} has more poses ({len(poses)}) than expected specs ({len(expected_specs)})."
        )

    if short_pose_alignment == "error":
        raise ValueError(
            f"Pose archive {archive_path} has only {len(poses)} poses for {len(expected_specs)} expected specs."
        )

    if short_pose_alignment == "suffix":
        aligned_specs = expected_specs[-len(poses) :]
    elif short_pose_alignment == "prefix":
        aligned_specs = expected_specs[: len(poses)]
    else:
        raise ValueError(f"Unknown short_pose_alignment={short_pose_alignment}")

    return list(zip(aligned_specs, poses)), short_pose_alignment


def expand_tto_camera_specs(
    *,
    cam_id: str,
    canonical_entries: list[FrameSpec],
    test_lookup: dict[int, FrameSpec],
    template_spec: FrameSpec,
    pose_archive_path: str,
    short_pose_alignment: str,
    initial_pose_fill: str,
) -> tuple[list[CameraRenderSpec], dict]:
    expected_specs = [test_lookup[spec.time_id] for spec in canonical_entries if spec.time_id in test_lookup]
    aligned_pose_pairs, alignment_mode = align_pose_archive_to_specs(
        pose_archive_path, expected_specs, short_pose_alignment
    )
    pose_by_time = {spec.time_id: (spec, pose) for spec, pose in aligned_pose_pairs}
    exact_pose_times = sorted(pose_by_time.keys())
    first_exact_spec, first_exact_pose = pose_by_time[exact_pose_times[0]]
    first_assignment = PoseAssignment(
        T_cw=first_exact_pose,
        source_kind="backfill_first_available",
        source_time_id=first_exact_spec.time_id,
        source_frame_name=first_exact_spec.frame_name,
        source_camera_json_path=first_exact_spec.camera_json_path,
        pose_archive_path=pose_archive_path,
        has_exact_tto_pose=False,
    )

    specs: list[CameraRenderSpec] = []
    num_exact = 0
    num_forward_filled = 0
    num_backfilled = 0
    last_assignment: PoseAssignment | None = None

    # Some result folders contain sparse or truncated TTO archives. We preserve the
    # available solved poses exactly, then fill the canonical training timeline using
    # the user's requested "last pose in time" rule.
    for canonical_spec in canonical_entries:
        actual_spec = test_lookup.get(canonical_spec.time_id)
        if canonical_spec.time_id in pose_by_time:
            matched_spec, pose = pose_by_time[canonical_spec.time_id]
            assignment = PoseAssignment(
                T_cw=pose.astype(np.float32),
                source_kind="exact_tto",
                source_time_id=matched_spec.time_id,
                source_frame_name=matched_spec.frame_name,
                source_camera_json_path=matched_spec.camera_json_path,
                pose_archive_path=pose_archive_path,
                has_exact_tto_pose=True,
            )
            num_exact += 1
            last_assignment = assignment
        elif last_assignment is not None:
            assignment = PoseAssignment(
                T_cw=last_assignment.T_cw,
                source_kind="carry_forward",
                source_time_id=last_assignment.source_time_id,
                source_frame_name=last_assignment.source_frame_name,
                source_camera_json_path=last_assignment.source_camera_json_path,
                pose_archive_path=pose_archive_path,
                has_exact_tto_pose=False,
            )
            num_forward_filled += 1
        elif initial_pose_fill == "first_available":
            assignment = first_assignment
            num_backfilled += 1
        else:
            raise ValueError(
                f"Camera {camera_label(cam_id)} has no prior pose for time_id={canonical_spec.time_id} "
                f"and initial fill mode is '{initial_pose_fill}'."
            )

        frame_spec = actual_spec if actual_spec is not None else template_spec
        frame_name = actual_spec.frame_name if actual_spec is not None else make_synthetic_frame_name(cam_id, canonical_spec.time_id)
        specs.append(
            CameraRenderSpec(
                cam_id=cam_id,
                time_id=canonical_spec.time_id,
                frame_name=frame_name,
                camera_json_path=actual_spec.camera_json_path if actual_spec is not None else template_spec.camera_json_path,
                image_path=actual_spec.image_path if actual_spec is not None else None,
                K_raw=frame_spec.K_raw.copy(),
                img_wh=frame_spec.img_wh,
                T_cw=assignment.T_cw.copy(),
                has_dataset_frame_at_time=actual_spec is not None,
                pose_assignment=assignment,
            )
        )

    stats = {
        "camera_id": cam_id,
        "alignment_mode": alignment_mode,
        "pose_archive_path": pose_archive_path,
        "num_pose_rows": len(aligned_pose_pairs),
        "num_expected_dataset_frames": len(expected_specs),
        "num_exact_pose_frames": num_exact,
        "num_forward_filled_frames": num_forward_filled,
        "num_backfilled_frames": num_backfilled,
    }
    return specs, stats


def load_fit_cfg(cfg_path: str) -> object:
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def build_interval_contexts(
    run_dir: str,
    process_height: int,
    cfg,
    device: torch.device,
) -> list[IntervalContext]:
    segtree = build_interval_tree(run_dir)
    post_cut = int(getattr(cfg, "min_interval", 12))
    nodes = find_all_nodes_and_intervals(segtree, 1, process_height, post_cut=post_cut)
    if not nodes:
        raise RuntimeError(
            f"No interval nodes found for process_height={process_height} under {run_dir}"
        )

    use_fachain = bool(getattr(cfg, "use_fachain", True))
    contexts: list[IntervalContext] = []
    for node in nodes:
        saved_dir = osp.join(run_dir, node["node_name"])
        cams, s_model, d_model = _load_eval_assets(saved_dir, device)
        fachain_d_model_list = _load_eval_fachain_models(
            cfg,
            saved_dir,
            node["left_t"],
            node["right_t"],
            use_fachain,
            segtree,
            device,
        )
        _freeze_eval_scene_models(s_model, d_model, fachain_d_model_list)
        contexts.append(
            IntervalContext(
                node_name=node["node_name"],
                saved_dir=saved_dir,
                left_bound=node["left_t"],
                right_bound=node["right_t"],
                cams=cams,
                s_model=s_model,
                d_model=d_model,
                fachain_d_model_list=fachain_d_model_list,
            )
        )
    return contexts


def locate_interval_context(contexts: list[IntervalContext], global_t: int) -> tuple[IntervalContext, int]:
    for ctx in contexts:
        if ctx.left_bound <= global_t <= ctx.right_bound:
            return ctx, global_t - ctx.left_bound
    raise KeyError(f"No interval context found for global time {global_t}")


def compute_reference_render_specs(
    canonical_entries: list[FrameSpec],
    contexts: list[IntervalContext],
    result_dir: str,
) -> list[CameraRenderSpec]:
    specs: list[CameraRenderSpec] = []
    for spec in canonical_entries:
        ctx, local_t = locate_interval_context(contexts, spec.time_id)
        T_cw = ctx.cams.T_cw(local_t).detach().cpu().numpy().astype(np.float32)
        specs.append(
            CameraRenderSpec(
                cam_id=spec.cam_id,
                time_id=spec.time_id,
                frame_name=spec.frame_name,
                camera_json_path=spec.camera_json_path,
                image_path=spec.image_path,
                K_raw=spec.K_raw.copy(),
                img_wh=spec.img_wh,
                T_cw=T_cw,
                has_dataset_frame_at_time=True,
                pose_assignment=PoseAssignment(
                    T_cw=T_cw,
                    source_kind="solved_reference_camera",
                    source_time_id=spec.time_id,
                    source_frame_name=spec.frame_name,
                    source_camera_json_path=spec.camera_json_path,
                    pose_archive_path=result_dir,
                    has_exact_tto_pose=True,
                ),
            )
        )
    return specs


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
        # Interval boundaries do not preserve a guaranteed one-to-one Gaussian ordering,
        # so scene flow falls back to zeros when the next timestep comes from a different
        # interval topology.
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


def build_gt_lookup(entries: list[FrameSpec]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for spec in entries:
        if spec.image_path is not None:
            lookup[spec.frame_name] = spec.image_path
    return lookup


def build_named_png_lookup(entries: list[FrameSpec], root_dir: str | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if root_dir is None or not osp.isdir(root_dir):
        return lookup
    for spec in entries:
        candidate = osp.join(root_dir, f"{spec.frame_name}.png")
        if osp.isfile(candidate):
            lookup[spec.frame_name] = candidate
    return lookup


def ensure_pair_dirs(
    pair_root: Path,
    pair: tuple[str, str],
    outputs: set[str],
    save_gt: bool,
) -> tuple[dict[str, dict[str, Path]], Path, Path, Path]:
    cam_dirs: dict[str, dict[str, Path]] = {}
    for cam_id in pair:
        per_cam: dict[str, Path] = {}
        for modality in IMAGE_MODALITIES:
            if modality in outputs:
                path = pair_root / modality / camera_label(cam_id)
                path.mkdir(parents=True, exist_ok=True)
                per_cam[modality] = path
        if save_gt:
            path = pair_root / "gt" / camera_label(cam_id)
            path.mkdir(parents=True, exist_ok=True)
            per_cam["gt"] = path
        for modality in TENSOR_MODALITIES:
            if modality in outputs:
                path = pair_root / modality / camera_label(cam_id)
                path.mkdir(parents=True, exist_ok=True)
                per_cam[modality] = path
        cam_dirs[cam_id] = per_cam

    cameras_dir = pair_root / "cameras"
    metadata_dir = pair_root / "metadata"
    frame_names_dir = pair_root / "frame_names"
    cameras_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    frame_names_dir.mkdir(parents=True, exist_ok=True)
    return cam_dirs, cameras_dir, metadata_dir, frame_names_dir


def write_pair_summary(
    *,
    pair: tuple[str, str],
    pair_name: str,
    reference_camera: str,
    canonical_entries: list[FrameSpec],
    interval_contexts: list[IntervalContext],
    test_pose_stats: dict[str, dict],
    result_dir: str,
    output_path: Path,
) -> None:
    payload = {
        "pair_name": pair_name,
        "pair_cam_ids": [pair[0], pair[1]],
        "pair_labels": [camera_label(pair[0]), camera_label(pair[1])],
        "reference_camera_id": reference_camera,
        "reference_camera_label": camera_label(reference_camera),
        "canonical_num_frames": len(canonical_entries),
        "canonical_time_ids": [spec.time_id for spec in canonical_entries],
        "canonical_reference_frame_names": [spec.frame_name for spec in canonical_entries],
        "result_dir": result_dir,
        "interval_nodes": [
            {
                "node_name": ctx.node_name,
                "left_bound": ctx.left_bound,
                "right_bound": ctx.right_bound,
            }
            for ctx in interval_contexts
        ],
        "test_pose_stats": test_pose_stats,
    }
    write_json(output_path, payload)


def write_camera_manifest(
    *,
    cam_id: str,
    pair_name: str,
    reference_camera: str,
    frame_records: list[dict],
    pose_stats: dict | None,
    scene_flow_rgb_hi: list[float] | None,
    out_path: Path,
) -> None:
    payload = {
        "pair_name": pair_name,
        "cam_id": cam_id,
        "cam_label": camera_label(cam_id),
        "reference_camera_id": reference_camera,
        "reference_camera_label": camera_label(reference_camera),
        "num_frames": len(frame_records),
        "num_valid_gt": sum(1 for record in frame_records if record["has_gt"]),
        "num_valid_test_dynamic": sum(
            1 for record in frame_records if record["has_test_dynamic"]
        ),
        "num_dataset_frames": sum(
            1 for record in frame_records if record["has_dataset_frame_at_time"]
        ),
        "num_exact_tto_pose_frames": sum(
            1 for record in frame_records if record["has_exact_tto_pose"]
        ),
        "pose_stats": pose_stats,
        "scene_flow_rgb_hi": scene_flow_rgb_hi,
        "frames": frame_records,
    }
    write_json(out_path, payload)


def main() -> None:
    args = parse_args()
    result_dir = osp.abspath(args.result_dir.rstrip("/"))
    run_dir = infer_run_dir(result_dir)
    cfg_path = infer_cfg_path(run_dir, args.cfg)
    ws = infer_workspace_path(run_dir, args.ws)
    process_height = infer_process_height(result_dir, args.process_height)
    output_dir = (
        osp.abspath(args.output_dir)
        if args.output_dir is not None
        else osp.join(result_dir, "benchmark_gbuffer")
    )
    reference_camera = normalize_camera_token(args.reference_camera)
    outputs = set(args.outputs)
    if "scene_flow_rgb" in outputs and "scene_flow" not in outputs:
        # scene_flow_rgb uses a shared per-camera scale over the full video, so we
        # materialize scene_flow tensors first and colorize them in a second pass.
        outputs.add("scene_flow")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_fit_cfg(cfg_path)
    train_camera_dir = osp.join(ws, "cameras")
    train_image_dir = osp.join(ws, "images")
    test_camera_dir = osp.join(ws, "test_cameras")
    test_image_dir = osp.join(ws, "test_images")
    test_dynamic_dir = osp.join(ws, args.test_dynamic_dirname)

    train_groups = discover_camera_frames(train_camera_dir, train_image_dir)
    test_groups = discover_camera_frames(test_camera_dir, test_image_dir)
    if reference_camera not in train_groups:
        raise KeyError(
            f"Reference camera {camera_label(reference_camera)} not found in {train_camera_dir}"
        )
    if not test_groups:
        raise RuntimeError(f"No test cameras discovered in {test_camera_dir}")
    if "test_dynamic" in outputs and not osp.isdir(test_dynamic_dir):
        raise FileNotFoundError(f"Test dynamic directory not found: {test_dynamic_dir}")

    canonical_entries = train_groups[reference_camera]
    if not canonical_entries:
        raise RuntimeError(f"No canonical reference frames found for {camera_label(reference_camera)}")

    all_test_camera_ids = sorted(test_groups.keys(), key=lambda cam_id: int(cam_id))
    pairs = parse_pairs(args.pairs, reference_camera, all_test_camera_ids)
    interval_contexts = build_interval_contexts(run_dir, process_height, cfg, device)

    log(f"result_dir: {result_dir}")
    log(f"run_dir: {run_dir}")
    log(f"ws: {ws}")
    log(f"cfg_path: {cfg_path}")
    log(f"output_dir: {output_dir}")
    log(f"device: {device}")
    log(f"process_height: {process_height}")
    log(f"reference_camera: {camera_label(reference_camera)}")
    log(f"pairs: {[(camera_label(a), camera_label(b)) for a, b in pairs]}")
    log(
        "intervals: "
        + ", ".join(
            f"{ctx.node_name}[{ctx.left_bound},{ctx.right_bound}]"
            for ctx in interval_contexts
        )
    )

    reference_specs = compute_reference_render_specs(canonical_entries, interval_contexts, result_dir)
    reference_by_time = {spec.time_id: spec for spec in reference_specs}
    train_gt_lookup = build_gt_lookup(canonical_entries)
    test_gt_lookups = {cam_id: build_gt_lookup(entries) for cam_id, entries in test_groups.items()}
    test_dynamic_lookups = {
        cam_id: build_named_png_lookup(entries, test_dynamic_dir)
        for cam_id, entries in test_groups.items()
    }

    test_render_specs: dict[str, list[CameraRenderSpec]] = {}
    test_pose_stats: dict[str, dict] = {}
    pose_index_by_camera = {cam_id: idx for idx, cam_id in enumerate(all_test_camera_ids)}
    for cam_id in all_test_camera_ids:
        pose_archive_path = osp.join(result_dir, f"tto_test_pose_{pose_index_by_camera[cam_id]}.npz")
        if not osp.isfile(pose_archive_path):
            raise FileNotFoundError(
                f"Missing TTO pose archive for {camera_label(cam_id)}: {pose_archive_path}"
            )
        template_spec = test_groups[cam_id][0]
        specs, stats = expand_tto_camera_specs(
            cam_id=cam_id,
            canonical_entries=canonical_entries,
            test_lookup=build_time_lookup(test_groups[cam_id]),
            template_spec=template_spec,
            pose_archive_path=pose_archive_path,
            short_pose_alignment=args.short_pose_alignment,
            initial_pose_fill=args.initial_pose_fill,
        )
        test_render_specs[cam_id] = specs
        test_pose_stats[cam_id] = stats
        log(
            f"{camera_label(cam_id)} pose archive: rows={stats['num_pose_rows']} "
            f"expected={stats['num_expected_dataset_frames']} "
            f"exact={stats['num_exact_pose_frames']} "
            f"carry_forward={stats['num_forward_filled_frames']} "
            f"backfilled={stats['num_backfilled_frames']} "
            f"alignment={stats['alignment_mode']}"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for pair in pairs:
            cam_a, cam_b = pair
            if cam_a != reference_camera:
                raise ValueError(
                    f"Only pairs that use reference camera {camera_label(reference_camera)} are supported."
                )
            if cam_b not in test_render_specs:
                raise KeyError(f"No test render specs prepared for {camera_label(cam_b)}")

            pair_name = f"pair_{camera_label(cam_a)}_{camera_label(cam_b)}"
            pair_root = Path(output_dir) / pair_name
            cam_dirs, cameras_dir, metadata_dir, frame_names_dir = ensure_pair_dirs(
                pair_root, pair, outputs, args.save_gt
            )
            write_pair_summary(
                pair=pair,
                pair_name=pair_name,
                reference_camera=reference_camera,
                canonical_entries=canonical_entries,
                interval_contexts=interval_contexts,
                test_pose_stats={cam_b: test_pose_stats[cam_b]},
                result_dir=result_dir,
                output_path=metadata_dir / f"{pair_name}_summary.json",
            )

            c2w_accum: dict[str, list[np.ndarray]] = {cam_a: [], cam_b: []}
            frame_records: dict[str, list[dict]] = {cam_a: [], cam_b: []}
            all_names: dict[str, list[str]] = {cam_a: [], cam_b: []}
            valid_names: dict[str, list[str]] = {cam_a: [], cam_b: []}
            scene_flow_rgb_hi_by_cam: dict[str, list[float] | None] = {
                cam_a: None,
                cam_b: None,
            }
            test_specs_by_time = {spec.time_id: spec for spec in test_render_specs[cam_b]}

            log(f"Rendering {pair_name} with {len(canonical_entries)} canonical frames")
            for frame_idx, canonical_spec in enumerate(canonical_entries):
                ctx, local_t = locate_interval_context(interval_contexts, canonical_spec.time_id)
                gs_list = _build_eval_gs_list(
                    ctx.s_model,
                    ctx.d_model,
                    ctx.fachain_d_model_list,
                    local_t,
                )

                next_t = min(canonical_spec.time_id + args.flow_step, canonical_entries[-1].time_id)
                next_ctx = None
                next_local_t = None
                if "scene_flow" in outputs and next_t > canonical_spec.time_id:
                    next_ctx, next_local_t = locate_interval_context(interval_contexts, next_t)
                    next_gs_list = _build_eval_gs_list(
                        next_ctx.s_model,
                        next_ctx.d_model,
                        next_ctx.fachain_d_model_list,
                        next_local_t,
                    )
                else:
                    next_gs_list = None

                render_specs = {
                    cam_a: reference_by_time[canonical_spec.time_id],
                    cam_b: test_specs_by_time[canonical_spec.time_id],
                }
                gt_lookup = {
                    cam_a: train_gt_lookup,
                    cam_b: test_gt_lookups[cam_b],
                }
                dynamic_lookup = {
                    cam_a: {},
                    cam_b: test_dynamic_lookups[cam_b],
                }

                for cam_id in pair:
                    render_spec = render_specs[cam_id]
                    frames, tensors, pose_meta = render_single_camera_frame(
                        spec=render_spec,
                        gs_list=gs_list,
                        next_gs_list=next_gs_list if next_ctx is ctx else None,
                        outputs=outputs,
                        scene_space=args.scene_space,
                        bg_color=args.bg_color,
                        alpha_threshold=args.alpha_threshold,
                        device=device,
                    )

                    gt_path = gt_lookup[cam_id].get(render_spec.frame_name)
                    has_gt = gt_path is not None and osp.isfile(gt_path)
                    gt_rgb_u8 = load_rgb_u8(gt_path) if has_gt else None
                    dynamic_mask_path = dynamic_lookup[cam_id].get(render_spec.frame_name)
                    has_test_dynamic = (
                        "test_dynamic" in outputs
                        and dynamic_mask_path is not None
                        and osp.isfile(dynamic_mask_path)
                    )

                    for modality, frame in frames.items():
                        if (
                            modality == "rgb"
                            and cam_id == reference_camera
                            and args.reference_rgb_source == "gt"
                            and gt_rgb_u8 is not None
                        ):
                            frame = gt_rgb_u8
                        save_frame_png(
                            frame,
                            cam_dirs[cam_id][modality] / f"{render_spec.frame_name}.png",
                        )
                    for modality, tensor in tensors.items():
                        save_frame_npz(
                            tensor,
                            modality,
                            cam_dirs[cam_id][modality] / f"{render_spec.frame_name}.npz",
                        )

                    if args.save_gt and has_gt:
                        save_frame_png(
                            gt_rgb_u8,
                            cam_dirs[cam_id]["gt"] / f"{render_spec.frame_name}.png",
                        )
                    if has_test_dynamic:
                        copy_png(
                            dynamic_mask_path,
                            cam_dirs[cam_id]["test_dynamic"] / f"{render_spec.frame_name}.png",
                        )

                    c2w_np = np.asarray(pose_meta["c2w"], dtype=np.float32)
                    c2w_accum[cam_id].append(c2w_np)
                    rgb_source_kind = "render"
                    if (
                        cam_id == reference_camera
                        and args.reference_rgb_source == "gt"
                        and gt_rgb_u8 is not None
                    ):
                        rgb_source_kind = "gt_reference_image"
                    record = {
                        "frame_idx": int(frame_idx),
                        "time_id": int(render_spec.time_id),
                        "frame_name": render_spec.frame_name,
                        "has_gt": bool(has_gt),
                        "has_test_dynamic": bool(has_test_dynamic),
                        "has_dataset_frame_at_time": bool(render_spec.has_dataset_frame_at_time),
                        "has_exact_tto_pose": bool(render_spec.pose_assignment.has_exact_tto_pose),
                        "render_pose_source_kind": render_spec.pose_assignment.source_kind,
                        "render_pose_source_time_id": render_spec.pose_assignment.source_time_id,
                        "render_pose_source_frame_name": render_spec.pose_assignment.source_frame_name,
                        "render_pose_archive_path": render_spec.pose_assignment.pose_archive_path,
                        "rgb_source_kind": rgb_source_kind,
                        "scene_flow_valid": pose_meta["scene_flow_valid"],
                    }
                    frame_records[cam_id].append(record)
                    all_names[cam_id].append(render_spec.frame_name)
                    if has_gt:
                        valid_names[cam_id].append(render_spec.frame_name)

            if "scene_flow_rgb" in outputs:
                for cam_id in pair:
                    flow_paths = [
                        cam_dirs[cam_id]["scene_flow"] / f"{frame_name}.npz"
                        for frame_name in all_names[cam_id]
                    ]
                    hi = compute_scene_flow_bounds(flow_paths)
                    scene_flow_rgb_hi_by_cam[cam_id] = hi.astype(np.float32).tolist()
                    log(
                        f"{pair_name} {camera_label(cam_id)} shared scene_flow_rgb hi={hi.tolist()}"
                    )
                    for frame_name, flow_path in zip(all_names[cam_id], flow_paths):
                        with np.load(flow_path) as data:
                            flow = data["scene_flow"].astype(np.float32)
                        flow_rgb = scene_flow_to_rgb_u8(flow, hi)
                        save_frame_png(
                            flow_rgb,
                            cam_dirs[cam_id]["scene_flow_rgb"] / f"{frame_name}.png",
                        )

            for cam_id in pair:
                write_camera_c2w_npz(
                    c2w_accum[cam_id],
                    cameras_dir / f"{camera_label(cam_id)}_c2w.npz",
                )
                pose_stats = None if cam_id == reference_camera else test_pose_stats.get(cam_id)
                write_camera_manifest(
                    cam_id=cam_id,
                    pair_name=pair_name,
                    reference_camera=reference_camera,
                    frame_records=frame_records[cam_id],
                    pose_stats=pose_stats,
                    scene_flow_rgb_hi=scene_flow_rgb_hi_by_cam[cam_id],
                    out_path=metadata_dir / f"{camera_label(cam_id)}_frames.json",
                )
                write_text_lines(
                    frame_names_dir / f"{camera_label(cam_id)}_all.txt",
                    all_names[cam_id],
                )
                write_text_lines(
                    frame_names_dir / f"{camera_label(cam_id)}_valid.txt",
                    valid_names[cam_id],
                )

    log(f"Saved benchmark gbuffers under {output_dir}")


if __name__ == "__main__":
    main()
