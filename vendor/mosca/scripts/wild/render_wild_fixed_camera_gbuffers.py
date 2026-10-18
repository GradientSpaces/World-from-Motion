#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_gen.mosca_render_types import CameraRenderSpec, PoseAssignment
from lib_gen.mosca_run_paths import infer_cfg_path
from lib_gen.worldtree_benchmark import validate_scene_layout
from wfm.camera import farthest_point_sample_positions


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
RENDER_IMAGE_MODALITIES = ("rgb", "depth", "normal", "alpha", "scene_flow_rgb")
RENDER_TENSOR_MODALITIES = ("scene_coords", "scene_flow")
REQUIRED_RENDER_OUTPUTS = ("rgb", "depth", "normal", "alpha")
SOURCE_CAMERA = "cam0"


@dataclass(frozen=True)
class CameraTimeline:
    name: str
    kind: str
    pose_frame_id: int | None
    t_cw_by_time: tuple[np.ndarray, ...]
    pose_frame_ids_by_time: tuple[int | None, ...] | None = None
    # When set, every rendered frame freezes the dynamic scene at this time id
    # (bullet-time) instead of advancing with the per-frame time id.
    fixed_scene_time_id: int | None = None
    # When set, the dynamic scene time for each rendered frame follows this explicit
    # per-frame schedule (scripted tours); scene flow is computed between consecutive
    # scheduled frames.
    scene_time_id_by_frame: tuple[int, ...] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render WorldTree/DiffSynth-compatible g-buffers from a wild MoSca "
            "photometric checkpoint using fixed target cameras sampled from the "
            "solved input trajectory."
        )
    )
    parser.add_argument("--ws", required=True, help="Wild workspace containing images/.")
    parser.add_argument("--saved-dir", "--saved_dir", required=True)
    parser.add_argument(
        "--cfg",
        default=None,
        help="MoSca config. Defaults to the --cfg recorded in fit_commandline_args.txt.",
    )
    parser.add_argument(
        "--output-root",
        "--output_dir",
        required=True,
        help="Benchmark root that will receive <scene-name>/.",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Scene directory name under --output-root. Defaults to the workspace name.",
    )
    parser.add_argument(
        "--fixed-frame-ids",
        type=int,
        nargs="+",
        default=None,
        help="Input trajectory timesteps to freeze as target cameras, e.g. 0 96 191.",
    )
    parser.add_argument(
        "--fixed-frame-preset",
        choices=["first_mid_last", "camera_fps"],
        default=None,
        help=(
            "Convenience preset. first_mid_last samples timesteps 0, T//2, and T-1; "
            "camera_fps samples solved camera centers with farthest-point sampling."
        ),
    )
    parser.add_argument(
        "--num-fps-cameras",
        type=int,
        default=3,
        help="Number of fixed cameras to sample when --fixed-frame-preset camera_fps is used.",
    )
    parser.add_argument(
        "--append-reverse-trajectory",
        action="store_true",
        help="Append one target camera that follows the solved source trajectory in reverse time.",
    )
    parser.add_argument(
        "--reverse-camera-name",
        default="cam_reverse",
        help="Camera name for --append-reverse-trajectory.",
    )
    parser.add_argument(
        "--interpolate-fps-anchors",
        action="store_true",
        help=(
            "Sample FPS anchors from the solved camera path and render smooth moving "
            "trajectories between adjacent anchors instead of rendering fixed anchors."
        ),
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=None,
        help=(
            "Target camera names corresponding to --fixed-frame-ids. Defaults to "
            "cam_first cam_mid cam_last for --fixed-frame-preset first_mid_last."
        ),
    )
    parser.add_argument(
        "--lyra-trajectory-modes",
        nargs="+",
        default=None,
        choices=[
            "left",
            "right",
            "up",
            "down",
            "zoom_in",
            "zoom_out",
            "clockwise",
            "counterclockwise",
        ],
        help=(
            "Generate Lyra-style local camera trajectories from FPS anchor poses. "
            "If set, this replaces fixed target cameras."
        ),
    )
    parser.add_argument(
        "--lyra-movement-distance",
        type=float,
        default=0.06,
        help="Motion distance as a fraction of estimated source camera trajectory radius.",
    )
    parser.add_argument(
        "--lyra-camera-rotation",
        choices=["no_rotation", "center_facing", "trajectory_aligned"],
        default="no_rotation",
        help="Rotation behavior for Lyra-style local trajectories.",
    )
    parser.add_argument(
        "--lyra-num-circles",
        type=float,
        default=1.0,
        help="Number of circles for clockwise/counterclockwise Lyra-style spiral paths.",
    )
    parser.add_argument(
        "--lyra-radius-x-factor",
        type=float,
        default=1.0,
        help="Horizontal radius multiplier for spiral Lyra-style paths.",
    )
    parser.add_argument(
        "--lyra-radius-y-factor",
        type=float,
        default=0.7,
        help="Vertical radius multiplier for spiral Lyra-style paths.",
    )
    parser.add_argument(
        "--photometric-dolly",
        action="store_true",
        help=(
            "Render the same center-depth orbit camera used by photometric_2dviz "
            "Novel/T panels."
        ),
    )
    parser.add_argument(
        "--photometric-dolly-angle-deg",
        type=float,
        default=10.0,
        help="Half-angle used to convert focus depth to orbit radius. Default matches photo_viz_move_angle_deg.",
    )
    parser.add_argument(
        "--photometric-dolly-focus-frame-id",
        type=int,
        default=None,
        help="Frame whose camera and center depth define the orbit. Defaults to the middle rendered timestep.",
    )
    parser.add_argument(
        "--photometric-dolly-fix-time",
        action="store_true",
        help=(
            "Bullet-time: freeze the dynamic scene at the dolly focus frame while the "
            "orbit camera moves (photometric_2dviz 'T=' panel). Default advances scene "
            "time with the camera (the 'Novel' panel)."
        ),
    )
    parser.add_argument(
        "--source-static-frame-id",
        type=int,
        default=None,
        help=(
            "Freeze the cam0 source/reference at this frame: the gt image, camera pose, "
            "and scene time are all held at this frame for every rendered timestep "
            "(static reference still). Use for bullet-time so the reference video is a "
            "frozen frame matching the focus frame. Default keeps the moving input clip."
        ),
    )
    parser.add_argument(
        "--photometric-dolly-base-cam-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional MonocularCameras checkpoint (e.g. new_traj.pth) whose pose at the "
            "dolly focus frame becomes the orbit anchor instead of the recon camera. The "
            "checkpoint must be 1:1 frame-aligned with the recon and live in the same "
            "world frame; the recon intrinsics and dynamic scene are still used."
        ),
    )
    parser.add_argument(
        "--external-camera-trajectory",
        type=Path,
        default=None,
        help=(
            "External TUM-style camera-to-world trajectory file to use as moving "
            "target cameras. Rows must be timestamp tx ty tz qx qy qz qw."
        ),
    )
    parser.add_argument(
        "--external-camera-npz",
        type=Path,
        default=None,
        help=(
            "External MultiCamVideo-style cameras_opencv.npz containing camXX_c2w "
            "trajectories to reuse as moving target cameras."
        ),
    )
    parser.add_argument(
        "--external-camera-ids",
        nargs="+",
        default=None,
        help=(
            "Camera ids to load from --external-camera-npz, e.g. cam01 cam02. "
            "Defaults to all ids in the npz."
        ),
    )
    parser.add_argument(
        "--external-camera-align-mode",
        choices=["none", "center_scale", "pca_similarity"],
        default="pca_similarity",
        help="How to align external npz trajectories to the solved source camera path.",
    )
    parser.add_argument(
        "--external-camera-ranges",
        nargs="+",
        default=None,
        help=(
            "Inclusive source pose ranges for external moving target cameras, e.g. "
            "0-146 200-346 last. The token 'last' uses the final range matching "
            "the rendered frame count and external stride."
        ),
    )
    parser.add_argument(
        "--external-camera-stride",
        type=int,
        default=1,
        help="Stride used within each external camera range.",
    )
    parser.add_argument(
        "--scripted-camera-npz",
        type=Path,
        default=None,
        help=(
            "Scripted per-frame trajectory npz with keys c2w:[N,4,4] (camera-to-world "
            "in the recon world frame) and time_ids:[N] (per-frame scene time id). "
            "Renders one target camera following this scripted (pose, scene-time) path; "
            "scene flow is computed between consecutive scheduled frames."
        ),
    )
    parser.add_argument(
        "--scripted-camera-name",
        default="cam_tour",
        help="Target camera name for --scripted-camera-npz.",
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
        ],
        choices=[
            "rgb",
            "depth",
            "normal",
            "alpha",
            "scene_coords",
            "scene_flow",
            "scene_flow_rgb",
        ],
    )
    parser.add_argument(
        "--scene-space",
        "--scene_space",
        default="world",
        choices=["world", "camera"],
        help="Coordinate space for scene_coords and scene_flow tensors.",
    )
    parser.add_argument("--flow-step", "--flow_step", type=int, default=1)
    parser.add_argument("--alpha-threshold", type=float, default=1e-3)
    parser.add_argument("--bg-color", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap after frame stride is applied, mainly for smoke tests.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth input timestep. Use 2 to render every other frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing scene directory.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip WorldTree layout validation after rendering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print the render plan without loading checkpoints or writing files.",
    )
    return parser.parse_args()


def normalize_outputs(outputs: Sequence[str]) -> set[str]:
    normalized = set(outputs)
    missing_required = sorted(set(REQUIRED_RENDER_OUTPUTS).difference(normalized))
    if missing_required:
        raise ValueError(
            "DiffSynth export requires these render outputs: "
            + ", ".join(REQUIRED_RENDER_OUTPUTS)
        )
    if "scene_flow_rgb" in normalized:
        normalized.add("scene_flow")
    return normalized


def normalize_camera_name(name: str) -> str:
    token = str(name).strip()
    if token == "":
        raise ValueError("Camera names must be non-empty.")
    if token.isdigit():
        return f"cam{int(token)}"
    return token if token.startswith("cam") else f"cam{token}"


def t_cw_to_c2w(t_cw: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(t_cw, dtype=np.float32)).astype(np.float32)


def camera_centers_from_t_cw(t_cw_by_time: Sequence[np.ndarray]) -> np.ndarray:
    centers = [t_cw_to_c2w(t_cw)[:3, 3] for t_cw in t_cw_by_time]
    return np.asarray(centers, dtype=np.float32)


def normalize_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {arr}")
    return (arr / norm).astype(np.float32)


def c2w_to_t_cw(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(c2w, dtype=np.float32)).astype(np.float32)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Invalid zero-norm quaternion")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        diag = np.diag(m)
        if diag[0] > diag[1] and diag[0] > diag[2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif diag[1] > diag[2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat.astype(np.float32)


def slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(alpha) * (q1 - q0)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        return q.astype(np.float32)
    theta_0 = np.arccos(dot)
    theta = theta_0 * float(alpha)
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0 + s1 * q1).astype(np.float32)


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def load_tum_c2w_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"Malformed trajectory row in {path}: {raw_line}")
        timestamp = float(fields[0])
        tx, ty, tz = (float(value) for value in fields[1:4])
        qx, qy, qz, qw = (float(value) for value in fields[4:8])
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = quat_xyzw_to_matrix(np.asarray([qx, qy, qz, qw], dtype=np.float64))
        c2w[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float32)
        timestamps.append(timestamp)
        poses.append(c2w)
    if not poses:
        raise ValueError(f"No trajectory poses found in {path}")
    return np.asarray(timestamps, dtype=np.float64), np.stack(poses, axis=0).astype(np.float32)


def estimate_camera_path_radius(t_cw_by_time: Sequence[np.ndarray]) -> float:
    centers = camera_centers_from_t_cw(t_cw_by_time)
    if len(centers) == 0:
        return 1.0
    center = np.mean(centers, axis=0, dtype=np.float32)
    dists = np.linalg.norm(centers - center[None], axis=1)
    return max(float(np.percentile(dists, 75)), 1e-3)


def look_at_c2w(center: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    forward = normalize_vector(np.asarray(target, dtype=np.float32) - center)
    up_hint = normalize_vector(up_hint)
    right = np.cross(up_hint, forward)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float32), forward)
    right = normalize_vector(right)
    down = normalize_vector(np.cross(forward, right))
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = center
    return c2w


def resolve_fixed_cameras(
    args: argparse.Namespace,
    total_frames: int,
    source_t_cw: Sequence[np.ndarray] | None = None,
    source_time_ids: Sequence[int] | None = None,
) -> tuple[list[int], list[str]]:
    if total_frames <= 0:
        raise ValueError("Cannot sample fixed cameras from an empty trajectory.")
    if args.fixed_frame_preset is not None and args.fixed_frame_ids is not None:
        raise ValueError("Use either --fixed-frame-preset or --fixed-frame-ids, not both.")
    if args.fixed_frame_preset == "first_mid_last":
        fixed_frame_ids = [0, total_frames // 2, total_frames - 1]
        default_camera_names = ["cam_first", "cam_mid", "cam_last"]
    elif args.fixed_frame_preset == "camera_fps":
        if args.num_fps_cameras <= 0:
            raise ValueError("--num-fps-cameras must be positive.")
        if source_t_cw is None:
            raise ValueError("--fixed-frame-preset camera_fps requires loaded camera poses.")
        selected_positions = farthest_point_sample_positions(
            camera_centers_from_t_cw(source_t_cw),
            num_samples=int(args.num_fps_cameras),
        )
        if source_time_ids is None:
            fixed_frame_ids = selected_positions
        else:
            fixed_frame_ids = [int(source_time_ids[pos]) for pos in selected_positions]
        default_camera_names = [f"cam_fps_{idx:02d}" for idx in range(len(fixed_frame_ids))]
    elif args.fixed_frame_ids is not None:
        fixed_frame_ids = [int(value) for value in args.fixed_frame_ids]
        default_camera_names = [f"cam_fixed_{value:05d}" for value in fixed_frame_ids]
    else:
        raise ValueError("Specify --fixed-frame-preset first_mid_last or --fixed-frame-ids.")

    camera_names = (
        default_camera_names
        if args.camera_names is None
        else [normalize_camera_name(name) for name in args.camera_names]
    )
    if len(fixed_frame_ids) != len(camera_names):
        raise ValueError("Fixed frame ids and camera names must have the same length.")
    return fixed_frame_ids, camera_names


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image frames found under {image_dir}")
    return images


def infer_num_frames_from_logs(saved_dir: Path) -> int | None:
    for log_name in ("moca_solve.log",):
        log_path = saved_dir / log_name
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        import re

        matches = re.findall(r"MonoCam Summary: T=(\d+)", text)
        if matches:
            return int(matches[-1])
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if lines:
            handle.write("\n".join(lines))
            handle.write("\n")


def write_camera_c2w_npz(t_cw_by_time: Sequence[np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c2w = [np.linalg.inv(t_cw).astype(np.float32) for t_cw in t_cw_by_time]
    np.savez_compressed(out_path, c2w=np.stack(c2w, axis=0))


def load_rgb_u8(path: Path) -> np.ndarray:
    rgb = imageio.imread(path)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def prepare_scene_root(scene_root: Path, overwrite: bool) -> None:
    if scene_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Scene directory already exists: {scene_root}. Pass --overwrite to replace it."
            )
        for attempt in range(5):
            shutil.rmtree(scene_root, ignore_errors=attempt > 0)
            if not scene_root.exists():
                break
            time.sleep(1.0)
        if scene_root.exists():
            subprocess.run(["rm", "-rf", str(scene_root)], check=True)
        if scene_root.exists():
            raise OSError(f"Failed to remove existing scene directory: {scene_root}")
    scene_root.mkdir(parents=True, exist_ok=True)


def ensure_modality_dirs(scene_root: Path, camera_names: Sequence[str], outputs: set[str]) -> None:
    for dirname in ("cameras", "metadata", "frame_names"):
        (scene_root / dirname).mkdir(parents=True, exist_ok=True)
    for modality in (*REQUIRED_RENDER_OUTPUTS, "gt"):
        for camera_name in camera_names:
            (scene_root / modality / camera_name).mkdir(parents=True, exist_ok=True)
    for modality in RENDER_IMAGE_MODALITIES:
        if modality in outputs:
            for camera_name in camera_names:
                (scene_root / modality / camera_name).mkdir(parents=True, exist_ok=True)
    for modality in RENDER_TENSOR_MODALITIES:
        if modality in outputs:
            for camera_name in camera_names:
                (scene_root / modality / camera_name).mkdir(parents=True, exist_ok=True)


def camera_spec(
    *,
    timeline: CameraTimeline,
    sample_idx: int,
    time_id: int,
    frame_name: str,
    image_path: Path | None,
    k_raw: np.ndarray,
    img_wh: tuple[int, int],
    saved_dir: Path,
) -> CameraRenderSpec:
    t_cw = timeline.t_cw_by_time[sample_idx]
    if timeline.kind == "source":
        source_kind = "solved_source_camera"
        # When the source is frozen (static reference), report the focus frame as the
        # pose/scene source time rather than the advancing per-frame time id.
        if timeline.fixed_scene_time_id is not None and timeline.pose_frame_id is not None:
            source_kind = "static_source_camera"
            source_time_id = timeline.pose_frame_id
        else:
            source_time_id = time_id
        source_frame_name = frame_name
        has_exact = True
    elif timeline.kind in {"external", "external_npz"}:
        source_kind = (
            "external_multicam_npz_trajectory_camera"
            if timeline.kind == "external_npz"
            else "external_mocap_trajectory_camera"
        )
        source_time_id = (
            None
            if timeline.pose_frame_ids_by_time is None
            else timeline.pose_frame_ids_by_time[sample_idx]
        )
        source_frame_name = None if source_time_id is None else f"{source_time_id:05d}"
        has_exact = False
    elif timeline.kind == "interpolated_fps":
        source_kind = "interpolated_fps_anchor_trajectory_camera"
        source_time_id = (
            timeline.pose_frame_id
            if timeline.pose_frame_ids_by_time is None
            else timeline.pose_frame_ids_by_time[sample_idx]
        )
        source_frame_name = None if source_time_id is None else f"{source_time_id:05d}"
        has_exact = False
    elif timeline.kind == "reverse_source":
        source_kind = "reverse_input_trajectory_camera"
        source_time_id = (
            timeline.pose_frame_id
            if timeline.pose_frame_ids_by_time is None
            else timeline.pose_frame_ids_by_time[sample_idx]
        )
        source_frame_name = None if source_time_id is None else f"{source_time_id:05d}"
        has_exact = False
    elif timeline.kind == "photometric_dolly":
        source_kind = "photometric_2dviz_center_depth_orbit_camera"
        source_time_id = timeline.pose_frame_id
        source_frame_name = (
            None if timeline.pose_frame_id is None else f"{timeline.pose_frame_id:05d}"
        )
        has_exact = False
    elif timeline.kind == "scripted":
        source_kind = "scripted_pose_time_trajectory_camera"
        source_time_id = (
            int(timeline.scene_time_id_by_frame[sample_idx])
            if timeline.scene_time_id_by_frame is not None
            else time_id
        )
        source_frame_name = None if source_time_id is None else f"{source_time_id:05d}"
        has_exact = False
    else:
        source_kind = "fixed_input_trajectory_camera"
        source_time_id = timeline.pose_frame_id
        source_frame_name = (
            None if timeline.pose_frame_id is None else f"{timeline.pose_frame_id:05d}"
        )
        has_exact = False

    return CameraRenderSpec(
        cam_id=timeline.name,
        time_id=time_id,
        frame_name=frame_name,
        camera_json_path=None,
        image_path=None if image_path is None else str(image_path),
        K_raw=k_raw.copy(),
        img_wh=img_wh,
        T_cw=t_cw.copy(),
        has_dataset_frame_at_time=timeline.kind == "source",
        pose_assignment=PoseAssignment(
            T_cw=t_cw.copy(),
            source_kind=source_kind,
            source_time_id=source_time_id,
            source_frame_name=source_frame_name,
            source_camera_json_path=None,
            pose_archive_path=str(saved_dir),
            has_exact_tto_pose=has_exact,
        ),
    )


def render_camera_timeline(
    *,
    timeline: CameraTimeline,
    frame_names: Sequence[str],
    time_ids: Sequence[int],
    max_time_id: int,
    image_paths: Sequence[Path],
    scene_root: Path,
    outputs: set[str],
    k_raw: np.ndarray,
    img_wh: tuple[int, int],
    saved_dir: Path,
    s_model,
    d_model,
    flow_step: int,
    scene_space: str,
    bg_color: list[float],
    alpha_threshold: float,
    device: torch.device,
) -> tuple[list[dict], list[float] | None]:
    from lib_gen.mosca_gbuffer_render import (
        compute_scene_flow_bounds,
        render_single_camera_frame,
        save_frame_npz,
        save_frame_png,
        scene_flow_to_rgb_u8,
    )

    records: list[dict] = []
    flow_paths: list[Path] | None = [] if "scene_flow_rgb" in outputs else None

    with torch.inference_mode():
        for frame_idx, frame_name in enumerate(frame_names):
            # record_t is the per-frame time id used for metadata / eval-unit pairing
            # (must match the reference camera). scene_t is the time the dynamic scene
            # is sampled at; for bullet-time it is frozen at the focus frame.
            record_t = int(time_ids[frame_idx])
            if timeline.scene_time_id_by_frame is not None:
                # Scripted tour: scene time follows an explicit per-frame schedule, and
                # scene flow is computed toward the NEXT scheduled frame (so a frozen
                # segment yields zero flow and a reversed segment yields negative flow).
                schedule = timeline.scene_time_id_by_frame
                scene_t = int(schedule[frame_idx])
                next_t = int(schedule[frame_idx + 1]) if frame_idx + 1 < len(schedule) else scene_t
            elif timeline.fixed_scene_time_id is not None:
                # Bullet-time: freeze the scene; camera moves but time does not, so
                # there is no temporal scene flow (next_gs_list stays None -> zero flow).
                scene_t = int(timeline.fixed_scene_time_id)
                next_t = scene_t
            else:
                scene_t = record_t
                next_t = min(scene_t + flow_step, int(max_time_id))
            s_gs = s_model()
            gs_list = [s_gs, d_model(scene_t)]
            # Compute flow toward next_t whenever it differs (forward or reverse);
            # equal times (frozen / clamped last frame) -> None -> zero flow.
            next_gs_list = [s_gs, d_model(next_t)] if next_t != scene_t else None
            spec = camera_spec(
                timeline=timeline,
                sample_idx=frame_idx,
                time_id=record_t,
                frame_name=frame_name,
                image_path=image_paths[frame_idx] if timeline.kind == "source" else None,
                k_raw=k_raw,
                img_wh=img_wh,
                saved_dir=saved_dir,
            )
            frames, tensors, pose_meta = render_single_camera_frame(
                spec=spec,
                gs_list=gs_list,
                next_gs_list=next_gs_list,
                outputs=outputs,
                scene_space=scene_space,
                bg_color=bg_color,
                alpha_threshold=alpha_threshold,
                device=device,
            )

            for modality, frame in frames.items():
                if modality == "scene_flow_rgb":
                    continue
                save_frame_png(
                    frame,
                    scene_root / modality / timeline.name / f"{frame_name}.png",
                )
            for modality, tensor in tensors.items():
                out_path = scene_root / modality / timeline.name / f"{frame_name}.npz"
                save_frame_npz(tensor, modality, out_path)
                if modality == "scene_flow" and flow_paths is not None:
                    flow_paths.append(out_path)

            has_gt = timeline.kind == "source"
            if has_gt:
                save_frame_png(
                    load_rgb_u8(image_paths[frame_idx]),
                    scene_root / "gt" / timeline.name / f"{frame_name}.png",
                )

            records.append(
                {
                    "frame_idx": int(frame_idx),
                    "time_id": int(record_t),
                    "frame_name": frame_name,
                    "reference_frame_name": frame_name,
                    "source_frame_name": frame_name,
                    "has_gt": bool(has_gt),
                    "has_test_dynamic": False,
                    "has_dataset_frame_at_time": bool(timeline.kind == "source"),
                    "has_exact_tto_pose": bool(timeline.kind == "source"),
                    "render_pose_source_kind": pose_meta["render_pose_source_kind"],
                    "render_pose_source_time_id": pose_meta["render_pose_source_time_id"],
                    "render_pose_archive_path": pose_meta["render_pose_archive_path"],
                    "rgb_source_kind": "gt_reference_image" if has_gt else "render",
                    "scene_flow_valid": bool(pose_meta["scene_flow_valid"]),
                }
            )

    if flow_paths is not None:
        hi = compute_scene_flow_bounds(flow_paths)
        for flow_path in flow_paths:
            with np.load(flow_path) as data:
                flow = data["scene_flow"].astype(np.float32)
            save_frame_png(
                scene_flow_to_rgb_u8(flow, hi),
                scene_root
                / "scene_flow_rgb"
                / timeline.name
                / f"{flow_path.stem}.png",
            )
        scene_flow_rgb_hi = hi.astype(np.float32).tolist()
    else:
        scene_flow_rgb_hi = None

    return records, scene_flow_rgb_hi


def write_camera_manifest(
    *,
    scene_root: Path,
    camera_name: str,
    reference_camera: str,
    records: Sequence[dict],
    scene_flow_rgb_hi: Sequence[float] | None,
) -> None:
    payload = {
        "scene_name": scene_root.name,
        "cam_id": camera_name,
        "cam_label": camera_name,
        "reference_camera_id": reference_camera,
        "reference_camera_label": reference_camera,
        "num_frames": len(records),
        "num_valid_gt": sum(1 for record in records if record["has_gt"]),
        "num_valid_test_dynamic": 0,
        "num_dataset_frames": sum(
            1 for record in records if record["has_dataset_frame_at_time"]
        ),
        "num_exact_tto_pose_frames": sum(
            1 for record in records if record["has_exact_tto_pose"]
        ),
        "pose_stats": None,
        "scene_flow_rgb_hi": None
        if scene_flow_rgb_hi is None
        else [float(value) for value in scene_flow_rgb_hi],
        "frames": list(records),
    }
    write_json(scene_root / "metadata" / f"{camera_name}_frames.json", payload)
    all_names = [str(record["frame_name"]) for record in records]
    valid_names = [str(record["frame_name"]) for record in records if record["has_gt"]]
    write_text_lines(scene_root / "frame_names" / f"{camera_name}_all.txt", all_names)
    write_text_lines(scene_root / "frame_names" / f"{camera_name}_valid.txt", valid_names)


def write_eval_units(
    scene_root: Path,
    reference_camera: str,
    target_cameras: Sequence[str],
) -> None:
    units = [
        {
            "unit_id": f"{reference_camera}_to_{target_camera}",
            "source_camera": reference_camera,
            "target_camera": target_camera,
            "reference_camera": reference_camera,
        }
        for target_camera in target_cameras
    ]
    write_json(
        scene_root / "metadata" / "eval_units.json",
        {"reference_camera": reference_camera, "evaluation_units": units},
    )


def frame_alpha(frame_idx: int, num_frames: int) -> float:
    if num_frames <= 1:
        return 0.0
    return float(frame_idx) / float(num_frames - 1)


def local_axis_for_lyra_mode(c2w: np.ndarray, mode: str) -> np.ndarray:
    right = c2w[:3, 0]
    down = c2w[:3, 1]
    forward = c2w[:3, 2]
    if mode == "left":
        return -right
    if mode == "right":
        return right
    if mode == "up":
        return -down
    if mode == "down":
        return down
    if mode == "zoom_in":
        return forward
    if mode == "zoom_out":
        return -forward
    raise ValueError(f"Unsupported local translation mode: {mode}")


def lyra_oriented_c2w(
    anchor_c2w: np.ndarray,
    center: np.ndarray,
    target: np.ndarray,
    motion_vec: np.ndarray,
    rotation_mode: str,
) -> np.ndarray:
    c2w = anchor_c2w.copy()
    c2w[:3, 3] = center
    if rotation_mode == "no_rotation":
        return c2w
    if rotation_mode == "center_facing":
        return look_at_c2w(center, target, -anchor_c2w[:3, 1])
    if rotation_mode == "trajectory_aligned":
        return look_at_c2w(center, target + motion_vec * 2.0, -anchor_c2w[:3, 1])
    raise ValueError(f"Unsupported Lyra camera rotation mode: {rotation_mode}")


def build_lyra_trajectory_timelines(
    *,
    modes: Sequence[str],
    source_t_cw: Sequence[np.ndarray],
    time_ids: Sequence[int],
    num_anchors: int,
    num_frames: int,
    movement_distance: float,
    rotation_mode: str,
    num_circles: float,
    radius_x_factor: float,
    radius_y_factor: float,
) -> tuple[list[CameraTimeline], list[int], dict]:
    selected_positions = farthest_point_sample_positions(
        camera_centers_from_t_cw(source_t_cw),
        num_samples=int(num_anchors),
    )
    anchors = [(int(time_ids[pos]), int(pos)) for pos in selected_positions]
    path_radius = estimate_camera_path_radius(source_t_cw)
    distance = float(movement_distance) * path_radius
    timelines: list[CameraTimeline] = []
    for anchor_idx, (frame_id, source_pos) in enumerate(anchors):
        anchor_c2w = t_cw_to_c2w(source_t_cw[source_pos])
        anchor_center = anchor_c2w[:3, 3]
        focus = anchor_center + anchor_c2w[:3, 2] * max(path_radius, 1e-3)
        for mode in modes:
            poses = []
            for frame_idx in range(num_frames):
                alpha = frame_alpha(frame_idx, num_frames)
                if mode in {"clockwise", "counterclockwise"}:
                    sign = 1.0 if mode == "clockwise" else -1.0
                    theta = 2.0 * np.pi * float(num_circles) * alpha
                    local_motion = (
                        anchor_c2w[:3, 0]
                        * (distance * float(radius_x_factor) * (np.cos(theta) - 1.0) * sign)
                        + anchor_c2w[:3, 1]
                        * (distance * float(radius_y_factor) * np.sin(theta))
                    )
                else:
                    local_motion = local_axis_for_lyra_mode(anchor_c2w, mode) * distance * alpha
                center = anchor_center + local_motion
                c2w = lyra_oriented_c2w(anchor_c2w, center, focus, local_motion, rotation_mode)
                poses.append(c2w_to_t_cw(c2w))
            timelines.append(
                CameraTimeline(
                    name=f"cam_lyra_{mode}_{anchor_idx:02d}",
                    kind="lyra",
                    pose_frame_id=int(frame_id),
                    t_cw_by_time=tuple(poses),
                )
            )
    metadata = {
        "lyra_trajectory_modes": list(modes),
        "anchor_frame_ids": [frame_id for frame_id, _ in anchors],
        "movement_distance_fraction": float(movement_distance),
        "movement_distance_world": float(distance),
        "camera_rotation": rotation_mode,
        "path_radius": float(path_radius),
    }
    return timelines, [frame_id for frame_id, _ in anchors], metadata


def resolve_photometric_dolly_camera_name(camera_names: Sequence[str] | None) -> str:
    if camera_names is None:
        return "cam_photometric_dolly"
    names = [normalize_camera_name(name) for name in camera_names]
    if len(names) != 1:
        raise ValueError("--photometric-dolly expects exactly one --camera-names value.")
    return names[0]


@torch.no_grad()
def estimate_center_depth_for_photometric_dolly(
    *,
    s_model,
    d_model,
    cams,
    focus_frame_id: int,
    base_t_cw: np.ndarray | None = None,
) -> float:
    from lib_render.render_helper import render

    gs_list = [s_model()]
    if d_model is not None:
        gs_list.append(d_model(focus_frame_id))
    # Render the frozen scene from the orbit anchor pose; defaults to the recon's own
    # camera at the focus frame, but may be an external anchor (e.g. new_traj.pth).
    if base_t_cw is None:
        render_T_cw = cams.T_cw(focus_frame_id)
    else:
        render_T_cw = torch.from_numpy(np.asarray(base_t_cw, dtype=np.float32)).to(
            cams.default_K
        )
    render_dict = render(
        gs_list,
        cams.default_H,
        cams.default_W,
        cams.default_K,
        render_T_cw,
    )
    depth = render_dict["dep"][0]
    center_dep = float(depth[depth.shape[0] // 2, depth.shape[1] // 2].item())
    if center_dep >= 1e-2:
        return center_dep

    alpha = render_dict.get("alpha")
    if alpha is not None:
        valid_depth = depth[alpha[0] > 0.1]
        if valid_depth.numel() > 0:
            return float(valid_depth.min().item())
    return 1.0


def build_photometric_dolly_timeline(
    *,
    camera_name: str,
    base_t_cw: np.ndarray,
    focus_depth: float,
    focus_frame_id: int,
    num_frames: int,
    angle_deg: float,
    fix_scene_time: bool = False,
) -> tuple[CameraTimeline, dict]:
    if num_frames <= 0:
        raise ValueError("Cannot build photometric dolly trajectory without frames.")

    focus_point = np.asarray([0.0, 0.0, float(focus_depth)], dtype=np.float32)
    radius = float(np.tan(np.deg2rad(float(angle_deg))) * focus_point[2])
    base_c2w = t_cw_to_c2w(base_t_cw)
    poses: list[np.ndarray] = []
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    for frame_idx in range(num_frames):
        alpha = frame_alpha(frame_idx, num_frames)
        theta = 2.0 * np.pi * alpha
        camera_in_base = np.eye(4, dtype=np.float32)
        camera_in_base[0, 3] = radius * np.cos(theta)
        camera_in_base[1, 3] = radius * np.sin(theta)

        z_dir = normalize_vector(focus_point - camera_in_base[:3, 3])
        x_dir = np.cross(up, z_dir)
        if np.linalg.norm(x_dir) < 1e-8:
            x_dir = np.cross(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), z_dir)
        x_dir = normalize_vector(x_dir)
        y_dir = normalize_vector(np.cross(z_dir, x_dir))
        camera_in_base[:3, 0] = x_dir
        camera_in_base[:3, 1] = y_dir
        camera_in_base[:3, 2] = z_dir

        c2w = base_c2w @ camera_in_base
        poses.append(c2w_to_t_cw(c2w))

    timeline = CameraTimeline(
        name=camera_name,
        kind="photometric_dolly",
        pose_frame_id=int(focus_frame_id),
        t_cw_by_time=tuple(poses),
        fixed_scene_time_id=int(focus_frame_id) if fix_scene_time else None,
    )
    metadata = {
        "camera_name": camera_name,
        "focus_frame_id": int(focus_frame_id),
        "focus_depth": float(focus_depth),
        "angle_deg": float(angle_deg),
        "orbit_radius": float(radius),
        "trajectory": "photometric_2dviz_center_depth_orbit",
        "fix_scene_time": bool(fix_scene_time),
        "mode": "bullet_time" if fix_scene_time else "vary_cam_time",
    }
    return timeline, metadata


def interpolate_c2w_poses(c2w_a: np.ndarray, c2w_b: np.ndarray, alpha: float) -> np.ndarray:
    alpha_s = smoothstep(alpha)
    center = (1.0 - alpha_s) * c2w_a[:3, 3] + alpha_s * c2w_b[:3, 3]
    q_a = matrix_to_quat_xyzw(c2w_a[:3, :3])
    q_b = matrix_to_quat_xyzw(c2w_b[:3, :3])
    q = slerp_quat_xyzw(q_a, q_b, alpha_s)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = quat_xyzw_to_matrix(q)
    c2w[:3, 3] = center.astype(np.float32)
    return c2w


def resample_c2w_sequence(c2w_by_frame: np.ndarray, num_frames: int) -> np.ndarray:
    c2w_by_frame = np.asarray(c2w_by_frame, dtype=np.float32)
    if c2w_by_frame.ndim != 3 or c2w_by_frame.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w sequence with shape (T,4,4), got {c2w_by_frame.shape}")
    if num_frames <= 0:
        raise ValueError("Cannot resample trajectory to a non-positive frame count.")
    if len(c2w_by_frame) == 0:
        raise ValueError("Cannot resample an empty trajectory.")
    if len(c2w_by_frame) == num_frames:
        return c2w_by_frame.astype(np.float32)
    if num_frames == 1:
        return c2w_by_frame[:1].astype(np.float32)
    src_pos = np.linspace(0.0, float(len(c2w_by_frame) - 1), num_frames)
    out = []
    for pos in src_pos:
        lo = int(np.floor(pos))
        hi = min(lo + 1, len(c2w_by_frame) - 1)
        alpha = float(pos - lo)
        out.append(interpolate_c2w_poses(c2w_by_frame[lo], c2w_by_frame[hi], alpha))
    return np.stack(out, axis=0).astype(np.float32)


def pca_basis(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh.T.astype(np.float32)
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return basis


def rms_radius(points: np.ndarray, center: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    return max(float(np.sqrt(np.mean(np.sum((points - center[None]) ** 2, axis=1)))), 1e-6)


def align_external_c2w_sequences(
    *,
    external_sequences: Sequence[np.ndarray],
    source_t_cw: Sequence[np.ndarray],
    mode: str,
) -> tuple[list[np.ndarray], dict]:
    source_centers = camera_centers_from_t_cw(source_t_cw)
    external_centers = np.concatenate([seq[:, :3, 3] for seq in external_sequences], axis=0)
    src_center = np.median(source_centers, axis=0).astype(np.float32)
    ext_center = np.median(external_centers, axis=0).astype(np.float32)
    if mode == "none":
        rotation = np.eye(3, dtype=np.float32)
        scale = 1.0
        translation = np.zeros(3, dtype=np.float32)
    else:
        scale = rms_radius(source_centers, src_center) / rms_radius(external_centers, ext_center)
        if mode == "center_scale":
            rotation = np.eye(3, dtype=np.float32)
        elif mode == "pca_similarity":
            rotation = (pca_basis(source_centers) @ pca_basis(external_centers).T).astype(np.float32)
            if np.linalg.det(rotation) < 0.0:
                rotation[:, -1] *= -1.0
        else:
            raise ValueError(f"Unknown external camera align mode: {mode}")
        translation = src_center - float(scale) * (rotation @ ext_center)

    aligned = []
    for seq in external_sequences:
        out = seq.copy().astype(np.float32)
        out[:, :3, :3] = np.einsum("ij,tjk->tik", rotation, out[:, :3, :3])
        out[:, :3, 3] = (float(scale) * (out[:, :3, 3] @ rotation.T)) + translation[None]
        aligned.append(out.astype(np.float32))
    metadata = {
        "align_mode": mode,
        "scale": float(scale),
        "source_center": src_center.astype(float).tolist(),
        "external_center": ext_center.astype(float).tolist(),
        "translation": translation.astype(float).tolist(),
        "rotation": rotation.astype(float).tolist(),
    }
    return aligned, metadata


def build_interpolated_fps_anchor_timelines(
    *,
    source_t_cw: Sequence[np.ndarray],
    time_ids: Sequence[int],
    num_anchors: int,
    num_frames: int,
) -> tuple[list[CameraTimeline], list[int], dict]:
    if int(num_anchors) < 2:
        raise ValueError("--interpolate-fps-anchors requires at least 2 FPS anchors.")
    selected_positions = farthest_point_sample_positions(
        camera_centers_from_t_cw(source_t_cw),
        num_samples=int(num_anchors),
    )
    anchors = [(int(time_ids[pos]), int(pos)) for pos in selected_positions]
    if len(anchors) < 2:
        raise ValueError("Need at least two sampled FPS anchors for interpolation.")

    timelines: list[CameraTimeline] = []
    for interp_idx, ((start_frame, start_pos), (end_frame, end_pos)) in enumerate(
        zip(anchors[:-1], anchors[1:])
    ):
        c2w_a = t_cw_to_c2w(source_t_cw[start_pos])
        c2w_b = t_cw_to_c2w(source_t_cw[end_pos])
        poses = []
        for frame_idx in range(num_frames):
            alpha = frame_alpha(frame_idx, num_frames)
            poses.append(c2w_to_t_cw(interpolate_c2w_poses(c2w_a, c2w_b, alpha)))
        timelines.append(
            CameraTimeline(
                name=f"cam_interp_{interp_idx:02d}",
                kind="interpolated_fps",
                pose_frame_id=int(start_frame),
                t_cw_by_time=tuple(poses),
                pose_frame_ids_by_time=tuple(
                    int(round((1.0 - frame_alpha(idx, num_frames)) * start_frame + frame_alpha(idx, num_frames) * end_frame))
                    for idx in range(num_frames)
                ),
            )
        )

    metadata = {
        "anchor_frame_ids": [frame_id for frame_id, _ in anchors],
        "anchor_positions": [pos for _, pos in anchors],
        "num_anchors": int(len(anchors)),
        "num_interpolated_trajectories": int(len(timelines)),
        "interpolation": "smoothstep_translation_slerp_rotation",
        "target_camera_names": [timeline.name for timeline in timelines],
    }
    return timelines, [frame_id for frame_id, _ in anchors], metadata


def resolve_external_camera_range(
    token: str,
    *,
    total_poses: int,
    num_frames: int,
    stride: int,
) -> list[int]:
    if stride <= 0:
        raise ValueError("--external-camera-stride must be positive.")
    if num_frames <= 0:
        raise ValueError("Cannot build external camera ranges without rendered frames.")
    if total_poses <= 0:
        raise ValueError("Cannot sample external camera ranges from an empty trajectory.")

    normalized = token.strip().lower()
    if normalized == "last":
        end = total_poses - 1
        start = end - (num_frames - 1) * stride
    elif "-" in normalized:
        start_text, end_text = normalized.split("-", 1)
        start = int(start_text)
        end = int(end_text)
    else:
        start = int(normalized)
        end = start + (num_frames - 1) * stride

    if start < 0 or start >= total_poses or end < 0 or end >= total_poses:
        raise ValueError(
            f"External camera range '{token}' is outside valid pose indices 0..{total_poses - 1}."
        )
    step = stride if end >= start else -stride
    stop = end + (1 if step > 0 else -1)
    frame_ids = list(range(start, stop, step))
    if len(frame_ids) != num_frames:
        raise ValueError(
            f"External camera range '{token}' with stride {stride} yields {len(frame_ids)} poses, "
            f"but rendering selected {num_frames} target timesteps."
        )
    return frame_ids


def build_external_trajectory_timelines(
    *,
    trajectory_path: Path,
    range_tokens: Sequence[str],
    camera_names: Sequence[str] | None,
    stride: int,
    num_frames: int,
) -> tuple[list[CameraTimeline], list[int], dict]:
    timestamps, c2w_by_frame = load_tum_c2w_trajectory(trajectory_path)
    resolved_ranges = [
        resolve_external_camera_range(
            token,
            total_poses=int(c2w_by_frame.shape[0]),
            num_frames=num_frames,
            stride=stride,
        )
        for token in range_tokens
    ]
    names = (
        [f"cam_mocap_{idx:02d}" for idx in range(len(resolved_ranges))]
        if camera_names is None
        else [normalize_camera_name(name) for name in camera_names]
    )
    if len(names) != len(resolved_ranges):
        raise ValueError("External camera ranges and camera names must have the same length.")

    timelines: list[CameraTimeline] = []
    for name, frame_ids in zip(names, resolved_ranges):
        c2w = c2w_by_frame[np.asarray(frame_ids, dtype=np.int64)]
        t_cw = tuple(c2w_to_t_cw(pose) for pose in c2w)
        timelines.append(
            CameraTimeline(
                name=name,
                kind="external",
                pose_frame_id=int(frame_ids[0]),
                t_cw_by_time=t_cw,
                pose_frame_ids_by_time=tuple(int(frame_id) for frame_id in frame_ids),
            )
        )

    metadata = {
        "trajectory_path": str(trajectory_path),
        "num_source_poses": int(c2w_by_frame.shape[0]),
        "range_tokens": list(range_tokens),
        "resolved_frame_ids": [[int(frame_id) for frame_id in frame_ids] for frame_ids in resolved_ranges],
        "source_timestamps": [
            [float(timestamps[frame_id]) for frame_id in frame_ids] for frame_ids in resolved_ranges
        ],
        "stride": int(stride),
    }
    return timelines, [int(frame_ids[0]) for frame_ids in resolved_ranges], metadata


def build_external_npz_trajectory_timelines(
    *,
    npz_path: Path,
    camera_ids: Sequence[str] | None,
    camera_names: Sequence[str] | None,
    num_frames: int,
    source_t_cw: Sequence[np.ndarray],
    align_mode: str,
) -> tuple[list[CameraTimeline], list[int], dict]:
    with np.load(npz_path) as data:
        if camera_ids is None:
            if "camera_ids" in data.files:
                ids = [str(value) for value in data["camera_ids"].tolist()]
            else:
                ids = sorted(key[:-4] for key in data.files if key.endswith("_c2w"))
        else:
            ids = [str(value) for value in camera_ids]
        raw_sequences = []
        for camera_id in ids:
            key = f"{camera_id}_c2w"
            if key not in data.files:
                raise KeyError(f"Missing {key} in {npz_path}")
            raw_sequences.append(resample_c2w_sequence(data[key].astype(np.float32), num_frames))

    aligned_sequences, align_metadata = align_external_c2w_sequences(
        external_sequences=raw_sequences,
        source_t_cw=source_t_cw,
        mode=align_mode,
    )
    names = (
        [f"cam_f18_{idx:02d}" for idx in range(len(aligned_sequences))]
        if camera_names is None
        else [normalize_camera_name(name) for name in camera_names]
    )
    if len(names) != len(aligned_sequences):
        raise ValueError("External npz camera ids and camera names must have the same length.")

    timelines: list[CameraTimeline] = []
    for name, camera_id, c2w in zip(names, ids, aligned_sequences):
        timelines.append(
            CameraTimeline(
                name=name,
                kind="external_npz",
                pose_frame_id=0,
                t_cw_by_time=tuple(c2w_to_t_cw(pose) for pose in c2w),
                pose_frame_ids_by_time=tuple(range(num_frames)),
            )
        )

    metadata = {
        "npz_path": str(npz_path),
        "camera_ids": ids,
        "target_camera_names": names,
        "num_frames": int(num_frames),
        "alignment": align_metadata,
    }
    return timelines, [0 for _ in timelines], metadata


def compute_alpha_coverage(
    scene_root: Path,
    camera_names: Sequence[str],
    frame_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for camera_name in camera_names:
        coverages = []
        for frame_name in frame_names:
            alpha_path = scene_root / "alpha" / camera_name / f"{frame_name}.png"
            if not alpha_path.is_file():
                continue
            alpha = imageio.imread(alpha_path)
            if alpha.ndim == 3:
                alpha = alpha[..., 0]
            coverages.append(float(np.mean(alpha > 0)))
        if coverages:
            arr = np.asarray(coverages, dtype=np.float32)
            stats[camera_name] = {
                "num_frames": int(len(coverages)),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            stats[camera_name] = {
                "num_frames": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
    return stats


def main() -> None:
    args = parse_args()

    ws = Path(args.ws).expanduser().resolve()
    saved_dir = Path(args.saved_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    scene_name = args.scene_name or ws.name
    scene_root = output_root / scene_name
    outputs = normalize_outputs(args.outputs)
    if args.flow_step <= 0:
        raise ValueError("--flow-step must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    using_external_trajectory = (
        args.external_camera_trajectory is not None
        or args.external_camera_ranges is not None
        or args.external_camera_npz is not None
        or args.external_camera_ids is not None
    )
    using_scripted = args.scripted_camera_npz is not None
    if using_scripted and (
        using_external_trajectory
        or args.interpolate_fps_anchors
        or args.lyra_trajectory_modes
        or args.photometric_dolly
        or args.fixed_frame_preset is not None
        or args.fixed_frame_ids
        or args.append_reverse_trajectory
    ):
        raise ValueError(
            "--scripted-camera-npz cannot be combined with external/lyra/interpolated-fps/"
            "photometric-dolly/fixed-frame/reverse trajectory options."
        )
    if args.external_camera_stride <= 0:
        raise ValueError("--external-camera-stride must be positive.")
    if using_external_trajectory:
        using_tum_external = (
            args.external_camera_trajectory is not None or args.external_camera_ranges is not None
        )
        using_npz_external = args.external_camera_npz is not None or args.external_camera_ids is not None
        if using_tum_external and using_npz_external:
            raise ValueError("Use either TUM external trajectory or external npz, not both.")
        if using_tum_external and (
            args.external_camera_trajectory is None or args.external_camera_ranges is None
        ):
            raise ValueError(
                "Use --external-camera-trajectory and --external-camera-ranges together."
            )
        if using_npz_external and args.external_camera_npz is None:
            raise ValueError("Use --external-camera-npz when passing --external-camera-ids.")
        if (
            args.lyra_trajectory_modes
            or args.interpolate_fps_anchors
            or args.photometric_dolly
            or args.fixed_frame_preset is not None
            or args.fixed_frame_ids
        ):
            raise ValueError(
                "External camera trajectories cannot be combined with Lyra, photometric dolly, interpolated FPS, or fixed-frame cameras."
            )
    if args.interpolate_fps_anchors and (
        args.lyra_trajectory_modes
        or args.photometric_dolly
        or args.fixed_frame_preset is not None
        or args.fixed_frame_ids
    ):
        raise ValueError(
            "--interpolate-fps-anchors cannot be combined with Lyra, photometric dolly, or fixed-frame cameras."
        )
    if args.photometric_dolly and (
        args.lyra_trajectory_modes or args.fixed_frame_preset is not None or args.fixed_frame_ids
    ):
        raise ValueError("--photometric-dolly cannot be combined with Lyra or fixed-frame cameras.")
    if args.append_reverse_trajectory and (
        using_external_trajectory
        or args.interpolate_fps_anchors
        or args.lyra_trajectory_modes
        or args.photometric_dolly
    ):
        raise ValueError(
            "--append-reverse-trajectory is currently supported with fixed-frame/FPS cameras only."
        )

    if args.dry_run:
        image_paths = list_images(ws / "images")
        total_frames = infer_num_frames_from_logs(saved_dir) or len(image_paths)
        selected_indices = list(range(0, len(image_paths), int(args.frame_stride)))
        if args.max_frames is not None:
            if args.max_frames <= 0:
                raise ValueError("--max-frames must be positive.")
            selected_indices = selected_indices[: int(args.max_frames)]
        print(f"Dry run scene: {scene_root}", flush=True)
        print(f"Input frames: {len(image_paths)}", flush=True)
        print(f"Frame stride: {args.frame_stride}", flush=True)
        print(f"Rendered frames: {len(selected_indices)}", flush=True)
        print(
            "Rendered timestep range: "
            f"{selected_indices[0] if selected_indices else 'none'}.."
            f"{selected_indices[-1] if selected_indices else 'none'}",
            flush=True,
        )
        print(f"Solved trajectory frames: {total_frames}", flush=True)
        print(f"Source camera: {SOURCE_CAMERA}", flush=True)
        if using_scripted:
            data = np.load(Path(args.scripted_camera_npz).expanduser().resolve())
            c2w_script = np.asarray(data["c2w"])
            sched = [int(x) for x in np.asarray(data["time_ids"]).reshape(-1)]
            print(f"Scripted camera: {normalize_camera_name(args.scripted_camera_name)}", flush=True)
            print(f"Scripted npz: {args.scripted_camera_npz}", flush=True)
            print(f"Scripted frames: {c2w_script.shape[0]} poses, {len(sched)} time ids", flush=True)
            print(f"Scripted scene-time schedule: {sched[0]}..{sched[-1]} (first {min(6,len(sched))}: {sched[:6]})", flush=True)
        elif using_external_trajectory:
            if args.external_camera_npz is not None:
                print(f"External npz: {args.external_camera_npz}", flush=True)
                print(f"External camera ids: {args.external_camera_ids or 'all'}", flush=True)
                print(f"External align mode: {args.external_camera_align_mode}", flush=True)
                target_timelines = []
                external_summary = {
                    "npz_path": str(args.external_camera_npz),
                    "camera_ids": args.external_camera_ids,
                    "align_mode": args.external_camera_align_mode,
                }
            else:
                target_timelines, _, external_summary = build_external_trajectory_timelines(
                    trajectory_path=Path(args.external_camera_trajectory).expanduser().resolve(),
                    range_tokens=args.external_camera_ranges,
                    camera_names=args.camera_names,
                    stride=int(args.external_camera_stride),
                    num_frames=len(selected_indices),
                )
                print(f"External ranges: {external_summary['range_tokens']}", flush=True)
                print(
                    "Resolved external endpoints: "
                    f"{[(ids[0], ids[-1]) for ids in external_summary['resolved_frame_ids']]}",
                    flush=True,
                )
            print(
                f"External trajectory cameras: {[timeline.name for timeline in target_timelines]}",
                flush=True,
            )
        elif args.interpolate_fps_anchors:
            approx_count = max(int(args.num_fps_cameras) - 1, 0)
            print(f"Interpolated FPS anchor trajectories: {approx_count}", flush=True)
            print("Anchor frame IDs require loading checkpoint camera poses.", flush=True)
        elif args.lyra_trajectory_modes:
            approx_count = int(args.num_fps_cameras) * len(args.lyra_trajectory_modes)
            print(f"Lyra trajectory modes: {args.lyra_trajectory_modes}", flush=True)
            print(f"Approx target cameras: {approx_count}", flush=True)
            print("Anchor frame IDs require loading checkpoint camera poses.", flush=True)
        elif args.photometric_dolly:
            camera_name = resolve_photometric_dolly_camera_name(args.camera_names)
            focus_frame_id = (
                max(total_frames - 1, 0) // 2
                if args.photometric_dolly_focus_frame_id is None
                else int(args.photometric_dolly_focus_frame_id)
            )
            print(f"Photometric dolly camera: {camera_name}", flush=True)
            print(f"Photometric dolly focus frame: {focus_frame_id}", flush=True)
            print(f"Photometric dolly angle deg: {args.photometric_dolly_angle_deg}", flush=True)
            print("Focus depth requires loading the checkpoint.", flush=True)
        elif args.fixed_frame_preset == "camera_fps":
            print(f"Fixed cameras: {args.num_fps_cameras} camera_fps anchors", flush=True)
            print("Anchor frame IDs require loading checkpoint camera poses.", flush=True)
        else:
            fixed_frame_ids, target_cameras = resolve_fixed_cameras(args, total_frames=total_frames)
            print(f"Fixed cameras: {dict(zip(target_cameras, fixed_frame_ids))}", flush=True)
        if args.append_reverse_trajectory:
            print(
                f"Reverse source trajectory camera: {normalize_camera_name(args.reverse_camera_name)}",
                flush=True,
            )
        print(f"Outputs: {sorted(outputs)}", flush=True)
        return

    from mosca_viz import load_model_cfg

    cfg_path = infer_cfg_path(str(saved_dir), args.cfg)
    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg, d_model, s_model, cams = load_model_cfg(cfg_path, str(saved_dir), device=device)

    all_image_paths = list_images(ws / "images")
    if len(all_image_paths) > cams.T:
        raise ValueError(
            f"Workspace has {len(all_image_paths)} images, but the solved camera track has T={cams.T}."
        )
    if len(all_image_paths) < cams.T and args.max_frames is None and not using_scripted:
        raise ValueError(
            f"Workspace has {len(all_image_paths)} images, but the solved camera track has T={cams.T}. "
            "Use --max-frames only for prefix smoke tests."
        )
    scripted_c2w = None
    if using_scripted:
        data = np.load(Path(args.scripted_camera_npz).expanduser().resolve())
        scripted_c2w = np.asarray(data["c2w"], dtype=np.float32)
        schedule = [int(x) for x in np.asarray(data["time_ids"]).reshape(-1)]
        if scripted_c2w.ndim != 3 or scripted_c2w.shape[1:] != (4, 4):
            raise ValueError(f"Scripted c2w must be [N,4,4], got {scripted_c2w.shape}.")
        if scripted_c2w.shape[0] != len(schedule):
            raise ValueError(
                f"Scripted c2w ({scripted_c2w.shape[0]}) and time_ids ({len(schedule)}) length mismatch."
            )
        for t in schedule:
            if t < 0 or t >= int(cams.T) or t >= len(all_image_paths):
                raise ValueError(
                    f"Scripted time id {t} out of range (cams.T={int(cams.T)}, images={len(all_image_paths)})."
                )
        # Frame names are sequential render order (unique even when scene times repeat
        # or reverse); time ids carry the per-frame scene time for both cam0 and target.
        time_ids = schedule
        image_paths = [all_image_paths[t] for t in schedule]
        frame_names = [f"{i:05d}" for i in range(len(schedule))]
        num_frames = len(schedule)
    else:
        selected_indices = list(range(0, len(all_image_paths), int(args.frame_stride)))
        if args.max_frames is not None:
            if args.max_frames <= 0:
                raise ValueError("--max-frames must be positive.")
            selected_indices = selected_indices[: int(args.max_frames)]
        if not selected_indices:
            raise ValueError("No frames selected for rendering.")
        image_paths = [all_image_paths[idx] for idx in selected_indices]
        time_ids = [int(idx) for idx in selected_indices]
        num_frames = len(image_paths)
        frame_names = [path.stem for path in image_paths]

    k_raw = cams.default_K.detach().cpu().numpy().astype(np.float32)
    img_wh = (int(cams.default_W.item()), int(cams.default_H.item()))
    source_t_cw = tuple(
        cams.T_cw(t).detach().cpu().numpy().astype(np.float32) for t in time_ids
    )
    lyra_summary = {}
    external_summary = {}
    interp_summary = {}
    dolly_summary = {}
    scripted_summary = {}
    reverse_summary = {}
    reverse_camera_name = None
    if using_scripted:
        scripted_camera_name = normalize_camera_name(args.scripted_camera_name)
        target_t_cw = tuple(
            c2w_to_t_cw(scripted_c2w[i]).astype(np.float32) for i in range(num_frames)
        )
        target_timelines = [
            CameraTimeline(
                name=scripted_camera_name,
                kind="scripted",
                pose_frame_id=int(time_ids[0]),
                t_cw_by_time=target_t_cw,
                pose_frame_ids_by_time=tuple(int(t) for t in time_ids),
                scene_time_id_by_frame=tuple(int(t) for t in time_ids),
            )
        ]
        fixed_frame_ids = [int(time_ids[0])]
        target_cameras = [scripted_camera_name]
        scripted_summary = {
            "camera_name": scripted_camera_name,
            "npz_path": str(args.scripted_camera_npz),
            "num_frames": num_frames,
            "scene_time_ids": [int(t) for t in time_ids],
            "trajectory": "scripted_pose_time_tour",
        }
    elif using_external_trajectory:
        if args.external_camera_npz is not None:
            target_timelines, fixed_frame_ids, external_summary = build_external_npz_trajectory_timelines(
                npz_path=Path(args.external_camera_npz).expanduser().resolve(),
                camera_ids=args.external_camera_ids,
                camera_names=args.camera_names,
                num_frames=num_frames,
                source_t_cw=source_t_cw,
                align_mode=args.external_camera_align_mode,
            )
        else:
            target_timelines, fixed_frame_ids, external_summary = build_external_trajectory_timelines(
                trajectory_path=Path(args.external_camera_trajectory).expanduser().resolve(),
                range_tokens=args.external_camera_ranges,
                camera_names=args.camera_names,
                stride=int(args.external_camera_stride),
                num_frames=num_frames,
            )
        target_cameras = [timeline.name for timeline in target_timelines]
    elif args.interpolate_fps_anchors:
        target_timelines, fixed_frame_ids, interp_summary = build_interpolated_fps_anchor_timelines(
            source_t_cw=source_t_cw,
            time_ids=time_ids,
            num_anchors=int(args.num_fps_cameras),
            num_frames=num_frames,
        )
        target_cameras = [timeline.name for timeline in target_timelines]
    elif args.lyra_trajectory_modes:
        target_timelines, fixed_frame_ids, lyra_summary = build_lyra_trajectory_timelines(
            modes=list(dict.fromkeys(args.lyra_trajectory_modes)),
            source_t_cw=source_t_cw,
            time_ids=time_ids,
            num_anchors=int(args.num_fps_cameras),
            num_frames=num_frames,
            movement_distance=float(args.lyra_movement_distance),
            rotation_mode=args.lyra_camera_rotation,
            num_circles=float(args.lyra_num_circles),
            radius_x_factor=float(args.lyra_radius_x_factor),
            radius_y_factor=float(args.lyra_radius_y_factor),
        )
        target_cameras = [timeline.name for timeline in target_timelines]
    elif args.photometric_dolly:
        focus_frame_id = (
            (int(cams.T) - 1) // 2
            if args.photometric_dolly_focus_frame_id is None
            else int(args.photometric_dolly_focus_frame_id)
        )
        if focus_frame_id < 0 or focus_frame_id >= cams.T:
            raise ValueError(
                f"Photometric dolly focus frame id {focus_frame_id} is outside valid range 0..{cams.T - 1}."
            )
        camera_name = resolve_photometric_dolly_camera_name(args.camera_names)
        # The orbit anchor pose defaults to the recon camera at the focus frame, but may
        # be taken from an external 1:1-aligned trajectory (e.g. new_traj.pth).
        if args.photometric_dolly_base_cam_ckpt is None:
            base_t_cw = cams.T_cw(focus_frame_id).detach().cpu().numpy().astype(np.float32)
            base_cam_source = "recon_camera"
        else:
            from lib_moca.camera import MonocularCameras

            base_ckpt_path = Path(args.photometric_dolly_base_cam_ckpt).expanduser().resolve()
            if not base_ckpt_path.is_file():
                raise FileNotFoundError(
                    f"--photometric-dolly-base-cam-ckpt not found: {base_ckpt_path}"
                )
            base_cams = MonocularCameras.load_from_ckpt(
                torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
            ).to(device)
            if focus_frame_id >= int(base_cams.T):
                raise ValueError(
                    f"Focus frame {focus_frame_id} is outside the base camera track "
                    f"(T={int(base_cams.T)}) from {base_ckpt_path}."
                )
            if int(base_cams.T) != int(cams.T):
                print(
                    f"WARNING: base anchor camera track (T={int(base_cams.T)}) is not 1:1 "
                    f"with the recon (T={int(cams.T)}); the anchor pose and frozen scene "
                    "time may not refer to the same moment.",
                    flush=True,
                )
            base_t_cw = (
                base_cams.T_cw(focus_frame_id).detach().cpu().numpy().astype(np.float32)
            )
            base_cam_source = str(base_ckpt_path)
            print(
                f"Photometric dolly anchor from {base_ckpt_path} (frame {focus_frame_id})",
                flush=True,
            )
        focus_depth = estimate_center_depth_for_photometric_dolly(
            s_model=s_model,
            d_model=d_model,
            cams=cams,
            focus_frame_id=focus_frame_id,
            base_t_cw=base_t_cw,
        )
        dolly_timeline, dolly_summary = build_photometric_dolly_timeline(
            camera_name=camera_name,
            base_t_cw=base_t_cw,
            focus_depth=focus_depth,
            focus_frame_id=focus_frame_id,
            num_frames=num_frames,
            angle_deg=float(args.photometric_dolly_angle_deg),
            fix_scene_time=bool(args.photometric_dolly_fix_time),
        )
        dolly_summary["base_cam_source"] = base_cam_source
        target_timelines = [dolly_timeline]
        fixed_frame_ids = [focus_frame_id]
        target_cameras = [camera_name]
    else:
        fixed_frame_ids, target_cameras = resolve_fixed_cameras(
            args,
            total_frames=int(cams.T),
            source_t_cw=source_t_cw,
            source_time_ids=time_ids,
        )
        if args.append_reverse_trajectory:
            reverse_camera_name = normalize_camera_name(args.reverse_camera_name)
            target_cameras = [*target_cameras, reverse_camera_name]
            fixed_frame_ids = [*fixed_frame_ids, int(time_ids[-1])]
            reverse_summary = {
                "camera_name": reverse_camera_name,
                "trajectory": "reverse_input_trajectory",
                "source_time_ids": [int(value) for value in reversed(time_ids)],
            }
        target_timelines = []
    if len(set(target_cameras)) != len(target_cameras):
        raise ValueError(f"Duplicate target camera names: {target_cameras}")
    if SOURCE_CAMERA in target_cameras:
        raise ValueError(f"Target cameras must not include the source camera {SOURCE_CAMERA}.")

    if not using_external_trajectory:
        for frame_id in fixed_frame_ids:
            if frame_id < 0 or frame_id >= cams.T:
                raise ValueError(f"Fixed frame id {frame_id} is outside valid range 0..{cams.T - 1}.")
    # By default the source/reference (cam0) plays the moving input clip; with
    # --source-static-frame-id it is frozen to a single frame (still gt image,
    # fixed pose, frozen scene time) so the reference matches a bullet-time moment.
    source_image_paths = image_paths
    if args.source_static_frame_id is None:
        source_timeline = CameraTimeline(
            name=SOURCE_CAMERA,
            kind="source",
            pose_frame_id=None,
            t_cw_by_time=source_t_cw,
        )
    else:
        source_static_frame_id = int(args.source_static_frame_id)
        if (
            source_static_frame_id < 0
            or source_static_frame_id >= int(cams.T)
            or source_static_frame_id >= len(all_image_paths)
        ):
            raise ValueError(
                f"--source-static-frame-id {source_static_frame_id} is outside valid range "
                f"0..{min(int(cams.T), len(all_image_paths)) - 1}."
            )
        static_pose = (
            cams.T_cw(source_static_frame_id).detach().cpu().numpy().astype(np.float32)
        )
        source_timeline = CameraTimeline(
            name=SOURCE_CAMERA,
            kind="source",
            pose_frame_id=source_static_frame_id,
            t_cw_by_time=tuple(static_pose.copy() for _ in range(num_frames)),
            fixed_scene_time_id=source_static_frame_id,
        )
        source_image_paths = [all_image_paths[source_static_frame_id]] * num_frames
        print(
            f"Static reference: cam0 frozen at frame {source_static_frame_id}",
            flush=True,
        )
    if not (
        args.lyra_trajectory_modes
        or args.interpolate_fps_anchors
        or using_external_trajectory
        or args.photometric_dolly
        or using_scripted
    ):
        fixed_pairs = list(zip(fixed_frame_ids, target_cameras))
        if reverse_camera_name is not None:
            fixed_pairs = fixed_pairs[:-1]
        for frame_id, target_camera in fixed_pairs:
            fixed_pose = cams.T_cw(frame_id).detach().cpu().numpy().astype(np.float32)
            target_timelines.append(
                CameraTimeline(
                    name=target_camera,
                    kind="fixed",
                    pose_frame_id=int(frame_id),
                    t_cw_by_time=tuple(fixed_pose.copy() for _ in range(num_frames)),
                )
            )
        if reverse_camera_name is not None:
            target_timelines.append(
                CameraTimeline(
                    name=reverse_camera_name,
                    kind="reverse_source",
                    pose_frame_id=int(time_ids[-1]),
                    t_cw_by_time=tuple(pose.copy() for pose in reversed(source_t_cw)),
                    pose_frame_ids_by_time=tuple(int(value) for value in reversed(time_ids)),
                )
            )
    timelines = [*target_timelines, source_timeline]

    prepare_scene_root(scene_root, overwrite=bool(args.overwrite))
    camera_names = [timeline.name for timeline in timelines]
    ensure_modality_dirs(scene_root, camera_names, outputs)
    write_eval_units(scene_root, SOURCE_CAMERA, target_cameras)

    print(
        f"Rendering {num_frames} frames for {len(timelines)} cameras to {scene_root}",
        flush=True,
    )
    print(
        f"Target cameras: {dict(zip(target_cameras, fixed_frame_ids))}",
        flush=True,
    )
    for timeline in timelines:
        print(f"Rendering {timeline.name} ({timeline.kind})", flush=True)
        timeline_image_paths = (
            source_image_paths if timeline.kind == "source" else image_paths
        )
        records, scene_flow_rgb_hi = render_camera_timeline(
            timeline=timeline,
            frame_names=frame_names,
            time_ids=time_ids,
            max_time_id=int(cams.T) - 1,
            image_paths=timeline_image_paths,
            scene_root=scene_root,
            outputs=outputs,
            k_raw=k_raw,
            img_wh=img_wh,
            saved_dir=saved_dir,
            s_model=s_model,
            d_model=d_model,
            flow_step=int(args.flow_step),
            scene_space=args.scene_space,
            bg_color=[float(value) for value in args.bg_color],
            alpha_threshold=float(args.alpha_threshold),
            device=device,
        )
        write_camera_c2w_npz(timeline.t_cw_by_time, scene_root / "cameras" / f"{timeline.name}_c2w.npz")
        write_camera_manifest(
            scene_root=scene_root,
            camera_name=timeline.name,
            reference_camera=SOURCE_CAMERA,
            records=records,
            scene_flow_rgb_hi=scene_flow_rgb_hi,
        )

    alpha_coverage = compute_alpha_coverage(scene_root, camera_names, frame_names)
    write_json(scene_root / "metadata" / "alpha_coverage.json", alpha_coverage)

    summary = {
        "scene_name": scene_name,
        "workspace": str(ws),
        "saved_dir": str(saved_dir),
        "cfg_path": str(cfg_path),
        "num_frames": num_frames,
        "frame_stride": int(args.frame_stride),
        "time_ids": time_ids,
        "source_camera": SOURCE_CAMERA,
        "source_static_frame_id": (
            None if args.source_static_frame_id is None else int(args.source_static_frame_id)
        ),
        "target_cameras": target_cameras,
        "fixed_frame_ids": [int(value) for value in fixed_frame_ids],
        "interpolated_fps_trajectory_summary": interp_summary,
        "lyra_trajectory_summary": lyra_summary,
        "photometric_dolly_trajectory_summary": dolly_summary,
        "scripted_trajectory_summary": scripted_summary,
        "reverse_trajectory_summary": reverse_summary,
        "external_trajectory_summary": external_summary,
        "alpha_coverage": alpha_coverage,
        "img_wh": [int(img_wh[0]), int(img_wh[1])],
        "outputs": sorted(outputs),
    }
    write_json(scene_root / "metadata" / "scene_summary.json", summary)

    if not args.no_validate:
        required_optional = [
            modality
            for modality in ("scene_coords", "scene_flow", "scene_flow_rgb")
            if modality in outputs
        ]
        layout = validate_scene_layout(
            scene_root,
            strict_files=True,
            required_optional_modalities=required_optional,
        )
        print(
            f"Validated scene {layout.scene_name}: cameras={list(layout.camera_names)} "
            f"eval_units={list(layout.evaluation_unit_ids)}",
            flush=True,
        )

    low_coverage = [
        name for name, stats in alpha_coverage.items() if name != SOURCE_CAMERA and stats["mean"] < 0.1
    ]
    if low_coverage:
        print(f"Warning: low mean alpha coverage cameras: {low_coverage}", flush=True)
    print(f"Saved benchmark scene under {scene_root}", flush=True)
    print("Preview tip: generate RGB grids from the exported rgb/<camera> folders before DiffSynth.", flush=True)


if __name__ == "__main__":
    main()
