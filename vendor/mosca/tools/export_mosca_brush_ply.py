#!/usr/bin/env python3
"""Export MoSca dynamic 3DGS checkpoints as CAT4D/Brush delta-frame PLY."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ATTRS = ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2")
BASE_FLOAT_COUNT = 59
CAT4D_REST_COUNT = 45
UINT16_MAX = 65535.0


@dataclass
class CheckpointInfo:
    frames: int
    static_splats: int
    dynamic_splats: int
    total_splats: int
    static_sh_rest: int
    dynamic_sh_rest: int
    height: int
    width: int
    estimated_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MoSca photometric checkpoints to CAT4D/Brush dynamic PLY."
    )
    parser.add_argument("--saved-dir", required=True, type=Path)
    parser.add_argument("--backend", default="native_add3")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--ply-name",
        default="indoor_scene_with_interaction_06_deform3dgs_full.ply",
    )
    parser.add_argument("--include-static", action="store_true")
    parser.add_argument("--dynamic-only", action="store_true")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Export only the static Gaussian model as a one-frame Brush-compatible PLY.",
    )
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--frame-sampling",
        choices=("prefix", "linspace"),
        default="prefix",
        help="How to choose frames when --max-frames limits the export.",
    )
    parser.add_argument(
        "--sh-rest-mode",
        choices=("raw", "zero", "channel-major"),
        default="zero",
        help="How to write f_rest_* SH coefficients in the Brush PLY. "
        "'zero' exports RGB/DC color only with all higher-degree SH set to zero.",
    )
    parser.add_argument("--opacity-threshold", type=float, default=None)
    parser.add_argument("--initial-camera-frame", default="middle")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-samples", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def checkpoint_paths(saved_dir: Path, backend: str) -> dict[str, Path]:
    backend = backend.lower()
    return {
        "camera": saved_dir / "photometric_cam.pth",
        "static": saved_dir / f"photometric_s_model_{backend}.pth",
        "dynamic": saved_dir / f"photometric_d_model_{backend}.pth",
    }


def load_torch(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location=map_location)


def expected_size_bytes(total_splats: int, frames: int, header_bytes: int = 0) -> int:
    return int(header_bytes + total_splats * 236 + max(frames - 1, 0) * (80 + total_splats * 20))


def inspect_checkpoints(
    saved_dir: Path,
    backend: str,
    include_static: bool,
    include_dynamic: bool,
    frame_indices: list[int],
) -> CheckpointInfo:
    paths = checkpoint_paths(saved_dir, backend)
    cam = load_torch(paths["camera"], "cpu")
    d_ckpt = load_torch(paths["dynamic"], "cpu") if include_dynamic else None
    s_ckpt = load_torch(paths["static"], "cpu") if include_static else None

    frames_total = int(cam["q_wc"].shape[0])
    if d_ckpt is not None and int(d_ckpt["scf._node_xyz"].shape[0]) != frames_total:
        raise ValueError(
            f"Camera T={frames_total} differs from dynamic model T={d_ckpt['scf._node_xyz'].shape[0]}"
        )
    if frame_indices[-1] >= frames_total:
        raise ValueError(f"Requested frame {frame_indices[-1]} but checkpoint has {frames_total} frames")

    static_splats = int(s_ckpt["_xyz"].shape[0]) if s_ckpt is not None else 0
    dynamic_splats = int(d_ckpt["_xyz"].shape[0]) if d_ckpt is not None else 0
    total_splats = static_splats + dynamic_splats
    return CheckpointInfo(
        frames=len(frame_indices),
        static_splats=static_splats,
        dynamic_splats=dynamic_splats,
        total_splats=total_splats,
        static_sh_rest=int(s_ckpt["_features_rest"].shape[1]) if s_ckpt is not None else 0,
        dynamic_sh_rest=int(d_ckpt["_features_rest"].shape[1]) if d_ckpt is not None else 0,
        height=int(torch.as_tensor(cam["default_H"]).item()),
        width=int(torch.as_tensor(cam["default_W"]).item()),
        estimated_bytes=expected_size_bytes(total_splats, len(frame_indices)),
    )


def make_frame_indices(
    total_frames: int,
    frame_stride: int,
    max_frames: int | None,
    all_frames: bool,
    frame_sampling: str = "prefix",
) -> list[int]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    indices = list(range(0, total_frames, frame_stride))
    if not all_frames and max_frames is None:
        max_frames = min(len(indices), 64)
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames < 1:
            raise ValueError("--max-frames must be >= 1")
        if frame_sampling == "prefix":
            indices = indices[:max_frames]
        elif frame_sampling == "linspace":
            if max_frames < len(indices):
                sample_pos = np.linspace(0, len(indices) - 1, num=max_frames)
                indices = [indices[int(round(pos))] for pos in sample_pos]
        else:
            raise ValueError(f"Unknown frame sampling mode: {frame_sampling}")
    if not indices:
        raise ValueError("No frames selected")
    return indices


def make_header(total_splats: int, frames: int) -> str:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Vertical Axis: z",
        f"element vertex {total_splats}",
        "property float x",
        "property float y",
        "property float z",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    lines.extend(f"property float f_rest_{i}" for i in range(CAT4D_REST_COUNT))
    lines.append("property float opacity")

    for frame in range(1, frames):
        lines.append(f"element meta_delta_min_{frame} 1")
        lines.extend(f"property float {name}" for name in ATTRS)
        lines.append(f"element meta_delta_max_{frame} 1")
        lines.extend(f"property float {name}" for name in ATTRS)
        lines.append(f"element delta_vertex_{frame} {total_splats}")
        lines.extend(f"property ushort {name}" for name in ATTRS)

    lines.append("end_header")
    return "\n".join(lines) + "\n"


def pad_sh_rest(rest: torch.Tensor, mode: str = "raw") -> np.ndarray:
    rest_np = rest.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if rest_np.shape[1] > CAT4D_REST_COUNT:
        raise ValueError(f"Expected at most {CAT4D_REST_COUNT} SH rest channels, got {rest_np.shape[1]}")
    out = np.zeros((rest_np.shape[0], CAT4D_REST_COUNT), dtype=np.float32)
    if mode == "zero":
        return out
    if mode == "raw":
        out[:, : rest_np.shape[1]] = rest_np
    elif mode == "channel-major":
        if rest_np.shape[1] % 3 != 0:
            raise ValueError(f"Expected SH rest channels divisible by 3, got {rest_np.shape[1]}")
        coeff_count = rest_np.shape[1] // 3
        out[:, : rest_np.shape[1]] = rest_np.reshape(rest_np.shape[0], coeff_count, 3).transpose(0, 2, 1).reshape(
            rest_np.shape[0], rest_np.shape[1]
        )
    else:
        raise ValueError(f"Unknown SH rest mode: {mode}")
    return out


def matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    from pytorch3d.transforms import matrix_to_quaternion

    quat = matrix_to_quaternion(matrix.float())
    return torch.nn.functional.normalize(quat, dim=-1)


def same_hemisphere(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    sign = torch.where((current * reference).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    return current * sign


def tensor_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)


def log_scale_np(scale: torch.Tensor) -> np.ndarray:
    return tensor_np(torch.log(torch.clamp(scale.float(), min=1e-12)))


def build_base_rows(
    s_model: Any,
    d_model: Any | None,
    include_static: bool,
    include_dynamic: bool,
    sh_rest_mode: str,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor | None, int]:
    rows: list[np.ndarray] = []
    geom: list[np.ndarray] = []
    static_count = 0
    prev_dynamic_quat: torch.Tensor | None = None

    with torch.inference_mode():
        if include_static:
            s_xyz, s_frame, s_scale, _, _ = s_model.forward()
            s_quat = matrix_to_quat_wxyz(s_frame)
            s_scale_log = log_scale_np(s_scale)
            s_fdc = tensor_np(s_model._features_dc)
            s_frest = pad_sh_rest(s_model._features_rest, sh_rest_mode)
            s_opacity = tensor_np(s_model._opacity)
            s_geom = np.concatenate([tensor_np(s_xyz), tensor_np(s_quat), s_scale_log], axis=1)
            rows.append(np.concatenate([s_geom, s_fdc, s_frest, s_opacity], axis=1))
            geom.append(s_geom)
            static_count = int(s_geom.shape[0])

        if include_dynamic:
            if d_model is None:
                raise ValueError("Dynamic export requested but dynamic model was not loaded")
            d_xyz, d_frame, d_scale, _, _ = d_model.forward(0)
            d_quat = matrix_to_quat_wxyz(d_frame)
            d_scale_log = log_scale_np(d_scale)
            d_fdc = tensor_np(d_model._features_dc)
            d_frest = pad_sh_rest(d_model._features_rest, sh_rest_mode)
            d_opacity = tensor_np(d_model._opacity)
            d_geom = np.concatenate([tensor_np(d_xyz), tensor_np(d_quat), d_scale_log], axis=1)
            rows.append(np.concatenate([d_geom, d_fdc, d_frest, d_opacity], axis=1))
            geom.append(d_geom)
            prev_dynamic_quat = d_quat.detach()

    if not rows:
        raise ValueError("No splats selected for export")

    base_rows = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
    if base_rows.shape[1] != BASE_FLOAT_COUNT:
        raise ValueError(f"Expected {BASE_FLOAT_COUNT} base floats, got {base_rows.shape[1]}")
    base_geom = np.ascontiguousarray(np.concatenate(geom, axis=0), dtype=np.float32)
    return base_rows, base_geom, prev_dynamic_quat, static_count


def quantize_delta(
    dynamic_delta: np.ndarray,
    total_splats: int,
    static_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros((1, dynamic_delta.shape[1]), dtype=np.float32)
    mn = np.minimum(dynamic_delta.min(axis=0), zero[0]).astype(np.float32)
    mx = np.maximum(dynamic_delta.max(axis=0), zero[0]).astype(np.float32)
    span = (mx - mn).astype(np.float32)
    good = span > 1e-12

    q = np.zeros((total_splats, dynamic_delta.shape[1]), dtype=np.uint16)
    if np.any(good):
        q_zero_float = np.zeros(dynamic_delta.shape[1], dtype=np.float32)
        q_zero_float[good] = (0.0 - mn[good]) / span[good] * UINT16_MAX
        q_zero = np.clip(np.rint(q_zero_float), 0, UINT16_MAX).astype(np.uint16)
        if static_count > 0:
            q[:static_count, :] = q_zero[None, :]

        q_dyn_float = np.zeros_like(dynamic_delta, dtype=np.float32)
        q_dyn_float[:, good] = (dynamic_delta[:, good] - mn[good]) / span[good] * UINT16_MAX
        q[static_count:, :] = np.clip(np.rint(q_dyn_float), 0, UINT16_MAX).astype(np.uint16)
    return mn, mx, q, span, good


def decode_q(q: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return mn[None, :] + (q.astype(np.float32) / UINT16_MAX) * (mx - mn)[None, :]


def parse_initial_frame(initial_camera_frame: str, total_frames: int) -> int:
    if initial_camera_frame == "middle":
        return total_frames // 2
    frame = int(initial_camera_frame)
    if frame < 0 or frame >= total_frames:
        raise ValueError(f"Initial camera frame {frame} outside [0, {total_frames - 1}]")
    return frame


def normalize_np(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return (vec / norm).astype(np.float32)


def make_brush_camera(c2w: torch.Tensor, focus_target: torch.Tensor, fov_y_degrees: float) -> dict[str, Any]:
    c2w_np = tensor_np(c2w)
    center = c2w_np[:3, 3]
    target = tensor_np(focus_target).reshape(3)

    mosca_right = normalize_np(c2w_np[:3, 0])
    mosca_down = normalize_np(c2w_np[:3, 1])
    mosca_forward = normalize_np(c2w_np[:3, 2])
    pose_up = -mosca_down

    focus_vec = target - center
    focus_distance = float(np.linalg.norm(focus_vec))
    if focus_distance < 1e-8:
        focus_distance = 1.0
        focus_vec = mosca_forward
        target = center + focus_vec * focus_distance

    brush_forward = normalize_np(focus_vec)
    brush_backward = -brush_forward
    brush_up = pose_up - float(np.dot(pose_up, brush_forward)) * brush_forward
    if float(np.linalg.norm(brush_up)) < 1e-8:
        brush_up = mosca_right - float(np.dot(mosca_right, brush_forward)) * brush_forward
    if float(np.linalg.norm(brush_up)) < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(world_up, brush_forward))) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        brush_up = world_up - float(np.dot(world_up, brush_forward)) * brush_forward
    brush_up = normalize_np(brush_up)
    brush_right = normalize_np(np.cross(brush_up, brush_backward))
    if float(np.dot(brush_right, mosca_right)) < 0.0:
        brush_right = -brush_right
        brush_up = normalize_np(np.cross(brush_backward, brush_right))

    return {
        "position": center.tolist(),
        "right_world": brush_right.tolist(),
        "up_world": brush_up.tolist(),
        "backward_world": brush_backward.tolist(),
        "forward_world": brush_forward.tolist(),
        "focus_target_world": target.tolist(),
        "focus_distance": focus_distance,
        "fov_y_degrees": float(fov_y_degrees),
        "source_convention": "Derived from MoSca/OpenCV c2w with right=c2w[:,0], down=c2w[:,1], forward=c2w[:,2]",
    }


def dynamic_focus_target(d_model: Any, frame_index: int) -> torch.Tensor:
    with torch.inference_mode():
        d_xyz, _, _, d_opacity, _ = d_model.forward(int(frame_index))
        if len(d_xyz) == 0:
            raise ValueError("Cannot derive focus target from an empty dynamic model")
        opacity = d_opacity.reshape(-1)
        if opacity.numel() == len(d_xyz) and bool((opacity > 0.0).any().item()):
            visible = d_xyz[opacity > torch.quantile(opacity, 0.25)]
            if len(visible) > 0:
                d_xyz = visible
        return torch.median(d_xyz.float(), dim=0).values


def static_focus_target(s_model: Any) -> torch.Tensor:
    with torch.inference_mode():
        s_xyz, _, _, s_opacity, _ = s_model.forward()
        if len(s_xyz) == 0:
            raise ValueError("Cannot derive focus target from an empty static model")
        opacity = s_opacity.reshape(-1)
        if opacity.numel() == len(s_xyz) and bool((opacity > 0.0).any().item()):
            visible = s_xyz[opacity > torch.quantile(opacity, 0.25)]
            if len(visible) > 0:
                s_xyz = visible
        return torch.median(s_xyz.float(), dim=0).values


def camera_to_json(cams: Any, frame_index: int, focus_target: torch.Tensor | None = None) -> dict[str, Any]:
    from pytorch3d.transforms import matrix_to_quaternion

    with torch.inference_mode():
        c2w_all = cams.T_wc_list().float()
        c2w = c2w_all[frame_index]
        quat = torch.nn.functional.normalize(matrix_to_quaternion(c2w[:3, :3][None])[0], dim=-1)
        k = cams.default_K.float()
        width = int(cams.default_W.item())
        height = int(cams.default_H.item())
        fx = float(k[0, 0].item())
        fy = float(k[1, 1].item())
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    fov_y = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    ret = {
        "frame_index": int(frame_index),
        "convention": "MoSca camera-to-world, PyTorch3D quaternion wxyz",
        "c2w": tensor_np(c2w).tolist(),
        "w2c": tensor_np(torch.linalg.inv(c2w)).tolist(),
        "wxyz": tensor_np(quat).tolist(),
        "position": tensor_np(c2w[:3, 3]).tolist(),
        "K": tensor_np(k).tolist(),
        "img_wh": [width, height],
        "fov_degrees": [fov_x, fov_y],
    }
    if focus_target is not None:
        brush_camera = make_brush_camera(c2w, focus_target, fov_y)
        ret.update(
            {
                "camera_center_world": brush_camera["position"],
                "look_direction_world": brush_camera["forward_world"],
                "up_direction_world": brush_camera["up_world"],
                "focus_target_world": brush_camera["focus_target_world"],
                "focus_distance": brush_camera["focus_distance"],
                "fov_y_degrees": brush_camera["fov_y_degrees"],
                "brush_camera": brush_camera,
            }
        )
    return ret


def load_scene(saved_dir: Path, backend: str, device: torch.device, load_dynamic: bool = True) -> tuple[Any, Any, Any | None]:
    from lib_moca.camera import MonocularCameras
    from lib_mosca.static_gs import StaticGaussian

    paths = checkpoint_paths(saved_dir, backend)
    cams = MonocularCameras.load_from_ckpt(load_torch(paths["camera"], device)).to(device).eval()
    s_model = StaticGaussian.load_from_ckpt(load_torch(paths["static"], device), device=device).to(device).eval()
    d_model = None
    if load_dynamic:
        from lib_mosca.dynamic_gs import DynSCFGaussian

        d_model = DynSCFGaussian.load_from_ckpt(load_torch(paths["dynamic"], device), device=device).to(device).eval()
        with torch.inference_mode():
            d_model.set_inference_mode()
    return cams, s_model, d_model


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_ply(args: argparse.Namespace) -> dict[str, Any]:
    saved_dir = args.saved_dir.expanduser().resolve()
    output_dir = (args.output_dir or (saved_dir / "export_brush")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / args.ply_name
    tmp_path = ply_path.with_name(ply_path.name + ".tmp")
    if ply_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace: {ply_path}")
    if tmp_path.exists():
        tmp_path.unlink()

    device = resolve_device(args.device)
    cam_ckpt = load_torch(checkpoint_paths(saved_dir, args.backend)["camera"], "cpu")
    frame_indices = make_frame_indices(
        int(cam_ckpt["q_wc"].shape[0]),
        args.frame_stride,
        args.max_frames,
        args.all_frames,
        args.frame_sampling,
    )
    if args.static_only:
        frame_indices = frame_indices[:1]
    include_dynamic = not bool(args.static_only)
    include_static = bool((args.include_static or args.static_only) and not args.dynamic_only)

    print(f"Loading scene on {device} ...", flush=True)
    cams, s_model, d_model = load_scene(saved_dir, args.backend, device, load_dynamic=include_dynamic)
    if include_dynamic and d_model is not None and int(cams.T) != int(d_model.T):
        raise ValueError(f"Camera T={cams.T} differs from dynamic model T={d_model.T}")

    if args.opacity_threshold is not None:
        raise NotImplementedError("--opacity-threshold is reserved for later size-reduction exports")

    base_rows, base_geom, prev_dynamic_quat, static_count = build_base_rows(
        s_model=s_model,
        d_model=d_model,
        include_static=include_static,
        include_dynamic=include_dynamic,
        sh_rest_mode=args.sh_rest_mode,
    )
    total_splats = int(base_rows.shape[0])
    dynamic_count = int(d_model.N) if d_model is not None else 0
    header = make_header(total_splats, len(frame_indices))
    header_bytes = len(header.encode("ascii"))
    expected_bytes = expected_size_bytes(total_splats, len(frame_indices), header_bytes)

    validation_samples = min(int(args.validate_samples), dynamic_count)
    sample_indices = (
        np.linspace(0, dynamic_count - 1, validation_samples, dtype=np.int64)
        if validation_samples > 0
        else np.array([], dtype=np.int64)
    )
    max_roundtrip_error = 0.0
    max_roundtrip_bound = 0.0
    max_static_decode_error = 0.0
    per_frame_stats: list[dict[str, Any]] = []

    dynamic_base_geom = base_geom[static_count:, :] if include_dynamic else np.empty((0, len(ATTRS)), dtype=np.float32)
    print(
        f"Writing {total_splats} splats, {len(frame_indices)} frames to {ply_path} "
        f"(expected {expected_bytes / 1e9:.3f} GB)",
        flush=True,
    )

    start = time.time()
    with tmp_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(base_rows, dtype="<f4").tobytes())

        for out_frame, source_frame in enumerate(frame_indices[1:], start=1):
            if d_model is None or prev_dynamic_quat is None:
                raise ValueError("Delta frames require a dynamic model")
            with torch.inference_mode():
                d_xyz, d_frame, d_scale, _, _ = d_model.forward(int(source_frame))
                d_quat = matrix_to_quat_wxyz(d_frame)
                d_quat = same_hemisphere(d_quat, prev_dynamic_quat)
                prev_dynamic_quat = d_quat.detach()
                dynamic_geom = np.concatenate(
                    [tensor_np(d_xyz), tensor_np(d_quat), log_scale_np(d_scale)],
                    axis=1,
                )

            dynamic_delta = np.ascontiguousarray(dynamic_geom - dynamic_base_geom, dtype=np.float32)
            mn, mx, q, span, good = quantize_delta(dynamic_delta, total_splats, static_count)
            f.write(np.ascontiguousarray(mn, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(mx, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(q, dtype="<u2").tobytes())

            sample_q = q[static_count + sample_indices]
            sample_decoded = decode_q(sample_q, mn, mx)
            sample_error = float(np.max(np.abs(sample_decoded - dynamic_delta[sample_indices])))
            sample_bound = float(np.max(span[good] / UINT16_MAX) if np.any(good) else 0.0)
            max_roundtrip_error = max(max_roundtrip_error, sample_error)
            max_roundtrip_bound = max(max_roundtrip_bound, sample_bound)
            if static_count > 0:
                static_decoded = decode_q(q[:1], mn, mx)
                max_static_decode_error = max(max_static_decode_error, float(np.max(np.abs(static_decoded))))

            per_frame_stats.append(
                {
                    "delta_frame": out_frame,
                    "source_frame": int(source_frame),
                    "min": mn.tolist(),
                    "max": mx.tolist(),
                    "sample_roundtrip_error": sample_error,
                    "sample_roundtrip_bound": sample_bound,
                }
            )
            if out_frame == 1 or out_frame % 16 == 0 or out_frame == len(frame_indices) - 1:
                print(
                    f"frame {out_frame}/{len(frame_indices) - 1} "
                    f"(source {source_frame}) elapsed={(time.time() - start) / 60.0:.1f}m",
                    flush=True,
                )

    os.replace(tmp_path, ply_path)
    actual_bytes = ply_path.stat().st_size

    initial_frame = parse_initial_frame(args.initial_camera_frame, int(cams.T))
    if args.static_only:
        focus_target = static_focus_target(s_model)
        focus_target_method = "median center of upper-opacity static Gaussians"
    else:
        if d_model is None:
            raise ValueError("Dynamic focus target requires a dynamic model")
        focus_target = dynamic_focus_target(d_model, initial_frame)
        focus_target_method = "median center of upper-opacity dynamic Gaussians at initial camera frame"
    camera_data = camera_to_json(cams, initial_frame, focus_target)
    camera_data.update(
        {
            "ply_file": ply_path.name,
            "saved_dir": str(saved_dir),
            "focus_target_method": focus_target_method,
        }
    )
    camera_path = output_dir / "initial_camera.json"
    write_json(camera_path, camera_data)

    manifest = {
        "schema": "mosca_brush_release_manifest_v1",
        "saved_dir": str(saved_dir),
        "backend": args.backend.lower(),
        "ply_file": ply_path.name,
        "ply_path": str(ply_path),
        "initial_camera_file": camera_path.name,
        "include_static": include_static,
        "include_dynamic": include_dynamic,
        "static_only": bool(args.static_only),
        "frame_sampling": args.frame_sampling,
        "frame_stride": int(args.frame_stride),
        "max_frames": None if args.max_frames is None else int(args.max_frames),
        "sh_rest_mode": args.sh_rest_mode,
        "static_splats": int(static_count),
        "dynamic_splats": int(dynamic_count),
        "total_splats": total_splats,
        "frames": len(frame_indices),
        "source_frame_indices": frame_indices,
        "header_bytes": header_bytes,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "base_float_count": BASE_FLOAT_COUNT,
        "delta_attr_order": list(ATTRS),
        "f_rest_count": CAT4D_REST_COUNT,
        "scale_convention": "log(clamped MoSca physical scale)",
        "rotation_convention": "PyTorch3D wxyz quaternion with temporal sign continuity",
        "opacity_convention": "raw MoSca opacity logits, static across frames",
        "validation": {
            "sample_count": validation_samples,
            "max_roundtrip_error": max_roundtrip_error,
            "max_roundtrip_bound": max_roundtrip_bound,
            "max_static_decode_error": max_static_decode_error,
            "byte_size_match": actual_bytes == expected_bytes,
        },
        "per_frame_validation": per_frame_stats,
    }
    manifest_path = output_dir / "release_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Wrote {ply_path}", flush=True)
    print(f"Wrote {camera_path}", flush=True)
    print(f"Wrote {manifest_path}", flush=True)
    return manifest


def read_ply_header(path: Path) -> tuple[str, int]:
    marker = b"end_header\n"
    with path.open("rb") as f:
        data = bytearray()
        while True:
            chunk = f.read(4096)
            if not chunk:
                raise ValueError(f"PLY header missing end_header: {path}")
            data.extend(chunk)
            idx = data.find(marker)
            if idx >= 0:
                end = idx + len(marker)
                return bytes(data[:end]).decode("ascii"), end


def validate_export(output_dir: Path, ply_name: str) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    ply_path = output_dir / ply_name
    manifest_path = output_dir / "release_manifest.json"
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    header, header_bytes = read_ply_header(ply_path)
    total_splats = int(manifest["total_splats"])
    frames = int(manifest["frames"])
    expected_bytes = expected_size_bytes(total_splats, frames, header_bytes)
    actual_bytes = ply_path.stat().st_size
    checks = {
        "header_starts_with_ply": header.startswith("ply\nformat binary_little_endian 1.0\n"),
        "vertex_count": f"element vertex {total_splats}" in header,
        "first_delta": frames == 1 or f"element delta_vertex_1 {total_splats}" in header,
        "last_delta": frames == 1 or f"element delta_vertex_{frames - 1} {total_splats}" in header,
        "byte_size_match": actual_bytes == expected_bytes,
        "manifest_byte_size_match": actual_bytes == int(manifest["actual_bytes"]),
        "roundtrip_within_bound": float(manifest["validation"]["max_roundtrip_error"])
        <= float(manifest["validation"]["max_roundtrip_bound"]) + 1e-5,
        "static_decode_within_bound": float(manifest["validation"]["max_static_decode_error"])
        <= float(manifest["validation"]["max_roundtrip_bound"]) + 1e-5,
    }
    result = {
        "ply_path": str(ply_path),
        "header_bytes": header_bytes,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "checks": checks,
        "ok": all(checks.values()),
    }
    if not result["ok"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"Validation failed for {ply_path}: {failed}")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    saved_dir = args.saved_dir.expanduser().resolve()
    output_dir = (args.output_dir or (saved_dir / "export_brush")).expanduser().resolve()
    if args.static_only and args.dynamic_only:
        raise ValueError("--static-only and --dynamic-only are mutually exclusive")
    if args.dynamic_only:
        args.include_static = False
    if args.static_only:
        args.include_static = True

    if args.validate_only:
        validate_export(output_dir, args.ply_name)
        return

    cam_ckpt = load_torch(checkpoint_paths(saved_dir, args.backend)["camera"], "cpu")
    frame_indices = make_frame_indices(
        int(cam_ckpt["q_wc"].shape[0]),
        args.frame_stride,
        args.max_frames,
        args.all_frames,
        args.frame_sampling,
    )
    if args.static_only:
        frame_indices = frame_indices[:1]
    info = inspect_checkpoints(
        saved_dir,
        args.backend,
        bool(args.include_static),
        not bool(args.static_only),
        frame_indices,
    )
    print(json.dumps(info.__dict__, indent=2), flush=True)
    if args.dry_run:
        return

    manifest = write_ply(args)
    validate_export(output_dir, manifest["ply_file"])


if __name__ == "__main__":
    main()
