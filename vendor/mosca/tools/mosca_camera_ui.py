#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


C0 = 0.28209479177387814


@dataclass
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray
    opacity: np.ndarray


@dataclass
class MoscaScene:
    saved_dir: Path
    cams: object
    s_model: object
    d_model: object
    device: torch.device
    static_points: PointCloudData
    source_c2w: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int


@dataclass
class Keyframe:
    frame: int
    c2w: np.ndarray
    wxyz: np.ndarray
    position: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Viser camera UI for MoSca photometric checkpoints."
    )
    parser.add_argument(
        "--saved-dir",
        required=True,
        type=Path,
        help="Directory containing photometric_cam.pth and photometric_*_model_*.pth.",
    )
    parser.add_argument("--backend", default="native_add3")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9997)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--camera-id", default="cam_ui")
    parser.add_argument(
        "--export-path",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <saved-dir>/camera_ui_exports/cam_ui_c2w.npz.",
    )
    parser.add_argument("--static-point-budget", type=int, default=150_000)
    parser.add_argument("--dynamic-point-budget", type=int, default=150_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--camera-frustum-scale", type=float, default=0.08)
    parser.add_argument("--camera-stride", type=int, default=8)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument(
        "--no-fast-dynamic",
        action="store_true",
        help="Skip DynSCFGaussian.set_inference_mode() before interactive updates.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Load checkpoints, extract sample point clouds, export source cameras, then exit.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def sh_dc_to_rgb_u8(sh_dc: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(sh_dc, torch.Tensor):
        rgb = sh_dc.detach().float().cpu().numpy()
    else:
        rgb = np.asarray(sh_dc, dtype=np.float32)
    rgb = np.clip(rgb[:, :3] * C0 + 0.5, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def choose_points(opacity: np.ndarray, threshold: float, budget: int) -> np.ndarray:
    mask = np.asarray(opacity).reshape(-1) >= float(threshold)
    indices = np.flatnonzero(mask)
    if budget <= 0 or len(indices) <= budget:
        return indices
    picked = np.linspace(0, len(indices) - 1, int(budget), dtype=np.int64)
    return indices[picked]


def load_static_points(s_model: object) -> PointCloudData:
    with torch.inference_mode():
        points = tensor_to_numpy(s_model.get_x.float()).astype(np.float32)
        colors = sh_dc_to_rgb_u8(s_model.get_c[:, :3])
        opacity = tensor_to_numpy(s_model.get_o.float()).reshape(-1).astype(np.float32)
    return PointCloudData(points=points, colors=colors, opacity=opacity)


def load_dynamic_points(d_model: object, frame: int) -> PointCloudData:
    with torch.inference_mode():
        xyz, _, _, opacity, sph = d_model.forward(int(frame), active_sph_order=0)
        points = tensor_to_numpy(xyz.float()).astype(np.float32)
        colors = sh_dc_to_rgb_u8(sph[:, :3])
        opacity_np = tensor_to_numpy(opacity.float()).reshape(-1).astype(np.float32)
    return PointCloudData(points=points, colors=colors, opacity=opacity_np)


def import_mosca_loaders():
    try:
        from lib_moca.camera import MonocularCameras
        from lib_mosca.dynamic_gs import DynSCFGaussian
        from lib_mosca.static_gs import StaticGaussian
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency while importing MoSca loaders: {exc}.\n"
            "Launch from the repo environment, for example: conda activate mosca"
        ) from exc
    return MonocularCameras, StaticGaussian, DynSCFGaussian


def load_scene(
    saved_dir: Path,
    backend: str,
    device: torch.device,
    *,
    fast_dynamic: bool,
) -> MoscaScene:
    saved_dir = saved_dir.expanduser().resolve()
    if not saved_dir.is_dir():
        raise FileNotFoundError(f"Missing saved directory: {saved_dir}")

    cam_path = saved_dir / "photometric_cam.pth"
    static_path = saved_dir / f"photometric_s_model_{backend.lower()}.pth"
    dynamic_path = saved_dir / f"photometric_d_model_{backend.lower()}.pth"
    for path in (cam_path, static_path, dynamic_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required checkpoint: {path}")

    MonocularCameras, StaticGaussian, DynSCFGaussian = import_mosca_loaders()

    logging.info("Loading cameras from %s", cam_path)
    cams = MonocularCameras.load_from_ckpt(torch.load(cam_path, map_location=device)).to(
        device
    )
    cams.eval()

    logging.info("Loading static Gaussians from %s", static_path)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(static_path, map_location=device),
        device=device,
    ).to(device)
    s_model.eval()

    logging.info("Loading dynamic Gaussians from %s", dynamic_path)
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(dynamic_path, map_location=device),
        device=device,
    ).to(device)
    d_model.eval()
    if fast_dynamic:
        with torch.inference_mode():
            d_model.set_inference_mode()

    with torch.inference_mode():
        source_c2w = tensor_to_numpy(cams.T_wc_list().float()).astype(np.float32)
        intrinsics = tensor_to_numpy(cams.default_K.float()).astype(np.float32)
        width = int(cams.default_W.item())
        height = int(cams.default_H.item())
    static_points = load_static_points(s_model)

    if int(cams.T) != int(d_model.T):
        logging.warning("Camera T=%s differs from dynamic model T=%s", cams.T, d_model.T)

    return MoscaScene(
        saved_dir=saved_dir,
        cams=cams,
        s_model=s_model,
        d_model=d_model,
        device=device,
        static_points=static_points,
        source_c2w=source_c2w,
        intrinsics=intrinsics,
        width=width,
        height=height,
    )


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Cannot normalize zero-norm quaternion")
    return (quat / norm).astype(np.float32)


def wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(wxyz).astype(np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
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
    return normalize_quat_wxyz(np.asarray([qw, qx, qy, qz], dtype=np.float64))


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quat_wxyz(q0).astype(np.float64)
    q1 = normalize_quat_wxyz(q1).astype(np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_wxyz(q0 + float(alpha) * (q1 - q0))
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


def c2w_from_wxyz_position(wxyz: np.ndarray, position: np.ndarray) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = wxyz_to_matrix(wxyz)
    c2w[:3, 3] = np.asarray(position, dtype=np.float32)
    return c2w


def interpolate_keyframes(
    keyframes: Sequence[Keyframe],
    num_frames: int,
    fallback_c2w: np.ndarray,
) -> np.ndarray:
    if not keyframes:
        return np.asarray(fallback_c2w, dtype=np.float32).copy()
    ordered = sorted(keyframes, key=lambda item: item.frame)
    if len(ordered) == 1:
        return np.repeat(ordered[0].c2w[None].astype(np.float32), num_frames, axis=0)

    out = np.zeros((num_frames, 4, 4), dtype=np.float32)
    for t in range(num_frames):
        if t <= ordered[0].frame:
            out[t] = ordered[0].c2w
            continue
        if t >= ordered[-1].frame:
            out[t] = ordered[-1].c2w
            continue

        right_index = next(i for i, keyframe in enumerate(ordered) if keyframe.frame >= t)
        left = ordered[right_index - 1]
        right = ordered[right_index]
        span = max(right.frame - left.frame, 1)
        alpha = smoothstep((t - left.frame) / span)
        position = (1.0 - alpha) * left.position + alpha * right.position
        wxyz = slerp_wxyz(left.wxyz, right.wxyz, alpha)
        out[t] = c2w_from_wxyz_position(wxyz, position)
    return out


def export_camera_npz(
    path: Path,
    camera_id: str,
    c2w: np.ndarray,
    scene: MoscaScene,
    keyframes: Sequence[Keyframe],
) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    keyframe_frames = np.asarray([item.frame for item in keyframes], dtype=np.int32)
    keyframe_c2w = (
        np.stack([item.c2w for item in keyframes], axis=0).astype(np.float32)
        if keyframes
        else np.zeros((0, 4, 4), dtype=np.float32)
    )
    intrinsics = np.repeat(scene.intrinsics[None], c2w.shape[0], axis=0).astype(np.float32)
    np.savez_compressed(
        path,
        camera_ids=np.asarray([camera_id]),
        **{f"{camera_id}_c2w": c2w.astype(np.float32)},
        intrinsics=intrinsics,
        K=scene.intrinsics.astype(np.float32),
        img_wh=np.asarray([scene.width, scene.height], dtype=np.int32),
        keyframe_frames=keyframe_frames,
        keyframe_c2w=keyframe_c2w,
        source_cam_c2w=scene.source_c2w.astype(np.float32),
        saved_dir=np.asarray(str(scene.saved_dir)),
    )
    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "camera_id": camera_id,
                "num_frames": int(c2w.shape[0]),
                "saved_dir": str(scene.saved_dir),
                "img_wh": [scene.width, scene.height],
                "keyframe_frames": keyframe_frames.astype(int).tolist(),
                "npz_path": str(path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def default_export_path(saved_dir: Path) -> Path:
    return saved_dir / "camera_ui_exports" / "cam_ui_c2w.npz"


def import_viser():
    try:
        import viser
        import viser.transforms as viser_tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Viser dependency: {exc}.\n"
            "Install it in the active environment, for example: conda activate mosca && pip install viser"
        ) from exc
    return viser, viser_tf


def selected_cloud(data: PointCloudData, threshold: float, budget: int) -> tuple[np.ndarray, np.ndarray]:
    indices = choose_points(data.opacity, threshold, budget)
    return data.points[indices], data.colors[indices]


def run_self_test(scene: MoscaScene, args: argparse.Namespace) -> None:
    frame = int(np.clip(args.initial_frame, 0, len(scene.source_c2w) - 1))
    dynamic_points = load_dynamic_points(scene.d_model, frame)
    static_xyz, _ = selected_cloud(
        scene.static_points, args.opacity_threshold, args.static_point_budget
    )
    dynamic_xyz, _ = selected_cloud(
        dynamic_points, args.opacity_threshold, args.dynamic_point_budget
    )
    export_path = args.export_path or default_export_path(scene.saved_dir)
    out_path = export_camera_npz(
        export_path,
        args.camera_id,
        scene.source_c2w,
        scene,
        keyframes=[],
    )
    print(
        "self-test ok: "
        f"T={len(scene.source_c2w)} static={len(static_xyz)} dynamic_frame_{frame}={len(dynamic_xyz)} "
        f"export={out_path}"
    )


def run_ui(scene: MoscaScene, args: argparse.Namespace) -> None:
    _, viser_tf = import_viser()
    import viser

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    num_frames = min(int(scene.cams.T), int(scene.d_model.T), len(scene.source_c2w))
    initial_frame = int(np.clip(args.initial_frame, 0, num_frames - 1))
    aspect = scene.width / max(scene.height, 1)
    fy = float(scene.intrinsics[1, 1])
    fov = float(2.0 * np.arctan2(scene.height / 2.0, max(fy, 1e-6)))
    export_path = args.export_path or default_export_path(scene.saved_dir)

    keyframes: dict[int, Keyframe] = {}
    keyframe_handles: dict[int, tuple[object, object]] = {}
    trajectory_handle = None

    static_xyz, static_rgb = selected_cloud(
        scene.static_points, args.opacity_threshold, args.static_point_budget
    )
    dynamic_data = load_dynamic_points(scene.d_model, initial_frame)
    dynamic_xyz, dynamic_rgb = selected_cloud(
        dynamic_data, args.opacity_threshold, args.dynamic_point_budget
    )

    static_handle = server.scene.add_point_cloud(
        "/static_gaussians",
        points=static_xyz,
        colors=static_rgb,
        point_size=float(args.point_size),
        point_shape="circle",
    )
    dynamic_handle = server.scene.add_point_cloud(
        "/dynamic_gaussians",
        points=dynamic_xyz,
        colors=dynamic_rgb,
        point_size=float(args.point_size),
        point_shape="circle",
    )

    source_frame_handles = []
    source_frustum_handles = []
    for frame_id in range(0, num_frames, max(int(args.camera_stride), 1)):
        c2w = scene.source_c2w[frame_id]
        se3 = viser_tf.SE3.from_matrix(c2w[:3, :])
        frame_handle = server.scene.add_frame(
            f"/source_cameras/{frame_id:05d}",
            wxyz=se3.rotation().wxyz,
            position=se3.translation(),
            axes_length=float(args.camera_frustum_scale) * 0.7,
            axes_radius=float(args.camera_frustum_scale) * 0.02,
            origin_radius=float(args.camera_frustum_scale) * 0.03,
        )
        frustum_handle = server.scene.add_camera_frustum(
            f"/source_cameras/{frame_id:05d}/frustum",
            fov=fov,
            aspect=aspect,
            scale=float(args.camera_frustum_scale),
            line_width=1.0,
        )
        source_frame_handles.append(frame_handle)
        source_frustum_handles.append(frustum_handle)

        @frustum_handle.on_click
        def _(_, pose=c2w):
            for client in server.get_clients().values():
                client.camera.wxyz = matrix_to_wxyz(pose[:3, :3])
                client.camera.position = pose[:3, 3]
                client.camera.fov = fov

    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=num_frames - 1, step=1, initial_value=initial_frame
    )
    opacity_slider = server.gui.add_slider(
        "Opacity threshold", min=0.0, max=1.0, step=0.005, initial_value=float(args.opacity_threshold)
    )
    static_budget_slider = server.gui.add_slider(
        "Static point budget",
        min=0,
        max=max(int(scene.static_points.points.shape[0]), 1),
        step=1000,
        initial_value=min(int(args.static_point_budget), int(scene.static_points.points.shape[0])),
    )
    dynamic_budget_slider = server.gui.add_slider(
        "Dynamic point budget",
        min=0,
        max=max(int(dynamic_data.points.shape[0]), 1),
        step=1000,
        initial_value=min(int(args.dynamic_point_budget), int(dynamic_data.points.shape[0])),
    )
    point_size_slider = server.gui.add_slider(
        "Point size", min=0.0005, max=0.03, step=0.0005, initial_value=float(args.point_size)
    )
    show_static = server.gui.add_checkbox("Show static", initial_value=True)
    show_dynamic = server.gui.add_checkbox("Show dynamic", initial_value=True)
    show_source_cameras = server.gui.add_checkbox("Show source cameras", initial_value=True)
    export_text = server.gui.add_text("Export path", initial_value=str(export_path))
    camera_id_text = server.gui.add_text("Camera id", initial_value=str(args.camera_id))
    capture_button = server.gui.add_button("Capture current view")
    delete_button = server.gui.add_button("Delete keyframe at frame")
    snap_button = server.gui.add_button("Snap to output camera")
    export_button = server.gui.add_button("Export cameras")

    def update_static_cloud() -> None:
        xyz, rgb = selected_cloud(
            scene.static_points,
            float(opacity_slider.value),
            int(static_budget_slider.value),
        )
        static_handle.points = xyz
        static_handle.colors = rgb

    def update_dynamic_cloud() -> None:
        frame = int(frame_slider.value)
        data = load_dynamic_points(scene.d_model, frame)
        xyz, rgb = selected_cloud(data, float(opacity_slider.value), int(dynamic_budget_slider.value))
        dynamic_handle.points = xyz
        dynamic_handle.colors = rgb

    def update_point_size() -> None:
        static_handle.point_size = float(point_size_slider.value)
        dynamic_handle.point_size = float(point_size_slider.value)

    def current_trajectory() -> np.ndarray:
        return interpolate_keyframes(
            list(keyframes.values()),
            num_frames=num_frames,
            fallback_c2w=scene.source_c2w[:num_frames],
        )

    def refresh_trajectory() -> None:
        nonlocal trajectory_handle
        if trajectory_handle is not None:
            trajectory_handle.remove()
            trajectory_handle = None
        if len(keyframes) < 2:
            return
        c2w = current_trajectory()
        trajectory_handle = server.scene.add_spline_catmull_rom(
            "/authored_camera_path",
            positions=c2w[:, :3, 3],
            color=(255, 196, 64),
            line_width=2.0,
        )

    def refresh_keyframe_handle(frame: int) -> None:
        old_handles = keyframe_handles.pop(frame, None)
        if old_handles is not None:
            for handle in old_handles:
                handle.remove()
        keyframe = keyframes[frame]
        se3 = viser_tf.SE3.from_matrix(keyframe.c2w[:3, :])
        frame_handle = server.scene.add_frame(
            f"/keyframes/{frame:05d}",
            wxyz=se3.rotation().wxyz,
            position=se3.translation(),
            axes_length=float(args.camera_frustum_scale),
            axes_radius=float(args.camera_frustum_scale) * 0.03,
            origin_radius=float(args.camera_frustum_scale) * 0.04,
        )
        frustum_handle = server.scene.add_camera_frustum(
            f"/keyframes/{frame:05d}/frustum",
            fov=fov,
            aspect=aspect,
            scale=float(args.camera_frustum_scale) * 1.25,
            color=(255, 196, 64),
            line_width=2.0,
        )
        keyframe_handles[frame] = (frame_handle, frustum_handle)

    def first_client():
        clients = server.get_clients()
        return next(iter(clients.values()), None)

    @frame_slider.on_update
    def _(_) -> None:
        update_dynamic_cloud()

    @opacity_slider.on_update
    def _(_) -> None:
        update_static_cloud()
        update_dynamic_cloud()

    @static_budget_slider.on_update
    def _(_) -> None:
        update_static_cloud()

    @dynamic_budget_slider.on_update
    def _(_) -> None:
        update_dynamic_cloud()

    @point_size_slider.on_update
    def _(_) -> None:
        update_point_size()

    @show_static.on_update
    def _(_) -> None:
        static_handle.visible = bool(show_static.value)

    @show_dynamic.on_update
    def _(_) -> None:
        dynamic_handle.visible = bool(show_dynamic.value)

    @show_source_cameras.on_update
    def _(_) -> None:
        for handle in [*source_frame_handles, *source_frustum_handles]:
            handle.visible = bool(show_source_cameras.value)

    @capture_button.on_click
    def _(_) -> None:
        client = first_client()
        if client is None:
            print("No Viser client connected; cannot capture keyframe.", flush=True)
            return
        frame = int(frame_slider.value)
        wxyz = np.asarray(client.camera.wxyz, dtype=np.float32)
        position = np.asarray(client.camera.position, dtype=np.float32)
        c2w = c2w_from_wxyz_position(wxyz, position)
        keyframes[frame] = Keyframe(frame=frame, c2w=c2w, wxyz=wxyz, position=position)
        refresh_keyframe_handle(frame)
        refresh_trajectory()
        print(f"Captured keyframe at frame {frame}; total keyframes={len(keyframes)}", flush=True)

    @delete_button.on_click
    def _(_) -> None:
        frame = int(frame_slider.value)
        removed = keyframes.pop(frame, None)
        handles = keyframe_handles.pop(frame, None)
        if handles is not None:
            for handle in handles:
                handle.remove()
        refresh_trajectory()
        print(
            f"{'Deleted' if removed is not None else 'No'} keyframe at frame {frame}",
            flush=True,
        )

    @snap_button.on_click
    def _(_) -> None:
        client = first_client()
        if client is None:
            print("No Viser client connected; cannot snap camera.", flush=True)
            return
        c2w = current_trajectory()[int(frame_slider.value)]
        client.camera.wxyz = matrix_to_wxyz(c2w[:3, :3])
        client.camera.position = c2w[:3, 3]
        client.camera.fov = fov

    @export_button.on_click
    def _(_) -> None:
        camera_id = str(camera_id_text.value).strip() or args.camera_id
        c2w = current_trajectory()
        out_path = export_camera_npz(
            Path(str(export_text.value)),
            camera_id,
            c2w,
            scene,
            list(keyframes.values()),
        )
        print(
            f"Exported {len(c2w)} poses for {camera_id} to {out_path} "
            f"with {len(keyframes)} keyframes.",
            flush=True,
        )

    print(
        f"Loaded {num_frames} frames, {scene.static_points.points.shape[0]} static points, "
        f"{dynamic_data.points.shape[0]} dynamic points."
    )
    print(f"Open http://localhost:{args.port} or forward port {args.port}.")
    print("Use Capture current view to add keyframes, then Export cameras.")
    while True:
        time.sleep(0.05)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    scene = load_scene(
        args.saved_dir,
        args.backend,
        resolve_device(args.device),
        fast_dynamic=not args.no_fast_dynamic,
    )
    if args.self_test:
        run_self_test(scene, args)
    else:
        run_ui(scene, args)


if __name__ == "__main__":
    main()
