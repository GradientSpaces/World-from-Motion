#!/usr/bin/env python3
"""Render three UI-camera release videos from a MoSca photometric checkpoint.

Given a solved photometric log directory and an authored camera trajectory
(camera-to-world ``[T, 4, 4]`` in MoSca world coordinates), this renders the
same three camera/time combinations used by the ``photometric_2dviz`` panels,
but driven by the authored UI path instead of the auto orbit:

1. ``fix_cam_vary_time``  - hold one camera pose, advance scene time.
2. ``fix_time_vary_cam``  - freeze the scene at one time, move the camera.
3. ``vary_cam_time``      - move the camera while time advances.

Timesteps are sampled from the original reconstruction with ``--frame-stride``
and ``--max-frames`` (default first-73 stride-2 -> scene time ``t_i = 2*i``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mosca_camera_ui import load_scene, resolve_device  # noqa: E402


# Base modes follow the authored UI camera trajectory.
BASE_MODES = ("fix_cam_vary_time", "fix_time_vary_cam", "vary_cam_time")
# Spiral modes replace the camera with a bullet-time circular orbit anchored at
# the mid-point camera (the photometric_2dviz "move_around" look).
SPIRAL_MODES = ("fix_time_vary_cam_spiral", "vary_cam_time_spiral")
MODES = BASE_MODES + SPIRAL_MODES
MODE_FILE_SUFFIX = {
    "fix_cam_vary_time": "mode1_fix_cam_vary_time",
    "fix_time_vary_cam": "mode2_fix_time_vary_cam",
    "vary_cam_time": "mode3_vary_cam_time",
    "fix_time_vary_cam_spiral": "mode2_spiral_fix_time_vary_cam",
    "vary_cam_time_spiral": "mode3_spiral_vary_cam_time",
}
BG_COLORS = {"white": [1.0, 1.0, 1.0], "black": [0.0, 0.0, 0.0]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved-dir",
        "--saved_dir",
        required=True,
        type=Path,
        help="Photometric log directory with photometric_cam.pth and *_model_*.pth.",
    )
    parser.add_argument(
        "--camera-source",
        choices=["npz", "recon_input"],
        default="npz",
        help=(
            "npz: load the camera trajectory from --camera-npz. recon_input: use the "
            "recon's own solved input trajectory (cams.T_wc); no npz needed and "
            "camera-index-mode is forced to timeline."
        ),
    )
    parser.add_argument(
        "--camera-npz",
        "--camera_npz",
        default=None,
        type=Path,
        help="Authored camera trajectory npz (camera-to-world [T,4,4]). Required for --camera-source npz.",
    )
    parser.add_argument(
        "--camera-key",
        default=None,
        help="Array key inside --camera-npz. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--camera-index-mode",
        choices=["timeline", "presampled"],
        default="timeline",
        help=(
            "timeline: index the camera array by absolute scene time (full-length "
            "trajectory). presampled: the camera array is already stride/max sampled "
            "and is indexed by render-frame order."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap on rendered frames. <=0 -> use the full trajectory.",
    )
    parser.add_argument(
        "--fix-cam-index",
        type=int,
        default=-1,
        help="Render-frame index whose camera pose is held in fix_cam_vary_time. <0 -> middle (trajectory mid-point).",
    )
    parser.add_argument(
        "--fix-time-index",
        type=int,
        default=-1,
        help="Render-frame index whose scene time is frozen in fix_time_vary_cam. <0 -> middle (trajectory mid-point).",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--spiral-angle-deg",
        type=float,
        default=20.0,
        help="Half-angle converting center depth to orbit radius for spiral modes "
        "(radius = tan(angle)*center_depth). Matches photometric_2dviz move_around.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--bg", choices=list(BG_COLORS), default="white")
    parser.add_argument("--backend", default="native_add3")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the mp4s into, e.g. data/release_vid/<scene_key>.",
    )
    parser.add_argument(
        "--scene-key",
        default=None,
        help="Filename prefix for the outputs. Defaults to --output-dir name.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-fast-dynamic",
        action="store_true",
        help="Skip DynSCFGaussian.set_inference_mode() (slower, matches raw forward).",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip the 1x3 side-by-side montage video.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate camera/index bounds on CPU without rendering, then exit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_camera_c2w(npz_path: Path, camera_key: str | None) -> tuple[np.ndarray, str]:
    with np.load(npz_path) as data:
        files = list(data.files)
        key = None
        if camera_key and camera_key in files:
            key = camera_key
        elif "c2w" in files:
            key = "c2w"
        elif "cam_ui_c2w" in files:
            key = "cam_ui_c2w"
        elif "camera_ids" in files:
            cam_id = str(np.asarray(data["camera_ids"]).reshape(-1)[0])
            candidate = f"{cam_id}_c2w"
            if candidate in files:
                key = candidate
        if key is None:
            for name in files:
                arr = data[name]
                if getattr(arr, "ndim", 0) == 3 and arr.shape[1:] == (4, 4):
                    key = name
                    break
        if key is None:
            raise KeyError(
                f"No [T,4,4] camera-to-world array found in {npz_path}; keys={files}"
            )
        c2w = np.asarray(data[key], dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected c2w shape {c2w.shape} for key {key} in {npz_path}")
    return c2w, key


def sample_timesteps(total_frames: int, stride: int, max_frames: int | None) -> list[int]:
    if stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {stride}")
    indices = list(range(0, int(total_frames), int(stride)))
    if max_frames is not None and max_frames > 0:
        indices = indices[: int(max_frames)]
    if not indices:
        raise ValueError("No timesteps selected; check --frame-stride/--max-frames.")
    return indices


def cameras_for_render(
    c2w: np.ndarray, timesteps: list[int], index_mode: str
) -> tuple[list[int], np.ndarray]:
    """Return (timesteps_used, cam_render) aligned 1:1 by render-frame index."""
    if index_mode == "timeline":
        max_t = max(timesteps)
        if max_t >= c2w.shape[0]:
            raise ValueError(
                f"timeline mode needs >= {max_t + 1} camera poses, got {c2w.shape[0]}."
            )
        return list(timesteps), c2w[np.asarray(timesteps, dtype=np.int64)]
    # presampled: camera array is already sampled; render as many frames as it has.
    n = min(len(timesteps), c2w.shape[0])
    return list(timesteps[:n]), c2w[:n]


@torch.no_grad()
def render_rgb_from_tcw(scene, scene_time: int, t_cw: torch.Tensor, bg_color) -> np.ndarray:
    from lib_render.render_helper import render

    gs = [scene.s_model(), scene.d_model(int(scene_time))]
    out = render(
        gs,
        scene.cams.default_H,
        scene.cams.default_W,
        scene.cams.default_K,
        t_cw.float(),
        bg_color=bg_color,
    )
    rgb = out["rgb"].permute(1, 2, 0).clamp(0.0, 1.0).detach().cpu().numpy()
    return np.round(rgb * 255.0).astype(np.uint8)


def render_rgb_frame(scene, scene_time: int, c2w: np.ndarray, bg_color) -> np.ndarray:
    t_cw = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(scene.device)
    return render_rgb_from_tcw(scene, scene_time, t_cw, bg_color)


@torch.no_grad()
def build_orbit_t_cw(
    scene,
    anchor_c2w: np.ndarray,
    depth_scene_time: int,
    num_steps: int,
    angle_deg: float,
    bg_color,
) -> list[torch.Tensor]:
    """Bullet-time circular orbit around the anchor camera's optical axis.

    Mirrors ``__get_move_around_cam_T_cw__`` in photo_recon_viz_utils: the orbit
    radius is ``tan(angle)*center_depth`` and every pose looks at the center-depth
    focus point. One full loop over ``num_steps`` frames.
    """
    from lib_render.render_helper import render

    device = scene.device
    anchor_c2w_t = torch.from_numpy(np.asarray(anchor_c2w, dtype=np.float32)).to(device)
    anchor_t_cw = torch.inverse(anchor_c2w_t)
    render_dict = render(
        [scene.s_model(), scene.d_model(int(depth_scene_time))],
        scene.cams.default_H,
        scene.cams.default_W,
        scene.cams.default_K,
        anchor_t_cw.float(),
        bg_color=bg_color,
    )
    depth = render_dict["dep"][0]
    center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
    if center_dep < 1e-2:
        try:
            center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
        except Exception:
            center_dep = 1.0
    focus_point = torch.tensor([0.0, 0.0, center_dep], device=device, dtype=torch.float32)
    radius = float(np.tan(np.deg2rad(angle_deg)) * center_dep)
    up_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    denom = max(num_steps - 1, 1)

    t_cw_list: list[torch.Tensor] = []
    for i in range(num_steps):
        x = radius * np.cos(2.0 * np.pi * i / denom)
        y = radius * np.sin(2.0 * np.pi * i / denom)
        t_c_new = torch.eye(4, device=device, dtype=torch.float32)
        t_c_new[0, 3] = float(x)
        t_c_new[1, 3] = float(y)
        z_dir = F.normalize(focus_point - t_c_new[:3, 3], dim=0)
        x_dir = F.normalize(torch.cross(up_axis, z_dir, dim=0), dim=0)
        y_dir = F.normalize(torch.cross(z_dir, x_dir, dim=0), dim=0)
        t_c_new[:3, 0] = x_dir
        t_c_new[:3, 1] = y_dir
        t_c_new[:3, 2] = z_dir
        t_w_new = anchor_c2w_t @ t_c_new
        t_cw_list.append(torch.inverse(t_w_new))
    logging.info(
        "Orbit: center_depth=%.4f radius=%.4f angle=%.1fdeg steps=%d",
        center_dep,
        radius,
        angle_deg,
        num_steps,
    )
    return t_cw_list


def render_mode_frames(
    scene,
    mode: str,
    timesteps: list[int],
    cam_render: np.ndarray,
    orbit_t_cw,
    fix_cam_index: int,
    fix_time_index: int,
    bg_color,
) -> list[np.ndarray]:
    n = len(timesteps)
    frames: list[np.ndarray] = []
    if mode == "fix_cam_vary_time":
        c2w_fixed = cam_render[fix_cam_index]
        for i in range(n):
            frames.append(render_rgb_frame(scene, timesteps[i], c2w_fixed, bg_color))
    elif mode == "fix_time_vary_cam":
        t_fixed = timesteps[fix_time_index]
        for i in range(n):
            frames.append(render_rgb_frame(scene, t_fixed, cam_render[i], bg_color))
    elif mode == "vary_cam_time":
        for i in range(n):
            frames.append(render_rgb_frame(scene, timesteps[i], cam_render[i], bg_color))
    elif mode == "fix_time_vary_cam_spiral":
        t_fixed = timesteps[fix_time_index]
        for i in range(n):
            frames.append(render_rgb_from_tcw(scene, t_fixed, orbit_t_cw[i], bg_color))
    elif mode == "vary_cam_time_spiral":
        for i in range(n):
            frames.append(render_rgb_from_tcw(scene, timesteps[i], orbit_t_cw[i], bg_color))
    else:
        raise ValueError(f"Unknown mode {mode}")
    return frames


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=1)
    logging.info("Wrote %s (%d frames @ %d fps)", path, len(frames), fps)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    scene_key = args.scene_key or args.output_dir.name
    if args.camera_source == "npz" and args.camera_npz is None:
        raise ValueError("--camera-npz is required when --camera-source npz.")
    # The recon's own input trajectory is already a full-length per-frame path.
    camera_index_mode = (
        "timeline" if args.camera_source == "recon_input" else args.camera_index_mode
    )

    c2w = None
    if args.camera_source == "npz":
        c2w, camera_key = load_camera_c2w(args.camera_npz, args.camera_key)
        logging.info(
            "Loaded camera trajectory %s key=%s shape=%s", args.camera_npz, camera_key, c2w.shape
        )

    if args.dry_run:
        from lib_moca.camera import MonocularCameras

        cams = MonocularCameras.load_from_ckpt(
            torch.load(args.saved_dir / "photometric_cam.pth", map_location="cpu")
        )
        total_frames = int(cams.T)
        if args.camera_source == "recon_input":
            c2w = cams.T_wc_list().detach().float().cpu().numpy()
            logging.info("Using recon input trajectory cams.T_wc shape=%s", c2w.shape)
        timesteps = sample_timesteps(total_frames, args.frame_stride, args.max_frames)
        timesteps, cam_render = cameras_for_render(c2w, timesteps, camera_index_mode)
        fix_cam_index = args.fix_cam_index if args.fix_cam_index >= 0 else len(timesteps) // 2
        fix_time_index = args.fix_time_index if args.fix_time_index >= 0 else len(timesteps) // 2
        if not (0 <= fix_cam_index < len(timesteps)):
            raise ValueError(f"resolved fix-cam-index {fix_cam_index} out of range [0,{len(timesteps)})")
        if not (0 <= fix_time_index < len(timesteps)):
            raise ValueError(f"resolved fix-time-index {fix_time_index} out of range [0,{len(timesteps)})")
        logging.info(
            "[dry-run] scene_key=%s camera_source=%s recon_T=%d render_frames=%d scene_time=[%d..%d] "
            "cam_index_mode=%s cam_render=%s fix_cam_index=%d (cam_t=%d) fix_time_index=%d (t=%d) modes=%s",
            scene_key,
            args.camera_source,
            total_frames,
            len(timesteps),
            timesteps[0],
            timesteps[-1],
            camera_index_mode,
            cam_render.shape,
            fix_cam_index,
            timesteps[fix_cam_index],
            fix_time_index,
            timesteps[fix_time_index],
            args.modes,
        )
        print(f"dry-run ok: {scene_key} render_frames={len(timesteps)} cam={cam_render.shape}")
        return

    device = resolve_device(args.device)
    scene = load_scene(
        args.saved_dir, args.backend, device, fast_dynamic=not args.no_fast_dynamic
    )
    total_frames = min(int(scene.cams.T), int(scene.d_model.T))
    if args.camera_source == "recon_input":
        c2w = np.asarray(scene.source_c2w, dtype=np.float32)
        logging.info("Using recon input trajectory cams.T_wc shape=%s", c2w.shape)
    timesteps = sample_timesteps(total_frames, args.frame_stride, args.max_frames)
    timesteps, cam_render = cameras_for_render(c2w, timesteps, camera_index_mode)
    fix_cam_index = args.fix_cam_index if args.fix_cam_index >= 0 else len(timesteps) // 2
    fix_time_index = args.fix_time_index if args.fix_time_index >= 0 else len(timesteps) // 2
    if not (0 <= fix_cam_index < len(timesteps)):
        raise ValueError(f"resolved fix-cam-index {fix_cam_index} out of range [0,{len(timesteps)})")
    if not (0 <= fix_time_index < len(timesteps)):
        raise ValueError(f"resolved fix-time-index {fix_time_index} out of range [0,{len(timesteps)})")
    bg_color = BG_COLORS[args.bg]

    logging.info(
        "Scene %s: recon_T=%d render_frames=%d scene_time=[%d..%d] H=%d W=%d "
        "fix_cam_index=%d (cam_t=%d) fix_time_index=%d (t=%d)",
        scene_key,
        total_frames,
        len(timesteps),
        timesteps[0],
        timesteps[-1],
        int(scene.cams.default_H),
        int(scene.cams.default_W),
        fix_cam_index,
        timesteps[fix_cam_index],
        fix_time_index,
        timesteps[fix_time_index],
    )

    orbit_t_cw = None
    if any(m in SPIRAL_MODES for m in args.modes):
        logging.info("Building bullet-time orbit anchored at mid-point camera ...")
        orbit_t_cw = build_orbit_t_cw(
            scene,
            cam_render[fix_cam_index],
            timesteps[fix_time_index],
            len(timesteps),
            args.spiral_angle_deg,
            bg_color,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_frames: dict[str, list[np.ndarray]] = {}
    for mode in args.modes:
        out_path = args.output_dir / f"{scene_key}_{MODE_FILE_SUFFIX[mode]}.mp4"
        if out_path.exists() and not args.overwrite:
            logging.warning("Skipping existing %s (pass --overwrite to regenerate)", out_path)
            continue
        logging.info("Rendering mode %s ...", mode)
        frames = render_mode_frames(
            scene, mode, timesteps, cam_render, orbit_t_cw, fix_cam_index, fix_time_index, bg_color
        )
        mode_frames[mode] = frames
        save_video(out_path, frames, args.fps)

    def _save_montage(name: str, mode_keys: list[str]) -> None:
        if args.no_combined or not all(m in mode_frames for m in mode_keys):
            return
        montage = [
            np.concatenate([mode_frames[m][i] for m in mode_keys], axis=1)
            for i in range(len(timesteps))
        ]
        save_video(args.output_dir / f"{scene_key}_{name}.mp4", montage, args.fps)

    _save_montage("combined_1x3", list(BASE_MODES))
    _save_montage("combined_spiral_1x2", list(SPIRAL_MODES))

    logging.info("Done: %s", args.output_dir)


if __name__ == "__main__":
    main()
