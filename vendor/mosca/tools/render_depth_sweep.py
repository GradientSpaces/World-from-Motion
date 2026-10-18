#!/usr/bin/env python3
"""Render a Lyra/CAT4D-style depth-sweep clip from a single recon frame.

Renders ONE (rgb, raw depth, alpha) frame from a MoSca photometric recon at a given
camera pose + scene time, then animates a glowing "light plane" that sweeps through
depth (near->far by default): the slice of the scene at the current sweep depth is
brightened while the rest stays dim, revealing the 3D structure.

Pose source: by default the recon's own camera at --pose-frame-index; pass
--pose-camera-pth to take the pose from another MonocularCameras checkpoint (which
must live in the SAME world frame as --saved-dir, e.g. the refined-frame new_traj.pth
when --saved-dir is the refined recon).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_gen.mosca_run_paths import infer_cfg_path  # noqa: E402
from lib_moca.camera import MonocularCameras  # noqa: E402
from lib_render.render_helper import render  # noqa: E402
from mosca_viz import load_model_cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saved-dir", required=True, type=Path,
                   help="MoSca photometric recon dir (photometric_cam.pth + gaussian models).")
    p.add_argument("--cfg", default=None, type=Path,
                   help="MoSca config. Defaults to the config recorded for --saved-dir.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output dir; frames go to <output-dir>/<camera-name>/, mp4 alongside.")
    p.add_argument("--camera-name", default="cam_sweep")
    p.add_argument("--pose-camera-pth", default=None, type=Path,
                   help="Optional MonocularCameras ckpt to take the pose from (same world frame).")
    p.add_argument("--pose-frame-index", type=int, default=0)
    p.add_argument("--time-id", type=int, default=0, help="Scene time id for the dynamic model.")
    p.add_argument("--sweep-frames", type=int, default=24,
                   help="Frames for the moving light band (smaller = faster sweep).")
    p.add_argument("--pre-hold-frames", type=int, default=6,
                   help="Clean RGB frames shown before the sweep.")
    p.add_argument("--post-hold-frames", type=int, default=6,
                   help="Clean RGB frames shown after the sweep (return to RGB).")
    p.add_argument("--direction", choices=("near2far", "far2near"), default="near2far")
    p.add_argument("--sigma-frac", type=float, default=0.08,
                   help="Light band width as a fraction of the near..far depth range.")
    p.add_argument("--ambient", type=float, default=0.55,
                   help="Brightness of the un-lit scene during the sweep (0=black, 1=full).")
    p.add_argument("--glow", type=float, default=0.3,
                   help="Extra additive white glow at the band (0 disables).")
    p.add_argument("--percentile-lo", type=float, default=2.0)
    p.add_argument("--percentile-hi", type=float, default=98.0)
    p.add_argument("--depth-alpha-threshold", type=float, default=1e-3)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def scalar_int(value) -> int:
    return int(value.item() if hasattr(value, "item") else value)


@torch.no_grad()
def render_source_frame(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    saved_dir = args.saved_dir.expanduser().resolve()
    cfg_path = (
        Path(infer_cfg_path(str(saved_dir), None)).resolve()
        if args.cfg is None
        else args.cfg.expanduser().resolve()
    )
    device = torch.device(args.device)
    _, d_model, s_model, cams = load_model_cfg(str(cfg_path), str(saved_dir), device=device)
    height = scalar_int(cams.default_H)
    width = scalar_int(cams.default_W)
    total_t = int(cams.T)
    time_id = int(args.time_id)
    if time_id < 0 or time_id >= total_t:
        raise ValueError(f"--time-id {time_id} out of range [0,{total_t}) for {saved_dir}")

    if args.pose_camera_pth is not None:
        pose_cams = MonocularCameras.load_from_ckpt(
            torch.load(args.pose_camera_pth.expanduser().resolve(),
                       map_location="cpu", weights_only=False)
        )
        idx = int(args.pose_frame_index)
        if idx < 0 or idx >= int(pose_cams.T):
            raise ValueError(
                f"--pose-frame-index {idx} out of range [0,{int(pose_cams.T)}) "
                f"for {args.pose_camera_pth}"
            )
        t_cw = pose_cams.T_cw(idx).detach().to(device).float()
    else:
        idx = int(args.pose_frame_index)
        if idx < 0 or idx >= total_t:
            raise ValueError(f"--pose-frame-index {idx} out of range [0,{total_t})")
        t_cw = cams.T_cw(idx).detach().to(device).float()

    render_dict = render(
        [s_model(), d_model(time_id)],
        height,
        width,
        K=cams.K(height, width),
        T_cw=t_cw,
        bg_color=[0.0, 0.0, 0.0],
    )
    rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0).detach().cpu().numpy()
    depth = render_dict["dep"].detach().cpu().numpy().squeeze(0).astype(np.float32)
    alpha = render_dict["alpha"].detach().cpu().numpy().squeeze(0).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (alpha > float(args.depth_alpha_threshold))
    return rgb.astype(np.float32), depth, valid


def synthesize_sweep(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                     args: argparse.Namespace) -> list[np.ndarray]:
    if not valid.any():
        raise RuntimeError("No valid rendered depth pixels for the sweep frame.")
    vals = depth[valid]
    d_near = float(np.percentile(vals, args.percentile_lo))
    d_far = float(np.percentile(vals, args.percentile_hi))
    if not np.isfinite(d_near) or not np.isfinite(d_far) or d_far <= d_near:
        d_near, d_far = float(vals.min()), float(vals.max())
    if d_far <= d_near:
        d_far = d_near + 1e-3
    sigma = max(float(args.sigma_frac) * (d_far - d_near), 1e-6)

    clean_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    k = int(args.sweep_frames)
    if k <= 0:
        fracs = np.array([])  # hold-only clip (no moving band)
    elif k == 1:
        fracs = np.array([0.5])
    else:
        fracs = np.linspace(0.0, 1.0, k)
    if args.direction == "far2near":
        fracs = fracs[::-1]

    frames: list[np.ndarray] = []
    # Show the clean RGB before the sweep.
    frames.extend([clean_u8] * max(int(args.pre_hold_frames), 0))
    for frac in fracs:
        s = d_near + float(frac) * (d_far - d_near)
        w = np.exp(-((depth - s) ** 2) / (2.0 * sigma * sigma))
        w[~valid] = 0.0
        brightness = float(args.ambient) + (1.0 - float(args.ambient)) * w
        out = rgb * brightness[..., None]
        if args.glow > 0.0:
            out = out + float(args.glow) * w[..., None]
        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(out_u8)
    # Return to the clean RGB after the sweep.
    frames.extend([clean_u8] * max(int(args.post_hold_frames), 0))
    return frames


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    cam_dir = out_dir / args.camera_name
    if cam_dir.exists() and any(cam_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{cam_dir} not empty; pass --overwrite.")
        # Clear stale frames so a shorter clip never leaves leftovers behind.
        shutil.rmtree(cam_dir)
    cam_dir.mkdir(parents=True, exist_ok=True)

    rgb, depth, valid = render_source_frame(args)
    frames = synthesize_sweep(rgb, depth, valid, args)

    for i, frame in enumerate(frames):
        imageio.imwrite(cam_dir / f"{i:05d}.png", frame)
    mp4_path = out_dir / f"{args.camera_name}.mp4"
    writer = imageio.get_writer(mp4_path, fps=int(args.fps), macro_block_size=None)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    meta = {
        "saved_dir": str(args.saved_dir.expanduser().resolve()),
        "camera_name": args.camera_name,
        "pose_camera_pth": None if args.pose_camera_pth is None
        else str(args.pose_camera_pth.expanduser().resolve()),
        "pose_frame_index": int(args.pose_frame_index),
        "time_id": int(args.time_id),
        "sweep_frames": int(args.sweep_frames),
        "pre_hold_frames": int(args.pre_hold_frames),
        "post_hold_frames": int(args.post_hold_frames),
        "direction": args.direction,
        "sigma_frac": float(args.sigma_frac),
        "ambient": float(args.ambient),
        "glow": float(args.glow),
        "fps": int(args.fps),
        "num_frames": len(frames),
        "frame_dir": str(cam_dir),
        "mp4": str(mp4_path),
    }
    (out_dir / f"{args.camera_name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OK depth-sweep {args.camera_name}: {len(frames)} frames @ {args.fps} fps -> {mp4_path}")


if __name__ == "__main__":
    main()
