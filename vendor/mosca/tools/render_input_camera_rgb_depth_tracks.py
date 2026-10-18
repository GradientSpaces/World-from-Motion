#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import colormaps
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_gen.mosca_run_paths import infer_cfg_path
from lib_render.render_helper import render
from mosca_viz import load_model_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render RGB/depth videos from a MoSca photometric checkpoint along the "
            "optimized input camera trajectory (default) or an external camera "
            "trajectory NPZ, then optionally overlay persistent 3D dynamic tracker trails."
        )
    )
    parser.add_argument("--ws", "--workspace", required=True, type=Path)
    parser.add_argument("--saved-dir", required=True, type=Path)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=None,
        help="MoSca config. Defaults to the config recorded in fit_commandline_args.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <saved-dir>/input_camera_rgb_depth_spectral_tracks "
            "for the input camera, or <saved-dir>/<camera-name>_rgb_depth_spectral_tracks "
            "when --camera-npz is given."
        ),
    )
    parser.add_argument(
        "--camera-npz",
        type=Path,
        default=None,
        help=(
            "Optional external camera-to-world trajectory NPZ (key 'c2w' or "
            "'<camera-name>_c2w', shape (T,4,4)) already aligned to the solved world "
            "frame. When omitted, the optimized input camera trajectory is used."
        ),
    )
    parser.add_argument("--camera-name", default="input_cam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-colormap", default="Spectral")
    parser.add_argument("--depth-alpha-threshold", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-tracks", action="store_true")
    parser.add_argument("--num-tracks", type=int, default=128)
    parser.add_argument("--line-steps", type=int, default=32)
    parser.add_argument("--track-radius", type=float, default=0.0015)
    parser.add_argument("--track-opacity", type=float, default=0.999)
    parser.add_argument("--track-overlay-alpha", type=float, default=0.95)
    return parser.parse_args()


def scalar_int(value) -> int:
    return int(value.item() if hasattr(value, "item") else value)


def load_external_c2w(npz_path: Path, camera_name: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if "c2w" in data.files:
            c2w = data["c2w"]
        elif f"{camera_name}_c2w" in data.files:
            c2w = data[f"{camera_name}_c2w"]
        else:
            raise KeyError(
                f"Missing 'c2w' or '{camera_name}_c2w' in {npz_path}; found {data.files}"
            )
        c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected (T,4,4) c2w in {npz_path}, got {c2w.shape}")
    return c2w


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=int(fps), quality=8, macro_block_size=1)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is not empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def render_rgb_depth(args: argparse.Namespace) -> dict[str, object]:
    ws = args.ws.expanduser().resolve()
    saved_dir = args.saved_dir.expanduser().resolve()
    cfg_path = (
        Path(infer_cfg_path(str(saved_dir), None)).resolve()
        if args.cfg is None
        else args.cfg.expanduser().resolve()
    )
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif args.camera_npz is None:
        output_dir = saved_dir / "input_camera_rgb_depth_spectral_tracks"
    else:
        output_dir = saved_dir / f"{args.camera_name}_rgb_depth_spectral_tracks"
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    device = torch.device(args.device)
    _, d_model, s_model, cams = load_model_cfg(str(cfg_path), str(saved_dir), device=device)
    height = scalar_int(cams.default_H)
    width = scalar_int(cams.default_W)
    num_frames = int(cams.T)

    external_c2w = None
    if args.camera_npz is not None:
        external_c2w = load_external_c2w(args.camera_npz.expanduser().resolve(), args.camera_name)
        if external_c2w.shape[0] != num_frames:
            raise ValueError(
                f"External camera trajectory has {external_c2w.shape[0]} poses, but the "
                f"checkpoint has T={num_frames}. They must match (frame i renders scene time i)."
            )

    def pose_t_cw(time_id: int) -> torch.Tensor:
        if external_c2w is None:
            return cams.T_cw(time_id)
        t_cw = np.linalg.inv(external_c2w[time_id]).astype(np.float32)
        return torch.from_numpy(t_cw).to(device)

    rgb_dir = output_dir / "rgb" / args.camera_name
    depth_dir = output_dir / "depth_spectral" / args.camera_name
    raw_depth_dir = output_dir / "depth_raw_npz" / args.camera_name
    camera_dir = output_dir / "cameras"
    for path in (rgb_dir, depth_dir, raw_depth_dir, camera_dir):
        path.mkdir(parents=True, exist_ok=True)

    rgb_frames: list[np.ndarray] = []
    depth_maps: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    c2w_list: list[np.ndarray] = []

    pose_desc = "input camera" if external_c2w is None else f"{args.camera_name}"
    for time_id in tqdm(range(num_frames), desc=f"{pose_desc} rgb/depth"):
        t_cw = pose_t_cw(time_id)
        render_dict = render(
            [s_model(), d_model(time_id)],
            height,
            width,
            K=cams.K(height, width),
            T_cw=t_cw,
            bg_color=[0.0, 0.0, 0.0],
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_u8 = (rgb.detach().cpu().numpy() * 255.0).astype(np.uint8)
        depth = render_dict["dep"].detach().cpu().numpy().squeeze(0).astype(np.float32)
        alpha = render_dict["alpha"].detach().cpu().numpy().squeeze(0).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0.0) & (alpha > float(args.depth_alpha_threshold))

        frame_name = f"{time_id:05d}"
        imageio.imwrite(rgb_dir / f"{frame_name}.png", rgb_u8)
        np.savez_compressed(raw_depth_dir / f"{frame_name}.npz", depth=depth, valid=valid)

        rgb_frames.append(rgb_u8)
        depth_maps.append(depth)
        valid_masks.append(valid)
        c2w = torch.linalg.inv(t_cw).detach().cpu().numpy().astype(np.float32)
        c2w_list.append(c2w)

    valid_depths = [depth[valid] for depth, valid in zip(depth_maps, valid_masks) if valid.any()]
    if not valid_depths:
        raise RuntimeError("No valid rendered depth pixels found.")
    valid_depth = np.concatenate(valid_depths, axis=0)
    depth_min = float(valid_depth.min())
    depth_max = float(valid_depth.max())
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
        raise RuntimeError(f"Invalid depth range: min={depth_min}, max={depth_max}")

    cmap = colormaps.get_cmap(str(args.depth_colormap))
    depth_frames: list[np.ndarray] = []
    for time_id, (depth, valid) in enumerate(zip(depth_maps, valid_masks)):
        norm = np.zeros_like(depth, dtype=np.float32)
        norm[valid] = np.clip((depth[valid] - depth_min) / (depth_max - depth_min), 0.0, 1.0)
        depth_rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
        depth_rgb[~valid] = 0
        frame_name = f"{time_id:05d}"
        imageio.imwrite(depth_dir / f"{frame_name}.png", depth_rgb)
        depth_frames.append(depth_rgb)

    c2w = np.stack(c2w_list, axis=0).astype(np.float32)
    camera_npz = camera_dir / f"{args.camera_name}_c2w.npz"
    np.savez_compressed(camera_npz, c2w=c2w, **{f"{args.camera_name}_c2w": c2w})

    rgb_video = output_dir / f"{args.camera_name}_rgb.mp4"
    depth_video = output_dir / f"{args.camera_name}_depth_spectral.mp4"
    write_video(rgb_frames, rgb_video, int(args.fps))
    write_video(depth_frames, depth_video, int(args.fps))

    metadata: dict[str, object] = {
        "workspace": str(ws),
        "saved_dir": str(saved_dir),
        "cfg": str(cfg_path),
        "camera_name": str(args.camera_name),
        "pose_source": (
            "input_camera"
            if external_c2w is None
            else str(args.camera_npz.expanduser().resolve())
        ),
        "num_frames": num_frames,
        "fps": int(args.fps),
        "depth_colormap": str(args.depth_colormap),
        "depth_alpha_threshold": float(args.depth_alpha_threshold),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "rgb_dir": str(rgb_dir),
        "depth_dir": str(depth_dir),
        "raw_depth_dir": str(raw_depth_dir),
        "rgb_video": str(rgb_video),
        "depth_video": str(depth_video),
        "camera_npz": str(camera_npz),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def render_track_overlay(args: argparse.Namespace, metadata: dict[str, object]) -> Path:
    output_dir = Path(str(metadata["rgb_dir"])).parents[1]
    track_root = output_dir / "track_overlay"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "wild" / "render_track_overlay.py"),
        "--ws",
        str(args.ws.expanduser().resolve()),
        "--saved-dir",
        str(args.saved_dir.expanduser().resolve()),
        "--cfg",
        str(metadata["cfg"]),
        "--output-root",
        str(track_root),
        "--camera",
        str(args.camera_name),
        str(metadata["camera_npz"]),
        str(metadata["rgb_dir"]),
        "--num-tracks",
        str(args.num_tracks),
        "--line-steps",
        str(args.line_steps),
        "--track-radius",
        str(args.track_radius),
        "--track-opacity",
        str(args.track_opacity),
        "--overlay-alpha",
        str(args.track_overlay_alpha),
        "--video-fps",
        str(args.fps),
        "--device",
        str(args.device),
        "--overwrite",
    ]
    print("Track overlay command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    track_video = track_root / f"{args.camera_name}_track_overlay.mp4"
    track_overlay_rgb_video = output_dir / f"{args.camera_name}_track_overlay_rgb.mp4"
    shutil.copy2(track_video, track_overlay_rgb_video)
    return track_overlay_rgb_video


def main() -> None:
    args = parse_args()
    metadata = render_rgb_depth(args)
    if not args.skip_tracks:
        metadata["track_overlay_rgb_video"] = str(render_track_overlay(args, metadata))
        output_dir = Path(str(metadata["rgb_dir"])).parents[1]
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"rgb_video={metadata['rgb_video']}")
    print(f"depth_video={metadata['depth_video']}")
    if "track_overlay_rgb_video" in metadata:
        print(f"track_overlay_rgb_video={metadata['track_overlay_rgb_video']}")
    print(f"camera_npz={metadata['camera_npz']}")


if __name__ == "__main__":
    main()
