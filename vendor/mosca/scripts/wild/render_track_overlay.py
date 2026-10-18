#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import cm
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_mosca.photo_recon_viz_utils import BG_COLOR1
from lib_render.render_helper import render, render_cam_pcl
from lib_render.sh_utils import RGB2SH
from mosca_viz import load_model_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render no-text 3D dynamic track overlays.")
    parser.add_argument("--ws", required=True, type=Path)
    parser.add_argument("--saved-dir", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--camera", action="append", nargs=3, metavar=("NAME", "NPZ", "RGB_DIR"))
    parser.add_argument("--num-tracks", type=int, default=128)
    parser.add_argument("--line-steps", type=int, default=32)
    parser.add_argument("--track-radius", type=float, default=0.0015)
    parser.add_argument("--track-opacity", type=float, default=0.999)
    parser.add_argument("--overlay-alpha", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.camera:
        parser.error("At least one --camera NAME NPZ RGB_DIR entry is required.")
    return args


def list_images(image_dir: Path) -> list[Path]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return images


def load_c2w(npz_path: Path, camera_name: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if "c2w" in data.files:
            c2w = data["c2w"]
        else:
            key = f"{camera_name}_c2w"
            if key not in data.files:
                raise KeyError(f"Missing c2w or {key} in {npz_path}")
            c2w = data[key]
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected {npz_path} to contain (T, 4, 4) c2w, got {c2w.shape}")
    return c2w


def load_rgb_frames(rgb_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    paths = list_images(rgb_dir)
    names = [path.stem for path in paths]
    frames = [imageio.imread(path)[..., :3] for path in paths]
    return names, frames


def image_to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame[..., :3]
    arr = np.asarray(frame[..., :3], dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


@torch.no_grad()
def prepare_track_state(cams, s_model, d_model, num_tracks: int, line_radius: float, opacity: float, seed: int):
    device = d_model._xyz.device
    prev_mu, _, s_traj, o_traj, sph_traj = d_model(0, 0, nn_fusion=-1)
    s_traj = torch.ones_like(s_traj) * float(line_radius)
    o_traj = torch.ones_like(o_traj) * float(opacity)

    color_id_coord = d_model._xyz
    color_id = color_id_coord[:, 2] + color_id_coord[:, 1] * 1e6 + color_id_coord[:, 0] * 1e12
    color_id = color_id.argsort().float() / len(color_id)
    track_color = torch.from_numpy(cm.hsv(color_id.detach().cpu())[:, :3]).to(prev_mu)
    sph_traj = RGB2SH(track_color)

    render_dict = render(
        [s_model(0), d_model(0, 0, nn_fusion=-1)],
        cams.default_H,
        cams.default_W,
        cams.default_K,
        cams.T_cw(0),
    )
    visibility_mask = render_dict["visibility_filter"][-d_model.N :]
    valid_ind = torch.arange(len(visibility_mask), device=device)[visibility_mask]
    if len(valid_ind) == 0:
        raise RuntimeError("No visible dynamic Gaussians found at frame 0.")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    choice = valid_ind[torch.randperm(len(valid_ind), device=device, generator=generator)[: min(num_tracks, len(valid_ind))]]

    return {
        "choice": choice,
        "prev_mu": prev_mu[choice],
        "s_traj": s_traj[choice],
        "o_traj": o_traj[choice],
        "sph_traj": sph_traj[choice],
        "mu_w": torch.zeros(0, 3, device=device),
        "fr_w": torch.zeros(0, 3, 3, device=device),
        "scale": torch.zeros(0, 3, device=device),
        "opacity": torch.zeros(0, 1, device=device),
        "sph": torch.zeros(0, sph_traj.shape[-1], device=device),
    }


@torch.no_grad()
def advance_track_state(d_model, state: dict, time_id: int, line_steps: int, line_radius: float, opacity: float) -> dict:
    choice = state["choice"]
    d_mu_w, d_fr_w, _, _, _ = d_model(time_id, 0, nn_fusion=-1)
    src_mu = state["prev_mu"]
    dst_mu = d_mu_w[choice]
    line_dir = dst_mu - src_mu
    intermediate_mu = (
        src_mu[:, None]
        + torch.linspace(0, 1, int(line_steps), device=line_dir.device)[None, :, None] * line_dir[:, None]
    ).reshape(-1, 3)
    intermediate_fr = torch.eye(3, device=intermediate_mu.device)[None].expand(len(intermediate_mu), -1, -1)
    intermediate_scale = torch.ones_like(intermediate_mu) * float(line_radius) * 0.3
    intermediate_opacity = torch.ones_like(intermediate_mu[:, :1]) * float(opacity)
    intermediate_sph = (
        state["sph_traj"][:, None]
        .expand(-1, int(line_steps), -1)
        .reshape(-1, state["sph_traj"].shape[-1])
    )

    state["prev_mu"] = dst_mu
    state["mu_w"] = torch.cat([state["mu_w"], intermediate_mu.clone(), dst_mu.clone()], 0)
    state["fr_w"] = torch.cat([state["fr_w"], intermediate_fr.clone(), d_fr_w[choice].clone()], 0)
    state["scale"] = torch.cat([state["scale"], intermediate_scale.clone(), state["s_traj"].clone()], 0)
    state["opacity"] = torch.cat([state["opacity"], intermediate_opacity.clone(), state["o_traj"].clone()], 0)
    state["sph"] = torch.cat([state["sph"], intermediate_sph.clone(), state["sph_traj"].clone()], 0)
    return state


@torch.no_grad()
def render_track_layer(cams, state: dict, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = state["mu_w"].device
    t_cw = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)
    r_cw = t_cw[:3, :3]
    t_cw_vec = t_cw[:3, 3]
    mu_cur = torch.einsum("ij,nj->ni", r_cw, state["mu_w"].contiguous()) + t_cw_vec[None]
    fr_cur = torch.einsum("ij,njk->nik", r_cw, state["fr_w"].contiguous())
    render_dict = render_cam_pcl(
        mu_cur.contiguous(),
        fr_cur.contiguous(),
        state["scale"].contiguous(),
        state["opacity"].contiguous(),
        state["sph"].contiguous(),
        int(cams.default_H),
        int(cams.default_W),
        CAM_K=cams.default_K,
        bg_color=BG_COLOR1,
    )
    rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
    alpha = render_dict.get("alpha")
    if alpha is None:
        alpha_np = (np.abs(rgb - np.asarray(BG_COLOR1, dtype=np.float32)).max(axis=-1, keepdims=True) > 1e-3).astype(np.float32)
    else:
        alpha_np = alpha.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(rgb, 0.0, 1.0), np.clip(alpha_np, 0.0, 1.0)


def composite(base_u8: np.ndarray, track_rgb: np.ndarray, track_alpha: np.ndarray, alpha_scale: float) -> np.ndarray:
    base = image_to_uint8(base_u8).astype(np.float32) / 255.0
    alpha = np.clip(track_alpha * float(alpha_scale), 0.0, 1.0)
    out = track_rgb * alpha + base * (1.0 - alpha)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=int(fps))


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output root exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg, d_model, s_model, cams = load_model_cfg(str(args.cfg.expanduser().resolve()), str(args.saved_dir.expanduser().resolve()), device=device)
    if d_model is None:
        raise RuntimeError("Track overlay requires a dynamic model.")

    camera_specs = []
    for camera_name, npz_path, rgb_dir in args.camera:
        npz_path = Path(npz_path).expanduser().resolve()
        rgb_dir = Path(rgb_dir).expanduser().resolve()
        frame_names, rgb_frames = load_rgb_frames(rgb_dir)
        c2w = load_c2w(npz_path, camera_name)
        if len(rgb_frames) != len(c2w):
            raise ValueError(f"{camera_name}: RGB count {len(rgb_frames)} != c2w count {len(c2w)}")
        if len(rgb_frames) != int(cams.T):
            raise ValueError(f"{camera_name}: RGB count {len(rgb_frames)} != checkpoint T {int(cams.T)}")
        camera_specs.append((camera_name, frame_names, rgb_frames, c2w))

    state = prepare_track_state(
        cams=cams,
        s_model=s_model,
        d_model=d_model,
        num_tracks=int(args.num_tracks),
        line_radius=float(args.track_radius),
        opacity=float(args.track_opacity),
        seed=int(args.seed),
    )

    outputs: dict[str, list[np.ndarray]] = {camera_name: [] for camera_name, *_ in camera_specs}
    for time_id in tqdm(range(int(cams.T)), desc="track overlay"):
        state = advance_track_state(
            d_model=d_model,
            state=state,
            time_id=time_id,
            line_steps=int(args.line_steps),
            line_radius=float(args.track_radius),
            opacity=float(args.track_opacity),
        )
        for camera_name, frame_names, rgb_frames, c2w in camera_specs:
            track_rgb, track_alpha = render_track_layer(cams, state, c2w[time_id])
            out = composite(rgb_frames[time_id], track_rgb, track_alpha, float(args.overlay_alpha))
            out_dir = output_root / camera_name
            out_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(out_dir / f"{frame_names[time_id]}.png", out)
            outputs[camera_name].append(out)

    if not args.skip_video:
        for camera_name, frames in outputs.items():
            save_video(frames, output_root / f"{camera_name}_track_overlay.mp4", fps=int(args.video_fps))

    print(f"Saved overlays under {output_root}")
    for camera_name, frames in outputs.items():
        print(f"{camera_name}: {len(frames)} frames")


if __name__ == "__main__":
    main()
