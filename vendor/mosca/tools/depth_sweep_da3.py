#!/usr/bin/env python3
"""Lyra-style depth-sweep effect on a single image, using DepthAnything V3 depth.

Runs DepthAnything V3 on one RGB image to get a depth map, then animates a glowing
light "plane" sweeping near->far (the slice at the current depth is brightened while
the rest stays dim), with clean-RGB holds before/after. Writes the sweep frames and
an mp4. Mirrors the sweep synthesis in tools/render_depth_sweep.py but sources depth
from DA3 instead of a recon's rendered depth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path, help="Input RGB frame (frame A).")
    p.add_argument("--image-b", default=None, type=Path,
                   help="Optional second frame (frame B). If given, run a depth-WIPE "
                        "transition A->B (to mask a small inconsistency at a cut) instead "
                        "of a single-frame reveal. DA3 depth is taken from frame A "
                        "(or frame B with --depth-from-b).")
    p.add_argument("--depth-from-b", action="store_true",
                   help="In transition mode, take the DA3 depth from frame B (the revealed "
                        "frame, e.g. the first frame of the next segment) instead of frame A.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output dir; frames -> <output-dir>/<camera-name>/, mp4 alongside.")
    p.add_argument("--camera-name", default="da3_sweep")
    p.add_argument("--weights", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sweep-frames", type=int, default=24)
    p.add_argument("--pre-hold-frames", type=int, default=6)
    p.add_argument("--post-hold-frames", type=int, default=6)
    p.add_argument("--direction", choices=("near2far", "far2near"), default="near2far")
    p.add_argument("--sigma-frac", type=float, default=0.08)
    p.add_argument("--ambient", type=float, default=0.55)
    p.add_argument("--glow", type=float, default=0.3)
    p.add_argument("--percentile-lo", type=float, default=2.0)
    p.add_argument("--percentile-hi", type=float, default=98.0)
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def da3_depth(img_rgb_u8: np.ndarray, weights: str, device: str) -> np.ndarray:
    """Run DepthAnything V3 on a single image; return a [H,W] depth map."""
    from lib_prior.depth_models.depthanything3_wrapper import get_depthanything3_model
    model = get_depthanything3_model(device=device, weights=weights)
    h, w = img_rgb_u8.shape[:2]
    nh, nw = (h // 14) * 14, (w // 14) * 14
    work = cv2.resize(img_rgb_u8, (nw, nh))
    pred = model.inference(image=[work], extrinsics=None, intrinsics=None,
                           align_to_input_ext_scale=False)
    dep = np.asarray(pred.depth[0], dtype=np.float32)
    return cv2.resize(dep, (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.float32)


def synthesize_sweep(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                     a: argparse.Namespace) -> list[np.ndarray]:
    """Near->far light-band sweep (matches tools/render_depth_sweep.py)."""
    if not valid.any():
        raise RuntimeError("No valid DA3 depth pixels for the sweep frame.")
    vals = depth[valid]
    d_near = float(np.percentile(vals, a.percentile_lo))
    d_far = float(np.percentile(vals, a.percentile_hi))
    if not np.isfinite(d_near) or not np.isfinite(d_far) or d_far <= d_near:
        d_near, d_far = float(vals.min()), float(vals.max())
    if d_far <= d_near:
        d_far = d_near + 1e-3
    sigma = max(float(a.sigma_frac) * (d_far - d_near), 1e-6)
    clean_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    k = int(a.sweep_frames)
    if k <= 0:
        fracs = np.array([])
    elif k == 1:
        fracs = np.array([0.5])
    else:
        fracs = np.linspace(0.0, 1.0, k)
    if a.direction == "far2near":
        fracs = fracs[::-1]

    frames: list[np.ndarray] = [clean_u8] * max(int(a.pre_hold_frames), 0)
    for frac in fracs:
        s = d_near + float(frac) * (d_far - d_near)
        w = np.exp(-((depth - s) ** 2) / (2.0 * sigma * sigma))
        w[~valid] = 0.0
        brightness = float(a.ambient) + (1.0 - float(a.ambient)) * w
        out = rgb * brightness[..., None]
        if a.glow > 0.0:
            out = out + float(a.glow) * w[..., None]
        frames.append((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8))
    frames.extend([clean_u8] * max(int(a.post_hold_frames), 0))
    return frames


def synthesize_transition(rgb_a: np.ndarray, rgb_b: np.ndarray, depth: np.ndarray,
                          valid: np.ndarray, a: argparse.Namespace) -> list[np.ndarray]:
    """Depth wipe A->B near->far with a glowing boundary band (full brightness)."""
    if not valid.any():
        raise RuntimeError("No valid DA3 depth pixels for the transition frame.")
    vals = depth[valid]
    d_near = float(np.percentile(vals, a.percentile_lo))
    d_far = float(np.percentile(vals, a.percentile_hi))
    if not np.isfinite(d_near) or not np.isfinite(d_far) or d_far <= d_near:
        d_near, d_far = float(vals.min()), float(vals.max())
    if d_far <= d_near:
        d_far = d_near + 1e-3
    sigma = max(float(a.sigma_frac) * (d_far - d_near), 1e-6)
    # Invalid-depth pixels swap last (treated as far).
    depth_filled = np.where(valid, depth, d_far).astype(np.float32)
    clean_a = (np.clip(rgb_a, 0.0, 1.0) * 255.0).astype(np.uint8)
    clean_b = (np.clip(rgb_b, 0.0, 1.0) * 255.0).astype(np.uint8)

    k = max(int(a.sweep_frames), 1)
    fracs = np.linspace(0.0, 1.0, k)
    if a.direction == "far2near":
        fracs = fracs[::-1]
    frames: list[np.ndarray] = [clean_a] * max(int(a.pre_hold_frames), 0)
    for frac in fracs:
        s = d_near + float(frac) * (d_far - d_near)
        swept = (depth_filled < s)[..., None]
        out = np.where(swept, rgb_b, rgb_a)
        w = np.exp(-((depth - s) ** 2) / (2.0 * sigma * sigma))
        w[~valid] = 0.0
        if a.glow > 0.0:
            out = out + float(a.glow) * w[..., None]
        frames.append((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8))
    frames.extend([clean_b] * max(int(a.post_hold_frames), 0))
    return frames


def main() -> None:
    a = parse_args()
    img = imageio.imread(a.image.expanduser().resolve())
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = np.asarray(img)[..., :3]
    rgb = img.astype(np.float32) / 255.0
    if a.image_b is not None:
        imgb = imageio.imread(a.image_b.expanduser().resolve())
        if imgb.ndim == 2:
            imgb = np.stack([imgb] * 3, axis=-1)
        imgb = np.asarray(imgb)[..., :3]
        if imgb.shape[:2] != img.shape[:2]:
            imgb = cv2.resize(imgb, (img.shape[1], img.shape[0]))
        # Depth drives the wipe ordering; take it from the revealed frame (B) when asked.
        depth = da3_depth(imgb if a.depth_from_b else img, a.weights, a.device)
        valid = np.isfinite(depth) & (depth > 0.0)
        frames = synthesize_transition(rgb, imgb.astype(np.float32) / 255.0, depth, valid, a)
    else:
        depth = da3_depth(img, a.weights, a.device)
        valid = np.isfinite(depth) & (depth > 0.0)
        frames = synthesize_sweep(rgb, depth, valid, a)

    out_dir = a.output_dir.expanduser().resolve()
    cam_dir = out_dir / a.camera_name
    cam_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        imageio.imwrite(cam_dir / f"{i:05d}.png", f)
    mp4 = out_dir / f"{a.camera_name}.mp4"
    writer = imageio.get_writer(mp4, fps=int(a.fps), macro_block_size=None)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()
    print(f"OK DA3 depth sweep: {len(frames)} frames @ {a.fps} fps -> {mp4}")


if __name__ == "__main__":
    main()
