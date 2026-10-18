#!/usr/bin/env python3
"""Render a bullet-time clip from the ORIGINAL reconstruction of each release scene.

For a release scene, this freezes the dynamic scene at the last time step of the
diffsynth trajectory and sweeps the camera from the input camera pose to the
diffsynth (new-trajectory) camera pose, both expressed in the original-recon world
frame. This is the "bullet time" look (fixed scene time, moving camera), driven by
the two real benchmark cameras rather than an auto orbit.

The inputs come from ``data/release_ws/original/<scene>/meta.json``:
  - ``orig_saved_dir``           : the original recon log dir (photometric_*.pth).
  - ``new_traj_camera``          : the diffsynth target camera name (e.g. cam_ui).
  - ``new_traj_sampled_pose_npz``: the sampled new trajectory; its parent
                                   ``cameras/`` dir also holds ``cam0_c2w.npz``
                                   (the input camera) in the original-recon frame.

The frozen scene time is read from ``metadata/<cam>_frames.json`` (the last frame's
``time_id``, e.g. 144 for first73-stride2 benchmarks).

Unlike the existing ``camera_tour_v2/segB`` (cam_bullet) renders, this renders from
the ORIGINAL reconstruction (not the refined reopt model) and freezes at the
original-recon scene time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mosca_camera_ui import load_scene, resolve_device  # noqa: E402
from render_camera_ui_modes import render_rgb_frame  # noqa: E402
from build_camera_tour_segments import interp_c2w  # noqa: E402

BG_COLORS = {"white": [1.0, 1.0, 1.0], "black": [0.0, 0.0, 0.0]}
DEFAULT_SCENES = ("anymal", "astronaut", "ball", "person", "piano", "solo", "street", "tower")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scene",
        nargs="+",
        required=True,
        help="One or more release scene names (e.g. tower ball).",
    )
    parser.add_argument(
        "--original-root",
        "--original_root",
        default=Path("data/release_ws/original"),
        type=Path,
        help="Release root holding <scene>/meta.json.",
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        default=Path("data/release_ws/videos"),
        type=Path,
        help="Per-scene mp4s are written to <output-root>/<scene>/.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=60,
        help="Number of interpolated camera poses from input -> diffsynth.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bg", choices=list(BG_COLORS), default="white")
    parser.add_argument("--backend", default="native_add3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-fast-dynamic",
        action="store_true",
        help="Skip DynSCFGaussian.set_inference_mode() (slower, matches raw forward).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate meta/cameras/time without loading the model or rendering, then exit.",
    )
    return parser.parse_args()


def load_c2w(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        files = list(data.files)
        key = "c2w" if "c2w" in files else None
        if key is None:
            for name in files:
                arr = data[name]
                if getattr(arr, "ndim", 0) == 3 and arr.shape[1:] == (4, 4):
                    key = name
                    break
        if key is None:
            raise KeyError(f"No [T,4,4] c2w array found in {npz_path}; keys={files}")
        c2w = np.asarray(data[key], dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected c2w shape {c2w.shape} in {npz_path}")
    return c2w


def read_frozen_time(frames_json: Path, fallback: int) -> int:
    """Last frame's time_id from a benchmark *_frames.json, else fallback."""
    if not frames_json.is_file():
        logging.warning("Frames metadata missing (%s); using fallback time_id=%d",
                        frames_json, fallback)
        return int(fallback)
    payload = json.loads(frames_json.read_text())
    records = payload if isinstance(payload, list) else payload.get("frames", payload)
    last = records[-1]
    return int(last["time_id"])


def resolve_scene_paths(original_root: Path, scene: str) -> dict:
    meta_path = original_root / scene / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"[{scene}] meta.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    orig_saved_dir = (REPO_ROOT / meta["orig_saved_dir"]).resolve()
    cam_name = meta["new_traj_camera"]
    sampled_npz = (REPO_ROOT / meta["new_traj_sampled_pose_npz"]).resolve()
    cameras_dir = sampled_npz.parent
    bench_scene_dir = cameras_dir.parent

    input_npz = cameras_dir / "cam0_c2w.npz"
    diffsynth_npz = cameras_dir / f"{cam_name}_c2w.npz"
    frames_json = bench_scene_dir / "metadata" / f"{cam_name}_frames.json"

    for path in (orig_saved_dir, input_npz, diffsynth_npz):
        if not path.exists():
            raise FileNotFoundError(f"[{scene}] required path missing: {path}")

    num_diffsynth = int(meta.get("num_diffsynth_frames", 0) or 0)
    fallback_time = 2 * (num_diffsynth - 1) if num_diffsynth > 0 else 0
    frozen_time = read_frozen_time(frames_json, fallback_time)

    return {
        "scene": scene,
        "orig_saved_dir": orig_saved_dir,
        "cam_name": cam_name,
        "input_npz": input_npz,
        "diffsynth_npz": diffsynth_npz,
        "frozen_time": frozen_time,
    }


def build_bullet_poses(input_c2w: np.ndarray, diffsynth_c2w: np.ndarray,
                       num_frames: int) -> np.ndarray:
    """Interpolated camera-to-world poses from the input pose to the diffsynth pose."""
    a = input_c2w[-1]
    b = diffsynth_c2w[-1]
    if num_frames < 2:
        return np.stack([b], 0)
    return np.stack(
        [interp_c2w(a, b, j / (num_frames - 1)) for j in range(num_frames)], 0
    )


def render_scene(scene: str, args) -> Path:
    info = resolve_scene_paths(args.original_root, scene)
    input_c2w = load_c2w(info["input_npz"])
    diffsynth_c2w = load_c2w(info["diffsynth_npz"])
    frozen_time = info["frozen_time"]
    poses = build_bullet_poses(input_c2w, diffsynth_c2w, args.num_frames)

    out_dir = args.output_root / scene
    out_path = out_dir / f"{scene}_bullet_input2diffsynth.mp4"

    logging.info(
        "[%s] orig_saved_dir=%s cam=%s frozen_time=%d input_frames=%d diffsynth_frames=%d "
        "render_frames=%d -> %s",
        scene, info["orig_saved_dir"], info["cam_name"], frozen_time,
        input_c2w.shape[0], diffsynth_c2w.shape[0], poses.shape[0], out_path,
    )

    if args.dry_run:
        logging.info(
            "[dry-run] %s: input_last_t=%s diffsynth_last_t=%s pose_delta_max=%.4f",
            scene,
            np.round(input_c2w[-1][:3, 3], 4).tolist(),
            np.round(diffsynth_c2w[-1][:3, 3], 4).tolist(),
            float(np.abs(input_c2w[-1] - diffsynth_c2w[-1]).max()),
        )
        return out_path

    if out_path.exists() and not args.overwrite:
        logging.warning("[%s] skipping existing %s (pass --overwrite)", scene, out_path)
        return out_path

    device = resolve_device(args.device)
    scene_obj = load_scene(
        info["orig_saved_dir"], args.backend, device, fast_dynamic=not args.no_fast_dynamic
    )
    model_t = min(int(scene_obj.cams.T), int(scene_obj.d_model.T))
    if frozen_time >= model_t:
        raise ValueError(
            f"[{scene}] frozen_time {frozen_time} out of range for recon T={model_t}"
        )

    bg_color = BG_COLORS[args.bg]
    frames = [
        render_rgb_frame(scene_obj, frozen_time, poses[i], bg_color)
        for i in range(poses.shape[0])
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.fps, quality=8, macro_block_size=1)
    logging.info("[%s] wrote %s (%d frames @ %d fps)", scene, out_path, len(frames), args.fps)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    for scene in args.scene:
        render_scene(scene, args)
    logging.info("Done.")


if __name__ == "__main__":
    main()
