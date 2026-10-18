#!/usr/bin/env python3
"""Build a DA3-style init-recon .npy from known new-trajectory poses.

The wild pipeline (run_wild_video_stage12.py) reads camera init from
``da3_recon_path`` (see lite_moca_reconstruct.load_da3_camera): a pickled dict
``{"traj_c2w": [T,4,4], "intrinsics": [fx, fy, cx, cy]}``. This script writes that
file from the new-trajectory c2w poses plus intrinsics borrowed from the original
reconstruction's recon .npy, so the fixed-camera fit uses the known poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_intrinsics(src: Path) -> np.ndarray:
    r = np.load(src, allow_pickle=True)
    if isinstance(r, np.ndarray) and r.dtype == object:
        r = r.item()
    intr = np.asarray(r["intrinsics"], dtype=np.float32)
    if intr.ndim == 2:
        intr = intr.mean(axis=0)
    if intr.shape != (4,):
        raise ValueError(f"Expected intrinsics [fx,fy,cx,cy], got {intr.shape} from {src}")
    return intr


def frames_wh(video: Path) -> tuple[int, int]:
    from PIL import Image

    pngs = sorted(video.glob("*.png")) + sorted(video.glob("*.jpg"))
    if not pngs:
        raise FileNotFoundError(f"No frames under {video}")
    return Image.open(pngs[0]).size  # (W, H)


def scale_intrinsics(intr: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    return np.array([intr[0] * sx, intr[1] * sy, intr[2] * sx, intr[3] * sy], dtype=np.float32)


def load_poses(pose_npz: Path) -> np.ndarray:
    z = np.load(pose_npz, allow_pickle=True)
    key = "c2w" if "c2w" in z else ("traj_c2w" if "traj_c2w" in z else list(z.keys())[0])
    c2w = np.asarray(z[key], dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] not in [(3, 4), (4, 4)]:
        raise ValueError(f"Invalid c2w shape {c2w.shape} in {pose_npz}")
    if c2w.shape[1:] == (3, 4):
        full = np.tile(np.eye(4, dtype=np.float32)[None], (len(c2w), 1, 1))
        full[:, :3, :] = c2w
        c2w = full
    return c2w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-npz", required=True, type=Path)
    p.add_argument("--intrinsics-src", required=True, type=Path,
                   help="recon .npy with an 'intrinsics' [fx,fy,cx,cy] entry.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--expect-frames", type=int, default=None)
    p.add_argument("--src-wh", type=int, nargs=2, default=None,
                   help="Resolution the intrinsics were defined at, as W H. If the "
                        "target frames differ, K is scaled accordingly.")
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Frames dir to read target resolution from (defaults to no scaling).")
    args = p.parse_args()

    c2w = load_poses(args.pose_npz)
    intr = load_intrinsics(args.intrinsics_src)
    if args.src_wh is not None and args.frames_dir is not None:
        dst_wh = frames_wh(args.frames_dir)
        if tuple(args.src_wh) != dst_wh:
            old = intr.tolist()
            intr = scale_intrinsics(intr, tuple(args.src_wh), dst_wh)
            print(f"scaled K from src {tuple(args.src_wh)} -> frames {dst_wh}: {old} -> {intr.tolist()}")
    if args.expect_frames is not None and len(c2w) != args.expect_frames:
        raise ValueError(f"pose count {len(c2w)} != expected {args.expect_frames}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, {"traj_c2w": c2w, "intrinsics": intr}, allow_pickle=True)
    print(f"OK wrote {args.out} traj_c2w={c2w.shape} intrinsics={intr.tolist()}")


if __name__ == "__main__":
    main()
