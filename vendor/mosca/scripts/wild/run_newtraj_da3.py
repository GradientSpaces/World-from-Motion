#!/usr/bin/env python3
"""Run pose-conditioned DepthAnything3 on a new-trajectory (DiffSynth) clip.

The new-trajectory poses live in the original reconstruction's world frame, and
the intrinsics are inherited from the original photometric camera. We therefore
feed DA3 the per-frame world-to-camera extrinsics and the recon intrinsics so
its depth is aligned (up to a global similarity) to the recon scale.

Outputs depth npz files (key ``dep``) under ``{out_dir}/depthanything3_depth/``
in the layout consumed by ``MultiViewObservations`` / ``Saved2D.load_dep``.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMG_EXTS = (".png", ".jpg", ".jpeg")


def load_frames(frames_dir: Path):
    import imageio.v2 as imageio

    fns = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS))
    if not fns:
        raise FileNotFoundError(f"No image frames found under {frames_dir}")
    img_list = []
    for fn in fns:
        img = imageio.imread(osp.join(frames_dir, fn))
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        img_list.append(np.asarray(img)[..., :3])
    return img_list, fns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cam-ckpt",
        required=True,
        type=Path,
        help="photometric_cam.pth of the original recon (source of intrinsics).",
    )
    parser.add_argument(
        "--pose-npz",
        required=True,
        type=Path,
        help="New-trajectory camera-to-world npz with key 'c2w' [N,4,4].",
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        type=Path,
        help="Directory of new-trajectory RGB frames (DiffSynth predictions).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Destination; depth lands under {out_dir}/depthanything3_depth/.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weights", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    )
    args = parser.parse_args()

    import torch

    from lib_moca.camera import MonocularCameras
    from lib_prior.depth_models.depthanything3_wrapper import (
        depthanything3_process_folder,
        get_depthanything3_model,
    )

    frames_dir = args.frames_dir.expanduser().resolve()
    img_list, fn_list = load_frames(frames_dir)
    H, W = img_list[0].shape[:2]
    n_frames = len(img_list)
    print(f"Loaded {n_frames} new-traj frames at {H}x{W} from {frames_dir}")

    c2w = np.load(args.pose_npz.expanduser().resolve())["c2w"].astype(np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w [N,4,4], got {c2w.shape}")
    if c2w.shape[0] != n_frames:
        raise ValueError(
            f"Pose count ({c2w.shape[0]}) != frame count ({n_frames}); "
            "the new-traj poses must correspond 1:1 with the frames."
        )
    w2c = np.linalg.inv(c2w).astype(np.float32)

    cams = MonocularCameras.load_from_ckpt(
        torch.load(args.cam_ckpt.expanduser().resolve(), map_location="cpu")
    )
    K = cams.K(H, W).detach().cpu().numpy().astype(np.float32)
    print(
        "Recon intrinsics at new-traj resolution: "
        f"fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}"
    )
    intrinsics = np.repeat(K[None], n_frames, axis=0)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_depthanything3_model(device=args.device, weights=args.weights)
    depthanything3_process_folder(
        model=model,
        img_list=img_list,
        fn_list=fn_list,
        dst=str(out_dir),
        extrinsics=w2c,
        intrinsics=intrinsics,
    )
    print(
        f"Wrote new-traj DA3 depth to {osp.join(str(out_dir), 'depthanything3_depth')}"
    )


if __name__ == "__main__":
    main()
