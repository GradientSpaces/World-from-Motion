#!/usr/bin/env python3
"""Build the multiview manifest for continued (resumed) recon with a new traj.

This writes, under ``{ws}/continue_newtraj/``:
  - ``cameras/cam0_c2w.npz``      : anchor reference c2w [T,4,4] from the recon cameras
  - ``cameras/cam_newtraj_c2w.npz``: the new-trajectory c2w [N,4,4] (copied)
  - ``multiview_manifest.json``   : anchor cam0 + sparse aux cam_newtraj

The aux camera ``cam_newtraj`` covers only a subset of the anchor timeline. Its
``frame_time_ids`` map each new-traj frame to the recon time index, derived by
matching the new-traj frame stems against the anchor (original) frame names.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMG_EXTS = (".png", ".jpg", ".jpeg")


def list_stems(directory: Path, exts=IMG_EXTS) -> list[str]:
    fns = sorted(f for f in os.listdir(directory) if f.lower().endswith(exts))
    return [osp.splitext(f)[0] for f in fns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ws", required=True, type=Path, help="Workspace root (original recon ws)."
    )
    parser.add_argument(
        "--cam-ckpt",
        required=True,
        type=Path,
        help="photometric_cam.pth of the original recon (anchor poses + intrinsics).",
    )
    parser.add_argument(
        "--anchor-rgb-dirname",
        default="images",
        help="Anchor RGB dir (relative to ws); defines the full timeline / frame names.",
    )
    parser.add_argument(
        "--newtraj-pose-npz",
        required=True,
        type=Path,
        help="New-trajectory c2w npz with key 'c2w' [N,4,4].",
    )
    parser.add_argument(
        "--newtraj-frames-dir",
        required=True,
        type=Path,
        help="New-trajectory RGB frames dir (used for rgb_dir + time mapping).",
    )
    parser.add_argument(
        "--newtraj-depth-dir",
        required=True,
        type=Path,
        help="New-trajectory DA3 depth dir (the 'depthanything3_depth' folder).",
    )
    parser.add_argument("--anchor-camera-name", default="cam0")
    parser.add_argument("--newtraj-camera-name", default="cam_newtraj")
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--out-subdir", default="continue_newtraj")
    args = parser.parse_args()

    import torch

    from lib_moca.camera import MonocularCameras

    ws = args.ws.expanduser().resolve()
    out_root = ws / args.out_subdir
    cameras_dir = out_root / "cameras"
    cameras_dir.mkdir(parents=True, exist_ok=True)

    # Anchor timeline / frame names from the original RGB dir.
    anchor_rgb_dir = ws / args.anchor_rgb_dirname
    anchor_stems = list_stems(anchor_rgb_dir)
    T = len(anchor_stems)
    if T == 0:
        raise FileNotFoundError(f"No anchor frames under {anchor_rgb_dir}")
    stem_to_time = {stem: i for i, stem in enumerate(anchor_stems)}

    # Anchor reference c2w for the full timeline, generated from the recon cameras.
    cams = MonocularCameras.load_from_ckpt(
        torch.load(args.cam_ckpt.expanduser().resolve(), map_location="cpu")
    )
    if int(cams.T) != T:
        raise ValueError(
            f"Camera T ({int(cams.T)}) != anchor frame count ({T}); "
            "anchor RGB dir must match the recon timeline."
        )
    anchor_c2w = np.stack(
        [cams.T_wc(t).detach().cpu().numpy().astype(np.float32) for t in range(T)],
        axis=0,
    )
    anchor_c2w_rel = osp.join(args.out_subdir, "cameras", f"{args.anchor_camera_name}_c2w.npz")
    np.savez_compressed(ws / anchor_c2w_rel, c2w=anchor_c2w)

    # New-trajectory frame time ids (subset of the anchor timeline).
    newtraj_frames_dir = args.newtraj_frames_dir.expanduser().resolve()
    newtraj_stems = list_stems(newtraj_frames_dir)
    n_newtraj = len(newtraj_stems)
    if n_newtraj == 0:
        raise FileNotFoundError(f"No new-traj frames under {newtraj_frames_dir}")
    missing = [s for s in newtraj_stems if s not in stem_to_time]
    if missing:
        raise KeyError(
            f"{len(missing)} new-traj frame stems not found in anchor timeline "
            f"(e.g. {missing[:3]}); cannot map to recon time indices."
        )
    frame_time_ids = [stem_to_time[s] for s in newtraj_stems]

    # New-trajectory reference c2w (copied; must be ordered like the sorted frames).
    newtraj_c2w = np.load(args.newtraj_pose_npz.expanduser().resolve())["c2w"].astype(
        np.float32
    )
    if newtraj_c2w.shape[0] != n_newtraj:
        raise ValueError(
            f"New-traj pose count ({newtraj_c2w.shape[0]}) != frame count "
            f"({n_newtraj}); poses and frames must correspond 1:1 in sorted order."
        )
    newtraj_c2w_rel = osp.join(
        args.out_subdir, "cameras", f"{args.newtraj_camera_name}_c2w.npz"
    )
    np.savez_compressed(ws / newtraj_c2w_rel, c2w=newtraj_c2w)

    # Depth dir relative to ws (fall back to absolute if outside ws).
    depth_dir = args.newtraj_depth_dir.expanduser().resolve()

    def rel_to_ws(path: Path) -> str:
        try:
            return osp.relpath(path, ws)
        except ValueError:
            return str(path)

    manifest = {
        "anchor_camera": args.anchor_camera_name,
        "anchor_rgb_camera": args.anchor_camera_name,
        "override_anchor_rgb": False,
        "depth_npz_key": "dep",
        "cameras": [
            {
                "camera_name": args.anchor_camera_name,
                "rgb_dir": args.anchor_rgb_dirname,
                "reference_c2w_path": anchor_c2w_rel,
                "weight": float(args.anchor_weight),
            },
            {
                "camera_name": args.newtraj_camera_name,
                "rgb_dir": rel_to_ws(newtraj_frames_dir),
                "reference_c2w_path": newtraj_c2w_rel,
                "depth_dir": rel_to_ws(depth_dir),
                "depth_npz_key": "dep",
                "frame_time_ids": frame_time_ids,
                "weight": float(args.aux_weight),
            },
        ],
    }
    manifest_path = out_root / "multiview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote manifest: {manifest_path}")
    print(f"  anchor '{args.anchor_camera_name}': {T} frames (dense)")
    print(
        f"  aux '{args.newtraj_camera_name}': {n_newtraj} frames, "
        f"time ids [{frame_time_ids[0]}..{frame_time_ids[-1]}] "
        f"(stride ~{frame_time_ids[1] - frame_time_ids[0] if n_newtraj > 1 else 1})"
    )


if __name__ == "__main__":
    main()
