#!/usr/bin/env python3
"""Merge a reference camera (cam0) from one benchmark into another, with pose override.

For camera-tour v3 seg2: the TARGET (cam_bullet) is rendered from the refined recon,
but the REFERENCE (cam0 = the original input video) is rendered from the original
recon. This tool takes the original-recon "ref" benchmark's cam0 g-buffers and copies
them over the refined "target" benchmark's cam0, then overwrites cam0's recorded pose
(cameras/cam0_c2w.npz) with a provided refined-frame pose repeated for all frames so
DiffSynth's camera embedding (--split-cam-embed) stays consistent with the target.

Only cam0 is touched; the target camera (e.g. cam_bullet) and eval metadata are left
intact. Per-camera frame names match (both scripted 00000..) so the target's cam0
manifest/frame-name lists remain valid.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_moca.camera import MonocularCameras  # noqa: E402

# Per-camera modality dirs that may hold cam0 g-buffers (copied if present in ref).
MODALITIES = ("gt", "rgb", "depth", "normal", "alpha",
              "scene_flow_rgb", "scene_flow", "scene_coords")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-scene-root", required=True, type=Path,
                   help="Refined target benchmark scene root (<...>/segB_benchmark/<scene>).")
    p.add_argument("--ref-scene-root", required=True, type=Path,
                   help="Original-recon reference benchmark scene root (<...>/segB_ref_benchmark/<scene>).")
    p.add_argument("--source-cam", default="cam0",
                   help="Reference/source camera name to replace (default cam0).")
    p.add_argument("--pose-pth", required=True, type=Path,
                   help="MonocularCameras ckpt providing cam0's refined-frame pose (cameras/original.pth).")
    p.add_argument("--pose-index", type=int, default=72,
                   help="Frame index in --pose-pth to use for cam0's (frozen) pose.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target_scene_root.expanduser().resolve()
    ref = args.ref_scene_root.expanduser().resolve()
    cam = args.source_cam
    if not target.is_dir():
        raise FileNotFoundError(f"Target scene root not found: {target}")
    if not ref.is_dir():
        raise FileNotFoundError(f"Reference scene root not found: {ref}")

    # 1) Copy cam0 g-buffers from the reference benchmark over the target's cam0.
    copied = []
    for modality in MODALITIES:
        src_dir = ref / modality / cam
        if not src_dir.is_dir():
            continue
        dst_dir = target / modality / cam
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        copied.append(modality)
    if "rgb" not in copied or "gt" not in copied:
        raise RuntimeError(
            f"Reference benchmark {ref} is missing rgb/{cam} or gt/{cam}; copied={copied}"
        )

    # 2) Overwrite cam0's recorded pose with the provided refined-frame pose (frozen).
    cam_npz = target / "cameras" / f"{cam}_c2w.npz"
    if not cam_npz.exists():
        raise FileNotFoundError(f"Missing target camera npz: {cam_npz}")
    with np.load(cam_npz) as data:
        key = "c2w" if "c2w" in data.files else data.files[0]
        n = int(np.asarray(data[key]).shape[0])
    cams = MonocularCameras.load_from_ckpt(
        torch.load(args.pose_pth.expanduser().resolve(), map_location="cpu", weights_only=False)
    )
    with torch.no_grad():
        pose = cams.T_wc_list().detach().double().cpu().numpy()  # [T,4,4] c2w (refined frame)
    if args.pose_index < 0 or args.pose_index >= pose.shape[0]:
        raise ValueError(f"--pose-index {args.pose_index} out of range [0,{pose.shape[0]})")
    c2w = np.repeat(pose[args.pose_index][None].astype(np.float32), n, axis=0)
    np.savez_compressed(cam_npz, c2w=c2w)

    print(f"Merged {cam}: copied {copied} from {ref} -> {target}; "
          f"set {cam}_c2w.npz to pose[{args.pose_index}] x{n} from {args.pose_pth.name}")


if __name__ == "__main__":
    main()
