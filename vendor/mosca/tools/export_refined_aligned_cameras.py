#!/usr/bin/env python3
"""Export aligned original + new-trajectory cameras into the refined workspaces.

For each release scene this writes ``data/release_ws/refined/<scene>/cameras/``:
  original.pth  - the original input camera (cam0) in the refined recon frame
  new_traj.pth  - the new virtual trajectory in the refined recon frame

Both are ``MonocularCameras`` checkpoints (same OrderedDict layout as
``photometric_cam.pth``, 73 frames, ``delta_flag=False``).

The refined ``photometric_cam.pth`` is the "anchor" of the refined recon. Which
real camera it equals differs per scene:
  - anchor == NEW TRAJ: ball, solo, street, tower, astronaut, person
  - anchor == CAM0    : anymal, piano

We solve a Sim(3) that maps the anchor camera from the original-recon frame
(benchmark ``cameras/`` dir, 73 frames) onto the refined anchor, then apply it to
the *other* camera so both end up in the refined frame. For the "copy" scenes the
Sim(3) is identity (or a small BA delta for piano).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os.path as osp
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pytorch3d.transforms import matrix_to_quaternion  # noqa: E402

from lib_moca.camera import MonocularCameras  # noqa: E402
from eval_utils.campose_alignment import (  # noqa: E402
    align_ate_c2b_use_a2b,
    alignSIM3,
    SO3_to_quat,
)

# Scenes whose refined photometric_cam equals the original input camera (cam0).
ANCHOR_IS_CAM0 = {"anymal", "piano"}
SCENES = ["anymal", "astronaut", "ball", "person", "piano", "solo", "street", "tower"]


def find_refined_cam(refined_root: Path, scene: str) -> str | None:
    direct = refined_root / scene / "photometric_cam.pth"
    if direct.exists():
        return str(direct)
    cands = sorted(glob.glob(str(refined_root / scene / "logs" / "*" / "photometric_cam.pth")))
    return cands[-1] if cands else None


def load_mono(path: str) -> tuple[MonocularCameras, np.ndarray]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cams = MonocularCameras.load_from_ckpt(ckpt)
    with torch.no_grad():
        c2w = cams.T_wc_list().cpu().numpy().astype(np.float64)
    return cams, c2w


def load_npz_c2w(path: str) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    if "c2w" in d.files:
        key = "c2w"
    else:
        key = [
            k for k in d.files
            if k.endswith("c2w") and "source" not in k and "keyframe" not in k
        ][0]
    return d[key].astype(np.float64)


def sim3_scale_angle(src_c2w: np.ndarray, dst_c2w: np.ndarray) -> tuple[float, float]:
    """Diagnostic: recovered Sim(3) scale and rotation angle (src -> dst)."""
    s, R, _ = alignSIM3(
        src_c2w[:, :3, 3], dst_c2w[:, :3, 3],
        SO3_to_quat(src_c2w[:, :3, :3]), SO3_to_quat(dst_c2w[:, :3, :3]),
    )
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return float(s), ang


def apply_sim3(traj_a: np.ndarray, traj_b: np.ndarray, traj_c: np.ndarray) -> np.ndarray:
    """Sim(3) from a->b applied to c (first-party, NOPE-NeRF verified)."""
    out = align_ate_c2b_use_a2b(
        torch.from_numpy(traj_a), torch.from_numpy(traj_b), torch.from_numpy(traj_c)
    )
    return out.cpu().numpy().astype(np.float64)


def build_ckpt(c2w: np.ndarray, donor: MonocularCameras) -> OrderedDict:
    """Build a MonocularCameras state_dict from c2w poses + a donor's intrinsics."""
    R = torch.from_numpy(np.ascontiguousarray(c2w[:, :3, :3])).float()
    t = torch.from_numpy(np.ascontiguousarray(c2w[:, :3, 3])).float()
    q = matrix_to_quaternion(R)  # (N, 4) wxyz, matches pytorch3d convention
    ckpt = OrderedDict()
    ckpt["_rel_focal"] = donor._rel_focal.detach().cpu().clone()
    ckpt["cxcy_ratio"] = donor.cxcy_ratio.detach().cpu().clone()
    ckpt["q_wc"] = q
    ckpt["t_wc"] = t
    ckpt["delta_flag"] = torch.tensor(False)
    ckpt["iso_focal"] = donor.iso_focal.detach().cpu().clone()
    ckpt["default_H"] = donor.default_H.detach().cpu().clone()
    ckpt["default_W"] = donor.default_W.detach().cpu().clone()
    return ckpt


def process_scene(scene: str, args) -> dict:
    original_root = Path(args.original_root)
    refined_root = Path(args.refined_root)

    meta = json.load(open(original_root / scene / "meta.json"))
    new_traj_cam = meta["new_traj_camera"]
    sampled_npz = meta["new_traj_sampled_pose_npz"]
    bench_cam_dir = osp.dirname(sampled_npz)
    cam0_npz = osp.join(bench_cam_dir, "cam0_c2w.npz")
    orig_recon_cam = osp.join(meta["orig_saved_dir"], "photometric_cam.pth")

    refined_cam = find_refined_cam(refined_root, scene)
    if refined_cam is None:
        raise FileNotFoundError(f"[{scene}] refined photometric_cam.pth not found")
    if not osp.exists(cam0_npz):
        raise FileNotFoundError(f"[{scene}] benchmark cam0 not found: {cam0_npz}")
    if not osp.exists(sampled_npz):
        raise FileNotFoundError(f"[{scene}] sampled new_traj not found: {sampled_npz}")

    refined_cams, anchor_ref = load_mono(refined_cam)
    cam0_orig = load_npz_c2w(cam0_npz)           # original input cam, orig-recon frame
    newtraj_orig = load_npz_c2w(sampled_npz)     # new trajectory, orig-recon frame

    n = anchor_ref.shape[0]
    assert cam0_orig.shape[0] == n == newtraj_orig.shape[0], (
        f"[{scene}] frame count mismatch: refined={n} cam0={cam0_orig.shape[0]} "
        f"newtraj={newtraj_orig.shape[0]}"
    )

    anchor_is_cam0 = scene in ANCHOR_IS_CAM0
    if anchor_is_cam0:
        # Refined anchor IS cam0: original.pth is a direct copy of the refined anchor
        # (its intrinsics are already cam0's), and only the new trajectory is mapped
        # into the refined frame.
        anchor_orig = cam0_orig
        original_c2w, original_donor = anchor_ref, refined_cams
        new_traj_c2w = apply_sim3(cam0_orig, anchor_ref, newtraj_orig)
        new_traj_donor = refined_cams
        anchor_name = "cam0 (input)"
    else:
        # Refined anchor IS the new trajectory: new_traj.pth is a direct copy; cam0 is
        # mapped into the refined frame and keeps its original-recon intrinsics.
        if osp.exists(orig_recon_cam):
            cam0_donor, _ = load_mono(orig_recon_cam)
        else:
            logging.warning(f"[{scene}] orig recon cam missing ({orig_recon_cam}); "
                            f"using refined intrinsics for original.pth")
            cam0_donor = refined_cams
        anchor_orig = newtraj_orig
        original_c2w, original_donor = apply_sim3(newtraj_orig, anchor_ref, cam0_orig), cam0_donor
        new_traj_c2w, new_traj_donor = anchor_ref, refined_cams
        anchor_name = f"{new_traj_cam} (new traj)"

    scale, angle = sim3_scale_angle(anchor_orig, anchor_ref)

    out_dir = refined_root / scene / "cameras"
    original_ckpt = build_ckpt(original_c2w, original_donor)
    new_traj_ckpt = build_ckpt(new_traj_c2w, new_traj_donor)

    if args.dry_run:
        logging.info(f"[{scene}] DRY-RUN would write {out_dir}/original.pth + new_traj.pth")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(original_ckpt, out_dir / "original.pth")
        torch.save(new_traj_ckpt, out_dir / "new_traj.pth")
        logging.info(f"[{scene}] wrote {out_dir}/original.pth + new_traj.pth")

    return {
        "scene": scene, "n": n, "anchor": anchor_name,
        "sim3_scale": scale, "sim3_angle_deg": angle,
        "out_dir": str(out_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-root", default="data/release_ws/original")
    p.add_argument("--refined-root", default="data/release_ws/refined")
    p.add_argument("--scenes", nargs="+", default=None, help="Restrict to these scenes.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().setLevel(logging.INFO)

    scenes = args.scenes or SCENES
    rows = []
    for scene in scenes:
        rows.append(process_scene(scene, args))

    print("\n" + "=" * 88)
    print(f"{'scene':10s} {'n':>3s}  {'anchor=refined photometric_cam':32s} {'sim3_scale':>10s} {'angle°':>8s}")
    print("-" * 88)
    for r in rows:
        print(f"{r['scene']:10s} {r['n']:3d}  {r['anchor']:32s} {r['sim3_scale']:10.4f} {r['sim3_angle_deg']:8.2f}")
    print("=" * 88)
    print("sim3 maps the anchor camera from the original-recon frame onto the refined anchor;")
    print("applied to the other camera so both land in the refined recon frame.")


if __name__ == "__main__":
    main()
