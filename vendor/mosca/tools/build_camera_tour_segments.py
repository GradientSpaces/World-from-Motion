#!/usr/bin/env python3
"""Build the 3 scripted segment npzs for a camera tour from the aligned cameras.

Reads the two aligned camera checkpoints written by
``tools/export_refined_aligned_cameras.py`` into
``data/release_ws/refined/<scene>/cameras/``:

  original.pth  - the original input camera, refined recon frame (N frames)
  new_traj.pth  - the new virtual trajectory, refined recon frame (N frames)

and emits 3 scripted (pose, scene-time) npzs that compose one continuous tour.

--variant v1 (default):
  seg1 fix-cam / vary-time : camera C_orig[0]              , time t_0..t_{N-1}
  seg2 fix-time / vary-cam : time t_{N-1} (frozen)         , camera C_orig[0..N-1]
  seg3 fix-time / vary-cam : time t_{N-1} (frozen)         , camera interp C_orig[N-1]->C_new[N-1]
  Joins: (C_orig[0],t_{N-1}) -> (C_orig[N-1],t_{N-1}) -> (C_new[N-1],t_{N-1}).

--variant v2 (forward / bullet / reverse):
  segA fwd  vary-cam+time : camera C_orig[0..N-1]          , time t_0..t_{N-1}
  segB bullet fix-time    : time t_{N-1} (frozen)          , camera interp C_orig[N-1]->C_new[N-1]
  segC rev  vary-cam+time : camera C_new[N-1..0]           , time t_{N-1}..t_0
  Joins: (C_orig[N-1],t_{N-1}) -> (C_new[N-1],t_{N-1}) -> (C_new[0],t_0).

Each npz holds c2w:[N,4,4] (recon world frame, camera-to-world) and time_ids:[N].
These feed ``render_wild_fixed_camera_gbuffers.py --scripted-camera-npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_moca.camera import MonocularCameras  # noqa: E402

# Ordered tour segments: (segment id, scripted camera name, output mp4 basename).
V1_SEGMENTS = (
    ("seg1", "cam_fixcam_first", "01_fixcam_first"),
    ("seg2", "cam_sweep_orig", "02_sweep_orig"),
    ("seg3", "cam_sweep_orig2new", "03_sweep_orig2new"),
)
V2_SEGMENTS = (
    ("segA", "cam_fwd_orig", "01_fwd_orig"),
    ("segB", "cam_bullet", "02_bullet"),
    ("segC", "cam_rev_new", "03_rev_new"),
)
# v3 reuses the v2 tour camera names (segA/segB/segC) but renders seg1 + seg2's
# reference from the ORIGINAL recon. The reference render is an extra (non-tour) npz.
V3_SEGMENTS = V2_SEGMENTS
SEGMENTS_BY_VARIANT = {"v1": V1_SEGMENTS, "v2": V2_SEGMENTS, "v3": V3_SEGMENTS}
# Backward-compat alias (default variant).
SEGMENTS = V1_SEGMENTS
# v3 overwrites the v2 outputs per the user's request.
DEFAULT_SUBDIR = {"v1": "camera_tour", "v2": "camera_tour_v2", "v3": "camera_tour_v2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refined-root", default="data/release_ws/refined", type=Path,
                   help="Refined release root holding <scene>/cameras/{original,new_traj}.pth.")
    p.add_argument("--scene", required=True, help="Scene name, e.g. anymal.")
    p.add_argument("--variant", choices=("v1", "v2", "v3"), default="v1",
                   help="v1: fixcam/sweep. v2: forward/bullet/reverse (refined). "
                        "v3: like v2 but seg1 + seg2-reference rendered from the ORIGINAL recon.")
    p.add_argument("--original", default=None, type=Path,
                   help="Override path to original.pth (default: <refined-root>/<scene>/cameras/original.pth).")
    p.add_argument("--new-traj", default=None, type=Path,
                   help="Override path to new_traj.pth (default: <refined-root>/<scene>/cameras/new_traj.pth).")
    p.add_argument("--original-root", default="data/release_ws/original", type=Path,
                   help="Original release root (for v3 recon_logs/<run>/photometric_cam.pth).")
    p.add_argument("--orig-stride", type=int, default=2,
                   help="v3: stride mapping the 73 tour frames onto the original recon (default 2).")
    p.add_argument("--output-dir", default=None, type=Path,
                   help="Default: <refined-root>/<scene>/<variant subdir>/segments.")
    return p.parse_args()


def resolve_orig_recon(original_root: Path, scene: str) -> Path:
    """Return the original recon run dir under <original-root>/<scene>/recon_logs/."""
    base = original_root / scene / "recon_logs"
    subs = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if not subs:
        raise FileNotFoundError(f"No recon_logs run dir under {base}")
    if not (subs[0] / "photometric_cam.pth").exists():
        raise FileNotFoundError(f"Missing photometric_cam.pth in {subs[0]}")
    return subs[0]


def load_cam_c2w(path: Path) -> np.ndarray:
    """Load a MonocularCameras checkpoint and return camera-to-world poses [N,4,4]."""
    cams = MonocularCameras.load_from_ckpt(torch.load(path, map_location="cpu", weights_only=False))
    with torch.no_grad():
        c2w = cams.T_wc_list().detach().double().cpu().numpy()
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected c2w shape {c2w.shape} from {path}")
    return c2w


def quat_from_R(R: np.ndarray) -> np.ndarray:
    # xyzw quaternion from a 3x3 rotation matrix (copied from build_tour_segments.py).
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def R_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(d, -1.0, 1.0))
    th = th0 * a
    s1 = np.sin(th) / np.sin(th0)
    s0 = np.cos(th) - d * s1
    return s0 * q0 + s1 * q1


def smoothstep(a: float) -> float:
    a = float(np.clip(a, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def interp_c2w(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    s = smoothstep(alpha)
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]
    out[:3, :3] = R_from_quat(slerp(quat_from_R(a[:3, :3]), quat_from_R(b[:3, :3]), s))
    return out


def main() -> None:
    args = parse_args()
    scene_dir = args.refined_root / args.scene
    original_path = args.original or (scene_dir / "cameras" / "original.pth")
    new_traj_path = args.new_traj or (scene_dir / "cameras" / "new_traj.pth")
    for path in (original_path, new_traj_path):
        if not path.exists():
            raise FileNotFoundError(f"Camera checkpoint not found: {path}")

    orig_c2w = load_cam_c2w(original_path)
    new_c2w = load_cam_c2w(new_traj_path)
    n = int(orig_c2w.shape[0])
    if new_c2w.shape[0] != n:
        raise ValueError(
            f"Frame count mismatch: original={n} new_traj={new_c2w.shape[0]}"
        )
    if n < 2:
        raise ValueError(f"Need >=2 frames, got {n}")
    last = n - 1

    fwd = list(range(n))
    frozen = [last] * n
    rev = list(range(last, -1, -1))
    bullet = np.stack([interp_c2w(orig_c2w[last], new_c2w[last], j / (n - 1))
                       for j in range(n)], 0)
    extra_npzs: dict[str, tuple[str, tuple[np.ndarray, list[int]]]] = {}
    v3_info: dict = {}
    if args.variant == "v1":
        segments = {
            "seg1": (np.stack([orig_c2w[0]] * n, 0), fwd),
            "seg2": (orig_c2w.copy(), frozen),
            "seg3": (bullet, frozen),
        }
        continuity_pairs = [("seg1", "seg2"), ("seg2", "seg3")]
    elif args.variant == "v2":  # forward along original -> bullet -> reverse along new
        segments = {
            "segA": (orig_c2w.copy(), fwd),
            "segB": (bullet, frozen),
            "segC": (new_c2w[::-1].copy(), rev),
        }
        continuity_pairs = [("segA", "segB"), ("segB", "segC")]
    else:  # v3: seg1 + seg2-reference from ORIGINAL recon; seg2-target + seg3 refined
        orig_recon = resolve_orig_recon(args.original_root, args.scene)
        orig_full = load_cam_c2w(orig_recon / "photometric_cam.pth")  # [T_orig,4,4] original frame
        stride = int(args.orig_stride)
        selected = list(range(0, stride * n, stride))
        if selected[-1] >= orig_full.shape[0]:
            raise ValueError(
                f"stride {stride} x {n} frames needs original index {selected[-1]}, "
                f"but the original recon has only {orig_full.shape[0]} frames"
            )
        ref_idx = selected[-1]
        segments = {
            "segA": (orig_full[selected].copy(), list(selected)),  # original recon frame
            "segB": (bullet, frozen),                              # refined frame (target)
            "segC": (new_c2w[::-1].copy(), rev),                   # refined frame
        }
        # seg2 reference: original recon frozen at the last input frame (original frame).
        extra_npzs["segB_ref"] = (
            "cam_ref_dummy",
            (np.stack([orig_full[ref_idx]] * n, 0), [ref_idx] * n),
        )
        # segA is in the original recon frame; only segB->segC share the refined frame.
        continuity_pairs = [("segB", "segC")]
        v3_info = {
            "orig_recon_dir": str(orig_recon),
            "orig_recon_T": int(orig_full.shape[0]),
            "orig_stride": stride,
            "selected_indices": list(selected),
            "ref_index": int(ref_idx),
        }
    seg_specs = SEGMENTS_BY_VARIANT[args.variant]

    # Continuity: seg_k end == seg_{k+1} start (pose + scene time), within-frame only.
    def first_last(seg):
        c, t = seg
        return (c[0], t[0]), (c[-1], t[-1])

    for a, b in continuity_pairs:
        (_, _), (ca_end, ta_end) = first_last(segments[a])
        (cb_start, tb_start), (_, _) = first_last(segments[b])
        if ta_end != tb_start or not np.allclose(ca_end, cb_start, atol=1e-5):
            raise AssertionError(
                f"Continuity break {a}->{b}: time {ta_end}->{tb_start}, "
                f"pose_diff={float(np.abs(ca_end - cb_start).max()):.6f}"
            )

    out_dir = args.output_dir or (scene_dir / DEFAULT_SUBDIR[args.variant] / "segments")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for seg_id, cam_name, mp4_base in seg_specs:
        c, t = segments[seg_id]
        path = out_dir / f"{seg_id}_{cam_name}_c2w_time.npz"
        np.savez_compressed(path, c2w=c.astype(np.float32),
                            time_ids=np.asarray(t, dtype=np.int32))
        written[seg_id] = {
            "path": str(path),
            "camera_name": cam_name,
            "mp4_basename": mp4_base,
            "num_frames": int(len(t)),
            "time_first": int(t[0]),
            "time_last": int(t[-1]),
        }
    for seg_id, (cam_name, (c, t)) in extra_npzs.items():
        path = out_dir / f"{seg_id}_{cam_name}_c2w_time.npz"
        np.savez_compressed(path, c2w=np.asarray(c).astype(np.float32),
                            time_ids=np.asarray(t, dtype=np.int32))
        written[seg_id] = {
            "path": str(path),
            "camera_name": cam_name,
            "num_frames": int(len(t)),
            "time_first": int(t[0]),
            "time_last": int(t[-1]),
        }

    meta = {
        "scene": args.scene,
        "variant": args.variant,
        "original_pth": str(original_path),
        "new_traj_pth": str(new_traj_path),
        "num_frames": n,
        "segments": written,
        "continuity_ok": True,
    }
    if v3_info:
        meta["v3"] = v3_info
    (out_dir / "camera_tour_segments_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OK {args.scene}: n={n} -> {out_dir}")
    for seg_id, info in written.items():
        print(f"  {seg_id} ({info['camera_name']}): frames={info['num_frames']} "
              f"time {info['time_first']}..{info['time_last']} -> {Path(info['path']).name}")


if __name__ == "__main__":
    main()
