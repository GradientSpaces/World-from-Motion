#!/usr/bin/env python3
"""Build the 4 scripted segment npzs for a no-cut camera tour.

For a scene's initial reconstruction (input camera = cams.T_wc) and an authored
new trajectory (UI or MoCap, camera-to-world in the same recon world frame),
emit 4 scripted (pose, scene-time) npzs that compose one continuous tour:

  Seg1 fix-cam / vary-time : camera C_in[0]        , time t_0..t_72
  Seg2 fix-time / vary-cam : time t_72 (frozen)    , camera C_in[0]..C_in[72]
  Seg3 reverse-time bridge : camera interp C_in[72]->C_new[0], time t_72..t_0
  Seg4 new-traj vary-both  : camera C_new[0..72]   , time t_0..t_72

with t_i = sampled scene time (default first-73, stride 2 -> 0,2,...,144).
Each npz holds c2w:[N,4,4] (recon world frame) and time_ids:[N].
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--saved-dir", required=True, type=Path, help="Initial recon photometric log dir.")
    p.add_argument("--new-traj-npz", required=True, type=Path, help="New trajectory npz (UI/MoCap).")
    p.add_argument("--new-traj-key", default=None, help="Array key in --new-traj-npz (auto-detected).")
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=73)
    p.add_argument("--output-dir", default=None, type=Path, help="Default: <workspace>/tour_segments.")
    p.add_argument("--scene-key", default=None)
    return p.parse_args()


def load_new_traj_c2w(npz_path: Path, key: str | None) -> np.ndarray:
    with np.load(npz_path) as data:
        files = list(data.files)
        if key and key in files:
            k = key
        elif "c2w" in files:
            k = "c2w"
        elif "cam_ui_c2w" in files:
            k = "cam_ui_c2w"
        else:
            k = next((f for f in files if np.asarray(data[f]).ndim == 3
                      and np.asarray(data[f]).shape[1:] == (4, 4)), None)
        if k is None:
            raise KeyError(f"No [T,4,4] c2w array in {npz_path}; keys={files}")
        c2w = np.asarray(data[k], dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected new-traj c2w shape {c2w.shape}")
    return c2w


def quat_from_R(R: np.ndarray) -> np.ndarray:
    # xyzw quaternion from a 3x3 rotation matrix
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
    saved = args.saved_dir
    cams = MonocularCameras.load_from_ckpt(torch.load(saved / "photometric_cam.pth", map_location="cpu"))
    T = int(cams.T)
    in_c2w = cams.T_wc_list().detach().double().cpu().numpy()  # [T,4,4] camera-to-world

    new_c2w = load_new_traj_c2w(args.new_traj_npz, args.new_traj_key)

    idx = list(range(0, T, int(args.frame_stride)))
    if args.max_frames and args.max_frames > 0:
        idx = idx[: int(args.max_frames)]
    n = len(idx)
    if n < 2:
        raise ValueError(f"Need >=2 sampled frames, got {n}")

    C_in = np.stack([in_c2w[t] for t in idx], 0)  # [n,4,4]
    # New trajectory: full-length -> index by absolute time; presampled -> index by order.
    if new_c2w.shape[0] > n and max(idx) < new_c2w.shape[0]:
        C_new = np.stack([new_c2w[t] for t in idx], 0)
        new_mode = "timeline"
    elif new_c2w.shape[0] >= n:
        C_new = new_c2w[:n].astype(np.float64)
        new_mode = "presampled"
    else:
        raise ValueError(f"new traj has {new_c2w.shape[0]} poses, need >= {n}")

    fwd = [int(t) for t in idx]
    rev = fwd[::-1]
    segments = {
        "seg1": (np.stack([C_in[0]] * n, 0), fwd),
        "seg2": (C_in.copy(), [fwd[-1]] * n),
        "seg3": (np.stack([interp_c2w(C_in[-1], C_new[0], j / (n - 1)) for j in range(n)], 0), rev),
        "seg4": (C_new.copy(), fwd),
    }

    out_dir = args.output_dir or (saved.parent.parent / "tour_segments")
    out_dir.mkdir(parents=True, exist_ok=True)

    def first_last(seg):
        c, t = seg
        return (c[0], t[0]), (c[-1], t[-1])

    # Continuity: seg_k end == seg_{k+1} start (pose + time).
    order = ["seg1", "seg2", "seg3", "seg4"]
    for a, b in zip(order[:-1], order[1:]):
        (_, _), (ca_end, ta_end) = first_last(segments[a])
        (cb_start, tb_start), (_, _) = first_last(segments[b])
        if ta_end != tb_start or not np.allclose(ca_end, cb_start, atol=1e-5):
            raise AssertionError(
                f"Continuity break {a}->{b}: time {ta_end}->{tb_start}, "
                f"pose_diff={float(np.abs(ca_end - cb_start).max()):.6f}"
            )

    written = {}
    for name in order:
        c, t = segments[name]
        path = out_dir / f"{name}_c2w_time.npz"
        np.savez_compressed(path, c2w=c.astype(np.float32), time_ids=np.asarray(t, dtype=np.int32))
        written[name] = {"path": str(path), "num_frames": int(len(t)),
                         "time_first": int(t[0]), "time_last": int(t[-1])}

    meta = {
        "scene_key": args.scene_key or saved.parent.parent.name,
        "saved_dir": str(saved),
        "new_traj_npz": str(args.new_traj_npz),
        "new_traj_mode": new_mode,
        "input_T": T,
        "frame_stride": int(args.frame_stride),
        "num_frames_per_segment": n,
        "sampled_time_ids": fwd,
        "segments": written,
        "continuity_ok": True,
    }
    (out_dir / "tour_segments_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OK {meta['scene_key']}: n={n} idx[0..2]={fwd[:3]} idx[-1]={fwd[-1]} "
          f"new_mode={new_mode} -> {out_dir}")


if __name__ == "__main__":
    main()
