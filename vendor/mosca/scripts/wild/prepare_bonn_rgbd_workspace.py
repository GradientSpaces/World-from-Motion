#!/usr/bin/env python3
"""Prepare a Bonn RGB-D Dynamic (TUM-format) sequence as a MoSca workspace.

The Bonn Dynamic scenes follow the TUM RGB-D layout:
  <src>/rgb/<timestamp>.png        color frames
  <src>/depth/<timestamp>.png      16-bit depth, meters = pixel / 5000
  <src>/rgb.txt                    "timestamp rgb/<name>.png"
  <src>/depth.txt                  "timestamp depth/<name>.png"
  <src>/groundtruth.txt            "timestamp tx ty tz qx qy qz qw" (camera-to-world)

RGB / depth / pose streams are not 1:1 sampled, so each selected RGB frame is
associated with its nearest depth frame and nearest GT pose (within a small time
window). The workspace layout matches prepare_wildgs_mocap_workspace.py so the
sensor-depth + fixed-GT-camera fit profile can consume it directly:
  <ws>/images/frame_<idx>.png
  <ws>/sensor_depth/frame_<idx>.npz   key ``dep`` -> float32 (H, W) meters
  <ws>/gt_recon.npy                   {traj_c2w, intrinsics=[fx,fy,cx,cy], ...}
  <ws>/intrinsics.json
  <ws>/wild_stage12_manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


# Bonn RGB-D Dynamic Dataset RGB camera calibration (StachnissLab / ReFusion).
# Shared across all sequences; depth is already registered to the RGB frame.
BONN_FX = 542.822841
BONN_FY = 542.576870
BONN_CX = 315.593520
BONN_CY = 237.756098
# OpenCV order [k1, k2, p1, p2, k3].
BONN_DISTORTION = (0.039903, -0.099343, -0.000730, -0.000144, 0.000000)
# 16-bit depth PNG is scaled by 5000 (TUM convention); meters = pixel * DEPTH_SCALE.
BONN_DEPTH_SCALE = 1.0 / 5000.0
# Native sensor resolution.
BONN_WIDTH = 640
BONN_HEIGHT = 480

# groundtruth.txt does NOT store the RGB-D sensor optical-frame pose. It stores the
# OptiTrack MARKER pose in a coordinate frame mangled by a ROS-node axis bug (T_ROS),
# and the marker frame is related to the sensor optical frame by a fixed calibration
# similarity (T_m, scale ~1.0593). Per the dataset's official evaluation transform
# (T_g = T_ROS^-1 T_0 T_ROS T_m, https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/),
# the per-frame sensor->world camera-to-world pose is T_ROS^-1 G T_ROS T_m. Using the
# raw G directly makes posed-depth fusion inconsistent across frames. T_ROS is its own
# inverse. We normalize T_m's scale out of the result so MoSca receives a proper SE(3)
# and the world stays in the metric scale of the sensor depth.
BONN_T_ROS = np.array(
    [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
BONN_T_M = np.array(
    [
        [1.0157, 0.1828, -0.2389, 0.0113],
        [0.0009, -0.8431, -0.6413, -0.0098],
        [-0.3009, 0.6147, -0.8085, 0.0111],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def marker_pose_to_sensor_c2w(g_wc: np.ndarray) -> np.ndarray:
    """Convert a raw groundtruth.txt marker pose to the sensor optical-frame c2w.

    Applies the dataset's T_ROS^-1 G T_ROS T_m transform (T_ROS is its own inverse)
    and divides out T_m's uniform scale so the result is a proper SE(3) in the same
    metric scale as the sensor depth.
    """
    m = BONN_T_ROS @ g_wc.astype(np.float64) @ BONN_T_ROS @ BONN_T_M
    scale = float(np.mean(np.linalg.norm(m[:3, :3], axis=1)))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = m[:3, :3] / scale
    out[:3, 3] = m[:3, 3] / scale
    return out.astype(np.float32)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Invalid zero-norm quaternion")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def read_association_table(path: Path) -> list[tuple[float, str]]:
    """Read a TUM index file: ``timestamp relative/path``."""
    entries: list[tuple[float, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed row in {path}: {raw_line!r}")
        entries.append((float(fields[0]), fields[1]))
    if not entries:
        raise ValueError(f"No entries parsed from {path}")
    entries.sort(key=lambda item: item[0])
    return entries


def read_pose_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read groundtruth.txt into sorted timestamps and camera-to-world poses."""
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"Malformed pose row in {path}: {raw_line!r}")
        timestamp = float(fields[0])
        tx, ty, tz = (float(v) for v in fields[1:4])
        qx, qy, qz, qw = (float(v) for v in fields[4:8])
        t_wc = np.eye(4, dtype=np.float32)
        # Raw Bonn groundtruth marker pose (mocap-marker-to-bugged-mocap-world).
        # marker_pose_to_sensor_c2w() converts this to the sensor optical-frame c2w.
        t_wc[:3, :3] = quat_xyzw_to_matrix(np.array([qx, qy, qz, qw], dtype=np.float64))
        t_wc[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        timestamps.append(timestamp)
        poses.append(t_wc)
    if not timestamps:
        raise ValueError(f"No poses parsed from {path}")
    order = np.argsort(np.asarray(timestamps))
    ts_arr = np.asarray(timestamps, dtype=np.float64)[order]
    pose_arr = np.stack(poses, axis=0)[order]
    return ts_arr, pose_arr


def nearest_index(query_ts: float, target_ts: np.ndarray, max_diff: float) -> int | None:
    """Return index of the nearest target timestamp within ``max_diff`` seconds."""
    pos = int(np.searchsorted(target_ts, query_ts))
    candidates = []
    if pos < len(target_ts):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best_idx = None
    best_diff = None
    for idx in candidates:
        diff = abs(float(target_ts[idx]) - query_ts)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_idx = idx
    if best_idx is None or best_diff is None or best_diff > max_diff:
        return None
    return best_idx


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output workspace exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resized_intrinsics(
    k: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int
) -> np.ndarray:
    scaled = k.copy()
    scaled[0, 0] *= float(dst_width) / float(src_width)
    scaled[0, 2] *= float(dst_width) / float(src_width)
    scaled[1, 1] *= float(dst_height) / float(src_height)
    scaled[1, 2] *= float(dst_height) / float(src_height)
    return scaled


def write_intrinsics_json(
    path: Path,
    k: np.ndarray,
    width: int,
    height: int,
    depth_scale: float,
    undistorted: bool,
) -> None:
    entry = {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "ppx": float(k[0, 2]),
        "ppy": float(k[1, 2]),
        "coeffs": [0.0] * 5 if undistorted else list(BONN_DISTORTION),
    }
    payload = {
        "color": entry,
        "depth": entry,
        "depth_scale": float(depth_scale),
        "image_size": [int(width), int(height)],
        "undistorted_for_pinhole_mosca": bool(undistorted),
        "source": "bonn_rgbd_dynamic",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Prepare a Bonn RGB-D Dynamic (TUM-format) sequence for the wild MoSca pipeline."
    )
    parser.add_argument("--src", required=True, type=Path, help="Bonn scene directory.")
    parser.add_argument("--ws", required=True, type=Path, help="Output MoSca workspace.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--target-width", type=int, default=BONN_WIDTH)
    parser.add_argument("--target-height", type=int, default=BONN_HEIGHT)
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.05,
        help="Max |timestamp| difference (s) for RGB<->depth and RGB<->pose association.",
    )
    parser.add_argument(
        "--keep-distortion",
        action="store_true",
        help="Skip undistortion (only if the source is already pinhole rectified).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    ws = args.ws.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source scene not found: {src}")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be positive")

    rgb_entries = read_association_table(src / "rgb.txt")
    depth_entries = read_association_table(src / "depth.txt")
    pose_ts, pose_c2w = read_pose_table(src / "groundtruth.txt")

    depth_ts = np.asarray([ts for ts, _ in depth_entries], dtype=np.float64)

    # Build the ordered list of RGB frames that have both a depth and a pose match.
    associated: list[tuple[float, str, str, np.ndarray]] = []
    n_missing_depth = 0
    n_missing_pose = 0
    for rgb_ts, rgb_rel in rgb_entries:
        depth_idx = nearest_index(rgb_ts, depth_ts, args.max_time_diff)
        if depth_idx is None:
            n_missing_depth += 1
            continue
        pose_idx = nearest_index(rgb_ts, pose_ts, args.max_time_diff)
        if pose_idx is None:
            n_missing_pose += 1
            continue
        depth_rel = depth_entries[depth_idx][1]
        associated.append((rgb_ts, rgb_rel, depth_rel, pose_c2w[pose_idx]))

    if not associated:
        raise RuntimeError(
            "No RGB frames could be associated with both depth and pose; "
            "consider increasing --max-time-diff."
        )

    selected = associated[args.start_frame :: args.frame_stride][: args.max_frames]
    if len(selected) < args.max_frames:
        print(
            f"Warning: requested {args.max_frames} frames but only {len(selected)} "
            f"available after stride/association (start={args.start_frame}, "
            f"stride={args.frame_stride})."
        )
    if not selected:
        raise RuntimeError("Frame selection is empty; check --start-frame/--frame-stride.")

    ensure_output(ws, overwrite=args.overwrite)
    images_dir = ws / "images"
    depth_out_dir = ws / "sensor_depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)

    k_native = np.array(
        [[BONN_FX, 0.0, BONN_CX], [0.0, BONN_FY, BONN_CY], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist = np.asarray(BONN_DISTORTION, dtype=np.float32)
    use_undistort = not args.keep_distortion and float(np.max(np.abs(dist))) > 0.0
    if use_undistort:
        map_x, map_y = cv2.initUndistortRectifyMap(
            k_native, dist, None, k_native, (BONN_WIDTH, BONN_HEIGHT), cv2.CV_32FC1
        )
    else:
        map_x = map_y = None

    target_width = int(args.target_width)
    target_height = int(args.target_height)
    out_k = resized_intrinsics(k_native, BONN_WIDTH, BONN_HEIGHT, target_width, target_height)

    timestamps: list[float] = []
    c2w_list: list[np.ndarray] = []
    frame_names: list[str] = []
    source_rgb_names: list[str] = []
    source_depth_names: list[str] = []
    for out_idx, (rgb_ts, rgb_rel, depth_rel, t_wc) in enumerate(selected):
        rgb_path = src / rgb_rel
        depth_path = src / depth_rel
        rgb = imageio.imread(rgb_path)[..., :3]
        depth_raw = imageio.imread(depth_path)
        depth_m = depth_raw.astype(np.float32) * BONN_DEPTH_SCALE

        if use_undistort:
            rgb = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.remap(
                depth_m,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        if rgb.shape[1] != target_width or rgb.shape[0] != target_height:
            rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
            depth_m = cv2.resize(
                depth_m, (target_width, target_height), interpolation=cv2.INTER_NEAREST
            )

        frame_name = f"frame_{out_idx:05d}.png"
        imageio.imwrite(images_dir / frame_name, rgb)
        np.savez_compressed(depth_out_dir / f"frame_{out_idx:05d}.npz", dep=depth_m)

        timestamps.append(rgb_ts)
        c2w_list.append(marker_pose_to_sensor_c2w(np.asarray(t_wc, dtype=np.float64)))
        frame_names.append(frame_name)
        source_rgb_names.append(Path(rgb_rel).name)
        source_depth_names.append(Path(depth_rel).name)

    recon = {
        "tstamps": np.arange(len(frame_names), dtype=np.float32),
        "source_timestamps": np.asarray(timestamps, dtype=np.float64),
        "traj_c2w": np.stack(c2w_list, axis=0).astype(np.float32),
        "intrinsics": np.array(
            [out_k[0, 0], out_k[1, 1], out_k[0, 2], out_k[1, 2]], dtype=np.float32
        ),
        "img_shape": np.array([target_height, target_width], dtype=np.int64),
        "pose_convention": (
            "T_wc camera-to-world, OpenCV optical frame; converted from Bonn marker "
            "groundtruth via SE3norm(T_ROS G T_ROS T_m)"
        ),
    }
    np.save(ws / "gt_recon.npy", np.array(recon, dtype=object))

    write_intrinsics_json(
        ws / "intrinsics.json", out_k, target_width, target_height, BONN_DEPTH_SCALE, use_undistort
    )

    manifest = {
        "source_sequence": str(src),
        "workspace": str(ws),
        "dataset": "bonn_rgbd_dynamic",
        "rgb_dirname": "images",
        "depth_dirname": "sensor_depth",
        "camera_recon": "gt_recon.npy",
        "source_image_size": [BONN_WIDTH, BONN_HEIGHT],
        "target_image_size": [target_width, target_height],
        "depth_scale": BONN_DEPTH_SCALE,
        "start_frame": int(args.start_frame),
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "num_frames": len(frame_names),
        "num_associated_available": len(associated),
        "num_rgb_missing_depth": int(n_missing_depth),
        "num_rgb_missing_pose": int(n_missing_pose),
        "max_time_diff": float(args.max_time_diff),
        "undistorted_for_pinhole_mosca": bool(use_undistort),
        "frame_names": frame_names,
        "source_rgb_names": source_rgb_names,
        "source_depth_names": source_depth_names,
        "source_timestamps": timestamps,
    }
    (ws / "wild_stage12_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(frame_names)} frames in {ws}")
    print(
        f"Associated {len(associated)} RGB frames (skipped {n_missing_depth} w/o depth, "
        f"{n_missing_pose} w/o pose); undistort={use_undistort}; "
        f"output {target_width}x{target_height}"
    )
    print("Wrote sensor depth to sensor_depth/ and GT camera poses to gt_recon.npy")


if __name__ == "__main__":
    main()
