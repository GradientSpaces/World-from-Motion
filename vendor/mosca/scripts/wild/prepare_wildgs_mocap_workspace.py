#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def parse_index_range(value: str) -> tuple[int, int]:
    if "-" not in value:
        index = int(value)
        return index, index
    start, end = value.split("-", 1)
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        raise argparse.ArgumentTypeError("frame range end must be >= start")
    return start_i, end_i


def load_pose_table(path: Path) -> dict[int, tuple[float, np.ndarray]]:
    poses: dict[int, tuple[float, np.ndarray]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"Malformed pose row in {path}: {raw_line}")
        timestamp = float(fields[0])
        tx, ty, tz = (float(v) for v in fields[1:4])
        qx, qy, qz, qw = (float(v) for v in fields[4:8])
        frame_id = len(poses)
        t_wc = np.eye(4, dtype=np.float32)
        # WildGS Mocap uses the TUM RGB-D trajectory layout:
        # timestamp tx ty tz qx qy qz qw, where the pose is camera-to-world.
        t_wc[:3, :3] = quat_xyzw_to_matrix(np.array([qx, qy, qz, qw], dtype=np.float64))
        t_wc[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        poses[frame_id] = (timestamp, t_wc)
    return poses


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


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    color = payload["color"]
    k = np.array(
        [
            [float(color["fx"]), 0.0, float(color["ppx"])],
            [0.0, float(color["fy"]), float(color["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist = np.array(color.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    depth_scale = float(payload.get("depth_scale", 0.001))
    return k, dist, depth_scale, payload


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output workspace exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_intrinsics_payload(
    path: Path, source_payload: dict, k: np.ndarray, undistorted: bool, depth_scale: float
) -> None:
    payload = dict(source_payload)
    coeffs = [0.0] * 5 if undistorted else source_payload["color"].get("coeffs", [0.0] * 5)
    for key in ("color", "depth"):
        entry = dict(source_payload[key])
        entry["fx"] = float(k[0, 0])
        entry["fy"] = float(k[1, 1])
        entry["ppx"] = float(k[0, 2])
        entry["ppy"] = float(k[1, 2])
        entry["coeffs"] = coeffs
        payload[key] = entry
    payload["depth_scale"] = depth_scale
    payload["undistorted_for_pinhole_mosca"] = bool(undistorted)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resized_intrinsics(k: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int) -> np.ndarray:
    scaled = k.copy()
    scaled[0, 0] *= float(dst_width) / float(src_width)
    scaled[0, 2] *= float(dst_width) / float(src_width)
    scaled[1, 1] *= float(dst_height) / float(src_height)
    scaled[1, 2] *= float(dst_height) / float(src_height)
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser("Prepare a WildGS Mocap sequence for the wild MoSca pipeline.")
    parser.add_argument("--src", required=True, type=Path, help="WildGS Mocap sequence directory.")
    parser.add_argument("--ws", required=True, type=Path, help="Output MoSca workspace.")
    parser.add_argument("--frame-range", default="100-246", type=parse_index_range)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-distortion",
        action="store_true",
        help="Do not undistort RGB/depth. Only use this if the source is already pinhole rectified.",
    )
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    ws = args.ws.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source sequence not found: {src}")
    ensure_output(ws, overwrite=args.overwrite)

    images_dir = ws / "images"
    depth_dir = ws / "sensor_depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    k, dist, depth_scale, intrinsics_payload = load_intrinsics(src / "intrinsics.json")
    poses = load_pose_table(src / "groundtruth.txt")
    start_id, end_id = args.frame_range
    frame_ids = list(range(start_id, end_id + 1))

    sample = imageio.imread(src / "rgb" / f"frame_{frame_ids[0]:05d}.png")
    src_height, src_width = sample.shape[:2]
    target_width = int(args.target_width)
    target_height = int(args.target_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("--target-width and --target-height must be positive")
    out_k = resized_intrinsics(k, src_width, src_height, target_width, target_height)
    use_undistort = not args.keep_distortion and np.max(np.abs(dist)) > 0.0
    if use_undistort:
        map_x, map_y = cv2.initUndistortRectifyMap(
            k, dist, None, k, (src_width, src_height), cv2.CV_32FC1
        )
    else:
        map_x = map_y = None

    timestamps: list[float] = []
    c2w: list[np.ndarray] = []
    frame_names: list[str] = []
    for frame_id in frame_ids:
        rgb_path = src / "rgb" / f"frame_{frame_id:05d}.png"
        depth_path = src / "depth" / f"depth_{frame_id:05d}.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        if frame_id not in poses:
            raise KeyError(f"No GT pose for frame {frame_id}")

        rgb = imageio.imread(rgb_path)[..., :3]
        depth_raw = imageio.imread(depth_path)
        depth_m = depth_raw.astype(np.float32) * depth_scale
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

        frame_name = f"frame_{frame_id:05d}.png"
        imageio.imwrite(images_dir / frame_name, rgb)
        np.savez_compressed(depth_dir / f"frame_{frame_id:05d}.npz", dep=depth_m)

        timestamp, t_wc = poses[frame_id]
        timestamps.append(timestamp)
        c2w.append(t_wc)
        frame_names.append(frame_name)

    recon = {
        "tstamps": np.arange(len(frame_ids), dtype=np.float32),
        "source_timestamps": np.asarray(timestamps, dtype=np.float64),
        "source_frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "traj_c2w": np.stack(c2w, axis=0).astype(np.float32),
        "intrinsics": np.array([out_k[0, 0], out_k[1, 1], out_k[0, 2], out_k[1, 2]], dtype=np.float32),
        "img_shape": np.array([target_height, target_width], dtype=np.int64),
        "pose_convention": "T_wc camera-to-world, OpenCV optical camera frame (+x right, +y down, +z forward)",
    }
    np.save(ws / "gt_recon.npy", np.array(recon, dtype=object))

    write_intrinsics_payload(ws / "intrinsics.json", intrinsics_payload, out_k, use_undistort, depth_scale)
    manifest = {
        "source_sequence": str(src),
        "workspace": str(ws),
        "rgb_dirname": "images",
        "depth_dirname": "sensor_depth",
        "camera_recon": "gt_recon.npy",
        "source_image_size": [src_width, src_height],
        "target_image_size": [target_width, target_height],
        "num_frames": len(frame_ids),
        "frame_names": frame_names,
        "source_frame_ids": frame_ids,
        "source_timestamps": timestamps,
        "undistorted_for_pinhole_mosca": bool(use_undistort),
    }
    (ws / "wild_stage12_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(frame_ids)} frames in {ws}")
    print("Wrote sensor depth to sensor_depth/ and GT camera poses to gt_recon.npy")


if __name__ == "__main__":
    main()
