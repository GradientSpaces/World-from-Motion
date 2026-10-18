#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output workspace exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_wildgs_iphone_intrinsics(path: Path) -> tuple[np.ndarray, tuple[int, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = payload["cameras"][0]
    k = np.array(
        [
            [float(camera["focalLengthX"]), 0.0, float(camera["principalPointX"])],
            [0.0, float(camera["focalLengthY"]), float(camera["principalPointY"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    image_size = (int(camera["imageWidth"]), int(camera["imageHeight"]))
    return k, image_size, payload


def resize_intrinsics(
    k: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]
) -> np.ndarray:
    src_width, src_height = src_size
    dst_width, dst_height = dst_size
    scaled = k.copy()
    scaled[0, 0] *= float(dst_width) / float(src_width)
    scaled[0, 2] *= float(dst_width) / float(src_width)
    scaled[1, 1] *= float(dst_height) / float(src_height)
    scaled[1, 2] *= float(dst_height) / float(src_height)
    return scaled


def list_rgb_frames(rgb_dir: Path, start_frame: int, max_frames: int) -> list[Path]:
    frames = sorted(
        path for path in rgb_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    end_frame = start_frame + max_frames
    if len(frames) < end_frame:
        raise ValueError(
            f"Requested frames {start_frame}:{end_frame}, but found only {len(frames)} in {rgb_dir}"
        )
    return frames[start_frame:end_frame]


def depth_path_for_rgb(depth_dir: Path, rgb_path: Path) -> Path:
    stem = rgb_path.stem
    candidates = [
        depth_dir / f"{stem}.png",
        depth_dir / f"{stem}.tiff",
        depth_dir / f"{stem}.tif",
        depth_dir / f"{stem.replace('frame_', 'depth_')}.png",
        depth_dir / f"{stem.replace('frame_', 'depth_')}.tiff",
        depth_dir / f"{stem.replace('frame_', 'depth_')}.tif",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No depth file found for {rgb_path.name} under {depth_dir}")


def write_camera_intrinsics(path: Path, k: np.ndarray, width: int, height: int) -> None:
    payload = {
        "width": int(width),
        "height": int(height),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "K": k.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Prepare a WildGS RGB-D image sequence as a MoSca workspace."
    )
    parser.add_argument("--src", required=True, type=Path, help="WildGS iPhone scene directory.")
    parser.add_argument("--ws", required=True, type=Path, help="Output MoSca workspace.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--target-height", type=int, default=384)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    ws = args.ws.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source scene not found: {src}")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be positive")

    rgb_dir = src / "rgb"
    depth_dir = src / "depth"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not depth_dir.is_dir():
        raise FileNotFoundError(depth_dir)

    k_src, calib_size, calibration_payload = load_wildgs_iphone_intrinsics(
        src / "calibration.json"
    )
    target_size = (int(args.target_width), int(args.target_height))
    k_out = resize_intrinsics(k_src, calib_size, target_size)

    ensure_output(ws, overwrite=args.overwrite)
    images_dir = ws / "images"
    sensor_depth_dir = ws / "sensor_depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    sensor_depth_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list_rgb_frames(rgb_dir, int(args.start_frame), int(args.max_frames))
    frame_names: list[str] = []
    source_depth_names: list[str] = []
    for rgb_path in frame_paths:
        depth_path = depth_path_for_rgb(depth_dir, rgb_path)
        rgb = imageio.imread(rgb_path)[..., :3]
        depth = imageio.imread(depth_path).astype(np.float32) * float(args.depth_scale)

        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

        frame_name = rgb_path.name
        imageio.imwrite(images_dir / frame_name, rgb)
        np.savez_compressed(sensor_depth_dir / f"{rgb_path.stem}.npz", dep=depth)
        frame_names.append(frame_name)
        source_depth_names.append(depth_path.name)

    write_camera_intrinsics(ws / "camera_intrinsics.json", k_out, *target_size)
    write_camera_intrinsics(ws / "depthanything3_input_intrinsics.json", k_out, *target_size)
    (ws / "source_calibration.json").write_text(
        json.dumps(calibration_payload, indent=2), encoding="utf-8"
    )

    manifest = {
        "source_sequence": str(src),
        "workspace": str(ws),
        "rgb_dirname": "images",
        "depth_dirname": "sensor_depth",
        "num_frames": len(frame_names),
        "source_start_frame": int(args.start_frame),
        "source_end_frame_inclusive": int(args.start_frame) + len(frame_names) - 1,
        "frame_names": frame_names,
        "source_depth_names": source_depth_names,
        "source_image_size": list(calib_size),
        "target_image_size": [target_size[0], target_size[1]],
        "depth_scale": float(args.depth_scale),
        "camera_model": "pinhole",
    }
    (ws / "wild_stage12_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(frame_names)} frames in {ws}")
    print(f"Images and sensor depth resized to {target_size[0]}x{target_size[1]}")


if __name__ == "__main__":
    main()
