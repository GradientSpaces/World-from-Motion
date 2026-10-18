#!/usr/bin/env python3
"""Materialize a DyCheck-style MoSca workspace from an ``iphone_extra`` scene.

The ``iphone_extra`` scenes (e.g. ``sriracha-tree``, ``haru-sit``) follow the
DyCheck/Nerfies on-disk layout::

    <source>/rgb/<scale>/<id>.png      RGB or RGBA frames
    <source>/depth/<scale>/<id>.npy    metric sensor depth, (H, W) or (H, W, 1)
    <source>/camera/<id>.json          Nerfies camera json (resolution-independent
                                       FOV + principal-point ratios)

MoSca's iphone pipeline (``mosca_precompute.py`` + ``mosca_reconstruct.py`` with
``profile/iphone/iphone_*.yaml``) instead expects::

    <out_ws>/images/<id>.png           RGB frames (alpha ignored downstream)
    <out_ws>/cameras/<id>.json         per-frame camera json
    <out_ws>/sensor_depth/<id>.npz     key ``dep`` holding float32 (H, W) depth
    <out_ws>/test_cameras/             present (may be empty) so the GT-pose
                                       loader does not raise

This script wires the former into the latter at a chosen ``--scale`` (default
``2x``). Depth ``.npy`` files are converted to ``.npz`` with key ``dep`` and
squeezed to 2D; RGB and camera JSONs are symlinked by default (use ``--copy`` to
copy instead). The camera intrinsics are stored at full (1x) resolution, but the
iphone loader derives FOV angles and principal-point ratios that are resolution
independent, so the same JSONs are valid at any scale.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a MoSca-ready iphone workspace (images/, cameras/, "
            "sensor_depth/) from an iphone_extra DyCheck scene at a given scale."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Source iphone_extra scene root (contains rgb/, depth/, camera/).",
    )
    parser.add_argument(
        "--out-ws",
        required=True,
        type=Path,
        help="Output MoSca workspace directory to create.",
    )
    parser.add_argument(
        "--scale",
        default="2x",
        help="rgb/depth sub-resolution to use (e.g. 1x, 2x). Default: 2x.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Optional cap on the number of (prefix) frames to materialize.",
    )
    parser.add_argument(
        "--depth-npz-key",
        default="dep",
        help="Key used to store depth inside the per-frame .npz. Default: dep.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy RGB/camera assets instead of symlinking them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output workspace before materializing.",
    )
    return parser.parse_args()


def _frame_stem(path: Path) -> str:
    return path.name.split(".", 1)[0]


def _list_rgb_frames(rgb_dir: Path) -> list[Path]:
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    frames = sorted(
        path for path in rgb_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        raise FileNotFoundError(f"No RGB frames found under {rgb_dir}")
    return frames


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _convert_depth(npy_path: Path, out_path: Path, depth_npz_key: str) -> tuple[int, int]:
    depth = np.load(npy_path)
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(
            f"Expected 2D depth after squeeze for {npy_path}, got shape {depth.shape}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **{depth_npz_key: depth})
    return int(depth.shape[0]), int(depth.shape[1])


def main() -> None:
    args = parse_args()

    source = args.source.expanduser().resolve()
    out_ws = args.out_ws.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source scene not found: {source}")

    rgb_dir = source / "rgb" / args.scale
    depth_dir = source / "depth" / args.scale
    camera_dir = source / "camera"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")

    rgb_frames = _list_rgb_frames(rgb_dir)
    if args.num_frames is not None:
        if args.num_frames <= 0:
            raise ValueError("--num-frames must be positive")
        rgb_frames = rgb_frames[: args.num_frames]
    frame_stems = [_frame_stem(path) for path in rgb_frames]

    missing_depth = [stem for stem in frame_stems if not (depth_dir / f"{stem}.npy").is_file()]
    missing_camera = [stem for stem in frame_stems if not (camera_dir / f"{stem}.json").is_file()]
    if missing_depth:
        raise FileNotFoundError(
            f"Missing depth .npy for {len(missing_depth)} frames, e.g. {missing_depth[:5]}"
        )
    if missing_camera:
        raise FileNotFoundError(
            f"Missing camera .json for {len(missing_camera)} frames, e.g. {missing_camera[:5]}"
        )

    if out_ws.exists():
        if not args.force:
            raise FileExistsError(
                f"Output workspace already exists: {out_ws}. Pass --force to replace it."
            )
        shutil.rmtree(out_ws)
    out_ws.mkdir(parents=True, exist_ok=False)

    images_out = out_ws / "images"
    cameras_out = out_ws / "cameras"
    sensor_depth_out = out_ws / "sensor_depth"
    for directory in (images_out, cameras_out, sensor_depth_out):
        directory.mkdir(parents=True, exist_ok=True)
    # Present (possibly empty) so data_utils.iphone_helpers.load_iphone_gt_poses
    # can call os.listdir without raising; these scenes ship no held-out views.
    (out_ws / "test_cameras").mkdir(parents=True, exist_ok=True)
    (out_ws / "test_images").mkdir(parents=True, exist_ok=True)

    depth_hw: tuple[int, int] | None = None
    for rgb_path, stem in zip(rgb_frames, frame_stems):
        _link_or_copy(rgb_path, images_out / f"{stem}.png", copy=args.copy)
        _link_or_copy(camera_dir / f"{stem}.json", cameras_out / f"{stem}.json", copy=args.copy)
        hw = _convert_depth(
            depth_dir / f"{stem}.npy",
            sensor_depth_out / f"{stem}.npz",
            args.depth_npz_key,
        )
        if depth_hw is None:
            depth_hw = hw
        elif depth_hw != hw:
            print(
                f"Warning: depth resolution {hw} for {stem} differs from {depth_hw}",
                file=sys.stderr,
            )

    print(f"Materialized workspace: {out_ws}")
    print(f"  scale:        {args.scale}")
    print(f"  frames:       {len(frame_stems)} ({frame_stems[0]} .. {frame_stems[-1]})")
    print(f"  depth (H,W):  {depth_hw}")
    print(f"  assets:       {'copied' if args.copy else 'symlinked'} RGB + cameras")
    print(f"  sensor_depth: converted .npy -> .npz key='{args.depth_npz_key}'")


if __name__ == "__main__":
    main()
