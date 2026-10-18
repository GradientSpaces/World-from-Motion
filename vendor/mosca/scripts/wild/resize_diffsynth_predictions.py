#!/usr/bin/env python3
"""Resize DiffSynth virtual-camera predictions back to the native benchmark size.

DiffSynth inference runs at a fixed generation resolution (e.g. 512x384) and
writes predicted RGB frames at that size. The downstream MoSca reoptimization,
however, renders the auxiliary (virtual) cameras with the anchor camera's native
intrinsics/resolution (``multiview_tie_anchor_intrinsics=True``), so the
predicted RGB must be resized back to the native benchmark resolution before the
reopt workspace is materialized.

For each target camera this script reads the predicted frames from
``<inference_root>/<scene>/<src_pred_subdir>/<camera>`` and writes resized copies
to ``<inference_root>/<scene>/<dst_pred_subdir>/<camera>``. The per-frame target
size defaults to the matching native benchmark frame in
``<benchmark_root>/<scene>/rgb/<camera>``; pass ``--size WxH`` to override.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

RESAMPLE_FILTERS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize DiffSynth predictions back to the native benchmark resolution."
    )
    parser.add_argument("--inference-root", required=True, type=Path)
    parser.add_argument("--benchmark-root", default=None, type=Path)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--target-cameras", nargs="+", required=True)
    parser.add_argument("--src-pred-subdir", required=True)
    parser.add_argument("--dst-pred-subdir", required=True)
    parser.add_argument(
        "--size",
        default=None,
        help="Explicit target size as WxH; overrides the benchmark RGB size.",
    )
    parser.add_argument(
        "--resample",
        choices=tuple(RESAMPLE_FILTERS),
        default="bicubic",
        help="Resampling filter used when resizing. Default: bicubic.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-resize and overwrite frames that already exist in the destination.",
    )
    return parser.parse_args()


def parse_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if "x" not in text:
        raise ValueError(f"--size must look like WxH, got {value!r}")
    width_text, height_text = text.split("x", 1)
    return int(width_text), int(height_text)


def resolve_scene_root(root: Path, scene_name: str, marker: str) -> Path:
    root = root.expanduser().resolve()
    if (root / scene_name / marker).exists():
        return root / scene_name
    if (root / marker).exists():
        return root
    # Fall back to the scene subdir if present, else the root itself.
    if (root / scene_name).is_dir():
        return root / scene_name
    return root


def main() -> None:
    args = parse_args()

    inference_scene = resolve_scene_root(
        args.inference_root, args.scene_name, args.src_pred_subdir
    )
    explicit_size = parse_size(args.size)
    benchmark_scene = None
    if explicit_size is None:
        if args.benchmark_root is None:
            raise ValueError("Provide --benchmark-root or --size to determine target size.")
        benchmark_scene = resolve_scene_root(args.benchmark_root, args.scene_name, "rgb")

    resample = RESAMPLE_FILTERS[args.resample]
    total_written = 0
    for camera in args.target_cameras:
        src_dir = inference_scene / args.src_pred_subdir / camera
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing prediction directory: {src_dir}")
        dst_dir = inference_scene / args.dst_pred_subdir / camera
        dst_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".png")
        if not frames:
            raise FileNotFoundError(f"No predicted .png frames found in {src_dir}")

        camera_written = 0
        for frame_path in frames:
            dst_path = dst_dir / frame_path.name
            if dst_path.exists() and not args.overwrite:
                continue
            if explicit_size is not None:
                target_size = explicit_size
            else:
                ref_path = benchmark_scene / "rgb" / camera / frame_path.name
                if not ref_path.is_file():
                    raise FileNotFoundError(
                        f"Missing benchmark reference frame for sizing: {ref_path}"
                    )
                with Image.open(ref_path) as ref_image:
                    target_size = ref_image.size  # (W, H)
            with Image.open(frame_path) as pred_image:
                pred_image = pred_image.convert("RGB")
                if pred_image.size != target_size:
                    pred_image = pred_image.resize(target_size, resample=resample)
                pred_image.save(dst_path)
            camera_written += 1
        total_written += camera_written
        print(f"{camera}: wrote {camera_written}/{len(frames)} frames -> {dst_dir}")

    print(
        f"Resized predictions for {len(args.target_cameras)} cameras "
        f"({total_written} frames) into '{args.dst_pred_subdir}'."
    )


if __name__ == "__main__":
    main()
