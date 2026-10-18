#!/usr/bin/env python3
"""Encode DiffSynth predicted PNG frame sequences into per-camera mp4 videos.

Given a prediction scene dir ``<output_root>/<scene>/<pred_subdir>`` containing one
subdirectory per target camera (each holding ``<frame_name>.png`` files), write a
``<camera>.mp4`` next to each camera subdir, in sorted (temporal) frame order.
"""

from __future__ import annotations

import argparse
import logging
import os.path as osp
from pathlib import Path

import imageio.v2 as imageio


def encode_camera(cam_dir: Path, fps: int) -> Path | None:
    frames = sorted(cam_dir.glob("*.png"))
    if not frames:
        logging.warning(f"No PNG frames in {cam_dir}; skipping")
        return None
    out_path = cam_dir.parent / f"{cam_dir.name}.mp4"
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
    try:
        for f in frames:
            writer.append_data(imageio.imread(f))
    finally:
        writer.close()
    logging.info(f"Wrote {out_path} ({len(frames)} frames @ {fps} fps)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-scene-dir", required=True,
                   help="Dir containing per-camera prediction subdirs (output_root/scene/pred_subdir).")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="Camera subdir names to encode. Default: every subdir found.")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pred_dir = Path(args.pred_scene_dir).expanduser()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction scene dir not found: {pred_dir}")

    if args.cameras:
        cam_dirs = [pred_dir / c for c in args.cameras]
    else:
        cam_dirs = sorted(d for d in pred_dir.iterdir() if d.is_dir())

    n_ok = 0
    for cam_dir in cam_dirs:
        if not cam_dir.is_dir():
            logging.warning(f"Camera dir missing: {cam_dir}; skipping")
            continue
        if encode_camera(cam_dir, args.fps) is not None:
            n_ok += 1
    logging.info(f"Encoded {n_ok}/{len(cam_dirs)} camera videos under {pred_dir}")


if __name__ == "__main__":
    main()
