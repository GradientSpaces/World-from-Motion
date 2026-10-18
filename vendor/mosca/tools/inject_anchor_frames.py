#!/usr/bin/env python3
"""Overwrite the first N frames of a target-camera RGB g-buffer with an anchor image.

Used to make a DiffSynth segment start from the previous segment's last generated
frame: the target RGB g-buffer (``rgb/<target_cam>``) is a VACE control input the
model was trained on, so replacing its leading frames with the anchor image biases
the generation toward continuity -- without the (unsupported) image-to-video path.

The true originals are backed up once into ``<dst-cam-dir>/.anchor_orig`` so repeated
runs always re-inject from the originals (idempotent) and the effect is reversible.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-frame", required=True, type=Path,
                   help="Anchor image (e.g. the previous segment's last predicted frame).")
    p.add_argument("--dst-cam-dir", required=True, type=Path,
                   help="Target-camera RGB g-buffer dir holding 00000.png.. (overwritten).")
    p.add_argument("--num-frames", type=int, default=4,
                   help="Number of leading frames to overwrite with the anchor image.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src_frame.exists():
        raise FileNotFoundError(f"Anchor src frame not found: {args.src_frame}")
    dst = args.dst_cam_dir
    frames = sorted(dst.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames in target g-buffer dir: {dst}")
    n = max(min(int(args.num_frames), len(frames)), 0)
    if n == 0:
        print(f"num-frames=0; nothing to inject into {dst}")
        return

    backup = dst / ".anchor_orig"
    backup.mkdir(exist_ok=True)
    # 1) Restore ALL previously backed-up originals first. This makes re-injection
    #    idempotent and correctly handles a previous run with a larger --num-frames
    #    (frames above the current n are returned to their originals).
    for bf in sorted(backup.glob("*.png")):
        target = dst / bf.name
        if target.exists():
            shutil.copy2(bf, target)
    # 2) Ensure the first n frames (now true originals on disk) are backed up before
    #    we overwrite them, so a later run can always restore.
    for f in frames[:n]:
        bf = backup / f.name
        if not bf.exists():
            shutil.copy2(f, bf)

    with Image.open(frames[0]) as im:
        width, height = im.convert("RGB").size
    src = Image.open(args.src_frame).convert("RGB").resize((width, height), Image.BILINEAR)
    src_arr = np.asarray(src)
    for f in frames[:n]:
        imageio.imwrite(f, src_arr)

    print(f"Injected {n} anchor frame(s) from {args.src_frame} into {dst} "
          f"(originals backed up in {backup.name})")


if __name__ == "__main__":
    main()
