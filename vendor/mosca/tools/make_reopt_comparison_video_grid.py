#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a no-text 1x3 video grid from initial, DiffSynth, and optimized RGB frames."
    )
    parser.add_argument("--initial-dir", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--optimized-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def list_pngs(directory: Path) -> list[Path]:
    frames = sorted(directory.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No .png frames found in {directory}")
    return frames


def main() -> None:
    args = parse_args()
    initial_dir = resolve(args.initial_dir)
    prediction_dir = resolve(args.prediction_dir)
    optimized_dir = resolve(args.optimized_dir)
    output = resolve(args.output)
    tmp_dir = (
        resolve(args.tmp_dir)
        if args.tmp_dir is not None
        else output.parent / f".{output.stem}_frames"
    )

    prediction_paths = list_pngs(prediction_dir)
    frame_names = [path.name for path in prediction_paths]
    missing = []
    for name in frame_names:
        for directory in (initial_dir, optimized_dir):
            path = directory / name
            if not path.is_file():
                missing.append(path)
                if len(missing) >= 8:
                    break
        if len(missing) >= 8:
            break
    if missing:
        preview = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing paired frame(s):\n{preview}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    for out_idx, name in enumerate(frame_names):
        paths = (initial_dir / name, prediction_dir / name, optimized_dir / name)
        images = [Image.open(path).convert("RGB") for path in paths]
        target_size = images[1].size
        images = [
            image
            if image.size == target_size
            else image.resize(target_size, Image.Resampling.BILINEAR)
            for image in images
        ]
        canvas = Image.new("RGB", (target_size[0] * 3, target_size[1]), (0, 0, 0))
        for panel_idx, image in enumerate(images):
            canvas.paste(image, (panel_idx * target_size[0], 0))
        canvas.save(tmp_dir / f"{out_idx:05d}.png")

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(tmp_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(args.crf),
        str(output),
    ]
    print(f"Frame count: {len(frame_names)}")
    print(f"Grid frames: {tmp_dir}")
    print("FFmpeg command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Video: {output}")


if __name__ == "__main__":
    main()
