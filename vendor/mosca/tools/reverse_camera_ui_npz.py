#!/usr/bin/env python3
"""Create a reverse-time copy of a camera UI trajectory NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse selected {camera_id}_c2w arrays in a camera UI NPZ.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input camera UI .npz.")
    parser.add_argument("--output", required=True, type=Path, help="Output reversed .npz.")
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        default=None,
        help="Camera ids to reverse. Defaults to camera_ids in the file, or all *_c2w keys.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def discover_camera_ids(data: np.lib.npyio.NpzFile, camera_ids: Sequence[str] | None) -> list[str]:
    if camera_ids is not None:
        return [str(camera_id) for camera_id in camera_ids]
    if "camera_ids" in data.files:
        return [str(camera_id) for camera_id in data["camera_ids"].tolist()]
    return sorted(key[:-4] for key in data.files if key.endswith("_c2w"))


def validate_c2w(key: str, c2w: np.ndarray) -> None:
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected {key} to have shape (T, 4, 4), got {c2w.shape}.")
    if c2w.shape[0] == 0:
        raise ValueError(f"Cannot reverse empty trajectory {key}.")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as data:
        camera_ids = discover_camera_ids(data, args.camera_ids)
        if not camera_ids:
            raise ValueError(f"No camera ids found in {input_path}")

        payload = {key: data[key] for key in data.files}
        for camera_id in camera_ids:
            key = f"{camera_id}_c2w"
            if key not in payload:
                raise KeyError(f"Missing {key} in {input_path}")
            c2w = np.asarray(payload[key])
            validate_c2w(key, c2w)
            payload[key] = c2w[::-1].copy()

    np.savez_compressed(output_path, **payload)
    print(f"Wrote reversed camera NPZ: {output_path}")
    print(f"Reversed camera ids: {' '.join(camera_ids)}")


if __name__ == "__main__":
    main()
