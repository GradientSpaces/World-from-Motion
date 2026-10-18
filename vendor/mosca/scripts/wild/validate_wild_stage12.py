#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_workspace(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    ws = Path(args.ws).expanduser().resolve()
    require(ws.is_dir(), f"workspace missing: {ws}", failures)

    image_dir = ws / args.rgb_dirname
    images = list_images(image_dir)
    require(len(images) > 0, f"no RGB frames under {image_dir}", failures)

    manifest = ws / "wild_stage12_manifest.json"
    if manifest.exists():
        with open(manifest, "r", encoding="utf-8") as handle:
            manifest_data = json.load(handle)
        require(
            int(manifest_data.get("num_frames", -1)) == len(images),
            "manifest num_frames does not match images",
            failures,
        )

    depth_dir = ws / args.depth_dirname
    require(depth_dir.is_dir(), f"depth directory missing: {depth_dir}", failures)
    depth_files = sorted(depth_dir.glob("*.npz"))
    require(len(depth_files) == len(images), "depth count does not match images", failures)
    if depth_files:
        sample_depth = np.load(depth_files[0])
        require("dep" in sample_depth.files, f"{depth_files[0]} missing dep key", failures)

    recon_path = ws / args.da3_recon
    require(recon_path.is_file(), f"camera recon missing: {recon_path}", failures)
    if recon_path.is_file():
        recon = np.load(recon_path, allow_pickle=True).item()
        require("traj_c2w" in recon, "camera recon missing traj_c2w", failures)
        require("intrinsics" in recon, "camera recon missing intrinsics", failures)
        if "traj_c2w" in recon:
            require(
                len(recon["traj_c2w"]) == len(images),
                "camera pose count does not match images",
                failures,
            )

    epi_error_dir = ws / "epi" / "error"
    require(epi_error_dir.is_dir(), f"epi/error dir missing: {epi_error_dir}", failures)
    epi_files = sorted(epi_error_dir.glob("*.npy"))
    require(len(epi_files) == len(images), "epi/error count does not match images", failures)

    if args.dynamic_mask_mode == "sam3":
        mask_dir = ws / "masks"
        require(mask_dir.is_dir(), f"SAM3 PNG mask dir missing: {mask_dir}", failures)

    uniform_taps = glob.glob(str(ws / "*uniform*cowtracker_tap.npz"))
    dynamic_taps = glob.glob(str(ws / "*dynamic*cowtracker_tap.npz"))
    require(len(uniform_taps) == 1, f"expected one uniform CoWTracker TAP, found {len(uniform_taps)}", failures)
    require(len(dynamic_taps) == 1, f"expected one dynamic CoWTracker TAP, found {len(dynamic_taps)}", failures)

    if args.logdir is not None:
        logdir = Path(args.logdir).expanduser().resolve()
        require(logdir.is_dir(), f"logdir missing: {logdir}", failures)
        require((logdir / "bundle" / "bundle.pth").is_file(), "bundle.pth missing", failures)
        require(
            (logdir / "bundle" / "bundle_cams.pth").is_file(),
            "bundle_cams.pth missing",
            failures,
        )
        require((logdir / "mosca" / "mosca.pth").is_file(), "mosca scaffold missing", failures)

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate wild stage 1-2 workspace artifacts.")
    parser.add_argument("--ws", required=True)
    parser.add_argument("--rgb_dirname", default="images")
    parser.add_argument("--depth_dirname", default="depthanything3_depth")
    parser.add_argument(
        "--da3_recon",
        "--recon_path",
        default="depthanything3_recon.npy",
        help="Workspace-relative camera recon .npy to validate.",
    )
    parser.add_argument(
        "--dynamic-mask-mode",
        default="sam3",
        choices=["sam3", "raft_epi"],
        help="Dynamic mask source used during preprocessing.",
    )
    parser.add_argument("--logdir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = validate_workspace(args)
    if failures:
        print("Wild stage 1-2 validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("Wild stage 1-2 validation passed.")


if __name__ == "__main__":
    main()
