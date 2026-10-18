#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_prior.depth_models.depth_utils import save_depth_list, viz_depth_list  # noqa: E402
from lib_prior.depth_models.vggt_omega_wrapper import (  # noqa: E402
    _write_recon_payload,
    get_vggt_omega_model,
    run_vggt_omega_inference,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VGGT-Omega jointly over a prepared wild DiffSynth multiview "
            "reoptimization workspace and write per-camera depth plus pose artifacts."
        )
    )
    parser.add_argument("--ws", required=True, type=Path)
    parser.add_argument("--manifest-name", default="multiview_manifest.json")
    parser.add_argument("--depth-root", default="multiview/vggt_omega_depth")
    parser.add_argument("--anchor-depth-dirname", default="multiview/vggt_omega_depth/cam0")
    parser.add_argument("--anchor-recon-name", default="vggt_omega_recon.npy")
    parser.add_argument("--target-cameras", nargs="+", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument(
        "--preprocess-mode",
        choices=["balanced", "max_size"],
        default="balanced",
    )
    parser.add_argument("--depth-boundary-th", type=float, default=1.0)
    parser.add_argument(
        "--normalize-anchor-depth-median",
        type=float,
        default=1.0,
        help=(
            "Apply one shared scale to all depths and camera translations so the "
            "anchor-camera valid depth median equals this value. Set <0 to disable."
        ),
    )
    parser.add_argument("--save-confidence", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def frame_stem(frame_name: str) -> str:
    return Path(str(frame_name)).stem


def resolve_frame_path(ws: Path, camera: dict[str, Any], frame_name: str) -> Path:
    rgb_dir = ws / str(camera["rgb_dir"])
    stem = frame_stem(frame_name)
    raw = Path(str(frame_name))
    candidates = [rgb_dir / raw.name]
    if raw.suffix == "":
        candidates.extend(rgb_dir / f"{stem}{suffix}" for suffix in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates[0])


def load_image(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return image[..., :3].astype(np.uint8)


def output_depth_dir(
    ws: Path,
    depth_root: Path,
    anchor_camera: str,
    anchor_depth_dirname: str,
    camera_name: str,
) -> Path:
    if camera_name == anchor_camera:
        return ws / str(anchor_depth_dirname)
    return ws / depth_root / camera_name


def clean_depth_outputs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.npz"):
        old.unlink()


def save_camera_depths(
    *,
    ws: Path,
    depth_root: Path,
    anchor_camera: str,
    anchor_depth_dirname: str,
    camera_name: str,
    frame_names: list[str],
    depths: np.ndarray,
    confidences: np.ndarray | None,
) -> str:
    depth_dir = output_depth_dir(
        ws,
        depth_root,
        anchor_camera,
        anchor_depth_dirname,
        camera_name,
    )
    clean_depth_outputs(depth_dir)
    save_depth_list(depths.copy(), [f"{frame_stem(name)}.png" for name in frame_names], str(depth_dir), None)
    viz_depth_list(list(depths), str(depth_dir.with_suffix(".mp4")))
    if confidences is not None:
        conf_dir = depth_dir.parent / f"{depth_dir.name}_conf"
        clean_depth_outputs(conf_dir)
        for frame_name, confidence in zip(frame_names, confidences):
            np.savez_compressed(
                conf_dir / f"{frame_stem(frame_name)}.npz",
                conf=confidence.astype(np.float32),
            )
    return depth_dir.relative_to(ws).as_posix()


def main() -> None:
    args = parse_args()
    ws = resolve_repo_path(args.ws)
    manifest_path = ws / str(args.manifest_name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    anchor_camera = str(manifest.get("anchor_camera", "cam0"))
    cameras = [dict(camera) for camera in manifest["cameras"]]
    if args.target_cameras is not None:
        requested = {anchor_camera, *[str(camera) for camera in args.target_cameras]}
        cameras = [camera for camera in cameras if str(camera["camera_name"]) in requested]
    if anchor_camera not in {str(camera["camera_name"]) for camera in cameras}:
        raise ValueError(f"Anchor camera {anchor_camera!r} is not selected")
    if len(cameras) < 2:
        raise ValueError("VGGT-Omega multiview inference needs anchor plus at least one auxiliary camera")

    frame_count = min(len(camera["frame_names"]) for camera in cameras)
    if args.max_frames is not None:
        frame_count = min(frame_count, int(args.max_frames))
    depth_root = Path(str(args.depth_root))

    images: list[np.ndarray] = []
    flat_items: list[dict[str, Any]] = []
    for frame_idx in range(frame_count):
        for camera in cameras:
            camera_name = str(camera["camera_name"])
            frame_name = str(camera["frame_names"][frame_idx])
            images.append(load_image(resolve_frame_path(ws, camera, frame_name)))
            flat_items.append(
                {
                    "camera_name": camera_name,
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                }
            )

    model = get_vggt_omega_model(
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    try:
        prediction = run_vggt_omega_inference(
            model,
            images,
            image_resolution=int(args.image_resolution),
            preprocess_mode=str(args.preprocess_mode),
            device=args.device,
        )
    finally:
        del model
        torch.cuda.empty_cache()

    depths = np.asarray(prediction["depth"], dtype=np.float32)
    confidences = np.asarray(prediction["depth_conf"], dtype=np.float32)
    extrinsics_w2c = np.asarray(prediction["extrinsic"], dtype=np.float32)
    intrinsics = np.asarray(prediction["intrinsic"], dtype=np.float32)
    c2w_all = np.linalg.inv(extrinsics_w2c).astype(np.float32)

    camera_to_indices: dict[str, list[int]] = {
        str(camera["camera_name"]): [] for camera in cameras
    }
    for flat_idx, item in enumerate(flat_items):
        camera_to_indices[str(item["camera_name"])].append(flat_idx)

    shared_scale = 1.0
    if float(args.normalize_anchor_depth_median) > 0.0:
        anchor_indices = np.asarray(camera_to_indices[anchor_camera], dtype=np.int64)
        anchor_depth = depths[anchor_indices]
        valid_anchor_depth = anchor_depth[anchor_depth > 1e-3]
        if valid_anchor_depth.size == 0:
            raise ValueError("Cannot normalize VGGT-Omega output: anchor depth has no valid values")
        anchor_median = float(np.median(valid_anchor_depth))
        shared_scale = float(args.normalize_anchor_depth_median) / max(anchor_median, 1e-6)
        depths = depths * np.float32(shared_scale)
        c2w_all[:, :3, 3] = c2w_all[:, :3, 3] * np.float32(shared_scale)
        extrinsics_w2c = np.linalg.inv(c2w_all).astype(np.float32)
        print(
            "Normalize VGGT-Omega joint output with shared scale "
            f"{shared_scale:.6f} (anchor median {anchor_median:.6f} -> "
            f"{float(args.normalize_anchor_depth_median):.6f})"
        )

    camera_c2w: dict[str, np.ndarray] = {}
    for camera in cameras:
        camera_name = str(camera["camera_name"])
        indices = np.asarray(camera_to_indices[camera_name], dtype=np.int64)
        frame_names = [str(name) for name in camera["frame_names"][:frame_count]]
        camera["frame_names"] = frame_names
        camera["depth_dir"] = save_camera_depths(
            ws=ws,
            depth_root=depth_root,
            anchor_camera=anchor_camera,
            anchor_depth_dirname=str(args.anchor_depth_dirname),
            camera_name=camera_name,
            frame_names=frame_names,
            depths=depths[indices],
            confidences=confidences[indices] if args.save_confidence else None,
        )
        camera["depth_npz_key"] = "dep"
        camera["depth_boundary_th"] = float(args.depth_boundary_th)
        camera_c2w[camera_name] = c2w_all[indices]

        reference_c2w_path = ws / str(camera["reference_c2w_path"])
        reference_c2w_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(reference_c2w_path, c2w=camera_c2w[camera_name])

    anchor_indices = np.asarray(camera_to_indices[anchor_camera], dtype=np.int64)
    anchor_depth_shape = depths[anchor_indices][0].shape
    _write_recon_payload(
        ws / str(args.anchor_recon_name),
        extrinsics_w2c=extrinsics_w2c[anchor_indices],
        intrinsics=intrinsics[anchor_indices],
        image_hw=(int(anchor_depth_shape[0]), int(anchor_depth_shape[1])),
    )

    manifest["multiview_depth_root"] = str(depth_root)
    manifest["depth_npz_key"] = "dep"
    manifest["multiview_depth_boundary_th"] = float(args.depth_boundary_th)
    manifest["vggt_omega_anchor_recon"] = str(args.anchor_recon_name)
    manifest["vggt_omega_image_resolution"] = int(args.image_resolution)
    manifest["vggt_omega_preprocess_mode"] = str(args.preprocess_mode)
    manifest["vggt_omega_shared_scale"] = float(shared_scale)
    manifest["cameras"] = [
        next(
            (updated for updated in cameras if str(updated["camera_name"]) == str(original["camera_name"])),
            original,
        )
        for original in manifest["cameras"]
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote VGGT-Omega depths, poses, and {ws / str(args.anchor_recon_name)}")


if __name__ == "__main__":
    main()
