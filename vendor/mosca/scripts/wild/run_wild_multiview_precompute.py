#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_prior.moca_processor import MoCaPrep, normalize_sam3_prompts  # noqa: E402
from lib_prior.sam3_wrapper import (  # noqa: E402
    generate_sam3_epi,
    save_sam3_epi_artifacts,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract auxiliary-camera SAM3 masks and TAP priors for a prepared "
            "wild multiview reoptimization workspace."
        )
    )
    parser.add_argument("--ws", required=True, type=Path)
    parser.add_argument("--manifest-name", default="multiview_manifest.json")
    parser.add_argument("--cfg", type=Path, default=None)
    parser.add_argument("--target-cameras", nargs="+", default=None)
    parser.add_argument(
        "--include-anchor",
        action="store_true",
        help=(
            "Also regenerate root Saved2D SAM3 masks and TAP priors for the anchor "
            "camera. Use this for scratch joint-DA3 reoptimization."
        ),
    )
    parser.add_argument("--sam3-text", default=None)
    parser.add_argument(
        "--sam3-lib-prior-root",
        default=None,
    )
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe", default=None)
    parser.add_argument("--sam3-gpus", default=None)
    parser.add_argument("--sam3-fps", type=float, default=10.0)
    parser.add_argument("--tap-mode", default=None)
    parser.add_argument("--tap-save-name", default="uniform_dep=multiview")
    parser.add_argument("--dynamic-tap-save-name", default="dynamic_dep=multiview")
    parser.add_argument("--tap-chunk-size", type=int, default=5000)
    parser.add_argument("--n-track-uniform", type=int, default=8192)
    parser.add_argument("--n-track-dynamic", type=int, default=24576)
    parser.add_argument("--depth-boundary-filter-th", type=float, default=1.0)
    parser.add_argument("--max-viz-cnt", type=int, default=512)
    parser.add_argument(
        "--with-dynamic-tap",
        action="store_true",
        help="Also extract a dense dynamic TAP prior using the SAM3 mask.",
    )
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--skip-tap", action="store_true")
    parser.add_argument("--force", action="store_true")
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
    raw = Path(str(frame_name))
    candidates = [rgb_dir / raw.name]
    stem = frame_stem(frame_name)
    if raw.suffix == "":
        candidates.extend(rgb_dir / f"{stem}{suffix}" for suffix in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing RGB for {camera['camera_name']} frame {frame_name}")


def load_camera_images(
    ws: Path, camera: dict[str, Any], frame_names: Sequence[str]
) -> tuple[list[np.ndarray], list[str]]:
    images = []
    image_names = []
    for frame_name in frame_names:
        image_path = resolve_frame_path(ws, camera, frame_name)
        image = imageio.imread(image_path)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        images.append(image[..., :3].astype(np.uint8))
        image_names.append(f"{frame_stem(frame_name)}.png")
    return images, image_names


def load_camera_depths(
    ws: Path, camera: dict[str, Any], frame_names: Sequence[str]
) -> np.ndarray:
    depth_dir = camera.get("depth_dir")
    if depth_dir is None:
        raise KeyError(f"No depth_dir configured for {camera['camera_name']}")
    depth_root = ws / str(depth_dir)
    depth_npz_key = str(camera.get("depth_npz_key", "dep"))
    depths = []
    for frame_name in frame_names:
        depth_path = depth_root / f"{frame_stem(frame_name)}.npz"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        depths.append(np.load(depth_path)[depth_npz_key].astype(np.float32))
    return np.stack(depths, axis=0)


def load_camera_masks(ws: Path, mask_dir: str, frame_names: Sequence[str]) -> np.ndarray:
    mask_root = ws / mask_dir
    masks = []
    for frame_name in frame_names:
        stem = frame_stem(frame_name)
        candidates = [mask_root / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg")]
        mask_path = next((path for path in candidates if path.is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for frame {frame_name} under {mask_root}")
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., :3].sum(axis=-1) > 0
        else:
            mask = mask > 0
        masks.append(mask)
    return np.stack(masks, axis=0).astype(bool)


def copy_mask_dir(src_dir: Path, dst_dir: Path, force: bool) -> None:
    if dst_dir.exists() or dst_dir.is_symlink():
        if not force:
            return
        if dst_dir.is_dir() and not dst_dir.is_symlink():
            shutil.rmtree(dst_dir)
        else:
            dst_dir.unlink()
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst_dir)


def run_sam3(
    *,
    ws: Path,
    camera: dict[str, Any],
    frame_names: Sequence[str],
    anchor_camera: str,
    args: argparse.Namespace,
) -> str:
    prompt_list = normalize_sam3_prompts(args.sam3_text)
    if len(prompt_list) == 0:
        raise ValueError("--sam3-text is required unless --skip-sam3 is set")
    camera_name = str(camera["camera_name"])
    images, image_names = load_camera_images(ws, camera, frame_names)
    is_anchor = camera_name == anchor_camera
    sam3_dir = ws if is_anchor else ws / "multiview" / "sam3" / camera_name
    sam3_dir.mkdir(parents=True, exist_ok=True)

    combined_mask_list = None
    for prompt in prompt_list:
        logging.info("Running SAM3 for %s with prompt %r", camera_name, prompt)
        mask_list = generate_sam3_epi(
            save_dir=str(sam3_dir),
            img_name_list=image_names,
            img_list=images,
            text_prompt=prompt,
            gpus=args.sam3_gpus,
            checkpoint_path=args.sam3_checkpoint,
            bpe_path=args.sam3_bpe,
            sam3_lib_prior_root=args.sam3_lib_prior_root,
            fps=float(args.sam3_fps),
            save_mask_png=len(prompt_list) == 1,
        )
        if combined_mask_list is None:
            combined_mask_list = [mask.astype(bool).copy() for mask in mask_list]
        else:
            combined_mask_list = [
                combined | mask.astype(bool)
                for combined, mask in zip(combined_mask_list, mask_list)
            ]
    if len(prompt_list) > 1:
        assert combined_mask_list is not None
        save_sam3_epi_artifacts(
            save_dir=str(sam3_dir),
            img_name_list=image_names,
            img_list=images,
            mask_list=combined_mask_list,
            fps=float(args.sam3_fps),
        )

    mask_src = sam3_dir / "masks"
    if not mask_src.is_dir():
        raise FileNotFoundError(f"SAM3 did not write masks for {camera_name}: {mask_src}")
    if is_anchor:
        return "masks"
    mask_rel = Path("multiview") / "masks" / camera_name
    copy_mask_dir(mask_src, ws / mask_rel, force=bool(args.force))
    return mask_rel.as_posix()


def run_tap(
    *,
    ws: Path,
    camera: dict[str, Any],
    frame_names: Sequence[str],
    anchor_camera: str,
    tap_mode: str,
    sample_mask: np.ndarray,
    save_name: str,
    n_track: int,
    args: argparse.Namespace,
) -> str:
    camera_name = str(camera["camera_name"])
    is_anchor = camera_name == anchor_camera
    images, _ = load_camera_images(ws, camera, frame_names)
    image_array = np.stack(images, axis=0)
    if tuple(sample_mask.shape) != tuple(image_array.shape[:3]):
        raise ValueError(
            f"Sample mask shape {sample_mask.shape} does not match images {image_array.shape[:3]}"
        )
    depth_array = None
    if tap_mode == "spatracker":
        depth_array = load_camera_depths(ws, camera, frame_names)

    tap_rel = Path("multiview") / "tap_priors" / camera_name
    tap_dir = ws if is_anchor else ws / tap_rel
    tap_dir.mkdir(parents=True, exist_ok=True)

    processor = MoCaPrep(
        dep_mode="sensor",
        tap_mode=tap_mode,
        flow_mode="none",
        align_metric_flag=False,
    )
    processor.compute_tap(
        ws=str(tap_dir),
        save_name=str(save_name),
        n_track=int(n_track),
        img_list=image_array,
        mask_list=sample_mask,
        dep_list=depth_array,
        chunk_size=int(args.tap_chunk_size),
        depth_boundary_filter_th=float(args.depth_boundary_filter_th),
        max_viz_cnt=int(args.max_viz_cnt),
    )
    return "." if is_anchor else tap_rel.as_posix()


def clear_tap_outputs(ws: Path, camera: dict[str, Any], anchor_camera: str) -> None:
    camera_name = str(camera["camera_name"])
    is_anchor = camera_name == anchor_camera
    if is_anchor:
        for old_tap in ws.glob("*tap.npz"):
            old_tap.unlink()
        for old_tap_viz in ws.glob("*tap.mp4"):
            old_tap_viz.unlink()
        viz_dir = ws / "cowtracker_viz"
        if viz_dir.is_dir():
            shutil.rmtree(viz_dir)
        return
    tap_rel = Path("multiview") / "tap_priors" / camera_name
    tap_dir = ws / tap_rel
    if tap_dir.exists():
        shutil.rmtree(tap_dir)


def selected_auxiliary_cameras(
    manifest: dict[str, Any],
    requested: Sequence[str] | None,
    include_anchor: bool,
) -> list[dict[str, Any]]:
    anchor = str(manifest.get("anchor_camera", "cam0"))
    cameras = [
        camera
        for camera in manifest["cameras"]
        if str(camera["camera_name"]) != anchor
    ]
    if include_anchor:
        cameras = [
            camera
            for camera in manifest["cameras"]
            if str(camera["camera_name"]) == anchor
        ] + cameras
    if requested is None:
        return cameras
    requested_set = {str(camera) for camera in requested}
    if include_anchor:
        requested_set.add(anchor)
    selected = [
        camera
        for camera in cameras
        if str(camera["camera_name"]) in requested_set
    ]
    missing = requested_set - {str(camera["camera_name"]) for camera in selected}
    if missing:
        raise ValueError(f"Requested cameras not in manifest: {sorted(missing)}")
    return selected


def infer_tap_mode(args: argparse.Namespace) -> str:
    if args.tap_mode is not None:
        return str(args.tap_mode)
    if args.cfg is not None:
        cfg_path = resolve_repo_path(args.cfg)
        cfg = OmegaConf.load(cfg_path)
        if hasattr(cfg, "tap_mode"):
            return str(cfg.tap_mode)
    return "spatracker"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(asctime)s:%(message)s",
    )
    ws = resolve_repo_path(args.ws)
    manifest_path = ws / str(args.manifest_name)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchor_camera = str(manifest.get("anchor_camera", "cam0"))
    target_cameras = selected_auxiliary_cameras(
        manifest, args.target_cameras, include_anchor=bool(args.include_anchor)
    )
    if not target_cameras:
        raise ValueError("No auxiliary cameras selected")
    tap_mode = infer_tap_mode(args)

    for camera in target_cameras:
        camera_name = str(camera["camera_name"])
        frame_names = [str(frame) for frame in camera["frame_names"]]
        logging.info("Precomputing auxiliary camera %s", camera_name)
        if not args.skip_sam3:
            camera["mask_dir"] = run_sam3(
                ws=ws,
                camera=camera,
                frame_names=frame_names,
                anchor_camera=anchor_camera,
                args=args,
            )
        if not args.skip_tap:
            if args.force:
                clear_tap_outputs(ws, camera, anchor_camera)
            images, _ = load_camera_images(ws, camera, frame_names)
            uniform_mask = np.ones(np.stack(images, axis=0).shape[:3], dtype=bool)
            tap_prior_dir = run_tap(
                ws=ws,
                camera=camera,
                frame_names=frame_names,
                anchor_camera=anchor_camera,
                tap_mode=tap_mode,
                sample_mask=uniform_mask,
                save_name=str(args.tap_save_name),
                n_track=int(args.n_track_uniform),
                args=args,
            )
            if args.with_dynamic_tap:
                mask_dir = str(camera.get("mask_dir", "masks" if camera_name == anchor_camera else ""))
                if not mask_dir:
                    raise KeyError(f"No mask_dir configured for dynamic TAP on {camera_name}")
                dynamic_mask = load_camera_masks(ws, mask_dir, frame_names)
                tap_prior_dir = run_tap(
                    ws=ws,
                    camera=camera,
                    frame_names=frame_names,
                    anchor_camera=anchor_camera,
                    tap_mode=tap_mode,
                    sample_mask=dynamic_mask,
                    save_name=str(args.dynamic_tap_save_name),
                    n_track=int(args.n_track_dynamic),
                    args=args,
                )
            if camera_name != anchor_camera:
                camera["tap_prior_dir"] = tap_prior_dir

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logging.info("Updated %s", manifest_path)


if __name__ == "__main__":
    main()
