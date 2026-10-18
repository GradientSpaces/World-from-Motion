#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.iphone_helpers import (  # noqa: E402
    load_iphone_camera_json,
    normalize_iphone_focal_length,
)
from lib_prior.depth_models.depthanything3_wrapper import get_depthanything3_model  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DepthAnything3 jointly over a prepared wild DiffSynth multiview "
            "reoptimization workspace and write auxiliary-camera depths."
        )
    )
    parser.add_argument("--ws", required=True, type=Path)
    parser.add_argument("--manifest-name", default="multiview_manifest.json")
    parser.add_argument("--depth-root", default="multiview/da3_depth")
    parser.add_argument(
        "--include-anchor",
        action="store_true",
        help=(
            "Also write the jointly estimated anchor-camera depth. DA3 is always run "
            "with all manifest cameras in each batch; this only controls whether the "
            "anchor result is saved."
        ),
    )
    parser.add_argument(
        "--anchor-depth-dirname",
        default=None,
        help=(
            "Workspace-relative directory for saved anchor depth when --include-anchor "
            "is set. Defaults to <depth-root>/<anchor-camera>."
        ),
    )
    parser.add_argument(
        "--target-cameras",
        nargs="+",
        default=None,
        help="Auxiliary cameras to write depth for. Defaults to all non-anchor cameras.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    parser.add_argument("--num-anchors", type=int, default=3)
    parser.add_argument("--ref-view-strategy", default="saddle_sim_range")
    parser.add_argument("--da3-weights", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-boundary-th", type=float, default=1.0)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_image(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return image[..., :3].astype(np.uint8)


def resize_to_hw(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


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
    return candidates[0]


def resolve_camera_json_path(ws: Path, camera: dict[str, Any], frame_name: str) -> Path:
    camera_json_dir = ws / str(camera["camera_json_dir"])
    stem = frame_stem(frame_name)
    direct = camera_json_dir / f"{stem}.json"
    if direct.is_file():
        return direct
    if str(camera["camera_name"]) == "cam0" and stem.startswith("frame_"):
        suffix = stem.split("_", 1)[1]
        anchor = camera_json_dir / f"0_{suffix}.json"
        if anchor.is_file():
            return anchor
    return direct


def c2w_from_camera_json(path: Path) -> np.ndarray:
    payload = load_iphone_camera_json(path)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(payload["orientation"], dtype=np.float32).T
    c2w[:3, 3] = np.asarray(payload["position"], dtype=np.float32)
    return c2w


def k_from_camera_json(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    payload = load_iphone_camera_json(path)
    focal_x, focal_y = normalize_iphone_focal_length(payload["focal_length"])
    declared_w, declared_h = payload.get("image_size", [image_hw[1], image_hw[0]])
    actual_h, actual_w = image_hw
    scale_x = actual_w / float(declared_w)
    scale_y = actual_h / float(declared_h)
    cx, cy = payload["principal_point"]
    return np.asarray(
        [
            [float(focal_x) * scale_x, 0.0, float(cx) * scale_x],
            [0.0, float(focal_y) * scale_y, float(cy) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def run_da3_batch(
    model: Any,
    images: Sequence[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    ref_view_strategy: str,
) -> np.ndarray:
    images_np = np.asarray(images)
    _, h, w, _ = images_np.shape
    new_h = h // 14 * 14
    new_w = w // 14 * 14
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid image size for DA3: {(h, w)}")
    working_images = [cv2.resize(image, (new_w, new_h)) for image in images_np]
    working_intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    working_intrinsics[:, 0, 0] *= new_w / float(w)
    working_intrinsics[:, 1, 1] *= new_h / float(h)
    working_intrinsics[:, 0, 2] *= new_w / float(w)
    working_intrinsics[:, 1, 2] *= new_h / float(h)

    try:
        prediction = model.inference(
            image=working_images,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=working_intrinsics,
            align_to_input_ext_scale=True,
            ref_view_strategy=ref_view_strategy,
        )
    except Exception as exc:
        print(
            "Warning: DA3 pose alignment failed; using unaligned camera-conditioned "
            f"depth prediction. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        imgs_cpu, ex_pre, in_pre = model._preprocess_inputs(
            working_images,
            np.asarray(extrinsics, dtype=np.float32),
            working_intrinsics,
        )
        imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, ex_pre, in_pre)
        ex_t_norm = model._normalize_extrinsics(ex_t.clone())
        raw_output = model._run_model_forward(
            imgs,
            ex_t_norm,
            in_t,
            [],
            False,
            False,
            ref_view_strategy,
        )
        prediction = model._convert_to_prediction(raw_output)

    return np.stack(
        [cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST_EXACT) for depth in prediction.depth],
        axis=0,
    ).astype(np.float32)


def build_chunks(num_frames: int, chunk_size: int, chunk_overlap: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("--chunk-overlap must satisfy 0 <= overlap < chunk-size")
    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < num_frames:
        end = min(num_frames, start + chunk_size)
        chunks.append(list(range(start, end)))
        if end == num_frames:
            break
        start += step
    return chunks


def choose_anchor_positions(num_frames: int, num_anchors: int) -> list[int]:
    if num_frames <= 1 or num_anchors <= 0:
        return []
    start = max(1, num_frames // 2)
    candidates = np.arange(start, num_frames)
    if candidates.size == 0:
        candidates = np.array([num_frames - 1], dtype=np.int64)
    sample = np.linspace(
        0,
        candidates.size - 1,
        num=min(num_anchors, candidates.size),
        dtype=np.int64,
    )
    return sorted(np.unique(candidates[sample]).tolist())


def main() -> None:
    args = parse_args()
    ws = resolve_repo_path(args.ws)
    manifest_path = ws / str(args.manifest_name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cameras = [dict(camera) for camera in manifest["cameras"]]
    anchor_camera = str(manifest.get("anchor_camera", "cam0"))
    aux_cameras = [camera for camera in cameras if str(camera["camera_name"]) != anchor_camera]
    if args.target_cameras is not None:
        selected = set(str(camera) for camera in args.target_cameras)
        aux_cameras = [camera for camera in aux_cameras if str(camera["camera_name"]) in selected]
    if not aux_cameras:
        raise ValueError(f"No auxiliary cameras selected in {manifest_path}")

    frame_count = min(len(cameras[0]["frame_names"]), *(len(camera["frame_names"]) for camera in cameras))
    if args.max_frames is not None:
        frame_count = min(frame_count, int(args.max_frames))

    first_aux = aux_cameras[0]
    first_aux_image = load_image(resolve_frame_path(ws, first_aux, first_aux["frame_names"][0]))
    target_hw = first_aux_image.shape[:2]
    model = get_depthanything3_model(device=args.device, weights=args.da3_weights)

    depth_root = Path(str(args.depth_root))
    output_camera_names = {str(camera["camera_name"]) for camera in aux_cameras}
    if args.include_anchor:
        output_camera_names.add(anchor_camera)

    def output_depth_dir(camera_name: str) -> Path:
        if camera_name == anchor_camera and args.anchor_depth_dirname:
            return ws / str(args.anchor_depth_dirname)
        return ws / depth_root / camera_name

    for camera_name in output_camera_names:
        out_dir = output_depth_dir(camera_name)
        if out_dir.exists():
            for old in out_dir.glob("*.npz"):
                old.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        chunks = build_chunks(frame_count, args.chunk_size, args.chunk_overlap)
        anchor_positions = choose_anchor_positions(frame_count, args.num_anchors)
        for chunk_positions in tqdm(chunks, desc=f"DA3 {ws.name}"):
            chunk_position_set = set(chunk_positions)
            effective_positions = [(idx, False) for idx in chunk_positions] + [
                (idx, True) for idx in anchor_positions if idx not in chunk_position_set
            ]
            images = []
            extrinsics = []
            intrinsics = []
            batch_items = []
            for frame_idx, _is_anchor in effective_positions:
                for camera in cameras:
                    camera_name = str(camera["camera_name"])
                    frame_name = str(camera["frame_names"][frame_idx])
                    image_path = resolve_frame_path(ws, camera, frame_name)
                    camera_json_path = resolve_camera_json_path(ws, camera, frame_name)
                    image = resize_to_hw(load_image(image_path), target_hw)
                    c2w = c2w_from_camera_json(camera_json_path)
                    batch_items.append(
                        {
                            "camera_name": camera_name,
                            "frame_idx": frame_idx,
                            "frame_name": frame_stem(frame_name),
                            "is_target_output": (
                                camera_name in output_camera_names
                                and frame_idx in chunk_position_set
                            ),
                        }
                    )
                    images.append(image)
                    extrinsics.append(np.linalg.inv(c2w).astype(np.float32))
                    intrinsics.append(k_from_camera_json(camera_json_path, target_hw))

            depths = run_da3_batch(
                model=model,
                images=images,
                extrinsics=np.stack(extrinsics, axis=0),
                intrinsics=np.stack(intrinsics, axis=0),
                ref_view_strategy=str(args.ref_view_strategy),
            )
            for item_idx, item in enumerate(batch_items):
                if not item["is_target_output"]:
                    continue
                out_dir = output_depth_dir(str(item["camera_name"]))
                np.savez_compressed(
                    out_dir / f"{item['frame_name']}.npz",
                    dep=depths[item_idx].astype(np.float32),
                )
    finally:
        del model
        torch.cuda.empty_cache()

    for camera in manifest["cameras"]:
        camera_name = str(camera["camera_name"])
        if camera_name not in output_camera_names:
            continue
        if camera_name == anchor_camera and args.anchor_depth_dirname:
            camera["depth_dir"] = str(args.anchor_depth_dirname)
        else:
            camera["depth_dir"] = str(depth_root / camera_name)
        camera["depth_npz_key"] = "dep"
        camera["depth_boundary_th"] = float(args.depth_boundary_th)
    manifest["multiview_depth_root"] = str(depth_root)
    manifest["depth_npz_key"] = "dep"
    manifest["multiview_depth_boundary_th"] = float(args.depth_boundary_th)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote DA3 depths under {ws / depth_root} and updated {manifest_path}")


if __name__ == "__main__":
    main()
