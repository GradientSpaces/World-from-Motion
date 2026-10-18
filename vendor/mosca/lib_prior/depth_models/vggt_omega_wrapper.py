from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
VGGT_OMEGA_ROOT = THIS_DIR / "vggt-omega"
for path in (THIS_DIR, VGGT_OMEGA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from depth_utils import save_depth_list, viz_depth_list  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


DEFAULT_CHECKPOINT = VGGT_OMEGA_ROOT / "checkpoints" / "vggt_omega_1b_512.pt"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class VGGTOmegaPreprocessMeta:
    original_wh: tuple[int, int]
    crop_box_ltrb: tuple[int, int, int, int]
    target_wh: tuple[int, int]
    padded_wh: tuple[int, int]
    pad_xy: tuple[int, int]


def get_vggt_omega_model(
    device: str | torch.device = "cuda",
    checkpoint_path: str | os.PathLike[str] | None = None,
    *,
    enable_alignment: bool = False,
) -> VGGTOmega:
    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"VGGT-Omega checkpoint not found: {checkpoint}. "
            "Download vggt_omega_1b_512.pt from facebook/VGGT-Omega."
        )

    model = VGGTOmega(enable_alignment=enable_alignment).eval()
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device)


def _crop_to_supported_aspect_ratio(
    image: Image.Image,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        crop_box = (left, 0, left + crop_width, height)
        return image.crop(crop_box), crop_box

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        crop_box = (0, top, width, top + crop_height)
        return image.crop(crop_box), crop_box

    return image, (0, 0, width, height)


def _balanced_target_shape(
    aspect_ratio: float,
    image_resolution: int,
    patch_size: int,
) -> tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    width_patches = np.sqrt(token_number / aspect_ratio)
    height_patches = token_number / width_patches
    width_patches = max(1, int(np.round(width_patches)))
    height_patches = max(1, int(np.round(height_patches)))
    return height_patches * patch_size, width_patches * patch_size


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def _max_size_target_shape(
    aspect_ratio: float,
    image_resolution: int,
    patch_size: int,
) -> tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _preprocess_rgb_images(
    img_list: Sequence[np.ndarray],
    *,
    image_resolution: int,
    mode: str,
    patch_size: int,
) -> tuple[torch.Tensor, list[VGGTOmegaPreprocessMeta]]:
    if mode not in {"balanced", "max_size"}:
        raise ValueError(f"Unsupported VGGT-Omega preprocessing mode: {mode}")
    if image_resolution <= 0 or image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be positive and divisible by patch_size")

    tensors: list[torch.Tensor] = []
    metas: list[VGGTOmegaPreprocessMeta] = []
    target_shapes: list[tuple[int, int]] = []

    for image_np in img_list:
        image_np = np.asarray(image_np)
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=2)
        if image_np.ndim != 3 or image_np.shape[2] < 3:
            raise ValueError(f"Expected RGB images, got shape {image_np.shape}")

        image = Image.fromarray(image_np[..., :3].astype(np.uint8), mode="RGB")
        original_wh = image.size
        cropped, crop_box = _crop_to_supported_aspect_ratio(image)
        crop_width, crop_height = cropped.size
        aspect_ratio = crop_height / max(crop_width, 1)
        if mode == "balanced":
            target_h, target_w = _balanced_target_shape(
                aspect_ratio,
                image_resolution,
                patch_size,
            )
        else:
            target_h, target_w = _max_size_target_shape(
                aspect_ratio,
                image_resolution,
                patch_size,
            )

        resized = cropped.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensors.append(_image_to_tensor(resized))
        target_shapes.append((target_h, target_w))
        metas.append(
            VGGTOmegaPreprocessMeta(
                original_wh=(int(original_wh[0]), int(original_wh[1])),
                crop_box_ltrb=tuple(int(value) for value in crop_box),
                target_wh=(int(target_w), int(target_h)),
                padded_wh=(int(target_w), int(target_h)),
                pad_xy=(0, 0),
            )
        )

    max_h = max(shape[0] for shape in target_shapes)
    max_w = max(shape[1] for shape in target_shapes)
    padded_tensors = []
    padded_metas = []
    for tensor, meta in zip(tensors, metas):
        _, target_h, target_w = tensor.shape
        pad_h = max_h - target_h
        pad_w = max_w - target_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded_tensors.append(tensor)
        padded_metas.append(
            VGGTOmegaPreprocessMeta(
                original_wh=meta.original_wh,
                crop_box_ltrb=meta.crop_box_ltrb,
                target_wh=meta.target_wh,
                padded_wh=(int(max_w), int(max_h)),
                pad_xy=(int(pad_left), int(pad_top)),
            )
        )

    return torch.stack(padded_tensors, dim=0), padded_metas


def _as_numpy_no_batch(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.shape[0] == 1:
        array = array[0]
    return array


def _ensure_extrinsics_4x4(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3:
        raise ValueError(f"Expected extrinsics with shape (N, *, *), got {extrinsics.shape}")
    if extrinsics.shape[1:] == (4, 4):
        return extrinsics
    if extrinsics.shape[1:] == (3, 4):
        out = np.tile(np.eye(4, dtype=np.float32)[None], (extrinsics.shape[0], 1, 1))
        out[:, :3, :4] = extrinsics
        return out
    raise ValueError(f"Expected extrinsics with shape (N, 3, 4) or (N, 4, 4), got {extrinsics.shape}")


def _transform_intrinsics_to_original(
    intrinsic: np.ndarray,
    meta: VGGTOmegaPreprocessMeta,
) -> np.ndarray:
    left, top, right, bottom = meta.crop_box_ltrb
    crop_w = max(right - left, 1)
    crop_h = max(bottom - top, 1)
    target_w, target_h = meta.target_wh
    pad_left, pad_top = meta.pad_xy

    scale_x = crop_w / float(target_w)
    scale_y = crop_h / float(target_h)

    K = np.asarray(intrinsic, dtype=np.float32).copy()
    K[0, 2] -= float(pad_left)
    K[1, 2] -= float(pad_top)
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] = K[0, 2] * scale_x + float(left)
    K[1, 2] = K[1, 2] * scale_y + float(top)
    return K


def _resize_prediction_to_original(
    value: np.ndarray,
    meta: VGGTOmegaPreprocessMeta,
    *,
    interpolation: int,
) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D prediction map, got shape {value.shape}")

    original_w, original_h = meta.original_wh
    left, top, right, bottom = meta.crop_box_ltrb
    crop_w = max(right - left, 1)
    crop_h = max(bottom - top, 1)
    target_w, target_h = meta.target_wh
    pad_left, pad_top = meta.pad_xy

    target_value = value[
        pad_top : pad_top + target_h,
        pad_left : pad_left + target_w,
    ]
    resized_crop = cv2.resize(
        target_value,
        (int(crop_w), int(crop_h)),
        interpolation=interpolation,
    )

    output = np.zeros((int(original_h), int(original_w)), dtype=np.float32)
    output[top:bottom, left:right] = resized_crop.astype(np.float32)
    return output


def _unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    ).astype(np.float32)


def _write_recon_payload(
    output_path: str | os.PathLike[str],
    *,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
) -> None:
    extrinsics_w2c = _ensure_extrinsics_4x4(extrinsics_w2c)
    c2w = np.linalg.inv(extrinsics_w2c).astype(np.float32)
    intrinsics_fxfycxcy = np.stack(
        [
            intrinsics[:, 0, 0],
            intrinsics[:, 1, 1],
            intrinsics[:, 0, 2],
            intrinsics[:, 1, 2],
        ],
        axis=1,
    ).astype(np.float32)
    recon_dict = {
        "tstamps": np.arange(len(c2w), dtype=np.float32),
        "traj_c2w": c2w,
        "intrinsics": intrinsics_fxfycxcy,
        "img_shape": np.array(image_hw, dtype=np.int32),
    }
    np.save(output_path, np.array(recon_dict, dtype=object))


def run_vggt_omega_inference(
    model: VGGTOmega,
    img_list: Sequence[np.ndarray],
    *,
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray | list[VGGTOmegaPreprocessMeta]]:
    images, metas = _preprocess_rgb_images(
        img_list,
        image_resolution=image_resolution,
        mode=preprocess_mode,
        patch_size=16,
    )
    if device is None:
        device = next(model.parameters()).device
    images = images.to(device)
    print(f"VGGT-Omega input tensor: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    depth = _as_numpy_no_batch(predictions["depth"])
    depth_conf = _as_numpy_no_batch(predictions["depth_conf"])
    extrinsics_np = _as_numpy_no_batch(extrinsics)
    intrinsics_np = _as_numpy_no_batch(intrinsics)

    original_depths = []
    original_confidences = []
    original_intrinsics = []
    for frame_idx, meta in enumerate(metas):
        original_depths.append(
            _resize_prediction_to_original(
                depth[frame_idx],
                meta,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
        )
        original_confidences.append(
            _resize_prediction_to_original(
                depth_conf[frame_idx],
                meta,
                interpolation=cv2.INTER_LINEAR,
            )
        )
        original_intrinsics.append(_transform_intrinsics_to_original(intrinsics_np[frame_idx], meta))

    return {
        "depth": np.stack(original_depths, axis=0).astype(np.float32),
        "depth_conf": np.stack(original_confidences, axis=0).astype(np.float32),
        "extrinsic": _ensure_extrinsics_4x4(extrinsics_np).astype(np.float32),
        "intrinsic": np.stack(original_intrinsics, axis=0).astype(np.float32),
        "preprocess_meta": metas,
    }


def vggt_omega_process_folder(
    model: VGGTOmega,
    img_list: Sequence[np.ndarray],
    fn_list: Sequence[str],
    dst: str | os.PathLike[str],
    *,
    invalid_mask_list: Sequence[np.ndarray] | None = None,
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    output_prefix: str = "vggt_omega",
    save_depth_confidence: bool = True,
    save_pointmaps: bool = False,
    output_indices: Sequence[int] | None = None,
    output_fn_list: Sequence[str] | None = None,
) -> dict[str, str]:
    """Run VGGT-Omega and save MoSca-compatible depth/camera artifacts.

    Outputs:
    - {dst}/{output_prefix}_depth/{stem}.npz with key "dep"
    - {dst}/{output_prefix}_disp/{stem}.npy
    - {dst}/{output_prefix}_intrins.json
    - {dst}/{output_prefix}_recon.npy
    """
    if len(img_list) != len(fn_list):
        raise ValueError(f"Image/name length mismatch: {len(img_list)} vs {len(fn_list)}")
    if len(img_list) == 0:
        raise ValueError("img_list is empty")
    if output_indices is None:
        output_indices = list(range(len(fn_list)))
    else:
        output_indices = [int(index) for index in output_indices]
    if any(index < 0 or index >= len(fn_list) for index in output_indices):
        raise ValueError(f"output_indices are out of range for {len(fn_list)} input images")
    if output_fn_list is None:
        output_fn_list = [fn_list[index] for index in output_indices]
    else:
        output_fn_list = [str(filename) for filename in output_fn_list]
    if len(output_fn_list) != len(output_indices):
        raise ValueError(
            f"output_fn_list length {len(output_fn_list)} does not match "
            f"output_indices length {len(output_indices)}"
        )
    output_invalid_mask_list = None
    if invalid_mask_list is not None:
        if len(invalid_mask_list) == len(img_list):
            output_invalid_mask_list = [invalid_mask_list[index] for index in output_indices]
        elif len(invalid_mask_list) == len(output_indices):
            output_invalid_mask_list = list(invalid_mask_list)
        else:
            raise ValueError(
                "invalid_mask_list must match either the full input length or the output subset length"
            )

    dst = Path(dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print("Running VGGT-Omega...")
    prediction = run_vggt_omega_inference(
        model,
        img_list,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
    )
    depths = np.asarray(prediction["depth"], dtype=np.float32)
    confidences = np.asarray(prediction["depth_conf"], dtype=np.float32)
    extrinsics = np.asarray(prediction["extrinsic"], dtype=np.float32)
    intrinsics = np.asarray(prediction["intrinsic"], dtype=np.float32)

    output_indices_np = np.asarray(output_indices, dtype=np.int64)
    output_depths = depths[output_indices_np]
    output_confidences = confidences[output_indices_np]
    output_extrinsics = extrinsics[output_indices_np]
    output_intrinsics = intrinsics[output_indices_np]

    depth_dir = dst / f"{output_prefix}_depth"
    disp_dir = dst / f"{output_prefix}_disp"
    conf_dir = dst / f"{output_prefix}_depth_conf"
    pointmap_dir = dst / f"{output_prefix}_pointmap"
    intrins_path = dst / f"{output_prefix}_intrins.json"
    recon_path = dst / f"{output_prefix}_recon.npy"
    depth_viz_path = dst / f"{output_prefix}.mp4"
    manifest_path = dst / f"{output_prefix}_manifest.json"

    print(f"Saving VGGT-Omega depth to {depth_dir}...")
    save_depth_list(
        output_depths.copy(),
        list(output_fn_list),
        str(depth_dir),
        output_invalid_mask_list,
    )

    disp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving VGGT-Omega disparity to {disp_dir}...")
    for frame_idx, depth in enumerate(output_depths):
        depth_out = depth.copy()
        if output_invalid_mask_list is not None:
            depth_out[np.asarray(output_invalid_mask_list[frame_idx]) > 0] = 0.0
        disparity = 1.0 / np.clip(depth_out, a_min=1e-6, a_max=1e6)
        np.save(
            disp_dir / f"{Path(output_fn_list[frame_idx]).stem}.npy",
            disparity.astype(np.float32),
        )

    if save_depth_confidence:
        conf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving VGGT-Omega depth confidence to {conf_dir}...")
        for frame_idx, confidence in enumerate(output_confidences):
            np.savez_compressed(
                conf_dir / f"{Path(output_fn_list[frame_idx]).stem}.npz",
                conf=confidence.astype(np.float32),
            )

    viz_depth_list(list(output_depths), str(depth_viz_path))

    intrins_dict = {}
    for filename, K in zip(output_fn_list, output_intrinsics):
        intrins_dict[Path(filename).stem] = [
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        ]
    with open(intrins_path, "w", encoding="utf-8") as handle:
        json.dump(intrins_dict, handle, indent=1)

    first_h, first_w = output_depths[0].shape
    _write_recon_payload(
        recon_path,
        extrinsics_w2c=output_extrinsics,
        intrinsics=output_intrinsics,
        image_hw=(int(first_h), int(first_w)),
    )

    if save_pointmaps:
        pointmap_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving VGGT-Omega pointmaps to {pointmap_dir}...")
        pointmaps = _unproject_depth_map_to_point_map(
            output_depths,
            output_extrinsics,
            output_intrinsics,
        )
        for frame_idx, pointmap in enumerate(pointmaps):
            np.savez_compressed(
                pointmap_dir / f"{Path(output_fn_list[frame_idx]).stem}.npz",
                xyz=pointmap.astype(np.float32),
            )

    manifest = {
        "mode": "vggt_omega_folder",
        "num_input_frames": int(len(fn_list)),
        "num_output_frames": int(len(output_fn_list)),
        "image_resolution": int(image_resolution),
        "preprocess_mode": preprocess_mode,
        "output_prefix": output_prefix,
        "input_filenames": list(fn_list),
        "output_filenames": list(output_fn_list),
        "output_indices": [int(index) for index in output_indices],
        "depth_dir": depth_dir.name,
        "depth_npz_key": "dep",
        "disp_dir": disp_dir.name,
        "confidence_dir": conf_dir.name if save_depth_confidence else None,
        "pointmap_dir": pointmap_dir.name if save_pointmaps else None,
        "intrinsics_path": intrins_path.name,
        "recon_path": recon_path.name,
        "preprocess_meta": [
            asdict(meta) for meta in prediction["preprocess_meta"]
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "depth_dir": str(depth_dir),
        "disp_dir": str(disp_dir),
        "intrinsics_path": str(intrins_path),
        "recon_path": str(recon_path),
        "manifest_path": str(manifest_path),
    }


def load_images_from_dir(src: str | os.PathLike[str]) -> tuple[list[np.ndarray], list[str]]:
    src_path = Path(src).expanduser().resolve()
    if not src_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {src_path}")
    filenames = sorted(
        filename
        for filename in os.listdir(src_path)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not filenames:
        raise ValueError(f"No RGB images found in {src_path}")
    images = []
    for filename in filenames:
        image = imageio.imread(src_path / filename)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        images.append(image[..., :3])
    return images, filenames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run VGGT-Omega on an image workspace.")
    parser.add_argument("--ws", type=Path, default=None, help="Workspace root. Defaults dst to ws.")
    parser.add_argument("--rgb-dirname", default="images")
    parser.add_argument("--src", type=Path, default=None, help="Image directory. Overrides --ws/--rgb-dirname.")
    parser.add_argument("--dst", type=Path, default=None, help="Output directory. Defaults to --ws or --src parent.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--output-prefix", default="vggt_omega")
    parser.add_argument("--save-pointmaps", action="store_true")
    parser.add_argument("--skip-depth-confidence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.src is None:
        if args.ws is None:
            raise ValueError("Pass either --src or --ws")
        src = args.ws.expanduser().resolve() / args.rgb_dirname
    else:
        src = args.src.expanduser().resolve()

    if args.dst is not None:
        dst = args.dst.expanduser().resolve()
    elif args.ws is not None:
        dst = args.ws.expanduser().resolve()
    else:
        dst = src.parent

    img_list, fn_list = load_images_from_dir(src)
    model = get_vggt_omega_model(device=args.device, checkpoint_path=args.checkpoint)
    try:
        outputs = vggt_omega_process_folder(
            model=model,
            img_list=img_list,
            fn_list=fn_list,
            dst=dst,
            image_resolution=args.image_resolution,
            preprocess_mode=args.preprocess_mode,
            output_prefix=args.output_prefix,
            save_depth_confidence=not args.skip_depth_confidence,
            save_pointmaps=args.save_pointmaps,
        )
    finally:
        del model
        torch.cuda.empty_cache()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
