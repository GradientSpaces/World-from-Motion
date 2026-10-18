from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_PRIOR_ROOT = Path(__file__).resolve().parent
DA3_SRC_ROOT = LIB_PRIOR_ROOT / "Depth-Anything-3" / "src"
for path in (REPO_ROOT, LIB_PRIOR_ROOT, DA3_SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from lib_prior.depth_utils import check_trajectory_quality, compute_max_baseline, viz_depth_list


BATCH_ROLE_LOCAL = "local"
BATCH_ROLE_ANCHOR = "anchor"
BATCH_ROLE_GLOBAL_CONTEXT = "global_context"
PRED_ALIGNED_RECON_NAME = "depthanything3_pred_aligned_recon.npy"
FRAMEWISE_DEPTH_NPZ_KEY = "dep"


@dataclass(frozen=True)
class CameraSequenceSpec:
    camera_id: str
    image_dir: str
    filenames: tuple[str, ...]
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    image_wh: tuple[int, int]


@dataclass(frozen=True)
class BatchItem:
    batch_index: int
    camera_id: str
    sequence_index: int
    filename: str
    role: str
    is_output: bool

    @property
    def is_anchor(self) -> bool:
        return self.role == BATCH_ROLE_ANCHOR

    @property
    def is_global_context(self) -> bool:
        return self.role == BATCH_ROLE_GLOBAL_CONTEXT

    @property
    def is_local(self) -> bool:
        return self.role == BATCH_ROLE_LOCAL


def load_image_filenames(image_dir: str | Path) -> tuple[str, ...]:
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_paths.extend(image_dir.glob(suffix))
    image_paths = sorted(image_paths, key=lambda path: path.name)
    if not image_paths:
        raise ValueError(f"No RGB images found in {image_dir}")
    return tuple(path.name for path in image_paths)


def read_image_rgb(path: str | Path) -> np.ndarray:
    image = np.asarray(iio.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image at {path}, got shape {image.shape}")
    return image


def resize_image_rgb(
    image: np.ndarray,
    dst_wh: tuple[int, int],
    *,
    interpolation: int | None = None,
) -> np.ndarray:
    dst_w, dst_h = [int(value) for value in dst_wh]
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"Invalid destination image size: {dst_wh}")
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (dst_w, dst_h):
        return np.asarray(image)
    if interpolation is None:
        interpolation = cv2.INTER_AREA if dst_w <= src_w and dst_h <= src_h else cv2.INTER_LINEAR
    return np.asarray(cv2.resize(image, (dst_w, dst_h), interpolation=interpolation))


def infer_image_wh(image_dir: str | Path, filenames: Sequence[str]) -> tuple[int, int]:
    first_image = Path(image_dir).expanduser().resolve() / filenames[0]
    image = read_image_rgb(first_image)
    height, width = image.shape[:2]
    return (int(width), int(height))


def scale_intrinsics_matrix(
    K_raw: np.ndarray,
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
) -> np.ndarray:
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source image size: {src_wh}")
    K_scaled = np.asarray(K_raw, dtype=np.float32).copy()
    K_scaled[0, 0] *= float(dst_w) / float(src_w)
    K_scaled[1, 1] *= float(dst_h) / float(src_h)
    K_scaled[0, 2] *= float(dst_w) / float(src_w)
    K_scaled[1, 2] *= float(dst_h) / float(src_h)
    return K_scaled


def build_camera_sequence_spec_from_filenames(
    *,
    camera_id: str,
    image_dir: str | Path,
    filenames: Sequence[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_wh_override: tuple[int, int] | None = None,
) -> CameraSequenceSpec:
    image_dir_path = Path(image_dir).expanduser().resolve()
    filenames = tuple(str(filename) for filename in filenames)
    if not filenames:
        raise ValueError(f"{camera_id} must provide at least one filename")
    missing_files = [filename for filename in filenames if not (image_dir_path / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{camera_id} is missing {len(missing_files)} image files under {image_dir_path}; "
            f"first missing file: {missing_files[0]}"
        )
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(
            f"{camera_id} extrinsics must have shape (T, 4, 4), got {extrinsics.shape}"
        )
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(
            f"{camera_id} intrinsics must have shape (T, 3, 3), got {intrinsics.shape}"
        )
    if len(filenames) != extrinsics.shape[0]:
        raise ValueError(
            f"{camera_id} image/pose length mismatch: {len(filenames)} images vs "
            f"{extrinsics.shape[0]} extrinsics"
        )
    if len(filenames) != intrinsics.shape[0]:
        raise ValueError(
            f"{camera_id} image/intrinsic length mismatch: {len(filenames)} images vs "
            f"{intrinsics.shape[0]} intrinsics"
        )

    image_wh = (
        tuple(int(value) for value in image_wh_override)
        if image_wh_override is not None
        else infer_image_wh(image_dir_path, filenames)
    )
    return CameraSequenceSpec(
        camera_id=str(camera_id),
        image_dir=str(image_dir_path),
        filenames=filenames,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_wh=image_wh,
    )


def build_camera_sequence_spec(
    *,
    camera_id: str,
    image_dir: str | Path,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_wh_override: tuple[int, int] | None = None,
) -> CameraSequenceSpec:
    return build_camera_sequence_spec_from_filenames(
        camera_id=camera_id,
        image_dir=image_dir,
        filenames=load_image_filenames(image_dir),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_wh_override=image_wh_override,
    )


class ImageFolderFrameLoader:
    def __init__(
        self,
        image_dir: str,
        filenames: Sequence[str],
        target_wh: tuple[int, int] | None = None,
    ):
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.filenames = tuple(filenames)
        self.target_wh = None if target_wh is None else tuple(int(value) for value in target_wh)

    def read(self, sequence_index: int) -> np.ndarray:
        image = read_image_rgb(self.image_dir / self.filenames[sequence_index])
        if self.target_wh is not None:
            image = resize_image_rgb(image, self.target_wh)
        return image


def build_chunks(num_frames: int, chunk_size: int, chunk_overlap: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[list[int]] = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < num_frames:
        end = min(num_frames, start + chunk_size)
        chunks.append(list(range(start, end)))
        if end == num_frames:
            break
        start += step
    return chunks


def _compute_framewise_all_camera_baseline_scores(
    extrinsics_by_camera: Mapping[str, np.ndarray],
    context_camera_ids: Sequence[str],
    num_frames: int,
) -> np.ndarray:
    """Score each frame by the mean baseline to the start frame across all cameras."""
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)

    camera_positions = []
    for camera_id in context_camera_ids:
        extrinsics = np.asarray(extrinsics_by_camera[camera_id], dtype=np.float32)
        if extrinsics.shape[0] != num_frames:
            raise ValueError(
                f"{camera_id} has {extrinsics.shape[0]} poses, expected {num_frames}"
            )
        camera_positions.append(np.linalg.inv(extrinsics)[:, :3, 3])

    if len(camera_positions) == 1:
        positions = camera_positions[0]
        return np.linalg.norm(positions - positions[0], axis=1).astype(np.float32)

    stacked_positions = np.stack(camera_positions, axis=1)  # (T, C, 3)
    start_positions = stacked_positions[0]  # (C, 3)
    framewise_diffs = stacked_positions[:, :, None, :] - start_positions[None, None, :, :]
    framewise_dists = np.linalg.norm(framewise_diffs, axis=-1)
    return framewise_dists.mean(axis=(1, 2)).astype(np.float32)


def choose_anchor_positions(
    extrinsics_by_camera: Mapping[str, np.ndarray],
    context_camera_ids: Sequence[str],
    num_frames: int,
    num_anchors: int,
    anchor_stride: int | None = None,
    explicit_anchor_positions: Sequence[int] | None = None,
) -> tuple[list[int], dict[str, float], str]:
    baselines = {
        camera_id: float(compute_max_baseline(extrinsics_by_camera[camera_id]))
        for camera_id in context_camera_ids
    }
    best_camera_id = max(baselines, key=baselines.get)
    if num_frames < 2:
        return [], baselines, best_camera_id

    if explicit_anchor_positions is not None:
        anchor_positions = sorted(
            {int(pos) for pos in explicit_anchor_positions if 0 <= int(pos) < num_frames}
        )
        if not anchor_positions:
            raise ValueError(
                "Explicit anchor positions do not overlap the sequence: "
                f"{list(explicit_anchor_positions)}"
            )
        return anchor_positions, baselines, best_camera_id

    if anchor_stride is not None:
        if anchor_stride <= 0:
            raise ValueError("anchor_stride must be positive")
        start_pos = max(1, num_frames // 2)
        anchor_positions = list(range(start_pos, num_frames, anchor_stride))
        if not anchor_positions:
            anchor_positions = [num_frames - 1]
        return anchor_positions, baselines, best_camera_id

    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive when anchor_stride is not set")

    framewise_baseline_scores = _compute_framewise_all_camera_baseline_scores(
        extrinsics_by_camera=extrinsics_by_camera,
        context_camera_ids=context_camera_ids,
        num_frames=num_frames,
    )
    candidate_positions = np.arange(1, num_frames)
    if candidate_positions.size == 0:
        return [], baselines, best_camera_id

    candidate_scores = framewise_baseline_scores[candidate_positions]
    if np.allclose(candidate_scores, 0.0):
        candidate_positions = np.arange(max(1, num_frames // 2), num_frames)
    else:
        threshold = float(np.quantile(candidate_scores, 0.75))
        candidate_positions = candidate_positions[candidate_scores >= threshold]
        if candidate_positions.size < num_anchors:
            candidate_positions = np.arange(max(1, num_frames // 2), num_frames)

    if candidate_positions.size == 0:
        candidate_positions = np.array([num_frames - 1], dtype=np.int64)

    sample_indices = np.linspace(
        0,
        candidate_positions.size - 1,
        num=min(num_anchors, candidate_positions.size),
        dtype=np.int64,
    )
    anchor_positions = sorted(np.unique(candidate_positions[sample_indices]).tolist())
    return anchor_positions, baselines, best_camera_id


def farthest_point_sample_positions(points: np.ndarray, num_samples: int) -> list[int]:
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected points with shape (N, D), got {points.shape}")
    if points.shape[0] == 0 or num_samples == 0:
        return []

    target_count = min(int(num_samples), int(points.shape[0]))
    centroid = np.mean(points, axis=0, dtype=np.float32)
    centroid_dists = np.linalg.norm(points - centroid[None, :], axis=1)
    selected_indices = [int(np.argmax(centroid_dists))]

    min_dists = np.linalg.norm(points - points[selected_indices[0]][None, :], axis=1)
    min_dists[selected_indices[0]] = -np.inf

    while len(selected_indices) < target_count:
        next_index = int(np.argmax(min_dists))
        selected_indices.append(next_index)
        dists_to_new = np.linalg.norm(points - points[next_index][None, :], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[np.asarray(selected_indices, dtype=np.int64)] = -np.inf

    return sorted(selected_indices)


def choose_global_context_positions(
    *,
    extrinsics: np.ndarray,
    num_context_frames: int,
) -> list[int]:
    if num_context_frames < 0:
        raise ValueError(f"num_context_frames must be non-negative, got {num_context_frames}")
    if num_context_frames == 0:
        return []
    camera_centers = _extrinsics_to_camera_centers(extrinsics)
    return farthest_point_sample_positions(camera_centers, num_samples=num_context_frames)


def resolve_inference_image_wh(
    camera_specs: Mapping[str, CameraSequenceSpec],
    inference_image_wh: tuple[int, int] | None,
) -> tuple[int, int]:
    if inference_image_wh is not None:
        width, height = inference_image_wh
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid inference image size: {inference_image_wh}")
        return (int(width), int(height))

    widths = [spec.image_wh[0] for spec in camera_specs.values()]
    heights = [spec.image_wh[1] for spec in camera_specs.values()]
    aspect_ratios = {
        round(float(spec.image_wh[0]) / float(spec.image_wh[1]), 4)
        for spec in camera_specs.values()
    }
    if len(aspect_ratios) > 1:
        raise ValueError(
            "Camera image aspect ratios differ. Pass an explicit inference_image_wh to "
            "choose a common DA3 batch resolution."
        )
    return (max(widths), max(heights))


def _ensure_homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsics, dtype=np.float32)
    if ext.ndim != 3:
        raise ValueError(f"Expected extrinsics with shape (N, *, *), got {ext.shape}")
    if ext.shape[1:] == (4, 4):
        return ext
    if ext.shape[1:] == (3, 4):
        ext44 = np.zeros((ext.shape[0], 4, 4), dtype=np.float32)
        ext44[:, :3, :4] = ext
        ext44[:, 3, 3] = 1.0
        return ext44
    raise ValueError(f"Expected extrinsics with shape (N, 4, 4) or (N, 3, 4), got {ext.shape}")


def _extrinsics_to_camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    ext44 = _ensure_homogeneous_extrinsics(extrinsics)
    return np.linalg.inv(ext44)[:, :3, 3].astype(np.float32)


def _compute_static_camera_depth_scale(
    *,
    prediction_extrinsics: np.ndarray,
    batch_extrinsics: np.ndarray,
    batch_items: Sequence[BatchItem],
    scale_alignment_camera_ids: Sequence[str],
) -> dict[str, Any]:
    if len(scale_alignment_camera_ids) != 2:
        raise ValueError(
            "scale_alignment_camera_ids must contain exactly two camera IDs for "
            "baseline-ratio alignment."
        )
    camera_a, camera_b = [str(camera_id) for camera_id in scale_alignment_camera_ids]

    pred_centers = _extrinsics_to_camera_centers(prediction_extrinsics)
    gt_centers = _extrinsics_to_camera_centers(batch_extrinsics)

    by_camera_and_sequence: dict[tuple[str, int], int] = {}
    for item in batch_items:
        if not item.is_local:
            continue
        if item.camera_id not in {camera_a, camera_b}:
            continue
        by_camera_and_sequence[(item.camera_id, int(item.sequence_index))] = int(item.batch_index)

    shared_sequence_indices = sorted(
        set(
            sequence_index
            for camera_id, sequence_index in by_camera_and_sequence
            if camera_id == camera_a
        ).intersection(
            sequence_index
            for camera_id, sequence_index in by_camera_and_sequence
            if camera_id == camera_b
        )
    )
    if not shared_sequence_indices:
        raise RuntimeError(
            "Could not find overlapping non-anchor frames for "
            f"{camera_a} and {camera_b} to estimate depth scale."
        )

    min_baseline = 1e-6
    per_pair_scales: list[float] = []
    gt_baselines: list[float] = []
    pred_baselines: list[float] = []
    valid_sequence_indices: list[int] = []
    skipped_sequence_indices: list[int] = []

    for sequence_index in shared_sequence_indices:
        idx_a = by_camera_and_sequence[(camera_a, sequence_index)]
        idx_b = by_camera_and_sequence[(camera_b, sequence_index)]
        gt_baseline = float(np.linalg.norm(gt_centers[idx_a] - gt_centers[idx_b]))
        pred_baseline = float(np.linalg.norm(pred_centers[idx_a] - pred_centers[idx_b]))
        if gt_baseline <= min_baseline or pred_baseline <= min_baseline:
            skipped_sequence_indices.append(int(sequence_index))
            continue
        gt_baselines.append(gt_baseline)
        pred_baselines.append(pred_baseline)
        per_pair_scales.append(gt_baseline / pred_baseline)
        valid_sequence_indices.append(int(sequence_index))

    if not per_pair_scales:
        raise RuntimeError(
            f"Predicted baseline between {camera_a} and {camera_b} collapsed for every "
            "non-anchor frame in the chunk."
        )

    depth_scale = float(np.median(np.asarray(per_pair_scales, dtype=np.float32)))
    gt_baselines_arr = np.asarray(gt_baselines, dtype=np.float32)
    pred_baselines_arr = np.asarray(pred_baselines, dtype=np.float32)
    scale_arr = np.asarray(per_pair_scales, dtype=np.float32)
    return {
        "mode": "static_camera_baseline",
        "camera_ids": [camera_a, camera_b],
        "num_pairs": int(len(valid_sequence_indices)),
        "valid_sequence_indices": valid_sequence_indices,
        "skipped_sequence_indices": skipped_sequence_indices,
        "gt_baseline_median": float(np.median(gt_baselines_arr)),
        "pred_baseline_median": float(np.median(pred_baselines_arr)),
        "depth_scale": depth_scale,
        "scale_median_abs_dev": float(np.median(np.abs(scale_arr - depth_scale))),
    }


def _align_prediction_extrinsics_to_input(
    *,
    prediction_extrinsics: np.ndarray | None,
    batch_extrinsics: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if prediction_extrinsics is None:
        return None, None

    from depth_anything_3.utils.pose_align import align_poses_umeyama

    ransac = len(batch_extrinsics) >= 10
    _, _, scale, aligned_prediction_extrinsics = align_poses_umeyama(
        batch_extrinsics,
        prediction_extrinsics,
        ransac=ransac,
        return_aligned=True,
        random_state=42,
    )
    return _ensure_homogeneous_extrinsics(aligned_prediction_extrinsics).astype(np.float32), {
        "mode": "umeyama_pred_to_input",
        "ransac": bool(ransac),
        "scale": float(scale),
    }


def run_da3_multiview_batch(
    model: Any,
    batch_images: Sequence[np.ndarray],
    batch_extrinsics: np.ndarray,
    batch_intrinsics: np.ndarray,
    batch_image_wh: Sequence[tuple[int, int]],
    batch_items: Sequence[BatchItem],
    inference_image_wh: tuple[int, int],
    ref_view_strategy: str,
    scale_alignment_camera_ids: Sequence[str] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any] | None, np.ndarray | None, dict[str, Any] | None]:
    inference_w, inference_h = inference_image_wh
    working_h = inference_h // 14 * 14
    working_w = inference_w // 14 * 14
    if working_h <= 0 or working_w <= 0:
        raise ValueError(
            f"Invalid 14-divisible inference size {(working_w, working_h)} from "
            f"{inference_image_wh}"
        )

    working_images = []
    working_intrinsics = []
    for image, K, image_wh in zip(batch_images, batch_intrinsics, batch_image_wh):
        working_images.append(cv2.resize(image.copy(), (working_w, working_h)))
        working_intrinsics.append(
            scale_intrinsics_matrix(K, src_wh=image_wh, dst_wh=(working_w, working_h))
        )
    working_intrinsics = np.stack(working_intrinsics, axis=0).astype(np.float32)

    prediction = model.inference(
        image=working_images,
        extrinsics=batch_extrinsics,
        intrinsics=working_intrinsics,
        align_to_input_ext_scale=scale_alignment_camera_ids is None,
        skip_input_pose_alignment=scale_alignment_camera_ids is not None,
        ref_view_strategy=ref_view_strategy,
    )

    aligned_prediction_extrinsics, pose_alignment = _align_prediction_extrinsics_to_input(
        prediction_extrinsics=prediction.extrinsics,
        batch_extrinsics=batch_extrinsics,
    )

    scale_alignment = None
    if scale_alignment_camera_ids is not None:
        if prediction.extrinsics is None:
            raise RuntimeError(
                "DA3 did not return predicted extrinsics, so chunk depth scale could not "
                f"be estimated from {list(scale_alignment_camera_ids)}."
            )
        scale_alignment = _compute_static_camera_depth_scale(
            prediction_extrinsics=prediction.extrinsics,
            batch_extrinsics=batch_extrinsics,
            batch_items=batch_items,
            scale_alignment_camera_ids=scale_alignment_camera_ids,
        )
        prediction.depth = np.asarray(prediction.depth, dtype=np.float32) * float(
            scale_alignment["depth_scale"]
        )

    resized_depths = []
    for depth, image_wh in zip(prediction.depth, batch_image_wh):
        image_w, image_h = image_wh
        resized = cv2.resize(
            depth,
            (int(image_w), int(image_h)),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        resized_depths.append(np.asarray(resized, dtype=np.float32))
    return resized_depths, scale_alignment, aligned_prediction_extrinsics, pose_alignment


def build_chunk_batch(
    *,
    chunk_positions: Sequence[int],
    global_anchor_positions: Sequence[int],
    context_camera_ids: Sequence[str],
    output_camera_ids: set[str],
    frame_loaders: Mapping[str, ImageFolderFrameLoader],
    anchor_frame_cache: Mapping[tuple[str, int], np.ndarray],
    global_context_camera_id: str | None,
    global_context_positions: Sequence[int],
    global_context_frame_cache: Mapping[int, np.ndarray] | None,
    camera_specs: Mapping[str, CameraSequenceSpec],
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    list[BatchItem],
    list[int],
    list[int],
]:
    chunk_position_set = set(chunk_positions)
    effective_anchor_positions = [
        pos for pos in global_anchor_positions if pos not in chunk_position_set
    ]
    effective_global_context_positions: list[int] = []

    if global_context_camera_id is not None:
        if global_context_camera_id not in camera_specs:
            raise ValueError(
                f"Unknown global context camera {global_context_camera_id!r}. "
                f"Known cameras: {sorted(camera_specs)}"
            )
        global_context_spec = camera_specs[global_context_camera_id]
        reserved_positions = chunk_position_set.union(effective_anchor_positions)
        seen_context_positions: set[int] = set()
        for raw_pos in global_context_positions:
            sequence_index = int(raw_pos)
            if sequence_index < 0 or sequence_index >= len(global_context_spec.filenames):
                raise ValueError(
                    f"Global context frame {sequence_index} is out of range for "
                    f"{global_context_camera_id} ({len(global_context_spec.filenames)} frames)."
                )
            if sequence_index in reserved_positions or sequence_index in seen_context_positions:
                continue
            seen_context_positions.add(sequence_index)
            effective_global_context_positions.append(sequence_index)

    ordered_positions = [(pos, False) for pos in chunk_positions] + [
        (pos, True) for pos in effective_anchor_positions
    ]

    batch_images: list[np.ndarray] = []
    batch_extrinsics: list[np.ndarray] = []
    batch_intrinsics: list[np.ndarray] = []
    batch_image_wh: list[tuple[int, int]] = []
    batch_items: list[BatchItem] = []

    def append_batch_item(
        *,
        camera_id: str,
        sequence_index: int,
        image: np.ndarray,
        role: str,
    ) -> None:
        spec = camera_specs[camera_id]
        batch_index = len(batch_images)
        batch_images.append(image)
        batch_extrinsics.append(spec.extrinsics[sequence_index].copy())
        batch_intrinsics.append(spec.intrinsics[sequence_index].copy())
        batch_image_wh.append(spec.image_wh)
        batch_items.append(
            BatchItem(
                batch_index=batch_index,
                camera_id=camera_id,
                sequence_index=int(sequence_index),
                filename=spec.filenames[sequence_index],
                role=role,
                is_output=(camera_id in output_camera_ids and role == BATCH_ROLE_LOCAL),
            )
        )

    for sequence_index, is_anchor in ordered_positions:
        role = BATCH_ROLE_ANCHOR if is_anchor else BATCH_ROLE_LOCAL
        for camera_id in context_camera_ids:
            image = (
                anchor_frame_cache[(camera_id, sequence_index)]
                if is_anchor
                else frame_loaders[camera_id].read(sequence_index)
            )
            append_batch_item(
                camera_id=camera_id,
                sequence_index=sequence_index,
                image=image,
                role=role,
            )

    if global_context_camera_id is not None:
        for sequence_index in effective_global_context_positions:
            if global_context_frame_cache is not None and sequence_index in global_context_frame_cache:
                image = global_context_frame_cache[sequence_index]
            else:
                image = frame_loaders[global_context_camera_id].read(sequence_index)
            append_batch_item(
                camera_id=global_context_camera_id,
                sequence_index=sequence_index,
                image=image,
                role=BATCH_ROLE_GLOBAL_CONTEXT,
            )

    return (
        batch_images,
        np.stack(batch_extrinsics, axis=0).astype(np.float32),
        np.stack(batch_intrinsics, axis=0).astype(np.float32),
        batch_image_wh,
        batch_items,
        effective_anchor_positions,
        effective_global_context_positions,
    )


def _write_recon_payload(
    *,
    output_path: str | Path,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    image_wh: tuple[int, int],
    num_frames: int,
) -> None:
    c2w = np.linalg.inv(extrinsics_w2c).astype(np.float32)
    K_avg = intrinsics.mean(axis=0)
    intrinsics_arr = np.array(
        [
            float(K_avg[0, 0]),
            float(K_avg[1, 1]),
            float(K_avg[0, 2]),
            float(K_avg[1, 2]),
        ],
        dtype=np.float32,
    )
    recon_dict = {
        "tstamps": np.arange(num_frames, dtype=np.float32),
        "traj_c2w": c2w,
        "intrinsics": intrinsics_arr,
        "img_shape": np.array([image_wh[1], image_wh[0]], dtype=np.int32),
    }
    np.save(output_path, np.array(recon_dict, dtype=object))


def _write_framewise_depth_npz_files(
    *,
    output_dir: str | Path,
    spec: CameraSequenceSpec,
    merged_depths: Sequence[np.ndarray],
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    for filename, depth in zip(spec.filenames, merged_depths):
        stem = Path(filename).stem
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            **{FRAMEWISE_DEPTH_NPZ_KEY: np.asarray(depth, dtype=np.float32)},
        )


def save_camera_outputs(
    output_dir: str | Path,
    spec: CameraSequenceSpec,
    merged_depths: Sequence[np.ndarray],
    pred_aligned_extrinsics_w2c: np.ndarray | None = None,
    save_depth_npz_files: bool = False,
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_depth_npz_files:
        _write_framewise_depth_npz_files(
            output_dir=output_dir,
            spec=spec,
            merged_depths=merged_depths,
        )
    disp_dir = output_dir / "depthanything3_disp"
    disp_dir.mkdir(parents=True, exist_ok=True)

    for filename, depth in zip(spec.filenames, merged_depths):
        stem = Path(filename).stem
        disparity = 1.0 / np.clip(depth, a_min=1e-6, a_max=1e6)
        np.save(disp_dir / f"{stem}.npy", disparity.astype(np.float32))

    viz_depth_list(list(merged_depths), str(output_dir / "depth_anything3.mp4"))

    intrins_dict = {}
    for filename, K in zip(spec.filenames, spec.intrinsics):
        stem = Path(filename).stem
        intrins_dict[stem] = [
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        ]
    with open(output_dir / "depthanything3_intrins.json", "w", encoding="utf-8") as handle:
        json.dump(intrins_dict, handle, indent=1)

    _write_recon_payload(
        output_path=output_dir / "depthanything3_recon.npy",
        extrinsics_w2c=np.asarray(spec.extrinsics, dtype=np.float32),
        intrinsics=np.asarray(spec.intrinsics, dtype=np.float32),
        image_wh=spec.image_wh,
        num_frames=len(spec.filenames),
    )
    if pred_aligned_extrinsics_w2c is not None:
        _write_recon_payload(
            output_path=output_dir / PRED_ALIGNED_RECON_NAME,
            extrinsics_w2c=np.asarray(pred_aligned_extrinsics_w2c, dtype=np.float32),
            intrinsics=np.asarray(spec.intrinsics, dtype=np.float32),
            image_wh=spec.image_wh,
            num_frames=len(spec.filenames),
        )


def run_joint_multiview_depthanything3(
    *,
    camera_specs: Mapping[str, CameraSequenceSpec],
    output_root: str | Path,
    output_camera_ids: Sequence[str] | None = None,
    device: str = "cuda",
    da3_weights: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    chunk_size: int = 4,
    chunk_overlap: int = 1,
    num_anchors: int = 3,
    anchor_stride: int | None = None,
    anchor_positions: Sequence[int] | None = None,
    ref_view_strategy: str = "saddle_sim_range",
    merge_mode: str = "center_priority",
    scale_alignment_camera_ids: Sequence[str] | None = None,
    global_context_camera_id: str | None = None,
    global_context_count: int = 0,
    inference_image_wh: tuple[int, int] | None = None,
    save_pred_aligned_poses: bool = False,
    save_depth_npz_files: bool = False,
    save_manifest_name: str = "da3_joint_multiview_manifest.json",
) -> dict[str, Any]:
    if not camera_specs:
        raise ValueError("camera_specs cannot be empty")
    if merge_mode not in {"center_priority", "median"}:
        raise ValueError(f"Unsupported merge_mode: {merge_mode}")
    if global_context_count < 0:
        raise ValueError(f"global_context_count must be non-negative, got {global_context_count}")

    camera_ids = list(camera_specs.keys())
    output_camera_ids = camera_ids if output_camera_ids is None else list(output_camera_ids)
    output_camera_id_set = set(output_camera_ids)
    unknown_output_cameras = output_camera_id_set.difference(camera_ids)
    if unknown_output_cameras:
        raise ValueError(
            f"Unknown output cameras requested: {sorted(unknown_output_cameras)}. "
            f"Known cameras: {camera_ids}"
        )
    if scale_alignment_camera_ids is not None:
        if len(scale_alignment_camera_ids) != 2:
            raise ValueError(
                "scale_alignment_camera_ids must contain exactly two camera IDs."
            )
        unknown_scale_cameras = set(scale_alignment_camera_ids).difference(camera_ids)
        if unknown_scale_cameras:
            raise ValueError(
                "scale_alignment_camera_ids contains unknown cameras: "
                f"{sorted(unknown_scale_cameras)}. Known cameras: {camera_ids}"
            )
    if global_context_camera_id is not None and global_context_camera_id not in camera_ids:
        raise ValueError(
            f"Unknown global_context_camera_id {global_context_camera_id!r}. "
            f"Known cameras: {camera_ids}"
        )
    if global_context_count > 0 and global_context_camera_id is None:
        raise ValueError(
            "global_context_camera_id must be provided when global_context_count > 0."
        )

    num_frames = len(camera_specs[camera_ids[0]].filenames)
    for camera_id in camera_ids[1:]:
        if len(camera_specs[camera_id].filenames) != num_frames:
            raise ValueError(
                f"All cameras must have the same frame count. {camera_ids[0]} has "
                f"{num_frames}, {camera_id} has {len(camera_specs[camera_id].filenames)}"
            )

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inference_image_wh = resolve_inference_image_wh(camera_specs, inference_image_wh)
    chunks = build_chunks(num_frames=num_frames, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    anchor_positions, baselines, best_camera_id = choose_anchor_positions(
        extrinsics_by_camera={camera_id: camera_specs[camera_id].extrinsics for camera_id in camera_ids},
        context_camera_ids=camera_ids,
        num_frames=num_frames,
        num_anchors=num_anchors,
        anchor_stride=anchor_stride,
        explicit_anchor_positions=anchor_positions,
    )
    global_context_positions = (
        []
        if global_context_camera_id is None or global_context_count == 0
        else choose_global_context_positions(
            extrinsics=camera_specs[global_context_camera_id].extrinsics,
            num_context_frames=global_context_count,
        )
    )

    from lib_prior.depthanything3_wrapper import get_depthanything3_model

    model = get_depthanything3_model(device=device, weights=da3_weights)
    frame_loaders = {
        camera_id: ImageFolderFrameLoader(
            camera_specs[camera_id].image_dir,
            camera_specs[camera_id].filenames,
            target_wh=camera_specs[camera_id].image_wh,
        )
        for camera_id in camera_ids
    }
    candidate_counts = {
        camera_id: {sequence_index: 0 for sequence_index in range(num_frames)}
        for camera_id in output_camera_ids
    }
    candidate_depth_lists = {
        camera_id: {sequence_index: [] for sequence_index in range(num_frames)}
        for camera_id in output_camera_ids
    }
    best_depths = {
        camera_id: {} for camera_id in output_camera_ids
    }
    best_aligned_predicted_extrinsics = (
        {camera_id: {} for camera_id in output_camera_ids}
        if save_pred_aligned_poses
        else None
    )
    chunk_manifest: list[dict[str, Any]] = []

    try:
        anchor_frame_cache = {
            (camera_id, sequence_index): frame_loaders[camera_id].read(sequence_index)
            for camera_id in camera_ids
            for sequence_index in anchor_positions
        }
        global_context_frame_cache = (
            {}
            if global_context_camera_id is None
            else {
                sequence_index: frame_loaders[global_context_camera_id].read(sequence_index)
                for sequence_index in global_context_positions
            }
        )

        for chunk_index, chunk_positions in enumerate(tqdm(chunks, desc="DA3 chunks")):
            (
                batch_images,
                batch_extrinsics,
                batch_intrinsics,
                batch_image_wh,
                batch_items,
                effective_anchor_positions,
                effective_global_context_positions,
            ) = build_chunk_batch(
                chunk_positions=chunk_positions,
                global_anchor_positions=anchor_positions,
                context_camera_ids=camera_ids,
                output_camera_ids=output_camera_id_set,
                frame_loaders=frame_loaders,
                anchor_frame_cache=anchor_frame_cache,
                global_context_camera_id=global_context_camera_id,
                global_context_positions=global_context_positions,
                global_context_frame_cache=global_context_frame_cache,
                camera_specs=camera_specs,
            )

            quality = check_trajectory_quality(batch_extrinsics)
            if scale_alignment_camera_ids is None and not quality["can_align"]:
                raise RuntimeError(
                    f"Chunk {chunk_index} batch is low-rank: {quality['reason']}. "
                    "Increase anchors, reduce chunk size, or adjust the context cameras."
                )

            (
                resized_depths,
                scale_alignment,
                aligned_prediction_extrinsics,
                pose_alignment,
            ) = run_da3_multiview_batch(
                model=model,
                batch_images=batch_images,
                batch_extrinsics=batch_extrinsics,
                batch_intrinsics=batch_intrinsics,
                batch_image_wh=batch_image_wh,
                batch_items=batch_items,
                inference_image_wh=inference_image_wh,
                ref_view_strategy=ref_view_strategy,
                scale_alignment_camera_ids=scale_alignment_camera_ids,
            )

            chunk_center = 0.5 * (chunk_positions[0] + chunk_positions[-1])
            batch_role_counts = {
                BATCH_ROLE_LOCAL: 0,
                BATCH_ROLE_ANCHOR: 0,
                BATCH_ROLE_GLOBAL_CONTEXT: 0,
            }
            for item in batch_items:
                batch_role_counts[item.role] = batch_role_counts.get(item.role, 0) + 1
            selected_output_entries = []
            for item in batch_items:
                if not item.is_output:
                    continue

                candidate_counts[item.camera_id][item.sequence_index] += 1
                depth = resized_depths[item.batch_index]
                pose_merge_score = abs(item.sequence_index - chunk_center)
                if (
                    save_pred_aligned_poses
                    and aligned_prediction_extrinsics is not None
                    and best_aligned_predicted_extrinsics is not None
                ):
                    previous_pose = best_aligned_predicted_extrinsics[item.camera_id].get(
                        item.sequence_index
                    )
                    if previous_pose is None or pose_merge_score < previous_pose[0]:
                        best_aligned_predicted_extrinsics[item.camera_id][item.sequence_index] = (
                            float(pose_merge_score),
                            aligned_prediction_extrinsics[item.batch_index].astype(np.float32),
                            int(chunk_index),
                        )
                if merge_mode == "median":
                    candidate_depth_lists[item.camera_id][item.sequence_index].append(depth)
                    merge_score = None
                    selected_for_output = True
                else:
                    merge_score = pose_merge_score
                    previous = best_depths[item.camera_id].get(item.sequence_index)
                    selected_for_output = previous is None or merge_score < previous[0]
                    if selected_for_output:
                        best_depths[item.camera_id][item.sequence_index] = (
                            float(merge_score),
                            depth,
                            int(chunk_index),
                        )

                selected_output_entries.append(
                    {
                        "batch_index": int(item.batch_index),
                        "camera_id": item.camera_id,
                        "sequence_index": int(item.sequence_index),
                        "filename": item.filename,
                        "selected": bool(selected_for_output),
                        "merge_score": None if merge_score is None else float(merge_score),
                    }
                )

            chunk_manifest.append(
                {
                    "chunk_index": int(chunk_index),
                    "chunk_positions": [int(pos) for pos in chunk_positions],
                    "anchor_positions": [int(pos) for pos in effective_anchor_positions],
                    "global_context_positions": [
                        int(pos) for pos in effective_global_context_positions
                    ],
                    "batch_size": int(len(batch_items)),
                    "batch_role_counts": {
                        role: int(count) for role, count in batch_role_counts.items()
                    },
                    "batch_quality": {
                        "reason": quality["reason"],
                        "stats": quality["stats"],
                    },
                    "pred_pose_alignment": pose_alignment,
                    "scale_alignment": scale_alignment,
                    "selected_output_entries": selected_output_entries,
                }
            )
    finally:
        del model
        torch.cuda.empty_cache()

    saved_output_dirs: dict[str, str] = {}
    merge_manifest: dict[str, list[dict[str, Any]]] = {}
    for camera_id in output_camera_ids:
        spec = camera_specs[camera_id]
        merged_depths = []
        merged_pred_aligned_extrinsics = []
        merge_entries = []
        for sequence_index, filename in enumerate(spec.filenames):
            if merge_mode == "median":
                candidates = candidate_depth_lists[camera_id][sequence_index]
                if not candidates:
                    raise RuntimeError(
                        f"No predictions were produced for {camera_id} frame {filename}"
                    )
                merged_depth = np.median(np.stack(candidates, axis=0), axis=0).astype(np.float32)
                merge_entries.append(
                    {
                        "sequence_index": int(sequence_index),
                        "filename": filename,
                        "num_candidates": int(candidate_counts[camera_id][sequence_index]),
                        "merge_mode": merge_mode,
                    }
                )
            else:
                if sequence_index not in best_depths[camera_id]:
                    raise RuntimeError(
                        f"No predictions were produced for {camera_id} frame {filename}"
                    )
                merge_score, merged_depth, source_chunk = best_depths[camera_id][sequence_index]
                merged_depth = merged_depth.astype(np.float32)
                merge_entries.append(
                    {
                        "sequence_index": int(sequence_index),
                        "filename": filename,
                        "num_candidates": int(candidate_counts[camera_id][sequence_index]),
                        "merge_mode": merge_mode,
                        "chosen_chunk": int(source_chunk),
                        "merge_score": float(merge_score),
                    }
                )
            merged_depths.append(merged_depth)
            if (
                save_pred_aligned_poses
                and best_aligned_predicted_extrinsics is not None
                and sequence_index in best_aligned_predicted_extrinsics[camera_id]
            ):
                _, aligned_ext, pose_chunk = best_aligned_predicted_extrinsics[camera_id][sequence_index]
                merged_pred_aligned_extrinsics.append(aligned_ext.astype(np.float32))
                merge_entries[-1]["pred_aligned_pose_chunk"] = int(pose_chunk)

        camera_output_dir = output_root / camera_id
        camera_output_dir.mkdir(parents=True, exist_ok=True)
        pred_aligned_extrinsics_w2c = (
            None
            if not save_pred_aligned_poses or len(merged_pred_aligned_extrinsics) != len(spec.filenames)
            else np.stack(merged_pred_aligned_extrinsics, axis=0).astype(np.float32)
        )
        save_camera_outputs(
            camera_output_dir,
            spec,
            merged_depths,
            pred_aligned_extrinsics_w2c=pred_aligned_extrinsics_w2c,
            save_depth_npz_files=save_depth_npz_files,
        )
        saved_output_dirs[camera_id] = str(camera_output_dir)
        merge_manifest[camera_id] = merge_entries

    manifest = {
        "mode": "joint_multiview_folder_da3",
        "output_root": str(output_root),
        "camera_ids": camera_ids,
        "output_camera_ids": output_camera_ids,
        "num_frames": int(num_frames),
        "da3_weights": da3_weights,
        "device": device,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "num_anchors": int(num_anchors),
        "anchor_stride": None if anchor_stride is None else int(anchor_stride),
        "anchor_selection_mode": "all_camera_baseline_from_start",
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "best_anchor_camera": best_camera_id,
        "all_camera_baselines": {camera_id: float(value) for camera_id, value in baselines.items()},
        "global_context_mode": (
            None
            if global_context_camera_id is None or global_context_count == 0
            else "camera_farthest_point_sampling"
        ),
        "global_context_camera_id": (
            None if global_context_count == 0 else global_context_camera_id
        ),
        "global_context_count_requested": int(global_context_count),
        "global_context_positions": [int(pos) for pos in global_context_positions],
        "ref_view_strategy": ref_view_strategy,
        "merge_mode": merge_mode,
        "scale_alignment_mode": (
            None if scale_alignment_camera_ids is None else "static_camera_baseline"
        ),
        "scale_alignment_camera_ids": (
            None
            if scale_alignment_camera_ids is None
            else [str(camera_id) for camera_id in scale_alignment_camera_ids]
        ),
        "inference_image_wh": [int(inference_image_wh[0]), int(inference_image_wh[1])],
        "save_pred_aligned_poses": bool(save_pred_aligned_poses),
        "save_depth_npz_files": bool(save_depth_npz_files),
        "depth_npz_key": FRAMEWISE_DEPTH_NPZ_KEY if save_depth_npz_files else None,
        "cameras": {
            camera_id: {
                "image_dir": camera_specs[camera_id].image_dir,
                "num_frames": int(len(camera_specs[camera_id].filenames)),
                "image_wh": [
                    int(camera_specs[camera_id].image_wh[0]),
                    int(camera_specs[camera_id].image_wh[1]),
                ],
            }
            for camera_id in camera_ids
        },
        "output_dirs": saved_output_dirs,
        "chunks": chunk_manifest,
        "merge_manifest": merge_manifest,
        "pred_aligned_recon_name": (
            PRED_ALIGNED_RECON_NAME if save_pred_aligned_poses else None
        ),
    }
    with open(output_root / save_manifest_name, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
