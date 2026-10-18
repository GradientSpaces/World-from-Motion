from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from .worldtree_benchmark import (
    PREFERRED_NPZ_KEYS,
    FrameRecord,
    SceneLayout,
    discover_scene_layouts,
)

try:  # pragma: no cover - pillow compatibility
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - pillow compatibility
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST

_TORCH_GENERATOR_SEED_MASK = (1 << 63) - 1


@dataclass(frozen=True)
class SceneChunkSpec:
    global_index: int
    chunk_index: int
    scene_name: str
    scene_root: Path
    unit_id: str
    source_cam: str
    target_cam: str
    reference_camera: str
    window_start: int
    window_end: int
    keep_start: int
    keep_end: int
    records: tuple[FrameRecord, ...]


def build_chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int, int, int]]:
    if total < chunk_size:
        raise ValueError(f"Expected at least {chunk_size} frames, got {total}")
    ranges: list[tuple[int, int, int, int]] = []
    remainder = total % chunk_size
    full_limit = total - remainder
    for start in range(0, full_limit, chunk_size):
        ranges.append((start, start + chunk_size, 0, chunk_size))
    if remainder:
        start = total - chunk_size
        keep_start = chunk_size - remainder
        ranges.append((start, total, keep_start, chunk_size))
    return ranges


def benchmark_sequence_seed_key(chunk: SceneChunkSpec) -> str:
    return f"{chunk.scene_name}/{chunk.unit_id}/{chunk.target_cam}"


def resolve_scene_chunk_seed(base_seed: int | None, chunk: SceneChunkSpec) -> int | None:
    if base_seed is None:
        return None
    digest = hashlib.sha256(benchmark_sequence_seed_key(chunk).encode("utf-8")).digest()
    sequence_offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (int(base_seed) + sequence_offset) & _TORCH_GENERATOR_SEED_MASK


class SceneBenchmarkDataset:
    def __init__(
        self,
        benchmark_root: Path,
        scene_names: Sequence[str] | None,
        chunk_size: int,
        max_frames: int | None = None,
        frame_stride: int = 1,
        dense_prefix: int = 0,
    ) -> None:
        self.benchmark_root = benchmark_root
        self.chunk_size = int(chunk_size)
        self.max_frames = None if max_frames is None else int(max_frames)
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {self.max_frames}")
        self.frame_stride = int(frame_stride)
        self.dense_prefix = int(dense_prefix)
        if self.frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {self.frame_stride}")
        if self.dense_prefix < 0:
            raise ValueError(f"dense_prefix must be non-negative, got {self.dense_prefix}")
        self.layouts: tuple[SceneLayout, ...] = tuple(
            discover_scene_layouts(
                benchmark_root,
                scene_names=scene_names,
                strict_files=False,
            )
        )
        self.jobs: list[SceneChunkSpec] = []
        global_index = 0
        for layout in self.layouts:
            for unit in layout.evaluation_units:
                records = layout.manifest(unit.target_camera).records
                if self.max_frames is not None:
                    records = records[: self.max_frames]
                records = self._select_records(records)
                for chunk_index, (start, end, keep_start, keep_end) in enumerate(
                    build_chunk_ranges(len(records), self.chunk_size)
                ):
                    self.jobs.append(
                        SceneChunkSpec(
                            global_index=global_index,
                            chunk_index=chunk_index,
                            scene_name=layout.scene_name,
                            scene_root=layout.scene_root,
                            unit_id=unit.unit_id,
                            source_cam=unit.source_camera,
                            target_cam=unit.target_camera,
                            reference_camera=unit.reference_camera,
                            window_start=start,
                            window_end=end,
                            keep_start=keep_start,
                            keep_end=keep_end,
                            records=records[start:end],
                        )
                    )
                    global_index += 1

    def _select_records(
        self, records: tuple[FrameRecord, ...]
    ) -> tuple[FrameRecord, ...]:
        """Subsample records with a dense prefix followed by a constant stride.

        The first ``dense_prefix`` records are kept at stride 1; the remaining
        records are sampled every ``frame_stride`` frames. With ``frame_stride==1``
        and ``dense_prefix==0`` this is a no-op so existing runs are unaffected.
        Camera poses are later gathered via ``FrameRecord.frame_idx`` so the
        non-contiguous selection stays aligned with the original trajectory.
        """
        if self.frame_stride <= 1 and self.dense_prefix <= 0:
            return records
        total = len(records)
        prefix = min(self.dense_prefix, total)
        selected_indices = sorted(
            set(range(prefix)) | set(range(prefix, total, self.frame_stride))
        )
        return tuple(records[i] for i in selected_indices)

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int) -> SceneChunkSpec:
        return self.jobs[idx]


def load_pil_frame(path: Path, size: tuple[int, int] | None, resample: int) -> Image.Image:
    with Image.open(path) as image:
        frame = image.convert("RGB")
    if size is not None and frame.size != size:
        frame = frame.resize(size, resample=resample)
    return frame


def load_pil_sequence(
    directory: Path,
    frame_names: Sequence[str],
    size: tuple[int, int] | None,
    resample: int,
) -> list[Image.Image]:
    return [
        load_pil_frame(directory / f"{frame_name}.png", size=size, resample=resample)
        for frame_name in frame_names
    ]


def load_npz_array(path: Path, kind: str) -> np.ndarray:
    preferred_keys = PREFERRED_NPZ_KEYS[kind]
    with np.load(path) as data:
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])
        if len(data.files) == 0:
            raise ValueError(f"Empty npz file: {path}")
        return np.asarray(data[data.files[0]])


def stage_chunk_assets(
    chunk: SceneChunkSpec,
    output_root: Path,
    use_scene_coords: bool,
) -> Path:
    stage_root = (
        output_root / "_tmp" / chunk.scene_name / chunk.unit_id / f"chunk_{chunk.chunk_index:05d}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)

    source_c2w = load_npz_array(
        chunk.scene_root / "cameras" / f"{chunk.source_cam}_c2w.npz",
        kind="camera",
    )
    target_c2w = load_npz_array(
        chunk.scene_root / "cameras" / f"{chunk.target_cam}_c2w.npz",
        kind="camera",
    )
    cameras_path = stage_root / "cameras.npz"
    # Gather poses by each record's original trajectory index. This matches the
    # contiguous window slice for standard chunks (frame_idx == window position)
    # while staying correct for non-contiguous (strided) frame selections.
    frame_indices = [int(record.frame_idx) for record in chunk.records]
    camera_payload = {
        f"{chunk.source_cam}_c2w": source_c2w[frame_indices].astype(np.float32),
        f"{chunk.target_cam}_c2w": target_c2w[frame_indices].astype(np.float32),
    }
    np.savez_compressed(cameras_path, **camera_payload)

    if use_scene_coords:
        scene_coords_dir = stage_root / "scene_coords"
        scene_coords_dir.mkdir(parents=True, exist_ok=True)
        target_frame_names = [record.frame_name for record in chunk.records]
        source_frame_names = [record.source_input_frame_name for record in chunk.records]
        for cam_name in (chunk.source_cam, chunk.target_cam):
            frame_names = source_frame_names if cam_name == chunk.source_cam else target_frame_names
            arrays = [
                load_npz_array(
                    chunk.scene_root / "scene_coords" / cam_name / f"{frame_name}.npz",
                    kind="scene_coords",
                )
                for frame_name in frame_names
            ]
            stacked = np.stack(arrays, axis=0).astype(np.float16)
            np.savez_compressed(scene_coords_dir / f"{cam_name}.npz", scene_coords=stacked)
    return cameras_path


def resolve_inference_frame_size(
    chunk: SceneChunkSpec,
    inference_res: tuple[int, int] | None,
) -> tuple[int, int]:
    if inference_res is not None:
        height, width = inference_res
        return width, height
    first_frame = chunk.scene_root / "rgb" / chunk.target_cam / f"{chunk.records[0].frame_name}.png"
    with Image.open(first_frame) as image:
        return image.convert("RGB").size


def build_chunk_inputs(
    chunk: SceneChunkSpec,
    *,
    inference_res: tuple[int, int] | None,
    use_scene_flow: bool,
    cameras_path: Path,
) -> tuple[dict[str, object], list[str]]:
    frame_names = [record.frame_name for record in chunk.records]
    source_frame_names = [record.source_input_frame_name for record in chunk.records]
    size = resolve_inference_frame_size(chunk, inference_res)
    scene_root = chunk.scene_root

    inputs: dict[str, object] = {
        "source_video": load_pil_sequence(
            scene_root / "gt" / chunk.source_cam,
            source_frame_names,
            size,
            BILINEAR,
        ),
        "cameras_npz": str(cameras_path),
        "target_cam": chunk.target_cam,
        "source_cam": chunk.source_cam,
        "vace_video": load_pil_sequence(scene_root / "rgb" / chunk.target_cam, frame_names, size, BILINEAR),
        "vace_depth": load_pil_sequence(scene_root / "depth" / chunk.target_cam, frame_names, size, BILINEAR),
        "vace_normal": load_pil_sequence(
            scene_root / "normal" / chunk.target_cam,
            frame_names,
            size,
            BILINEAR,
        ),
        "vace_alpha": load_pil_sequence(
            scene_root / "alpha" / chunk.target_cam, frame_names, size, NEAREST
        ),
        "source_vace_video": load_pil_sequence(
            scene_root / "rgb" / chunk.source_cam,
            source_frame_names,
            size,
            BILINEAR,
        ),
        "source_vace_depth": load_pil_sequence(
            scene_root / "depth" / chunk.source_cam,
            source_frame_names,
            size,
            BILINEAR,
        ),
        "source_vace_normal": load_pil_sequence(
            scene_root / "normal" / chunk.source_cam,
            source_frame_names,
            size,
            BILINEAR,
        ),
        "source_vace_alpha": load_pil_sequence(
            scene_root / "alpha" / chunk.source_cam,
            source_frame_names,
            size,
            NEAREST,
        ),
    }
    if use_scene_flow:
        inputs["scene_flow_target"] = load_pil_sequence(
            scene_root / "scene_flow_rgb" / chunk.target_cam,
            frame_names,
            size,
            BILINEAR,
        )
        inputs["scene_flow_source"] = load_pil_sequence(
            scene_root / "scene_flow_rgb" / chunk.source_cam,
            source_frame_names,
            size,
            BILINEAR,
        )
    return inputs, frame_names


def tensor_to_uint8_frames(pred_tensor: torch.Tensor) -> np.ndarray:
    pred = pred_tensor.detach().float().clamp(-1.0, 1.0)
    pred = ((pred[0].permute(1, 2, 3, 0) + 1.0) * 127.5).round()
    pred = pred.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    return pred


def save_chunk_predictions(
    chunk: SceneChunkSpec,
    output_root: Path,
    pred_subdir: str,
    pred_frames: np.ndarray,
) -> int:
    pred_len = pred_frames.shape[0]
    keep_end = min(chunk.keep_end, pred_len)
    kept_records = list(chunk.records[chunk.keep_start : keep_end])
    kept_frames = pred_frames[chunk.keep_start : keep_end]
    out_dir = output_root / chunk.scene_name / pred_subdir / chunk.target_cam
    out_dir.mkdir(parents=True, exist_ok=True)
    for record, frame in zip(kept_records, kept_frames):
        iio.imwrite(out_dir / f"{record.frame_name}.png", frame)
    return len(kept_records)
