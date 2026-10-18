from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image


DEFAULT_BENCHMARK_ROOT = ""
DEFAULT_DATA_ROOT = ""
MASK_DIR_CANDIDATE_SUFFIXES = (
    ("covisible", "2x", "val"),
    ("covisible", "1x", "val"),
    ("flow3d_preprocessed", "covisible", "1x", "val"),
    ("flow3d_preprocessed", "covisible", "2x", "val"),
)
PAIR_RE = re.compile(r"^pair_(cam\d+)_(cam\d+)$")
PREFERRED_NPZ_KEYS = {
    "camera": ("c2w", "arr_0"),
    "scene_coords": ("scene_coords", "coords", "xyz", "arr_0"),
    "scene_flow": ("scene_flow", "flow", "arr_0"),
}
try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = 2
    NEAREST = 0

SYNC_POLL_SECONDS = 10.0
SYNC_LOG_INTERVAL_SECONDS = 60.0
SYNC_TIMEOUT_SECONDS = 6 * 60 * 60
VALID_OPACITY_THRESHOLD = 0.05
TEST_DYNAMIC_DIRNAME = "test_dynamic"
MASKED_METRIC_KEYS = ("masked_mPSNR", "masked_mSSIM", "masked_mLPIPS")
FULL_METRIC_KEYS = ("full_mPSNR", "full_mSSIM", "full_mLPIPS")
VALID_METRIC_KEYS = ("valid_mPSNR", "valid_mSSIM", "valid_mLPIPS")
DYNAMIC_VALID_METRIC_KEYS = (
    "dynamic_valid_mPSNR",
    "dynamic_valid_mSSIM",
    "dynamic_valid_mLPIPS",
)
STATIC_VALID_METRIC_KEYS = (
    "static_valid_mPSNR",
    "static_valid_mSSIM",
    "static_valid_mLPIPS",
)


@dataclass(frozen=True)
class FrameRecord:
    frame_idx: int
    frame_name: str
    source_frame_name: str
    source_input_frame_name: str
    has_gt: bool


@dataclass(frozen=True)
class BenchmarkSequence:
    scene_name: str
    pair_name: str
    pair_root: Path
    source_cam: str
    target_cam: str
    records: tuple[FrameRecord, ...]


@dataclass(frozen=True)
class ChunkSpec:
    global_index: int
    chunk_index: int
    scene_name: str
    pair_name: str
    pair_root: Path
    source_cam: str
    target_cam: str
    window_start: int
    window_end: int
    keep_start: int
    keep_end: int
    records: tuple[FrameRecord, ...]


@dataclass(frozen=True)
class DyCheckBenchmarkConfig:
    benchmark_root: Path
    data_root: Path
    output_root: Path
    seq_names: tuple[str, ...] | None = None
    chunk_size: int = 73
    pred_subdir: str = "pred_rgb"
    prompt: str = " "
    negative_prompt: str | None = None
    inference_res: tuple[int, int] | None = None
    eval_res: str | tuple[int, int] = "pred"
    num_inference_steps: int = 20
    cfg_scale: float = 1.0
    target_gbuffer_cfg_scale: float = 1.0
    target_gbuffer_dropout_scope: str = "target_only"
    target_gbuffer_uncond_fill_mode: str = "zero"
    target_gbuffer_guidance_mode: str = "cfg"
    target_gbuffer_apg_eta: float = 0.0
    target_gbuffer_apg_norm_threshold: float = 8.0
    target_gbuffer_apg_momentum: float | None = -0.5
    target_gbuffer_apg_space: str = "x0"
    vace_scale: float = 1.0
    seed: int | None = 0
    tiled: bool = False
    use_scene_flow: bool = False
    split_cam_embed: bool = False
    zero_source_cam_emb_when_missing: bool = False
    disable_gbuffer: bool = False
    # Keep source-video conditioning explicit so DyCheck can match runs that
    # intentionally disable source_latents / source_video at inference time.
    use_source_video: bool = True
    # When set, restrict the target/source VACE signals fed to the pipeline to
    # exactly these keys. This is required whenever the trained VACE patch
    # embedding was shrunk to a compact channel layout (e.g. vace_video only,
    # 16 channels) via --drop_cached_vace_groups / --strip_source_gbuffer during
    # training; otherwise this benchmark would load all 4 target + 4 source
    # gbuffer signals and build a 64-channel VACE context that cannot be
    # consumed by a 16-channel vace_patch_embedding. Leave as None to preserve
    # legacy behavior (load the full 4-signal target + 4-signal source set, with
    # scene_flow gated by use_scene_flow).
    active_target_vace_keys: tuple[str, ...] | None = None
    active_source_vace_keys: tuple[str, ...] | None = None
    scene_coord_pe: bool = False
    scene_coord_xyzt_pe: bool = False
    scene_coord_gated_residual: bool = False
    metrics_device: str = "auto"
    lpips_net: str = "alex"
    mask_frame_name_mode: str = "source"
    binarize_masks: bool = False

    def normalized_seq_names(self) -> list[str] | None:
        if self.seq_names is None:
            return None
        return [name.strip() for name in self.seq_names if str(name).strip()]


@dataclass
class DyCheckBenchmarkRunResult:
    dataset: "DyCheckBenchmarkDataset"
    metrics_dir: Path
    per_scene_payloads: list[dict]
    summary: dict | None
    summary_path: Path | None


_TORCH_GENERATOR_SEED_MASK = (1 << 63) - 1


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_resolution(resolution: str | tuple[int, int] | None) -> str:
    if resolution is None:
        return "benchmark"
    if isinstance(resolution, tuple):
        return f"{resolution[0]}x{resolution[1]}"
    return str(resolution)


def metrics_dir_name(eval_res: str | tuple[int, int]) -> str:
    if eval_res == "pred":
        return "metrics"
    return f"metrics_{format_resolution(eval_res)}"


def scene_metrics_path(metrics_dir: Path, scene_name: str) -> Path:
    return metrics_dir / f"{scene_name}.json"


def pair_metrics_path(metrics_dir: Path, scene_name: str, pair_name: str) -> Path:
    return metrics_dir / "_pair_metrics" / scene_name / f"{pair_name}.json"


def sync_dir(metrics_dir: Path, run_token: str | None = None) -> Path:
    if run_token is None:
        return metrics_dir
    return metrics_dir / "_sync" / run_token


def rank_metrics_done_path(metrics_dir: Path, rank: int, run_token: str | None = None) -> Path:
    return sync_dir(metrics_dir, run_token) / f"_rank_{rank:03d}_metrics_complete.json"


def dycheck_postprocess_done_path(metrics_dir: Path, run_token: str | None = None) -> Path:
    return sync_dir(metrics_dir, run_token) / "_dycheck_postprocess_complete.json"


def build_sync_run_token(accelerator) -> str:
    token = f"run_{time.time_ns()}" if accelerator.is_main_process else None
    if (
        accelerator.num_processes > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        payload = [token]
        torch.distributed.broadcast_object_list(payload, src=0)
        token = payload[0]
    if token is None:
        raise RuntimeError("Failed to establish DyCheck sync run token.")
    return str(token)


def wait_for_files(
    paths: list[Path],
    *,
    timeout_seconds: float = SYNC_TIMEOUT_SECONDS,
    poll_seconds: float = SYNC_POLL_SECONDS,
    log_interval_seconds: float = SYNC_LOG_INTERVAL_SECONDS,
    progress_prefix: str | None = None,
    print_fn=None,
) -> None:
    if len(paths) == 0:
        return
    if print_fn is None:
        print_fn = lambda msg: print(msg, flush=True)

    start_time = time.monotonic()
    last_log_time = start_time - log_interval_seconds
    pending = [path for path in paths if not path.is_file()]
    while pending:
        now = time.monotonic()
        elapsed = now - start_time
        if elapsed >= timeout_seconds:
            preview = ", ".join(path.name for path in pending[:5])
            if len(pending) > 5:
                preview += ", ..."
            raise TimeoutError(
                f"{progress_prefix or 'Timed out waiting for files'} after "
                f"{elapsed / 60.0:.1f} minutes. Missing: {preview}"
            )
        if progress_prefix is not None and now - last_log_time >= log_interval_seconds:
            preview = ", ".join(path.name for path in pending[:3])
            if len(pending) > 3:
                preview += ", ..."
            print_fn(
                f"{progress_prefix}: waiting on {len(pending)} file(s) "
                f"after {elapsed / 60.0:.1f} min. Pending: {preview}"
            )
            last_log_time = now
        time.sleep(poll_seconds)
        pending = [path for path in paths if not path.is_file()]


def parse_pair_name(pair_name: str) -> tuple[str, str]:
    match = PAIR_RE.match(pair_name)
    if match is None:
        raise ValueError(f"Unexpected pair directory name: {pair_name}")
    return match.group(1), match.group(2)


def infer_scene_names(benchmark_root: Path, seq_names: list[str] | None) -> list[str]:
    if seq_names is not None:
        return [name.strip() for name in seq_names if name.strip()]
    names = []
    for path in sorted(benchmark_root.iterdir()):
        if not path.is_dir():
            continue
        has_pair = any(child.is_dir() and PAIR_RE.match(child.name) for child in path.iterdir())
        if has_pair:
            names.append(path.name)
    return names


def _frame_name_suffix(frame_name: str) -> str:
    text = str(frame_name).strip()
    if "_" not in text:
        return text
    return text.split("_", 1)[1]


def _camera_name_to_prefix(camera_name: str | None, camera_id: str | None) -> str | None:
    if camera_name is not None:
        camera_text = str(camera_name).strip()
        match = re.fullmatch(r"cam(\d+)", camera_text)
        if match is not None:
            return match.group(1)
        if camera_text != "":
            return camera_text
    if camera_id is not None:
        camera_id_text = str(camera_id).strip()
        if camera_id_text != "":
            return camera_id_text
    return None


def infer_source_input_frame_name(frame: dict, metadata_payload: dict) -> str:
    reference_frame_name = frame.get("reference_frame_name")
    if reference_frame_name is not None and str(reference_frame_name).strip():
        return str(reference_frame_name)
    frame_name = str(frame["frame_name"])
    source_prefix = _camera_name_to_prefix(
        metadata_payload.get("reference_camera_label"),
        metadata_payload.get("reference_camera_id"),
    )
    if source_prefix is None:
        return frame_name
    return f"{source_prefix}_{_frame_name_suffix(frame_name)}"


def resolve_source_frame_name(frame: dict[str, object]) -> str:
    source_frame_name = frame.get("source_frame_name")
    if source_frame_name is None:
        source_frame_name = frame.get("render_pose_source_frame_name", frame["frame_name"])
    return str(source_frame_name)


def load_frame_records(metadata_path: Path) -> list[FrameRecord]:
    payload = load_json(metadata_path)
    return [
        FrameRecord(
            frame_idx=int(frame["frame_idx"]),
            frame_name=str(frame["frame_name"]),
            source_frame_name=resolve_source_frame_name(frame),
            source_input_frame_name=infer_source_input_frame_name(frame, payload),
            has_gt=bool(frame["has_gt"]),
        )
        for frame in payload["frames"]
    ]


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


def benchmark_sequence_seed_key(chunk: ChunkSpec) -> str:
    return f"{chunk.scene_name}/{chunk.pair_name}/{chunk.target_cam}"


def resolve_benchmark_chunk_seed(base_seed: int | None, chunk: ChunkSpec) -> int | None:
    if base_seed is None:
        return None

    # Keep all chunks from the same evaluated camera sequence on the same
    # latent-noise trajectory while still letting the caller vary a base seed.
    sequence_key = benchmark_sequence_seed_key(chunk)
    digest = hashlib.sha256(sequence_key.encode("utf-8")).digest()
    sequence_offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (int(base_seed) + sequence_offset) & _TORCH_GENERATOR_SEED_MASK


class DyCheckBenchmarkDataset:
    def __init__(
        self,
        benchmark_root: Path,
        seq_names: list[str] | None,
        chunk_size: int,
    ) -> None:
        self.benchmark_root = benchmark_root
        self.seq_names = infer_scene_names(benchmark_root, seq_names)
        self.chunk_size = chunk_size
        self.sequences: list[BenchmarkSequence] = []
        self.jobs: list[ChunkSpec] = []
        global_index = 0
        for scene_name in self.seq_names:
            scene_root = benchmark_root / scene_name
            for pair_root in sorted(scene_root.iterdir()):
                if not pair_root.is_dir() or PAIR_RE.match(pair_root.name) is None:
                    continue
                source_cam, target_cam = parse_pair_name(pair_root.name)
                metadata_path = pair_root / "metadata" / f"{target_cam}_frames.json"
                records = tuple(load_frame_records(metadata_path))
                sequence = BenchmarkSequence(
                    scene_name=scene_name,
                    pair_name=pair_root.name,
                    pair_root=pair_root,
                    source_cam=source_cam,
                    target_cam=target_cam,
                    records=records,
                )
                self.sequences.append(sequence)
                for chunk_index, (start, end, keep_start, keep_end) in enumerate(
                    build_chunk_ranges(len(records), chunk_size)
                ):
                    self.jobs.append(
                        ChunkSpec(
                            global_index=global_index,
                            chunk_index=chunk_index,
                            scene_name=scene_name,
                            pair_name=pair_root.name,
                            pair_root=pair_root,
                            source_cam=source_cam,
                            target_cam=target_cam,
                            window_start=start,
                            window_end=end,
                            keep_start=keep_start,
                            keep_end=keep_end,
                            records=records[start:end],
                        )
                    )
                    global_index += 1

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int) -> ChunkSpec:
        return self.jobs[idx]


def load_pil_frame(path: Path, size: tuple[int, int] | None, resample: int) -> Image.Image:
    with Image.open(path) as image:
        frame = image.convert("RGB")
    if size is not None and frame.size != size:
        frame = frame.resize(size, resample=resample)
    return frame


def load_pil_sequence(
    directory: Path,
    frame_names: list[str],
    size: tuple[int, int] | None,
    resample: int,
) -> list[Image.Image]:
    return [
        load_pil_frame(directory / f"{frame_name}.png", size=size, resample=resample)
        for frame_name in frame_names
    ]


def read_rgb_image(path: Path) -> np.ndarray:
    img = np.array(iio.imread(path), copy=True)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img[..., :3].astype(np.uint8, copy=False)


def read_mask_image(path: Path) -> np.ndarray:
    mask = np.array(iio.imread(path), copy=True)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8, copy=False)


def resolve_mask_frame_name(record: FrameRecord, mask_frame_name_mode: str) -> str:
    mode = str(mask_frame_name_mode).strip().lower()
    if mode == "source":
        return record.source_frame_name
    if mode == "target":
        return record.frame_name
    raise ValueError(
        f"Unsupported mask frame name mode: {mask_frame_name_mode!r}. "
        "Expected 'source' or 'target'."
    )


def mask_to_metric_tensor(
    mask: np.ndarray,
    device: torch.device,
    *,
    binarize_masks: bool,
) -> torch.Tensor:
    if binarize_masks:
        return torch.from_numpy((mask > 0).astype(np.float32, copy=False)).to(device)
    return torch.from_numpy(mask).to(device).float() / 255.0


def mask_array_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32, copy=False)).to(device)


def resize_rgb_array(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize(size, resample=BILINEAR))


def resize_mask_array(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask).resize(size, resample=NEAREST))


def resolve_pair_mask_path(
    pair_root: Path,
    mask_subdir: str,
    target_cam: str,
    frame_name: str,
) -> Path:
    return pair_root / mask_subdir / target_cam / f"{frame_name}.png"


def load_resized_mask(
    path: Path,
    size: tuple[int, int],
    *,
    description: str,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    mask = read_mask_image(path)
    if mask.shape[:2] != (size[1], size[0]):
        mask = resize_mask_array(mask, size)
    return mask


def opacity_mask_to_valid_mask(alpha_mask: np.ndarray) -> np.ndarray:
    alpha_01 = alpha_mask.astype(np.float32, copy=False) / 255.0
    return (alpha_01 > VALID_OPACITY_THRESHOLD).astype(np.float32, copy=False)


def positive_mask_to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32, copy=False)


def resolve_sequence_eval_size(
    sequence: BenchmarkSequence,
    output_root: Path,
    pred_subdir: str,
    eval_res: str | tuple[int, int],
) -> tuple[int, int] | None:
    eval_records = [record for record in sequence.records if record.has_gt]
    if len(eval_records) == 0:
        return None
    if isinstance(eval_res, tuple):
        eval_h, eval_w = eval_res
        return int(eval_w), int(eval_h)

    first_record = eval_records[0]
    pred_path = (
        output_root
        / sequence.scene_name
        / sequence.pair_name
        / pred_subdir
        / sequence.target_cam
        / f"{first_record.frame_name}.png"
    )
    gt_path = sequence.pair_root / "gt" / sequence.target_cam / f"{first_record.frame_name}.png"
    if eval_res == "gt":
        gt_img = read_rgb_image(gt_path)
        return gt_img.shape[1], gt_img.shape[0]
    if eval_res == "pred":
        pred_img = read_rgb_image(pred_path)
        return pred_img.shape[1], pred_img.shape[0]
    raise ValueError(f"Unsupported eval_res: {eval_res!r}")


def build_sequence_union_valid_mask(
    sequence: BenchmarkSequence,
    mask_dir: Path,
    size: tuple[int, int],
    mask_frame_name_mode: str,
) -> np.ndarray:
    union_mask = np.zeros((size[1], size[0]), dtype=np.float32)
    mask_frame_names = sorted({
        resolve_mask_frame_name(record, mask_frame_name_mode)
        for record in sequence.records
        if record.has_gt
    })
    for mask_frame_name in mask_frame_names:
        mask_path = mask_dir / f"{mask_frame_name}.png"
        mask = load_resized_mask(
            mask_path,
            size,
            description=f"covisible mask for valid union ({mask_frame_name})",
        )
        union_mask = np.maximum(union_mask, positive_mask_to_binary(mask))
    return union_mask


def load_npz_array(path: Path, kind: str) -> np.ndarray:
    preferred_keys = PREFERRED_NPZ_KEYS[kind]
    with np.load(path) as data:
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])
        return np.asarray(data[data.files[0]])


def stage_chunk_assets(
    chunk: ChunkSpec,
    output_root: Path,
    use_scene_coords: bool,
) -> Path:
    stage_root = (
        output_root
        / "_tmp"
        / chunk.scene_name
        / chunk.pair_name
        / chunk.target_cam
        / f"chunk_{chunk.chunk_index:05d}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)

    source_c2w = load_npz_array(
        chunk.pair_root / "cameras" / f"{chunk.source_cam}_c2w.npz",
        kind="camera",
    )
    target_c2w = load_npz_array(
        chunk.pair_root / "cameras" / f"{chunk.target_cam}_c2w.npz",
        kind="camera",
    )
    cameras_path = stage_root / "cameras.npz"
    camera_payload = {
        f"{chunk.source_cam}_c2w": source_c2w[chunk.window_start : chunk.window_end].astype(np.float32),
        f"{chunk.target_cam}_c2w": target_c2w[chunk.window_start : chunk.window_end].astype(np.float32),
    }
    np.savez_compressed(cameras_path, **camera_payload)  # pyright: ignore[reportArgumentType]

    if use_scene_coords:
        scene_coords_dir = stage_root / "scene_coords"
        scene_coords_dir.mkdir(parents=True, exist_ok=True)
        target_frame_names = [record.frame_name for record in chunk.records]
        source_frame_names = [record.source_input_frame_name for record in chunk.records]
        for cam_name in (chunk.source_cam, chunk.target_cam):
            frame_names = source_frame_names if cam_name == chunk.source_cam else target_frame_names
            arrays = [
                load_npz_array(
                    chunk.pair_root / "scene_coords" / cam_name / f"{frame_name}.npz",
                    kind="scene_coords",
                )
                for frame_name in frame_names
            ]
            stacked = np.stack(arrays, axis=0).astype(np.float16)
            np.savez_compressed(scene_coords_dir / f"{cam_name}.npz", scene_coords=stacked)
    return cameras_path


def resolve_inference_frame_size(
    chunk: ChunkSpec,
    inference_res: tuple[int, int] | None,
) -> tuple[int, int]:
    if inference_res is not None:
        height, width = inference_res
        return width, height
    first_frame = chunk.pair_root / "rgb" / chunk.target_cam / f"{chunk.records[0].frame_name}.png"
    with Image.open(first_frame) as image:
        return image.convert("RGB").size


def build_chunk_inputs(
    chunk: ChunkSpec,
    config: DyCheckBenchmarkConfig,
    cameras_path: Path,
) -> tuple[dict, list[str]]:
    frame_names = [record.frame_name for record in chunk.records]
    source_frame_names = [record.source_input_frame_name for record in chunk.records]
    size = resolve_inference_frame_size(chunk, config.inference_res)
    pair_root = chunk.pair_root

    inputs = {
        "cameras_npz": str(cameras_path),
        "target_cam": chunk.target_cam,
        "source_cam": chunk.source_cam,
    }
    if config.use_source_video:
        inputs["source_video"] = load_pil_sequence(
            pair_root / "gt" / chunk.source_cam,
            source_frame_names,
            size,
            BILINEAR,
        )
    # Loader map: key -> (relative_subdir, frame_names_list, interpolation).
    # Use lists so the resampling choice stays colocated with the path.
    target_loaders = {
        "vace_video": ("rgb", frame_names, BILINEAR),
        "vace_depth": ("depth", frame_names, BILINEAR),
        "vace_normal": ("normal", frame_names, BILINEAR),
        "vace_alpha": ("alpha", frame_names, NEAREST),
        "scene_flow_target": ("scene_flow_rgb", frame_names, BILINEAR),
    }
    source_loaders = {
        "source_vace_video": ("rgb", source_frame_names, BILINEAR),
        "source_vace_depth": ("depth", source_frame_names, BILINEAR),
        "source_vace_normal": ("normal", source_frame_names, BILINEAR),
        "source_vace_alpha": ("alpha", source_frame_names, NEAREST),
        "scene_flow_source": ("scene_flow_rgb", source_frame_names, BILINEAR),
    }

    # Resolve which VACE signals to actually load. When the caller pinned the
    # active keys (compact / partial-conditioning training), honor that
    # exactly; otherwise fall back to the legacy behavior (load the full
    # 4-signal gbuffer set, with scene_flow gated by use_scene_flow).
    legacy_target_keys = ("vace_video", "vace_depth", "vace_normal", "vace_alpha")
    legacy_source_keys = (
        "source_vace_video", "source_vace_depth", "source_vace_normal", "source_vace_alpha",
    )
    if config.active_target_vace_keys is None:
        resolved_target_keys = legacy_target_keys + (
            ("scene_flow_target",) if config.use_scene_flow else ()
        )
    else:
        resolved_target_keys = tuple(config.active_target_vace_keys)
    if config.active_source_vace_keys is None:
        resolved_source_keys = legacy_source_keys + (
            ("scene_flow_source",) if config.use_scene_flow else ()
        )
    else:
        resolved_source_keys = tuple(config.active_source_vace_keys)

    if not config.disable_gbuffer:
        for key in resolved_target_keys:
            if key not in target_loaders:
                raise ValueError(
                    f"Unknown active target VACE key for DyCheck benchmark: {key}. "
                    f"Expected one of {sorted(target_loaders)}."
                )
            subdir, names, interp = target_loaders[key]
            inputs[key] = load_pil_sequence(
                pair_root / subdir / chunk.target_cam, names, size, interp
            )
        for key in resolved_source_keys:
            if key not in source_loaders:
                raise ValueError(
                    f"Unknown active source VACE key for DyCheck benchmark: {key}. "
                    f"Expected one of {sorted(source_loaders)}."
                )
            subdir, names, interp = source_loaders[key]
            inputs[key] = load_pil_sequence(
                pair_root / subdir / chunk.source_cam, names, size, interp
            )
    return inputs, frame_names


def tensor_to_uint8_frames(pred_tensor: torch.Tensor) -> np.ndarray:
    pred = pred_tensor.detach().float().clamp(-1.0, 1.0)
    pred = ((pred[0].permute(1, 2, 3, 0) + 1.0) * 127.5).round()
    pred = pred.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    return pred


def save_chunk_predictions(
    chunk: ChunkSpec,
    output_root: Path,
    pred_subdir: str,
    pred_frames: np.ndarray,
) -> int:
    pred_len = pred_frames.shape[0]
    keep_end = min(chunk.keep_end, pred_len)
    kept_records = list(chunk.records[chunk.keep_start : keep_end])
    kept_frames = pred_frames[chunk.keep_start : keep_end]
    out_dir = output_root / chunk.scene_name / chunk.pair_name / pred_subdir / chunk.target_cam
    out_dir.mkdir(parents=True, exist_ok=True)
    for record, frame in zip(kept_records, kept_frames):
        iio.imwrite(out_dir / f"{record.frame_name}.png", frame)
    return len(kept_records)


def resolve_metrics_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA metrics requested but CUDA is not available.")
    return torch.device(device_name)


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_flow3d_root(benchmark_root: Path, data_root: Path) -> Path:
    candidates = unique_paths(
        [
            benchmark_root.resolve().parent.parent,
            data_root.resolve().parent.parent,
            Path(DEFAULT_BENCHMARK_ROOT).expanduser().resolve().parent.parent,
            Path(DEFAULT_DATA_ROOT).expanduser().resolve().parent.parent,
        ]
    )
    for root in candidates:
        if (root / "flow3d" / "metrics.py").is_file():
            return root
    checked = "\n".join(f" - {root / 'flow3d' / 'metrics.py'}" for root in candidates)
    raise ModuleNotFoundError(
        "Unable to locate flow3d.metrics. Checked the following roots:\n"
        f"{checked}"
    )


def import_dycheck_metrics(benchmark_root: Path, data_root: Path):
    flow3d_root = resolve_flow3d_root(benchmark_root, data_root)
    if str(flow3d_root) not in sys.path:
        sys.path.insert(0, str(flow3d_root))
    from flow3d.metrics import mLPIPS, mPSNR, mSSIM  # pyright: ignore[reportMissingImports]

    return mPSNR, mSSIM, mLPIPS


def resolve_scene_mask_dir(
    scene_name: str,
    benchmark_root: Path,
    data_root: Path,
) -> Path:
    scene_roots = unique_paths([benchmark_root / scene_name, data_root / scene_name])
    candidates = [
        scene_root.joinpath(*suffix)
        for scene_root in scene_roots
        for suffix in MASK_DIR_CANDIDATE_SUFFIXES
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    checked = "\n".join(f" - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"No covisible mask directory found for scene '{scene_name}'. Checked:\n{checked}"
    )


def update_metric_triplet(
    psnr_metric,
    ssim_metric,
    lpips_metric,
    gt_t: torch.Tensor,
    pred_t: torch.Tensor,
    mask_t: torch.Tensor | None = None,
) -> bool:
    if mask_t is not None and float(mask_t.sum().item()) <= 0.0:
        return False
    if mask_t is None:
        psnr_metric.update(gt_t, pred_t)
        ssim_metric.update(gt_t[None], pred_t[None])
        lpips_metric.update(gt_t[None], pred_t[None])
        return True
    psnr_metric.update(gt_t, pred_t, mask_t)
    ssim_metric.update(gt_t[None], pred_t[None], mask_t[None])
    lpips_metric.update(gt_t[None], pred_t[None], mask_t[None])
    return True


def finalize_metric_triplet(
    psnr_metric,
    ssim_metric,
    lpips_metric,
    metric_keys: tuple[str, str, str],
    update_count: int,
) -> tuple[dict[str, float], dict[str, int]]:
    psnr_key, ssim_key, lpips_key = metric_keys
    if update_count <= 0:
        metrics = {
            psnr_key: float("nan"),
            ssim_key: float("nan"),
            lpips_key: float("nan"),
        }
    else:
        metrics = {
            psnr_key: float(psnr_metric.compute().item()),
            ssim_key: float(ssim_metric.compute().item()),
            lpips_key: float(lpips_metric.compute().item()),
        }
    counts = {
        psnr_key: int(update_count),
        ssim_key: int(update_count),
        lpips_key: int(update_count),
    }
    return metrics, counts


def mean_ignore_nan(values: list[Any]) -> float:
    filtered = []
    for value in values:
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float):
            filtered.append(value_float)
    if not filtered:
        return float("nan")
    return float(np.mean(filtered))


def evaluate_sequence(
    *,
    sequence: BenchmarkSequence,
    benchmark_root: Path,
    data_root: Path,
    output_root: Path,
    pred_subdir: str,
    eval_res: str | tuple[int, int],
    metrics_device: torch.device,
    lpips_net: str,
    rank_label: str | None = None,
    progress_log_every: int = 100,
    mask_frame_name_mode: str = "source",
    binarize_masks: bool = False,
) -> dict:
    scene_name = sequence.scene_name
    pair_name = sequence.pair_name
    mPSNR, mSSIM, mLPIPS = import_dycheck_metrics(benchmark_root, data_root)
    # DyCheck metrics are aggregated explicitly via per-pair JSON payloads, so
    # torchmetrics must not synchronize on compute() across the global process group.
    metric_kwargs = {"sync_on_compute": False}
    masked_psnr = mPSNR(**metric_kwargs).to(metrics_device)
    masked_ssim = mSSIM(**metric_kwargs).to(metrics_device)
    masked_lpips = mLPIPS(net_type=lpips_net, **metric_kwargs).to(metrics_device)
    full_psnr = mPSNR(**metric_kwargs).to(metrics_device)
    full_ssim = mSSIM(**metric_kwargs).to(metrics_device)
    full_lpips = mLPIPS(net_type=lpips_net, **metric_kwargs).to(metrics_device)
    valid_psnr = mPSNR(**metric_kwargs).to(metrics_device)
    valid_ssim = mSSIM(**metric_kwargs).to(metrics_device)
    valid_lpips = mLPIPS(net_type=lpips_net, **metric_kwargs).to(metrics_device)
    dynamic_valid_psnr = mPSNR(**metric_kwargs).to(metrics_device)
    dynamic_valid_ssim = mSSIM(**metric_kwargs).to(metrics_device)
    dynamic_valid_lpips = mLPIPS(net_type=lpips_net, **metric_kwargs).to(metrics_device)
    static_valid_psnr = mPSNR(**metric_kwargs).to(metrics_device)
    static_valid_ssim = mSSIM(**metric_kwargs).to(metrics_device)
    static_valid_lpips = mLPIPS(net_type=lpips_net, **metric_kwargs).to(metrics_device)
    mask_dir = resolve_scene_mask_dir(scene_name, benchmark_root, data_root)
    camera_counts: dict[str, int] = {}
    num_eval_frames = 0
    masked_metric_updates = 0
    full_metric_updates = 0
    valid_metric_updates = 0
    dynamic_valid_metric_updates = 0
    static_valid_metric_updates = 0

    with torch.no_grad():
        pair_root = sequence.pair_root
        if not pair_root.is_dir() or PAIR_RE.match(pair_root.name) is None:
            raise ValueError(f"Unexpected benchmark pair root: {pair_root}")
        target_cam = sequence.target_cam
        eval_size = resolve_sequence_eval_size(
            sequence=sequence,
            output_root=output_root,
            pred_subdir=pred_subdir,
            eval_res=eval_res,
        )
        valid_mask = (
            build_sequence_union_valid_mask(
                sequence=sequence,
                mask_dir=mask_dir,
                size=eval_size,
                mask_frame_name_mode=mask_frame_name_mode,
            )
            if eval_size is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        valid_mask_t = (
            mask_array_to_tensor(valid_mask, metrics_device)
            if eval_size is not None
            else None
        )
        for record in sequence.records:
            if not record.has_gt:
                continue
            pred_path = output_root / scene_name / pair_root.name / pred_subdir / target_cam / f"{record.frame_name}.png"
            gt_path = pair_root / "gt" / target_cam / f"{record.frame_name}.png"
            mask_path = mask_dir / f"{resolve_mask_frame_name(record, mask_frame_name_mode)}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing covisible mask for scene '{scene_name}': {mask_path}"
                )

            pred_img = read_rgb_image(pred_path)
            gt_img = read_rgb_image(gt_path)
            mask = read_mask_image(mask_path)
            if eval_size is None:
                raise RuntimeError(
                    f"Sequence {scene_name}/{pair_name} has no GT frames but entered evaluation loop."
                )
            size = eval_size

            if pred_img.shape[:2] != (size[1], size[0]):
                pred_img = resize_rgb_array(pred_img, size)
            if gt_img.shape[:2] != (size[1], size[0]):
                gt_img = resize_rgb_array(gt_img, size)
            if mask.shape[:2] != (size[1], size[0]):
                mask = resize_mask_array(mask, size)
            dynamic_mask = load_resized_mask(
                resolve_pair_mask_path(
                    pair_root,
                    TEST_DYNAMIC_DIRNAME,
                    target_cam,
                    record.frame_name,
                ),
                size,
                description="target dynamic mask",
            )

            pred_t = torch.from_numpy(pred_img).to(metrics_device).float() / 255.0
            gt_t = torch.from_numpy(gt_img).to(metrics_device).float() / 255.0
            mask_t = mask_to_metric_tensor(
                mask,
                metrics_device,
                binarize_masks=binarize_masks,
            )
            dynamic_valid_mask = (valid_mask * positive_mask_to_binary(dynamic_mask)).astype(
                np.float32,
                copy=False,
            )
            static_valid_mask = (valid_mask * (1.0 - positive_mask_to_binary(dynamic_mask))).astype(
                np.float32,
                copy=False,
            )

            if update_metric_triplet(masked_psnr, masked_ssim, masked_lpips, gt_t, pred_t, mask_t):
                masked_metric_updates += 1
            if update_metric_triplet(full_psnr, full_ssim, full_lpips, gt_t, pred_t):
                full_metric_updates += 1

            if valid_mask_t is None:
                raise RuntimeError(
                    f"Sequence {scene_name}/{pair_name} failed to build a valid union mask."
                )
            if update_metric_triplet(valid_psnr, valid_ssim, valid_lpips, gt_t, pred_t, valid_mask_t):
                valid_metric_updates += 1

            dynamic_valid_mask_t = mask_array_to_tensor(dynamic_valid_mask, metrics_device)
            if update_metric_triplet(
                dynamic_valid_psnr,
                dynamic_valid_ssim,
                dynamic_valid_lpips,
                gt_t,
                pred_t,
                dynamic_valid_mask_t,
            ):
                dynamic_valid_metric_updates += 1

            static_valid_mask_t = mask_array_to_tensor(static_valid_mask, metrics_device)
            if update_metric_triplet(
                static_valid_psnr,
                static_valid_ssim,
                static_valid_lpips,
                gt_t,
                pred_t,
                static_valid_mask_t,
            ):
                static_valid_metric_updates += 1

            camera_counts[target_cam] = camera_counts.get(target_cam, 0) + 1
            num_eval_frames += 1
            if progress_log_every > 0 and num_eval_frames % progress_log_every == 0:
                prefix = "[DyCheck]" if rank_label is None else f"[DyCheck][{rank_label}]"
                print(
                    f"{prefix} Pair {scene_name}/{pair_name}: "
                    f"evaluated {num_eval_frames} frames...",
                    flush=True,
                )

    prefix = "[DyCheck]" if rank_label is None else f"[DyCheck][{rank_label}]"
    print(
        f"{prefix} Pair {scene_name}/{pair_name}: finalizing metric reductions "
        f"for {num_eval_frames} frames...",
        flush=True,
    )
    metrics: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for metric_values, count_values in (
        finalize_metric_triplet(
            masked_psnr,
            masked_ssim,
            masked_lpips,
            MASKED_METRIC_KEYS,
            masked_metric_updates,
        ),
        finalize_metric_triplet(
            full_psnr,
            full_ssim,
            full_lpips,
            FULL_METRIC_KEYS,
            full_metric_updates,
        ),
        finalize_metric_triplet(
            valid_psnr,
            valid_ssim,
            valid_lpips,
            VALID_METRIC_KEYS,
            valid_metric_updates,
        ),
        finalize_metric_triplet(
            dynamic_valid_psnr,
            dynamic_valid_ssim,
            dynamic_valid_lpips,
            DYNAMIC_VALID_METRIC_KEYS,
            dynamic_valid_metric_updates,
        ),
        finalize_metric_triplet(
            static_valid_psnr,
            static_valid_ssim,
            static_valid_lpips,
            STATIC_VALID_METRIC_KEYS,
            static_valid_metric_updates,
        ),
    ):
        metrics.update(metric_values)
        metric_counts.update(count_values)

    return {
        "scene_name": scene_name,
        "pair_name": pair_name,
        "num_eval_frames": num_eval_frames,
        "camera_counts": camera_counts,
        "metric_counts": metric_counts,
        "metrics": metrics,
    }


def aggregate_scene_payload(scene_name: str, pair_payloads: list[dict]) -> dict:
    if len(pair_payloads) == 0:
        raise ValueError(f"No pair payloads found for scene '{scene_name}'.")

    total_frames = sum(int(payload["num_eval_frames"]) for payload in pair_payloads)
    metric_keys = list(pair_payloads[0]["metrics"].keys())
    camera_counts: dict[str, int] = {}
    metric_counts = {key: 0 for key in metric_keys}
    metric_weighted_sums = {key: 0.0 for key in metric_keys}
    for payload in pair_payloads:
        for camera_name, count in payload["camera_counts"].items():
            camera_counts[camera_name] = camera_counts.get(camera_name, 0) + int(count)
        payload_metric_counts = payload.get("metric_counts", {})
        for key in metric_keys:
            count = int(payload_metric_counts.get(key, payload["num_eval_frames"]))
            if count <= 0:
                continue
            value = payload["metrics"].get(key)
            if value is None:
                continue
            value_float = float(value)
            if not np.isfinite(value_float):
                continue
            metric_weighted_sums[key] += value_float * count
            metric_counts[key] += count

    metrics = {
        key: (
            float(metric_weighted_sums[key] / metric_counts[key])
            if metric_counts[key] > 0
            else float("nan")
        )
        for key in metric_keys
    }

    return {
        "scene_name": scene_name,
        "num_eval_frames": total_frames,
        "camera_counts": camera_counts,
        "metric_counts": metric_counts,
        "metrics": metrics,
    }


def compute_mean_metrics(per_scene_payloads: list[dict]) -> dict[str, float]:
    if not per_scene_payloads:
        return {}
    keys = list(per_scene_payloads[0]["metrics"].keys())
    return {
        key: mean_ignore_nan([payload["metrics"].get(key) for payload in per_scene_payloads])
        for key in keys
    }


def build_benchmark_summary(
    config: DyCheckBenchmarkConfig,
    dataset: DyCheckBenchmarkDataset,
    world_size: int,
    per_scene_payloads: list[dict],
) -> dict:
    return {
        "benchmark_root": str(config.benchmark_root),
        "data_root": str(config.data_root),
        "output_root": str(config.output_root),
        "pred_subdir": config.pred_subdir,
        "inference_res": format_resolution(config.inference_res),
        "eval_res": format_resolution(config.eval_res),
        "cfg_scale": float(config.cfg_scale),
        "target_gbuffer_cfg_scale": float(config.target_gbuffer_cfg_scale),
        "target_gbuffer_dropout_scope": str(config.target_gbuffer_dropout_scope),
        "target_gbuffer_uncond_fill_mode": str(config.target_gbuffer_uncond_fill_mode),
        "target_gbuffer_guidance_mode": str(config.target_gbuffer_guidance_mode),
        "target_gbuffer_apg_eta": float(config.target_gbuffer_apg_eta),
        "target_gbuffer_apg_norm_threshold": float(config.target_gbuffer_apg_norm_threshold),
        "target_gbuffer_apg_momentum": (
            None
            if config.target_gbuffer_apg_momentum is None
            else float(config.target_gbuffer_apg_momentum)
        ),
        "target_gbuffer_apg_space": str(config.target_gbuffer_apg_space),
        "use_source_video": bool(config.use_source_video),
        "mask_frame_name_mode": config.mask_frame_name_mode,
        "binarize_masks": bool(config.binarize_masks),
        "seq_names": dataset.seq_names,
        "num_scenes": len(dataset.seq_names),
        "chunk_size": config.chunk_size,
        "num_chunks": len(dataset.jobs),
        "seed": None if config.seed is None else int(config.seed),
        "seed_strategy": (
            "stable_sequence"
            if config.seed is not None
            else "unseeded"
        ),
        "seed_scope": (
            "scene/pair/target_cam"
            if config.seed is not None
            else None
        ),
        "world_size": world_size,
        "mean_metrics": compute_mean_metrics(per_scene_payloads),
        "per_scene": {payload["scene_name"]: payload for payload in per_scene_payloads},
    }


def run_dycheck_benchmark(
    pipe,
    accelerator,
    config: DyCheckBenchmarkConfig,
) -> DyCheckBenchmarkRunResult:
    output_root = config.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_root = config.benchmark_root.expanduser().resolve()
    data_root = config.data_root.expanduser().resolve()
    dataset = DyCheckBenchmarkDataset(
        benchmark_root=benchmark_root,
        seq_names=config.normalized_seq_names(),
        chunk_size=int(config.chunk_size),
    )
    local_jobs = [
        job
        for idx, job in enumerate(dataset.jobs)
        if idx % accelerator.num_processes == accelerator.process_index
    ]

    if accelerator.is_main_process:
        accelerator.print(f"[DyCheck] Scenes: {dataset.seq_names}")
        accelerator.print(f"[DyCheck] Total chunks: {len(dataset.jobs)}")
        accelerator.print(f"[DyCheck] World size: {accelerator.num_processes}")
        accelerator.print(f"[DyCheck] Inference resolution: {format_resolution(config.inference_res)}")
        accelerator.print(f"[DyCheck] Eval resolution: {format_resolution(config.eval_res)}")
        accelerator.print(f"[DyCheck] Prompt cfg_scale: {float(config.cfg_scale):.3f}")
        accelerator.print(
            "[DyCheck] Target-gbuffer guidance: "
            f"mode={config.target_gbuffer_guidance_mode} "
            f"cfg_scale={float(config.target_gbuffer_cfg_scale):.3f} "
            f"scope={config.target_gbuffer_dropout_scope} "
            f"fill={config.target_gbuffer_uncond_fill_mode}"
        )
        accelerator.print(
            f"[DyCheck] Source-video conditioning: {bool(config.use_source_video)}"
        )
        if str(config.target_gbuffer_guidance_mode).strip().lower() == "apg":
            accelerator.print(
                "[DyCheck] Target-gbuffer APG: "
                f"space={config.target_gbuffer_apg_space} "
                f"eta={float(config.target_gbuffer_apg_eta):.3f} "
                f"norm_threshold={float(config.target_gbuffer_apg_norm_threshold):.3f} "
                f"momentum={config.target_gbuffer_apg_momentum}"
            )
        accelerator.print(f"[DyCheck] Mask frame mode: {config.mask_frame_name_mode}")
        accelerator.print(f"[DyCheck] Binarize masks: {bool(config.binarize_masks)}")
        if config.seed is None:
            accelerator.print("[DyCheck] Chunk seeding: disabled")
        else:
            accelerator.print(
                "[DyCheck] Chunk seeding: stable per scene/pair/target_cam "
                f"(base seed {int(config.seed)})"
            )
        accelerator.print(f"[DyCheck] Output root: {output_root}")

    with torch.no_grad():
        for local_idx, chunk in enumerate(local_jobs):
            if local_idx % 10 == 0:
                accelerator.print(
                    f"[Rank {accelerator.process_index}] chunk {local_idx + 1}/{len(local_jobs)} "
                    f"{chunk.scene_name}/{chunk.pair_name}/{chunk.target_cam}"
                )
            use_scene_coords = bool(config.scene_coord_pe or config.scene_coord_xyzt_pe)
            cameras_path = stage_chunk_assets(
                chunk=chunk,
                output_root=output_root,
                use_scene_coords=use_scene_coords,
            )
            inputs, frame_names = build_chunk_inputs(chunk, config, cameras_path)
            seed = resolve_benchmark_chunk_seed(config.seed, chunk)
            conditioning_frames = inputs.get("vace_video")
            if conditioning_frames is None:
                conditioning_frames = inputs.get("source_video")
            if conditioning_frames is None:
                raise ValueError(
                    "DyCheck benchmark requires at least one conditioning input "
                    "(target VACE or source_video)."
                )
            pred_video = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                split_cam_embed=config.split_cam_embed,
                zero_source_cam_emb_when_missing=config.zero_source_cam_emb_when_missing,
                height=conditioning_frames[0].size[1],
                width=conditioning_frames[0].size[0],
                num_frames=len(frame_names),
                cfg_scale=float(config.cfg_scale),
                target_gbuffer_cfg_scale=float(config.target_gbuffer_cfg_scale),
                target_gbuffer_dropout_scope=str(config.target_gbuffer_dropout_scope),
                target_gbuffer_uncond_fill_mode=str(config.target_gbuffer_uncond_fill_mode),
                target_gbuffer_guidance_mode=str(config.target_gbuffer_guidance_mode),
                target_gbuffer_apg_eta=float(config.target_gbuffer_apg_eta),
                target_gbuffer_apg_norm_threshold=float(config.target_gbuffer_apg_norm_threshold),
                target_gbuffer_apg_momentum=config.target_gbuffer_apg_momentum,
                target_gbuffer_apg_space=str(config.target_gbuffer_apg_space),
                tiled=bool(config.tiled),
                num_inference_steps=int(config.num_inference_steps),
                vace_scale=float(config.vace_scale),
                output_type="floatpoint",
                seed=seed,
                **inputs,
            )
            pred_frames = tensor_to_uint8_frames(pred_video)
            save_chunk_predictions(
                chunk=chunk,
                output_root=output_root,
                pred_subdir=config.pred_subdir,
                pred_frames=pred_frames,
            )

    accelerator.wait_for_everyone()
    metrics_dir = output_root / metrics_dir_name(config.eval_res)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sync_run_token = build_sync_run_token(accelerator)

    if accelerator.is_main_process:
        active_ranks = min(len(dataset.sequences), accelerator.num_processes)
        accelerator.print("[DyCheck] Prediction generation complete on all ranks.")
        accelerator.print(
            f"[DyCheck] Starting distributed pair metrics across "
            f"{active_ranks}/{accelerator.num_processes} rank(s)."
        )
        accelerator.print(f"[DyCheck] Metrics sync token: {sync_run_token}")

    local_sequences = [
        sequence
        for idx, sequence in enumerate(dataset.sequences)
        if idx % accelerator.num_processes == accelerator.process_index
    ]
    if len(local_sequences) > 0:
        print(
            f"[DyCheck][Rank {accelerator.process_index}] Assigned pairs: "
            f"{', '.join(f'{sequence.scene_name}/{sequence.pair_name}' for sequence in local_sequences)}",
            flush=True,
        )
    else:
        print(
            f"[DyCheck][Rank {accelerator.process_index}] No pairs assigned for metric evaluation.",
            flush=True,
        )

    metrics_device = resolve_metrics_device(str(config.metrics_device))
    local_pair_payloads = []
    for sequence in local_sequences:
        print(
            f"[DyCheck][Rank {accelerator.process_index}] Evaluating pair: "
            f"{sequence.scene_name}/{sequence.pair_name}",
            flush=True,
        )
        payload = evaluate_sequence(
            sequence=sequence,
            benchmark_root=benchmark_root,
            data_root=data_root,
            output_root=output_root,
            pred_subdir=config.pred_subdir,
            eval_res=config.eval_res,
            metrics_device=metrics_device,
            lpips_net=config.lpips_net,
            rank_label=f"Rank {accelerator.process_index}",
            mask_frame_name_mode=config.mask_frame_name_mode,
            binarize_masks=bool(config.binarize_masks),
        )
        local_pair_payloads.append(payload)
        write_json(
            pair_metrics_path(metrics_dir, sequence.scene_name, sequence.pair_name),
            payload,
        )
        print(
            f"[DyCheck][Rank {accelerator.process_index}] Finished pair: "
            f"{sequence.scene_name}/{sequence.pair_name} "
            f"({payload['num_eval_frames']} eval frames)",
            flush=True,
        )

    write_json(
        rank_metrics_done_path(metrics_dir, accelerator.process_index, sync_run_token),
        {
            "rank": accelerator.process_index,
            "sync_run_token": sync_run_token,
            "pair_names": [
                f"{sequence.scene_name}/{sequence.pair_name}" for sequence in local_sequences
            ],
            "num_pairs": len(local_sequences),
        },
    )
    print(
        f"[DyCheck][Rank {accelerator.process_index}] Local metric evaluation complete.",
        flush=True,
    )

    summary_path = metrics_dir / "summary.json"
    postprocess_done_path = dycheck_postprocess_done_path(metrics_dir, sync_run_token)
    if not accelerator.is_main_process:
        wait_for_files(
            [postprocess_done_path],
            progress_prefix=(
                f"[DyCheck][Rank {accelerator.process_index}] Waiting for rank 0 "
                "to aggregate summary"
            ),
        )
        return DyCheckBenchmarkRunResult(
            dataset=dataset,
            metrics_dir=metrics_dir,
            per_scene_payloads=[],
            summary=None,
            summary_path=summary_path,
        )

    accelerator.print("[DyCheck] Waiting for all rank-local pair metrics to complete...")
    wait_for_files(
        [
            rank_metrics_done_path(metrics_dir, rank, sync_run_token)
            for rank in range(accelerator.num_processes)
        ],
        progress_prefix="[DyCheck] Waiting for distributed pair metric workers",
        print_fn=accelerator.print,
    )

    per_scene_payloads = []
    for scene_name in dataset.seq_names:
        pair_payloads = []
        for sequence in dataset.sequences:
            if sequence.scene_name != scene_name:
                continue
            payload_path = pair_metrics_path(metrics_dir, sequence.scene_name, sequence.pair_name)
            if not payload_path.is_file():
                raise FileNotFoundError(
                    f"Missing pair metrics payload after distributed DyCheck evaluation: "
                    f"{payload_path}"
                )
            pair_payloads.append(load_json(payload_path))
        scene_payload = aggregate_scene_payload(scene_name, pair_payloads)
        write_json(scene_metrics_path(metrics_dir, scene_name), scene_payload)
        per_scene_payloads.append(scene_payload)

    summary = build_benchmark_summary(
        config=config,
        dataset=dataset,
        world_size=accelerator.num_processes,
        per_scene_payloads=per_scene_payloads,
    )
    summary["sync_run_token"] = sync_run_token
    write_json(summary_path, summary)
    write_json(
        postprocess_done_path,
        {
            "sync_run_token": sync_run_token,
            "summary_path": str(summary_path),
            "num_scenes": len(per_scene_payloads),
        },
    )
    accelerator.print(f"[DyCheck] Saved summary: {summary_path}")
    return DyCheckBenchmarkRunResult(
        dataset=dataset,
        metrics_dir=metrics_dir,
        per_scene_payloads=per_scene_payloads,
        summary=summary,
        summary_path=summary_path,
    )
