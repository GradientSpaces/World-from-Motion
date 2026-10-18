from __future__ import annotations

import importlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import imageio.v2 as imageio
import numpy as np
import torch

from .util import write_json_atomic


NATIVE_CHECKPOINTS = (
    "photometric_cam.pth",
    "photometric_s_model_native_add3.pth",
    "photometric_d_model_native_add3.pth",
)


@dataclass(frozen=True)
class DistillationConfig:
    benchmark_root: Path
    prediction_root: Path
    output_root: Path
    scene_name: str
    source_native: Path
    mosca_root: Path
    target_cameras: tuple[str, ...]
    reference_camera: str = "cam0"
    device: str = "cuda"
    seed: int = 0
    steps: int = 200
    source_weight: float = 1.0
    virtual_weight: float = 1.0
    ssim_lambda: float = 0.1
    lr_scale: float = 0.25
    grad_clip_norm: float = 0.0
    use_alpha_mask: bool = True
    debug_every: int = 25


@dataclass(frozen=True)
class FrameRecord:
    time_id: int
    frame_name: str
    source_frame_name: str


@dataclass(frozen=True)
class TargetSequence:
    camera: str
    records: tuple[FrameRecord, ...]
    t_cw: np.ndarray
    prediction_dir: Path
    alpha_dir: Path
    ignored_prediction_frames: tuple[str, ...]


@dataclass(frozen=True)
class DistillationObservation:
    time_id: int
    source_frame_name: str
    source_rgb: torch.Tensor
    source_mask: torch.Tensor
    targets: dict[str, tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class DistillationDataset:
    scene_root: Path
    prediction_scene_root: Path
    reference_camera: str
    reference_records: dict[str, FrameRecord]
    source_rgb_dir: Path
    source_alpha_dir: Path
    targets: dict[str, TargetSequence]
    time_ids: tuple[int, ...]
    time_id_to_index: dict[int, int]

    @property
    def num_frames(self) -> int:
        return len(self.time_ids)

    @property
    def target_cameras(self) -> tuple[str, ...]:
        return tuple(self.targets)

    def get_observation(
        self,
        time_id: int,
        *,
        device: torch.device,
        use_alpha_mask: bool,
    ) -> DistillationObservation:
        try:
            index = self.time_id_to_index[int(time_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown distillation time_id={time_id}") from exc
        first_target = self.targets[self.target_cameras[0]]
        first_record = first_target.records[index]
        source_record = self.reference_records[first_record.source_frame_name]
        source_rgb = _rgb_tensor(
            _read_rgb_cached(str(self.source_rgb_dir / f"{source_record.frame_name}.png")),
            device=device,
        )
        if use_alpha_mask:
            source_mask = _mask_tensor(
                _read_mask_cached(
                    str(self.source_alpha_dir / f"{source_record.frame_name}.png")
                ),
                device=device,
            )
        else:
            source_mask = torch.ones(source_rgb.shape[:2], device=device, dtype=torch.float32)
        _require_nonempty_mask(source_mask, self.reference_camera, time_id)

        target_observations = {}
        for camera, sequence in self.targets.items():
            record = sequence.records[index]
            if record.source_frame_name != first_record.source_frame_name:
                raise ValueError(
                    f"Target cameras disagree on the source frame for time_id={time_id}: "
                    f"{first_record.source_frame_name} vs {record.source_frame_name}"
                )
            target_mask_array = _read_mask_cached(
                str(sequence.alpha_dir / f"{record.frame_name}.png")
            )
            target_rgb_array = _read_rgb_cached(
                str(sequence.prediction_dir / f"{record.frame_name}.png")
            )
            if target_rgb_array.shape[:2] != target_mask_array.shape[:2]:
                target_rgb_array = _resize_rgb_uint8(
                    target_rgb_array, target_mask_array.shape[:2]
                )
            if target_rgb_array.shape[:2] != source_rgb.shape[:2]:
                raise ValueError(
                    f"Source/target size mismatch for {camera} time_id={time_id}: "
                    f"{tuple(source_rgb.shape[:2])} vs {target_rgb_array.shape[:2]}"
                )
            target_rgb = _rgb_tensor(target_rgb_array, device=device)
            if use_alpha_mask:
                target_mask = _mask_tensor(target_mask_array, device=device)
            else:
                target_mask = torch.ones(
                    target_rgb.shape[:2], device=device, dtype=torch.float32
                )
            _require_nonempty_mask(target_mask, camera, time_id)
            target_observations[camera] = (
                record.frame_name,
                target_rgb,
                target_mask,
                torch.from_numpy(sequence.t_cw[index]).to(
                    device=device, dtype=torch.float32
                ),
            )
        return DistillationObservation(
            time_id=int(time_id),
            source_frame_name=source_record.frame_name,
            source_rgb=source_rgb,
            source_mask=source_mask,
            targets=target_observations,
        )


@dataclass(frozen=True)
class _WorldTreeModules:
    MonocularCameras: Any
    StaticGaussian: Any
    DynSCFGaussian: Any
    render: Any
    ssim: Any
    gs_backend: str


@dataclass(frozen=True)
class _Loss:
    total: torch.Tensor
    l1: torch.Tensor
    ssim: torch.Tensor
    pred_rgb: torch.Tensor


def load_distillation_dataset(
    *,
    benchmark_root: Path,
    prediction_root: Path,
    scene_name: str,
    reference_camera: str,
    target_cameras: tuple[str, ...],
) -> DistillationDataset:
    if not target_cameras:
        raise ValueError("At least one target camera is required for distillation")
    scene_root = _resolve_scene_root(Path(benchmark_root), scene_name)
    prediction_scene_root = _resolve_prediction_scene_root(
        Path(prediction_root), scene_name
    )
    reference_camera = _validate_camera_name(reference_camera)
    target_cameras = tuple(_validate_camera_name(name) for name in target_cameras)
    if len(set(target_cameras)) != len(target_cameras):
        raise ValueError(f"Duplicate target cameras: {target_cameras}")

    reference_payload = _load_frame_manifest(scene_root, reference_camera)
    reference_records = {
        record.frame_name: record for record in _frame_records(reference_payload)
    }
    if not reference_records:
        raise ValueError(f"Reference camera {reference_camera} has no frames")

    targets: dict[str, TargetSequence] = {}
    expected_time_ids: tuple[int, ...] | None = None
    expected_sources: tuple[str, ...] | None = None
    for camera in target_cameras:
        payload = _load_frame_manifest(scene_root, camera)
        records = _frame_records(payload)
        if not records:
            raise ValueError(f"Target camera {camera} has no frames")
        time_ids = tuple(record.time_id for record in records)
        sources = tuple(record.source_frame_name for record in records)
        if len(set(time_ids)) != len(time_ids):
            raise ValueError(f"Target camera {camera} contains duplicate time IDs")
        if expected_time_ids is None:
            expected_time_ids = time_ids
            expected_sources = sources
        elif time_ids != expected_time_ids or sources != expected_sources:
            raise ValueError(
                f"Target camera {camera} does not align with the other pseudo views"
            )
        missing_sources = sorted(set(sources) - set(reference_records))
        if missing_sources:
            raise ValueError(
                f"Target camera {camera} references missing source frames: "
                f"{missing_sources[:5]}"
            )

        prediction_dir = prediction_scene_root / "pred_rgb" / camera
        expected_predictions = {record.frame_name for record in records}
        actual_predictions = {path.stem for path in prediction_dir.glob("*.png")}
        missing_predictions = sorted(expected_predictions - actual_predictions)
        if missing_predictions:
            raise FileNotFoundError(
                f"Missing generated frames for {camera}: {missing_predictions[:5]}"
            )
        ignored_predictions = tuple(sorted(actual_predictions - expected_predictions))
        alpha_dir = scene_root / "alpha" / camera
        _require_frames(alpha_dir, expected_predictions, f"alpha frames for {camera}")
        targets[camera] = TargetSequence(
            camera=camera,
            records=records,
            t_cw=np.linalg.inv(
                _load_camera_trajectory(
                    scene_root / "cameras" / f"{camera}_c2w.npz", records
                )
            ).astype(np.float32),
            prediction_dir=prediction_dir,
            alpha_dir=alpha_dir,
            ignored_prediction_frames=ignored_predictions,
        )

    assert expected_time_ids is not None
    source_rgb_dir = scene_root / "gt" / reference_camera
    source_alpha_dir = scene_root / "alpha" / reference_camera
    required_source_frames = set(expected_sources or ())
    _require_frames(source_rgb_dir, required_source_frames, "source RGB frames")
    _require_frames(source_alpha_dir, required_source_frames, "source alpha frames")
    return DistillationDataset(
        scene_root=scene_root,
        prediction_scene_root=prediction_scene_root,
        reference_camera=reference_camera,
        reference_records=reference_records,
        source_rgb_dir=source_rgb_dir,
        source_alpha_dir=source_alpha_dir,
        targets=targets,
        time_ids=expected_time_ids,
        time_id_to_index={time_id: index for index, time_id in enumerate(expected_time_ids)},
    )


def distill_generated_views(config: DistillationConfig) -> dict[str, Any]:
    cfg = _resolve_config(config)
    dataset = load_distillation_dataset(
        benchmark_root=cfg.benchmark_root,
        prediction_root=cfg.prediction_root,
        scene_name=cfg.scene_name,
        reference_camera=cfg.reference_camera,
        target_cameras=cfg.target_cameras,
    )
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4DGS distillation but is not available")
    modules = _load_worldtree_modules(cfg.mosca_root)
    output_root = _prepare_output_root(cfg.output_root)
    native_output = output_root / "native"
    shutil.copytree(cfg.source_native, native_output)
    cams, s_model, d_model = _load_worldtree_state(
        modules=modules,
        saved_dir=cfg.source_native,
        device=device,
    )
    for parameter in cams.parameters():
        parameter.requires_grad_(False)
    _freeze_dynamic_controls(d_model)
    optimizer = _build_optimizer(
        s_model=s_model,
        d_model=d_model,
        lr_scale=cfg.lr_scale,
    )

    history: list[dict[str, Any]] = []
    schedule = list(dataset.time_ids)
    rng = random.Random(cfg.seed)
    rng.shuffle(schedule)
    schedule_index = 0
    debug_dir = output_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    for step in range(cfg.steps):
        if schedule_index >= len(schedule):
            rng.shuffle(schedule)
            schedule_index = 0
        time_id = int(schedule[schedule_index])
        schedule_index += 1
        observation = dataset.get_observation(
            time_id,
            device=device,
            use_alpha_mask=cfg.use_alpha_mask,
        )
        optimizer.zero_grad(set_to_none=True)

        source_render = _render_observation(
            modules=modules,
            cams=cams,
            s_model=s_model,
            d_model=d_model,
            time_id=time_id,
            height=int(observation.source_rgb.shape[0]),
            width=int(observation.source_rgb.shape[1]),
            t_cw=cams.T_cw(time_id).detach(),
        )
        source_loss = _compute_rgb_loss(
            gt_rgb=observation.source_rgb,
            render_dict=source_render,
            mask=observation.source_mask,
            ssim_fn=modules.ssim,
            ssim_lambda=cfg.ssim_lambda,
        )
        (source_loss.total * cfg.source_weight).backward()
        source_pred_for_debug = source_loss.pred_rgb.detach()
        del source_render

        virtual_entries: dict[str, dict[str, float | str]] = {}
        debug_targets: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        num_targets = len(dataset.target_cameras)
        for camera in dataset.target_cameras:
            frame_name, target_rgb, target_mask, target_t_cw = observation.targets[camera]
            target_render = _render_observation(
                modules=modules,
                cams=cams,
                s_model=s_model,
                d_model=d_model,
                time_id=time_id,
                height=int(target_rgb.shape[0]),
                width=int(target_rgb.shape[1]),
                t_cw=target_t_cw,
            )
            target_loss = _compute_rgb_loss(
                gt_rgb=target_rgb,
                render_dict=target_render,
                mask=target_mask,
                ssim_fn=modules.ssim,
                ssim_lambda=cfg.ssim_lambda,
            )
            (target_loss.total * (cfg.virtual_weight / num_targets)).backward()
            virtual_entries[camera] = {
                "frame_name": frame_name,
                "loss": float(target_loss.total.detach().cpu()),
                "l1": float(target_loss.l1.detach().cpu()),
                "ssim": float(target_loss.ssim.detach().cpu()),
            }
            if _debug_due(step, cfg.steps, cfg.debug_every):
                debug_targets.append(
                    (camera, target_rgb, target_mask, target_loss.pred_rgb.detach())
                )
            del target_render

        if cfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ],
                max_norm=cfg.grad_clip_norm,
            )
        optimizer.step()
        virtual_mean = sum(float(entry["loss"]) for entry in virtual_entries.values()) / len(
            virtual_entries
        )
        entry = {
            "step": step + 1,
            "time_id": time_id,
            "source_frame_name": observation.source_frame_name,
            "loss_total": float(source_loss.total.detach().cpu()) * cfg.source_weight
            + virtual_mean * cfg.virtual_weight,
            "loss_source": float(source_loss.total.detach().cpu()),
            "loss_source_l1": float(source_loss.l1.detach().cpu()),
            "loss_source_ssim": float(source_loss.ssim.detach().cpu()),
            "loss_virtual": virtual_mean,
            "virtual_views": virtual_entries,
        }
        if not math.isfinite(float(entry["loss_total"])):
            raise FloatingPointError(f"Non-finite distillation loss at step {step + 1}")
        history.append(entry)

        if _debug_due(step, cfg.steps, cfg.debug_every):
            _save_debug_panel(
                debug_dir / f"step_{step + 1:06d}_time_{time_id:05d}.png",
                source_rgb=observation.source_rgb,
                source_mask=observation.source_mask,
                source_pred=source_pred_for_debug,
                targets=debug_targets,
            )
            print(
                f"[wfm] distill step={step + 1}/{cfg.steps} time={time_id} "
                f"source={entry['loss_source']:.6f} virtual={virtual_mean:.6f}",
                flush=True,
            )

    _save_worldtree_state(
        output_dir=native_output,
        modules=modules,
        cams=cams,
        s_model=s_model,
        d_model=d_model,
    )
    history_path = output_root / "distillation_history.json"
    write_json_atomic(history_path, history)
    summary = {
        "schema_version": 1,
        "scene_name": cfg.scene_name,
        "reference_camera": cfg.reference_camera,
        "target_cameras": list(dataset.target_cameras),
        "num_frames": dataset.num_frames,
        "num_steps": cfg.steps,
        "elapsed_seconds": time.time() - started_at,
        "source_native": _portable_path(cfg.source_native, output_root),
        "distilled_native": "native",
        "benchmark_root": _portable_path(cfg.benchmark_root, output_root),
        "prediction_root": _portable_path(cfg.prediction_root, output_root),
        "ignored_padded_prediction_frames": {
            camera: len(sequence.ignored_prediction_frames)
            for camera, sequence in dataset.targets.items()
        },
        "config": {
            "seed": cfg.seed,
            "source_weight": cfg.source_weight,
            "virtual_weight": cfg.virtual_weight,
            "ssim_lambda": cfg.ssim_lambda,
            "lr_scale": cfg.lr_scale,
            "grad_clip_norm": cfg.grad_clip_norm,
            "use_alpha_mask": cfg.use_alpha_mask,
        },
        "initial_metrics": history[0],
        "final_metrics": history[-1],
    }
    write_json_atomic(output_root / "distillation_summary.json", summary)
    return summary


def _resolve_config(config: DistillationConfig) -> DistillationConfig:
    cfg = DistillationConfig(
        benchmark_root=Path(config.benchmark_root).expanduser().resolve(),
        prediction_root=Path(config.prediction_root).expanduser().resolve(),
        output_root=Path(config.output_root).expanduser().resolve(),
        scene_name=str(config.scene_name),
        source_native=Path(config.source_native).expanduser().resolve(),
        mosca_root=Path(config.mosca_root).expanduser().resolve(),
        target_cameras=tuple(str(camera) for camera in config.target_cameras),
        reference_camera=str(config.reference_camera),
        device=str(config.device),
        seed=int(config.seed),
        steps=int(config.steps),
        source_weight=float(config.source_weight),
        virtual_weight=float(config.virtual_weight),
        ssim_lambda=float(config.ssim_lambda),
        lr_scale=float(config.lr_scale),
        grad_clip_norm=float(config.grad_clip_norm),
        use_alpha_mask=bool(config.use_alpha_mask),
        debug_every=int(config.debug_every),
    )
    if cfg.steps <= 0:
        raise ValueError("Distillation steps must be positive")
    if cfg.source_weight <= 0.0 or cfg.virtual_weight <= 0.0:
        raise ValueError("Source and virtual-view weights must be positive")
    if not 0.0 <= cfg.ssim_lambda <= 1.0:
        raise ValueError("Distillation SSIM lambda must be in [0, 1]")
    if cfg.lr_scale <= 0.0:
        raise ValueError("Distillation learning-rate scale must be positive")
    if not cfg.source_native.is_dir():
        raise FileNotFoundError(f"Initial native reconstruction not found: {cfg.source_native}")
    if not cfg.mosca_root.is_dir():
        raise FileNotFoundError(f"Vendored MoSca source not found: {cfg.mosca_root}")
    for name in NATIVE_CHECKPOINTS:
        if not (cfg.source_native / name).is_file():
            raise FileNotFoundError(f"Missing initial native checkpoint: {name}")
    return cfg


def _resolve_scene_root(root: Path, scene_name: str) -> Path:
    root = root.expanduser().resolve()
    direct = root / scene_name
    if (direct / "metadata").is_dir():
        return direct
    if root.name == scene_name and (root / "metadata").is_dir():
        return root
    raise FileNotFoundError(f"Cannot find benchmark scene {scene_name} under {root}")


def _resolve_prediction_scene_root(root: Path, scene_name: str) -> Path:
    root = root.expanduser().resolve()
    direct = root / scene_name
    if (direct / "pred_rgb").is_dir():
        return direct
    if root.name == scene_name and (root / "pred_rgb").is_dir():
        return root
    raise FileNotFoundError(f"Cannot find generated predictions for {scene_name} under {root}")


def _validate_camera_name(name: str) -> str:
    name = str(name).strip()
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid camera name: {name!r}")
    return name


def _load_frame_manifest(scene_root: Path, camera: str) -> dict[str, Any]:
    path = scene_root / "metadata" / f"{camera}_frames.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load camera metadata {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ValueError(f"Invalid camera metadata: {path}")
    return payload


def _frame_records(payload: dict[str, Any]) -> tuple[FrameRecord, ...]:
    records = []
    for index, raw in enumerate(payload["frames"]):
        if not isinstance(raw, dict):
            raise ValueError(f"Frame record {index} is not an object")
        frame_name = str(raw.get("frame_name", f"{index:05d}"))
        source_frame_name = str(
            raw.get("source_frame_name", raw.get("reference_frame_name", frame_name))
        )
        records.append(
            FrameRecord(
                time_id=int(raw.get("time_id", index)),
                frame_name=frame_name,
                source_frame_name=source_frame_name,
            )
        )
    declared = payload.get("num_frames")
    if declared is not None and int(declared) != len(records):
        raise ValueError(
            f"Camera metadata declares {declared} frames but contains {len(records)}"
        )
    return tuple(records)


def _require_frames(directory: Path, expected: set[str], label: str) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {directory}")
    actual = {path.stem for path in directory.glob("*.png")}
    missing = sorted(expected - actual)
    if missing:
        raise FileNotFoundError(f"Missing {label}: {missing[:5]}")


def _load_camera_trajectory(path: Path, records: tuple[FrameRecord, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing target camera trajectory: {path}")
    with np.load(path) as archive:
        if "c2w" in archive:
            c2w = np.asarray(archive["c2w"], dtype=np.float32)
        elif "arr_0" in archive:
            c2w = np.asarray(archive["arr_0"], dtype=np.float32)
        elif archive.files:
            c2w = np.asarray(archive[archive.files[0]], dtype=np.float32)
        else:
            raise ValueError(f"Empty target camera trajectory: {path}")
        stored_time_ids = (
            None
            if "time_ids" not in archive
            else tuple(int(value) for value in np.asarray(archive["time_ids"]).tolist())
        )
    if c2w.shape != (len(records), 4, 4):
        raise ValueError(
            f"Target camera trajectory {path} has shape {c2w.shape}; "
            f"expected {(len(records), 4, 4)}"
        )
    expected_time_ids = tuple(record.time_id for record in records)
    if stored_time_ids is not None and stored_time_ids != expected_time_ids:
        raise ValueError(f"Target camera time IDs do not match metadata: {path}")
    if not np.isfinite(c2w).all():
        raise ValueError(f"Target camera trajectory contains non-finite values: {path}")
    return c2w


@lru_cache(maxsize=256)
def _read_rgb_cached(path_text: str) -> np.ndarray:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Missing RGB frame: {path}")
    image = np.asarray(imageio.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB image at {path}, got {image.shape}")
    return _to_uint8(image[..., :3], path)


@lru_cache(maxsize=256)
def _read_mask_cached(path_text: str) -> np.ndarray:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask frame: {path}")
    image = np.asarray(imageio.imread(path))
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected single-channel mask at {path}, got {image.shape}")
    return _to_uint8(image, path)


def _to_uint8(image: np.ndarray, path: Path) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer):
        if int(image.min()) < 0 or int(image.max()) > 255:
            raise ValueError(f"Unsupported image range at {path}")
        return image.astype(np.uint8)
    raise TypeError(f"Unsupported image dtype at {path}: {image.dtype}")


def _resize_rgb_uint8(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    value = torch.from_numpy(image.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
    value = torch.nn.functional.interpolate(
        value, size=target_hw, mode="bilinear", align_corners=False
    )[0]
    return (
        value.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )


def _rgb_tensor(image: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).to(device=device, dtype=torch.float32).div_(255.0)


def _mask_tensor(image: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).to(device=device, dtype=torch.float32).div_(255.0)


def _require_nonempty_mask(mask: torch.Tensor, camera: str, time_id: int) -> None:
    if mask.sum() <= 0:
        raise ValueError(f"Empty supervision mask for {camera} time_id={time_id}")


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    text = str(path.expanduser().resolve())
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:
            pass


def _load_worldtree_modules(mosca_root: Path) -> _WorldTreeModules:
    with _prepend_sys_path(mosca_root):
        camera_module = importlib.import_module("lib_moca.camera")
        static_module = importlib.import_module("lib_mosca.static_gs")
        dynamic_module = importlib.import_module("lib_mosca.dynamic_gs")
        render_module = importlib.import_module("lib_render.render_helper")
        loss_module = importlib.import_module("lib_mosca.gs_utils.loss_helper")
    return _WorldTreeModules(
        MonocularCameras=camera_module.MonocularCameras,
        StaticGaussian=static_module.StaticGaussian,
        DynSCFGaussian=dynamic_module.DynSCFGaussian,
        render=render_module.render,
        ssim=loss_module.ssim,
        gs_backend=str(render_module.GS_BACKEND),
    )


def _load_worldtree_state(
    *, modules: _WorldTreeModules, saved_dir: Path, device: torch.device
):
    cam_path = saved_dir / "photometric_cam.pth"
    static_path = saved_dir / f"photometric_s_model_{modules.gs_backend}.pth"
    dynamic_path = saved_dir / f"photometric_d_model_{modules.gs_backend}.pth"
    for path in (cam_path, static_path, dynamic_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing 4DGS checkpoint: {path}")
    cams = modules.MonocularCameras.load_from_ckpt(
        torch.load(cam_path, map_location="cpu")
    ).to(device)
    s_model = modules.StaticGaussian.load_from_ckpt(
        torch.load(static_path, map_location="cpu"), device=device
    ).to(device)
    d_model = modules.DynSCFGaussian.load_from_ckpt(
        torch.load(dynamic_path, map_location="cpu"), device=device
    ).to(device)
    return cams, s_model, d_model


def _freeze_dynamic_controls(d_model: Any) -> None:
    for parameter in d_model.scf.parameters():
        parameter.requires_grad_(False)
    d_model._skinning_weight.requires_grad_(False)
    d_model._dynamic_logit.requires_grad_(False)


def _build_optimizer(*, s_model: Any, d_model: Any, lr_scale: float):
    groups = []
    groups.extend(
        s_model.get_optimizable_list(
            lr_p=0.00016 * lr_scale,
            lr_q=0.001 * lr_scale,
            lr_s=0.005 * lr_scale,
            lr_o=0.05 * lr_scale,
            lr_sph=0.0025 * lr_scale,
        )
    )
    groups.extend(
        d_model.get_optimizable_list(
            lr_p=0.00016 * lr_scale,
            lr_q=0.001 * lr_scale,
            lr_s=0.005 * lr_scale,
            lr_o=0.05 * lr_scale,
            lr_sph=0.0025 * lr_scale,
            lr_dyn=None,
            lr_w=None,
            lr_np=None,
            lr_nq=None,
            lr_nsig=None,
        )
    )
    if not groups:
        raise RuntimeError("No trainable 4DGS parameters were found")
    return torch.optim.Adam(groups)


def _render_observation(
    *,
    modules: _WorldTreeModules,
    cams: Any,
    s_model: Any,
    d_model: Any,
    time_id: int,
    height: int,
    width: int,
    t_cw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gs5 = [list(s_model()), list(d_model(time_id))]
    return modules.render(
        gs5,
        height,
        width,
        cams.K(height, width).detach(),
        t_cw,
        bg_color=[1.0, 1.0, 1.0],
    )


def _compute_rgb_loss(
    *,
    gt_rgb: torch.Tensor,
    render_dict: dict[str, torch.Tensor],
    mask: torch.Tensor,
    ssim_fn: Any,
    ssim_lambda: float,
) -> _Loss:
    pred_rgb_chw = render_dict["rgb"]
    pred_rgb = pred_rgb_chw.permute(1, 2, 0)
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError(
            f"Rendered/target RGB shape mismatch: {tuple(pred_rgb.shape)} vs "
            f"{tuple(gt_rgb.shape)}"
        )
    if mask.shape != pred_rgb.shape[:2]:
        raise ValueError(
            f"RGB/mask shape mismatch: {tuple(pred_rgb.shape[:2])} vs {tuple(mask.shape)}"
        )
    denominator = torch.clamp(mask.sum(), min=1e-6)
    l1 = (torch.abs(pred_rgb - gt_rgb.detach()) * mask[..., None]).sum() / denominator
    ssim_loss = torch.zeros((), device=pred_rgb.device, dtype=pred_rgb.dtype)
    if ssim_lambda > 0.0:
        ssim_loss = 1.0 - ssim_fn(
            (pred_rgb_chw * mask[None])[None],
            (gt_rgb.detach().permute(2, 0, 1) * mask[None])[None],
        )
    return _Loss(
        total=l1 + ssim_loss * ssim_lambda,
        l1=l1,
        ssim=ssim_loss,
        pred_rgb=pred_rgb,
    )


def _prepare_output_root(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.rename(path.with_name(f".{path.name}.invalid-{time.time_ns()}"))
    path.mkdir(parents=True)
    return path


def _save_worldtree_state(
    *, output_dir: Path, modules: _WorldTreeModules, cams: Any, s_model: Any, d_model: Any
) -> None:
    torch.save(cams.state_dict(), output_dir / "photometric_cam.pth")
    torch.save(
        s_model.state_dict(), output_dir / f"photometric_s_model_{modules.gs_backend}.pth"
    )
    torch.save(
        d_model.state_dict(), output_dir / f"photometric_d_model_{modules.gs_backend}.pth"
    )


def _debug_due(step: int, steps: int, every: int) -> bool:
    return every > 0 and (step == 0 or step + 1 == steps or (step + 1) % every == 0)


def _save_debug_panel(
    path: Path,
    *,
    source_rgb: torch.Tensor,
    source_mask: torch.Tensor,
    source_pred: torch.Tensor,
    targets: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    rows = [
        np.concatenate(
            [
                _to_rgb8(source_rgb),
                _to_rgb8(source_pred),
                _to_mask8(source_mask),
                _to_error8(source_pred, source_rgb),
            ],
            axis=1,
        )
    ]
    for _, target_rgb, target_mask, target_pred in targets:
        rows.append(
            np.concatenate(
                [
                    _to_rgb8(target_rgb),
                    _to_rgb8(target_pred),
                    _to_mask8(target_mask),
                    _to_error8(target_pred, target_rgb),
                ],
                axis=1,
            )
        )
    width = max(row.shape[1] for row in rows)
    rows = [
        np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=255)
        for row in rows
    ]
    imageio.imwrite(path, np.concatenate(rows, axis=0))


def _to_rgb8(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )


def _to_mask8(value: torch.Tensor) -> np.ndarray:
    mask = value.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return np.repeat(mask[..., None], 3, axis=2)


def _to_error8(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    error = torch.abs(pred.detach() - target.detach()).mean(dim=-1)
    return _to_mask8(error)


def _portable_path(path: Path, relative_to: Path) -> str:
    return Path(os.path.relpath(path, relative_to)).as_posix()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
