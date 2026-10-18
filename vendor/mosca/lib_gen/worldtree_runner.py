from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from .diffsynth_bridge import (
    DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS,
    DiffSynthCheckpointConfig,
    load_diffsynth_model,
)
from .scene_benchmark import (
    SceneBenchmarkDataset,
    build_chunk_inputs,
    resolve_scene_chunk_seed,
    save_chunk_predictions,
    stage_chunk_assets,
    tensor_to_uint8_frames,
)
from .worldtree_benchmark import SceneLayout, camera_label, validate_scene_layout


@dataclass(frozen=True)
class WorldTreeInferenceConfig:
    benchmark_root: Path
    output_root: Path
    diffsynth_root: Path
    checkpoint: Path | None = None
    scene_names: tuple[str, ...] | None = None
    reference_camera: str | None = None
    target_cameras: tuple[str, ...] | None = None
    evaluation_unit_ids: tuple[str, ...] | None = None
    chunk_size: int = 73
    max_frames: int | None = None
    frame_stride: int = 1
    dense_prefix: int = 0
    pred_subdir: str = "pred_rgb"
    prompt: str = " "
    negative_prompt: str | None = None
    inference_res: tuple[int, int] | None = None
    num_inference_steps: int = 20
    cfg_scale: float = 1.0
    target_gbuffer_cfg_scale: float = 1.0
    target_gbuffer_dropout_scope: str = "target_only"
    target_gbuffer_uncond_fill_mode: str = "zero"
    vace_scale: float = 1.0
    seed: int | None = 0
    tiled: bool = False
    input_image: str | None = None
    use_scene_flow: bool = False
    split_cam_embed: bool = False
    scene_coord_pe: bool = False
    scene_coord_xyzt_pe: bool = False
    scene_coord_gated_residual: bool = False
    model_paths: str | None = None
    model_id_with_origin_paths: str = DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS
    tokenizer_path: str | None = None
    audio_processor_path: str | None = None
    trainable_models: str = "vace"
    remove_cached_keys: str = ""
    fp8_models: str | None = None
    offload_models: str | None = None
    device: str | None = None
    strict_layout: bool = True

    def required_optional_modalities(self) -> tuple[str, ...]:
        required = []
        if self.use_scene_flow:
            required.append("scene_flow_rgb")
        if self.scene_coord_pe or self.scene_coord_xyzt_pe:
            required.append("scene_coords")
        return tuple(required)

    def checkpoint_config(self) -> DiffSynthCheckpointConfig:
        if self.checkpoint is None:
            raise ValueError("checkpoint is required unless an external pipe is supplied.")
        return DiffSynthCheckpointConfig(
            diffsynth_root=self.diffsynth_root,
            checkpoint=self.checkpoint,
            model_paths=self.model_paths,
            model_id_with_origin_paths=self.model_id_with_origin_paths,
            tokenizer_path=self.tokenizer_path,
            audio_processor_path=self.audio_processor_path,
            trainable_models=self.trainable_models,
            remove_cached_keys=self.remove_cached_keys,
            fp8_models=self.fp8_models,
            offload_models=self.offload_models,
            split_cam_embed=self.split_cam_embed,
            scene_coord_pe=self.scene_coord_pe,
            scene_coord_xyzt_pe=self.scene_coord_xyzt_pe,
            scene_coord_gated_residual=self.scene_coord_gated_residual,
            num_inference_steps=self.num_inference_steps,
            use_scene_flow=self.use_scene_flow,
            device=self.device,
        )


@dataclass
class WorldTreeInferenceResult:
    output_root: Path
    summary_path: Path
    validated_scenes: tuple[SceneLayout, ...]
    num_chunks: int
    num_saved_frames: int


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "none"):
        return None
    if "x" not in text:
        raise ValueError(f"Resolution must look like HxW, got {value!r}")
    height_text, width_text = text.split("x", 1)
    return int(height_text), int(width_text)


def _normalize_names(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    cleaned = tuple(str(value).strip() for value in values if str(value).strip())
    return cleaned or None


def _normalize_camera_names(values: Iterable[str] | None) -> tuple[str, ...] | None:
    normalized = _normalize_names(values)
    if normalized is None:
        return None
    return tuple(camera_label(value) for value in normalized)


def _select_jobs(dataset_jobs: list[Any], config: WorldTreeInferenceConfig) -> list[Any]:
    scene_names = set(config.scene_names or ())
    target_cameras = set(_normalize_camera_names(config.target_cameras) or ())
    evaluation_unit_ids = set(config.evaluation_unit_ids or ())
    reference_camera = None if config.reference_camera is None else camera_label(config.reference_camera)
    selected = []
    for job in dataset_jobs:
        if scene_names and job.scene_name not in scene_names:
            continue
        if target_cameras and job.target_cam not in target_cameras:
            continue
        if evaluation_unit_ids and job.unit_id not in evaluation_unit_ids:
            continue
        if reference_camera is not None and job.reference_camera != reference_camera:
            continue
        selected.append(job)
    return selected


def _coerce_uint8_frames(pred_video: Any) -> np.ndarray:
    if torch.is_tensor(pred_video):
        return tensor_to_uint8_frames(pred_video)
    if isinstance(pred_video, np.ndarray):
        if pred_video.ndim != 4:
            raise TypeError(f"Expected [T,H,W,C] ndarray prediction, got {pred_video.shape}")
        if pred_video.dtype == np.uint8:
            return pred_video
        if np.issubdtype(pred_video.dtype, np.floating):
            clipped = np.clip(pred_video, -1.0, 1.0)
            return ((clipped + 1.0) * 127.5).round().astype(np.uint8)
        raise TypeError(f"Unsupported ndarray dtype for prediction: {pred_video.dtype}")
    if isinstance(pred_video, (list, tuple)):
        frames = []
        for frame in pred_video:
            if isinstance(frame, Image.Image):
                frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
            else:
                array = np.asarray(frame)
                if array.ndim == 2:
                    array = np.repeat(array[..., None], 3, axis=2)
                if array.dtype != np.uint8:
                    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
                frames.append(array[..., :3])
        if not frames:
            raise TypeError("Prediction frame list was empty.")
        return np.stack(frames, axis=0)
    raise TypeError(f"Unsupported prediction type: {type(pred_video)!r}")


def _config_summary_dict(config: WorldTreeInferenceConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def run_worldtree_inference(
    config: WorldTreeInferenceConfig,
    *,
    pipe: Any | None = None,
    verbose: bool = True,
) -> WorldTreeInferenceResult:
    benchmark_root = config.benchmark_root.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = SceneBenchmarkDataset(
        benchmark_root=benchmark_root,
        scene_names=config.scene_names,
        chunk_size=int(config.chunk_size),
        max_frames=config.max_frames,
        frame_stride=int(config.frame_stride),
        dense_prefix=int(config.dense_prefix),
    )
    selected_jobs = _select_jobs(dataset.jobs, config)
    if not selected_jobs:
        raise RuntimeError(
            "No benchmark chunks matched the requested scene/unit filters. "
            f"Scenes={config.scene_names}, targets={config.target_cameras}, "
            f"units={config.evaluation_unit_ids}, reference={config.reference_camera}."
        )

    validated_scenes_by_name: dict[str, SceneLayout] = {}
    for job in selected_jobs:
        if job.scene_name in validated_scenes_by_name:
            continue
        validated_scenes_by_name[job.scene_name] = validate_scene_layout(
            job.scene_root,
            strict_files=config.strict_layout,
            required_optional_modalities=config.required_optional_modalities(),
        )
    validated_scenes = tuple(validated_scenes_by_name.values())

    if pipe is None:
        loaded = load_diffsynth_model(config.checkpoint_config(), verbose=verbose)
        pipe = loaded.pipe

    # Optional first-frame (image-to-video) condition: anchor the generated chunk's
    # first frame to a provided image (e.g. the previous segment's last frame).
    input_image_pil = None
    if config.input_image is not None:
        from PIL import Image
        input_image_pil = Image.open(config.input_image).convert("RGB")
        if verbose:
            print(f"[lib_gen] Using first-frame input_image condition: {config.input_image}")

    total_saved_frames = 0
    use_scene_coords = bool(config.scene_coord_pe or config.scene_coord_xyzt_pe)
    with torch.no_grad():
        for job_index, chunk in enumerate(selected_jobs, start=1):
            if verbose:
                print(
                    f"[lib_gen] Chunk {job_index}/{len(selected_jobs)} "
                    f"{chunk.scene_name}/{chunk.unit_id}/{chunk.target_cam}"
                )
            cameras_path = stage_chunk_assets(
                chunk=chunk,
                output_root=output_root,
                use_scene_coords=use_scene_coords,
            )
            inputs, frame_names = build_chunk_inputs(
                chunk,
                inference_res=config.inference_res,
                use_scene_flow=bool(config.use_scene_flow),
                cameras_path=cameras_path,
            )
            seed = resolve_scene_chunk_seed(config.seed, chunk)
            pipe_kwargs = dict(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                split_cam_embed=bool(config.split_cam_embed),
                height=inputs["vace_video"][0].size[1],
                width=inputs["vace_video"][0].size[0],
                num_frames=len(frame_names),
                cfg_scale=float(config.cfg_scale),
                target_gbuffer_cfg_scale=float(config.target_gbuffer_cfg_scale),
                target_gbuffer_dropout_scope=str(config.target_gbuffer_dropout_scope),
                target_gbuffer_uncond_fill_mode=str(config.target_gbuffer_uncond_fill_mode),
                tiled=bool(config.tiled),
                num_inference_steps=int(config.num_inference_steps),
                vace_scale=float(config.vace_scale),
                output_type="floatpoint",
                seed=seed,
            )
            # Only pass input_image when set, so the default path is byte-for-byte
            # identical to before this plumbing was added.
            if input_image_pil is not None:
                pipe_kwargs["input_image"] = input_image_pil
            pred_video = pipe(**pipe_kwargs, **inputs)
            pred_frames = _coerce_uint8_frames(pred_video)
            total_saved_frames += int(
                save_chunk_predictions(
                    chunk=chunk,
                    output_root=output_root,
                    pred_subdir=config.pred_subdir,
                    pred_frames=pred_frames,
                )
            )

    summary = {
        "benchmark_root": str(benchmark_root),
        "output_root": str(output_root),
        "num_chunks": len(selected_jobs),
        "num_saved_frames": total_saved_frames,
        "scenes": [
            {
                "scene_name": layout.scene_name,
                "camera_names": list(layout.camera_names),
                "evaluation_unit_ids": list(layout.evaluation_unit_ids),
                "available_optional_modalities": list(layout.available_optional_modalities),
            }
            for layout in validated_scenes
        ],
        "config": _config_summary_dict(config),
    }
    summary_path = output_root / "worldtree_inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if verbose:
        print(f"[lib_gen] Saved inference summary: {summary_path}")

    return WorldTreeInferenceResult(
        output_root=output_root,
        summary_path=summary_path,
        validated_scenes=validated_scenes,
        num_chunks=len(selected_jobs),
        num_saved_frames=total_saved_frames,
    )
