from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import imageio.v3 as iio
import numpy as np

from .worldtree_benchmark import (
    OPTIONAL_MODALITIES,
    REQUIRED_MODALITIES,
    SceneLayout,
    camera_label,
    strip_camera_prefix,
    validate_scene_layout,
)

IMAGE_MODALITIES = ("rgb", "depth", "normal", "alpha", "gt", "scene_flow_rgb", "test_dynamic")
TENSOR_MODALITIES = ("scene_coords", "scene_flow")
DEFAULT_EXPORT_OUTPUTS = REQUIRED_MODALITIES + OPTIONAL_MODALITIES


class WorldTreeExportError(RuntimeError):
    """Raised when an adapter cannot produce a valid WorldTree-style layout."""


@dataclass(frozen=True)
class ExportFrameRecord:
    frame_idx: int
    time_id: int
    frame_name: str
    has_gt: bool = True
    has_test_dynamic: bool = False
    has_dataset_frame_at_time: bool = True
    has_exact_tto_pose: bool = True
    render_pose_source_kind: str = "adapter"
    render_pose_source_time_id: int | None = None
    render_pose_source_frame_name: str | None = None
    render_pose_archive_path: str | None = None
    rgb_source_kind: str = "render"
    scene_flow_valid: bool = True
    reference_frame_name: str | None = None
    source_frame_name: str | None = None


@dataclass(frozen=True)
class RenderedFrame:
    frame_name: str
    c2w: np.ndarray
    images: Mapping[str, np.ndarray]
    tensors: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationUnitPlan:
    unit_id: str
    source_camera: str
    target_camera: str
    reference_camera: str


@dataclass(frozen=True)
class SceneExportPlan:
    scene_name: str
    reference_camera: str
    camera_records: Mapping[str, tuple[ExportFrameRecord, ...]]
    evaluation_units: tuple[EvaluationUnitPlan, ...]
    camera_pose_stats: Mapping[str, Mapping[str, Any] | None] = field(default_factory=dict)
    camera_scene_flow_rgb_hi: Mapping[str, Sequence[float] | None] = field(default_factory=dict)
    scene_summary_extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.camera_records))


@dataclass(frozen=True)
class WorldTreeExportConfig:
    output_root: Path
    scene_name: str
    reference_camera: str
    target_cameras: tuple[str, ...] | None = None
    frame_limit: int | None = None
    outputs: tuple[str, ...] = DEFAULT_EXPORT_OUTPUTS
    overwrite: bool = False


@dataclass
class WorldTreeExportResult:
    scene_root: Path
    validated_scene: SceneLayout


class WorldTreeExportAdapter(Protocol):
    def build_scene_plan(
        self,
        reference_camera: str,
        *,
        target_cameras: Sequence[str] | None = None,
        max_frames: int | None = None,
    ) -> SceneExportPlan:
        ...

    def render_camera_frames(
        self,
        camera_name: str,
        records: Sequence[ExportFrameRecord],
        outputs: Sequence[str],
    ) -> Iterable[RenderedFrame]:
        ...


def normalize_outputs(outputs: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    valid = set(DEFAULT_EXPORT_OUTPUTS)
    for output in outputs:
        name = str(output).strip()
        if not name:
            continue
        if name not in valid:
            raise ValueError(
                f"Unsupported export modality '{name}'. Valid outputs: {sorted(valid)}"
            )
        if name in seen:
            continue
        cleaned.append(name)
        seen.add(name)
    if not cleaned:
        raise ValueError("At least one export modality must be requested.")
    return tuple(cleaned)


def _normalize_camera_list(camera_names: Sequence[str] | None) -> tuple[str, ...] | None:
    if camera_names is None:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    for camera_name in camera_names:
        normalized = camera_label(camera_name)
        if normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
    return tuple(cleaned) or None


def save_frame_png(frame: np.ndarray, path: Path) -> None:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.0 and array.min(initial=0.0) >= 0.0:
            array = (array * 255.0).round()
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, array)


def save_frame_npz(data: np.ndarray, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: data})


def compute_scene_flow_bounds(flow_paths: Sequence[Path]) -> np.ndarray:
    if not flow_paths:
        raise ValueError("Cannot compute scene-flow bounds from an empty file list.")
    abs_chunks: list[np.ndarray] = []
    for flow_path in flow_paths:
        with np.load(flow_path) as data:
            if "scene_flow" not in data:
                raise KeyError(f"Missing 'scene_flow' key in {flow_path}")
            flow = np.asarray(data["scene_flow"], dtype=np.float32)
        abs_chunks.append(np.abs(flow.reshape(-1, 3)))
    flat_abs = np.concatenate(abs_chunks, axis=0)
    hi = np.percentile(flat_abs, 99, axis=0)
    return np.maximum(hi, 1e-6)


def scene_flow_to_rgb_u8(flow: np.ndarray, hi: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    zero_mask = np.all(np.isclose(flow, 0.0, atol=1e-8), axis=-1)
    if zero_mask.all():
        return np.zeros(flow.shape, dtype=np.uint8)
    normed = np.clip(flow / hi[None, None], -1.0, 1.0)
    rgb = ((normed * 0.5 + 0.5) * 255.0).astype(np.uint8)
    rgb[zero_mask] = 0
    return rgb


def write_text_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if lines:
            handle.write("\n".join(lines))
            handle.write("\n")


def write_camera_c2w_npz(c2w_list: Sequence[np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(c2w_list, axis=0).astype(np.float32)
    np.savez_compressed(out_path, c2w=stacked)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2)


def _camera_id(camera_name: str) -> str:
    return strip_camera_prefix(camera_name)


def _frame_record_payload(record: ExportFrameRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_idx": int(record.frame_idx),
        "time_id": int(record.time_id),
        "frame_name": record.frame_name,
        "has_gt": bool(record.has_gt),
        "has_test_dynamic": bool(record.has_test_dynamic),
        "has_dataset_frame_at_time": bool(record.has_dataset_frame_at_time),
        "has_exact_tto_pose": bool(record.has_exact_tto_pose),
        "render_pose_source_kind": record.render_pose_source_kind,
        "render_pose_source_time_id": (
            int(record.time_id)
            if record.render_pose_source_time_id is None
            else int(record.render_pose_source_time_id)
        ),
        "render_pose_source_frame_name": (
            record.frame_name
            if record.render_pose_source_frame_name is None
            else record.render_pose_source_frame_name
        ),
        "render_pose_archive_path": record.render_pose_archive_path,
        "rgb_source_kind": record.rgb_source_kind,
        "scene_flow_valid": bool(record.scene_flow_valid),
    }
    if record.reference_frame_name is not None:
        payload["reference_frame_name"] = record.reference_frame_name
    if record.source_frame_name is not None:
        payload["source_frame_name"] = record.source_frame_name
    return payload


def _write_camera_manifest(
    *,
    scene_name: str,
    cam_name: str,
    reference_camera: str,
    records: Sequence[ExportFrameRecord],
    pose_stats: Mapping[str, Any] | None,
    scene_flow_rgb_hi: Sequence[float] | None,
    out_path: Path,
) -> None:
    payload = {
        "scene_name": scene_name,
        "cam_id": _camera_id(cam_name),
        "cam_label": camera_label(cam_name),
        "reference_camera_id": _camera_id(reference_camera),
        "reference_camera_label": camera_label(reference_camera),
        "num_frames": len(records),
        "num_valid_gt": sum(1 for record in records if record.has_gt),
        "num_valid_test_dynamic": sum(1 for record in records if record.has_test_dynamic),
        "num_dataset_frames": sum(1 for record in records if record.has_dataset_frame_at_time),
        "num_exact_tto_pose_frames": sum(1 for record in records if record.has_exact_tto_pose),
        "pose_stats": None if pose_stats is None else dict(pose_stats),
        "scene_flow_rgb_hi": None if scene_flow_rgb_hi is None else list(scene_flow_rgb_hi),
        "frames": [_frame_record_payload(record) for record in records],
    }
    write_json(out_path, payload)


def _write_evaluation_units_manifest(
    *,
    scene_name: str,
    reference_camera: str,
    evaluation_units: Sequence[EvaluationUnitPlan],
    out_path: Path,
) -> None:
    payload = {
        "scene_name": scene_name,
        "reference_camera": camera_label(reference_camera),
        "evaluation_units": [
            {
                "unit_id": unit.unit_id,
                "source_camera": camera_label(unit.source_camera),
                "target_camera": camera_label(unit.target_camera),
                "reference_camera": camera_label(unit.reference_camera),
            }
            for unit in evaluation_units
        ],
    }
    write_json(out_path, payload)


def _write_scene_summary(
    *,
    scene_name: str,
    scene_root: Path,
    reference_camera: str,
    camera_names: Sequence[str],
    evaluation_units: Sequence[EvaluationUnitPlan],
    extra: Mapping[str, Any],
    out_path: Path,
) -> None:
    payload = {
        "scene_name": scene_name,
        "reference_camera_id": _camera_id(reference_camera),
        "reference_camera_label": camera_label(reference_camera),
        "camera_names": [camera_label(name) for name in camera_names],
        "evaluation_unit_ids": [unit.unit_id for unit in evaluation_units],
        "evaluation_units": [
            {
                "unit_id": unit.unit_id,
                "source_camera": camera_label(unit.source_camera),
                "target_camera": camera_label(unit.target_camera),
                "reference_camera": camera_label(unit.reference_camera),
            }
            for unit in evaluation_units
        ],
        "result_dir": str(scene_root),
    }
    payload.update(dict(extra))
    write_json(out_path, payload)


class WorldTreeLayoutExporter:
    def __init__(self, config: WorldTreeExportConfig) -> None:
        self.config = WorldTreeExportConfig(
            output_root=config.output_root.expanduser().resolve(),
            scene_name=str(config.scene_name),
            reference_camera=camera_label(config.reference_camera),
            target_cameras=_normalize_camera_list(config.target_cameras),
            frame_limit=config.frame_limit,
            outputs=normalize_outputs(config.outputs),
            overwrite=bool(config.overwrite),
        )

    def export(self, adapter: WorldTreeExportAdapter) -> WorldTreeExportResult:
        plan = adapter.build_scene_plan(
            self.config.reference_camera,
            target_cameras=self.config.target_cameras,
            max_frames=self.config.frame_limit,
        )
        self._validate_plan(plan)

        scene_root = self.config.output_root / self.config.scene_name
        self._ensure_scene_root(scene_root)
        self._ensure_core_dirs(scene_root)
        for camera_name in plan.camera_names:
            self._ensure_camera_output_dirs(scene_root, camera_name)

        c2w_by_camera: dict[str, list[np.ndarray]] = {}
        scene_flow_rgb_hi_by_camera: dict[str, tuple[float, ...] | None] = {}
        for camera_name in plan.camera_names:
            records = plan.camera_records[camera_name]
            c2w_by_camera[camera_name] = self._write_camera_outputs(
                scene_root=scene_root,
                camera_name=camera_name,
                records=records,
                rendered_frames=adapter.render_camera_frames(camera_name, records, self.config.outputs),
            )
            scene_flow_rgb_hi_by_camera[camera_name] = self._ensure_scene_flow_rgb_outputs(
                scene_root=scene_root,
                camera_name=camera_name,
                records=records,
                preferred_hi=plan.camera_scene_flow_rgb_hi.get(camera_name),
            )

        cameras_dir = scene_root / "cameras"
        frame_names_dir = scene_root / "frame_names"
        metadata_dir = scene_root / "metadata"
        for camera_name in plan.camera_names:
            records = plan.camera_records[camera_name]
            write_camera_c2w_npz(
                c2w_by_camera[camera_name], cameras_dir / f"{camera_name}_c2w.npz"
            )
            write_text_lines(
                frame_names_dir / f"{camera_name}_all.txt",
                [record.frame_name for record in records],
            )
            write_text_lines(
                frame_names_dir / f"{camera_name}_valid.txt",
                [record.frame_name for record in records if record.has_gt],
            )
            _write_camera_manifest(
                scene_name=self.config.scene_name,
                cam_name=camera_name,
                reference_camera=self.config.reference_camera,
                records=records,
                pose_stats=plan.camera_pose_stats.get(camera_name),
                scene_flow_rgb_hi=scene_flow_rgb_hi_by_camera[camera_name],
                out_path=metadata_dir / f"{camera_name}_frames.json",
            )

        _write_evaluation_units_manifest(
            scene_name=self.config.scene_name,
            reference_camera=self.config.reference_camera,
            evaluation_units=plan.evaluation_units,
            out_path=metadata_dir / "eval_units.json",
        )
        _write_scene_summary(
            scene_name=self.config.scene_name,
            scene_root=scene_root,
            reference_camera=self.config.reference_camera,
            camera_names=plan.camera_names,
            evaluation_units=plan.evaluation_units,
            extra=plan.scene_summary_extra,
            out_path=metadata_dir / "scene_summary.json",
        )

        required_optional = [name for name in OPTIONAL_MODALITIES if name in self.config.outputs]
        validated_scene = validate_scene_layout(
            scene_root,
            strict_files=True,
            required_optional_modalities=required_optional,
        )
        return WorldTreeExportResult(scene_root=scene_root, validated_scene=validated_scene)

    def _validate_plan(self, plan: SceneExportPlan) -> None:
        if plan.scene_name != self.config.scene_name:
            raise WorldTreeExportError(
                f"Adapter produced scene '{plan.scene_name}', expected '{self.config.scene_name}'."
            )
        if camera_label(plan.reference_camera) != self.config.reference_camera:
            raise WorldTreeExportError("Adapter reference camera does not match exporter config.")
        if not plan.evaluation_units:
            raise WorldTreeExportError("Adapter produced no evaluation units.")
        if self.config.reference_camera not in plan.camera_records:
            raise WorldTreeExportError(
                f"Adapter did not provide records for reference camera {self.config.reference_camera}."
            )

        requested_targets = set(self.config.target_cameras or ())
        seen_targets: set[str] = set()
        seen_units: set[str] = set()
        for unit in plan.evaluation_units:
            if unit.unit_id in seen_units:
                raise WorldTreeExportError(f"Duplicate evaluation unit id '{unit.unit_id}'.")
            seen_units.add(unit.unit_id)
            source_camera = camera_label(unit.source_camera)
            target_camera = camera_label(unit.target_camera)
            reference_camera = camera_label(unit.reference_camera)
            if reference_camera != self.config.reference_camera:
                raise WorldTreeExportError(
                    f"Evaluation unit {unit.unit_id} uses reference camera {reference_camera}, "
                    f"expected {self.config.reference_camera}."
                )
            if source_camera not in plan.camera_records:
                raise WorldTreeExportError(
                    f"Evaluation unit {unit.unit_id} references missing source camera {source_camera}."
                )
            if target_camera not in plan.camera_records:
                raise WorldTreeExportError(
                    f"Evaluation unit {unit.unit_id} references missing target camera {target_camera}."
                )
            source_records = plan.camera_records[source_camera]
            target_records = plan.camera_records[target_camera]
            if len(source_records) == 0 or len(target_records) == 0:
                raise WorldTreeExportError(
                    f"Evaluation unit {unit.unit_id} produced an empty frame sequence."
                )
            if len(source_records) != len(target_records):
                raise WorldTreeExportError(
                    f"Evaluation unit {unit.unit_id} requires aligned source/target frame counts."
                )
            seen_targets.add(target_camera)

        if requested_targets and seen_targets != requested_targets:
            raise WorldTreeExportError(
                "Adapter targets do not match exporter selection. "
                f"Requested={sorted(requested_targets)}, got={sorted(seen_targets)}."
            )

        for camera_name, records in plan.camera_records.items():
            if not records:
                raise WorldTreeExportError(f"Adapter produced zero records for {camera_name}.")

    def _ensure_scene_root(self, scene_root: Path) -> None:
        if scene_root.is_dir() and not self.config.overwrite and any(scene_root.iterdir()):
            raise WorldTreeExportError(
                f"Scene root already exists and is non-empty: {scene_root}. "
                "Pass overwrite=True to allow writing on top of it."
            )
        scene_root.mkdir(parents=True, exist_ok=True)

    def _ensure_core_dirs(self, scene_root: Path) -> None:
        for directory_name in ("cameras", "frame_names", "metadata"):
            (scene_root / directory_name).mkdir(parents=True, exist_ok=True)

    def _ensure_camera_output_dirs(self, scene_root: Path, camera_name: str) -> None:
        for modality in self.config.outputs:
            (scene_root / modality / camera_name).mkdir(parents=True, exist_ok=True)

    def _write_camera_outputs(
        self,
        *,
        scene_root: Path,
        camera_name: str,
        records: Sequence[ExportFrameRecord],
        rendered_frames: Iterable[RenderedFrame],
    ) -> list[np.ndarray]:
        rendered_list = list(rendered_frames)
        if len(rendered_list) != len(records):
            raise WorldTreeExportError(
                f"Adapter returned {len(rendered_list)} rendered frames for {camera_name}, "
                f"expected {len(records)}."
            )

        c2w_list: list[np.ndarray] = []
        for record, rendered in zip(records, rendered_list):
            if rendered.frame_name != record.frame_name:
                raise WorldTreeExportError(
                    f"Frame-name mismatch for {camera_name}: "
                    f"{rendered.frame_name} vs {record.frame_name}"
                )
            c2w = np.asarray(rendered.c2w, dtype=np.float32)
            if c2w.shape != (4, 4):
                raise WorldTreeExportError(
                    f"Expected 4x4 c2w matrix for {camera_name}/{record.frame_name}, got {c2w.shape}"
                )
            c2w_list.append(c2w)

            for modality in self.config.outputs:
                if modality in IMAGE_MODALITIES:
                    image = rendered.images.get(modality)
                    if image is None:
                        if modality == "scene_flow_rgb" and "scene_flow" in self.config.outputs:
                            continue
                        if modality == "gt" and not record.has_gt:
                            continue
                        if modality == "test_dynamic" and not record.has_test_dynamic:
                            continue
                        raise WorldTreeExportError(
                            f"Adapter did not provide image modality '{modality}' "
                            f"for {camera_name}/{record.frame_name}."
                        )
                    save_frame_png(
                        np.asarray(image),
                        scene_root / modality / camera_name / f"{record.frame_name}.png",
                    )
                else:
                    tensor = rendered.tensors.get(modality)
                    if tensor is None:
                        raise WorldTreeExportError(
                            f"Adapter did not provide tensor modality '{modality}' "
                            f"for {camera_name}/{record.frame_name}."
                        )
                    key = "scene_coords" if modality == "scene_coords" else "scene_flow"
                    save_frame_npz(
                        np.asarray(tensor),
                        key,
                        scene_root / modality / camera_name / f"{record.frame_name}.npz",
                    )
        return c2w_list

    def _ensure_scene_flow_rgb_outputs(
        self,
        *,
        scene_root: Path,
        camera_name: str,
        records: Sequence[ExportFrameRecord],
        preferred_hi: Sequence[float] | None,
    ) -> tuple[float, ...] | None:
        if "scene_flow_rgb" not in self.config.outputs:
            return None if preferred_hi is None else tuple(float(value) for value in preferred_hi)
        rgb_paths = [
            scene_root / "scene_flow_rgb" / camera_name / f"{record.frame_name}.png"
            for record in records
        ]
        if preferred_hi is not None and all(path.is_file() for path in rgb_paths):
            return tuple(float(value) for value in preferred_hi)
        if "scene_flow" not in self.config.outputs:
            if all(path.is_file() for path in rgb_paths):
                return None if preferred_hi is None else tuple(float(value) for value in preferred_hi)
            raise WorldTreeExportError(
                f"scene_flow_rgb requested for {camera_name} but no scene_flow tensors are available "
                "to synthesize the colorized visualization."
            )

        flow_paths = [
            scene_root / "scene_flow" / camera_name / f"{record.frame_name}.npz"
            for record in records
        ]
        hi = compute_scene_flow_bounds(flow_paths)
        hi_tuple = tuple(float(value) for value in hi.tolist())
        for flow_path, rgb_path in zip(flow_paths, rgb_paths):
            with np.load(flow_path) as data:
                flow = np.asarray(data["scene_flow"], dtype=np.float32)
            save_frame_png(scene_flow_to_rgb_u8(flow, hi), rgb_path)
        return hi_tuple
