from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REQUIRED_MODALITIES = ("rgb", "depth", "normal", "alpha", "gt")
OPTIONAL_MODALITIES = ("scene_coords", "scene_flow", "scene_flow_rgb", "test_dynamic")
IMAGE_MODALITIES = ("rgb", "depth", "normal", "alpha", "gt", "scene_flow_rgb", "test_dynamic")
TENSOR_MODALITIES = ("scene_coords", "scene_flow")
PREFERRED_NPZ_KEYS = {
    "camera": ("c2w", "arr_0"),
    "scene_coords": ("scene_coords", "coords", "xyz", "arr_0"),
    "scene_flow": ("scene_flow", "flow", "arr_0"),
}

SCENE_EVAL_UNITS_FILENAME = "eval_units.json"
SCENE_SUMMARY_FILENAME = "scene_summary.json"


class WorldTreeBenchmarkError(RuntimeError):
    """Raised when a scene root does not match the expected benchmark layout."""


@dataclass(frozen=True)
class FrameRecord:
    frame_idx: int
    time_id: int
    frame_name: str
    source_frame_name: str
    source_input_frame_name: str
    has_gt: bool
    has_test_dynamic: bool
    has_dataset_frame_at_time: bool
    has_exact_tto_pose: bool
    render_pose_source_kind: str | None
    render_pose_source_time_id: int | None
    render_pose_archive_path: str | None
    rgb_source_kind: str | None
    scene_flow_valid: bool


@dataclass(frozen=True)
class EvaluationUnit:
    unit_id: str
    source_camera: str
    target_camera: str
    reference_camera: str


@dataclass(frozen=True)
class SceneCameraManifest:
    scene_name: str
    camera_name: str
    metadata_path: Path
    frame_names_path: Path
    valid_frame_names_path: Path
    reference_camera: str
    pose_stats: dict[str, Any] | None
    scene_flow_rgb_hi: tuple[float, ...] | None
    records: tuple[FrameRecord, ...]

    @property
    def frame_names(self) -> tuple[str, ...]:
        return tuple(record.frame_name for record in self.records)

    @property
    def valid_frame_names(self) -> tuple[str, ...]:
        return tuple(record.frame_name for record in self.records if record.has_gt)

    @property
    def time_ids(self) -> tuple[int, ...]:
        return tuple(record.time_id for record in self.records)


@dataclass(frozen=True)
class SceneLayout:
    scene_name: str
    scene_root: Path
    reference_camera: str
    camera_manifests: dict[str, SceneCameraManifest]
    evaluation_units: tuple[EvaluationUnit, ...]
    available_optional_modalities: tuple[str, ...]

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.camera_manifests))

    @property
    def evaluation_unit_ids(self) -> tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.evaluation_units)

    def manifest(self, camera_name: str) -> SceneCameraManifest:
        return self.camera_manifests[camera_label(camera_name)]


def strip_camera_prefix(camera_name: str) -> str:
    camera_name = str(camera_name).strip()
    if camera_name.startswith("cam"):
        return camera_name[3:]
    return camera_name


def camera_label(cam_id: str | int) -> str:
    cam_text = strip_camera_prefix(str(cam_id))
    return f"cam{int(cam_text)}" if cam_text.isdigit() else f"cam{cam_text}"


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise WorldTreeBenchmarkError(f"Expected a JSON object in {path}")
    return payload


def read_text_lines(path: Path) -> tuple[str, ...]:
    with open(path, "r", encoding="utf-8") as handle:
        return tuple(line.strip() for line in handle.readlines() if line.strip())


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
            return strip_camera_prefix(camera_text)
    if camera_id is not None:
        camera_text = str(camera_id).strip()
        if camera_text != "":
            return strip_camera_prefix(camera_text)
    return None


def infer_source_input_frame_name(frame: dict[str, Any], metadata_payload: dict[str, Any]) -> str:
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


def resolve_source_frame_name(frame: dict[str, Any]) -> str:
    source_frame_name = frame.get("source_frame_name")
    if source_frame_name is None:
        source_frame_name = frame.get("render_pose_source_frame_name", frame["frame_name"])
    return str(source_frame_name)


def load_frame_records(metadata_path: Path) -> tuple[FrameRecord, ...]:
    payload = load_json(metadata_path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise WorldTreeBenchmarkError(f"Missing 'frames' list in {metadata_path}")
    records: list[FrameRecord] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise WorldTreeBenchmarkError(f"Invalid frame payload in {metadata_path}")
        render_pose_source_time_id = frame.get("render_pose_source_time_id")
        records.append(
            FrameRecord(
                frame_idx=int(frame["frame_idx"]),
                time_id=int(frame.get("time_id", frame["frame_idx"])),
                frame_name=str(frame["frame_name"]),
                source_frame_name=resolve_source_frame_name(frame),
                source_input_frame_name=infer_source_input_frame_name(frame, payload),
                has_gt=bool(frame.get("has_gt", False)),
                has_test_dynamic=bool(frame.get("has_test_dynamic", False)),
                has_dataset_frame_at_time=bool(frame.get("has_dataset_frame_at_time", False)),
                has_exact_tto_pose=bool(frame.get("has_exact_tto_pose", False)),
                render_pose_source_kind=(
                    None
                    if frame.get("render_pose_source_kind") is None
                    else str(frame["render_pose_source_kind"])
                ),
                render_pose_source_time_id=(
                    None if render_pose_source_time_id is None else int(render_pose_source_time_id)
                ),
                render_pose_archive_path=(
                    None
                    if frame.get("render_pose_archive_path") is None
                    else str(frame["render_pose_archive_path"])
                ),
                rgb_source_kind=(
                    None if frame.get("rgb_source_kind") is None else str(frame["rgb_source_kind"])
                ),
                scene_flow_valid=bool(frame.get("scene_flow_valid", False)),
            )
        )
    return tuple(records)


def _load_npz_array(path: Path, kind: str) -> np.ndarray:
    preferred_keys = PREFERRED_NPZ_KEYS[kind]
    with np.load(path) as data:
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])
        if len(data.files) == 0:
            raise WorldTreeBenchmarkError(f"Empty npz file: {path}")
        return np.asarray(data[data.files[0]])


def _ensure_camera_npz(path: Path, expected_length: int, camera_name: str) -> None:
    if not path.is_file():
        raise WorldTreeBenchmarkError(f"Missing camera trajectory file: {path}")
    c2w = _load_npz_array(path, kind="camera")
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise WorldTreeBenchmarkError(
            f"Unexpected camera trajectory shape in {path}: {tuple(c2w.shape)}"
        )
    if c2w.shape[0] != expected_length:
        raise WorldTreeBenchmarkError(
            f"Camera trajectory length mismatch for {camera_name}: "
            f"{c2w.shape[0]} poses vs {expected_length} metadata frames."
        )


def _expect_modality_dir(scene_root: Path, modality: str, cam_name: str) -> Path:
    path = scene_root / modality / cam_name
    if not path.is_dir():
        raise WorldTreeBenchmarkError(f"Missing {modality} directory: {path}")
    return path


def _missing_frames(directory: Path, frame_names: Iterable[str], suffix: str) -> list[str]:
    missing = []
    for frame_name in frame_names:
        if not (directory / f"{frame_name}{suffix}").is_file():
            missing.append(frame_name)
    return missing


def _load_scene_camera_manifest(scene_root: Path, cam_name: str) -> SceneCameraManifest:
    metadata_path = scene_root / "metadata" / f"{cam_name}_frames.json"
    all_path = scene_root / "frame_names" / f"{cam_name}_all.txt"
    valid_path = scene_root / "frame_names" / f"{cam_name}_valid.txt"
    if not metadata_path.is_file():
        raise WorldTreeBenchmarkError(f"Missing metadata file: {metadata_path}")
    if not all_path.is_file():
        raise WorldTreeBenchmarkError(f"Missing frame list: {all_path}")
    if not valid_path.is_file():
        raise WorldTreeBenchmarkError(f"Missing frame list: {valid_path}")
    payload = load_json(metadata_path)
    records = load_frame_records(metadata_path)
    listed_all = read_text_lines(all_path)
    expected_all = tuple(record.frame_name for record in records)
    if listed_all != expected_all:
        raise WorldTreeBenchmarkError(
            f"Frame list mismatch in {all_path}. "
            f"Metadata has {len(expected_all)} entries, txt has {len(listed_all)}."
        )
    listed_valid = read_text_lines(valid_path)
    expected_valid = tuple(record.frame_name for record in records if record.has_gt)
    if listed_valid != expected_valid:
        raise WorldTreeBenchmarkError(
            f"Valid-frame list mismatch in {valid_path}. "
            f"Metadata has {len(expected_valid)} valid entries, txt has {len(listed_valid)}."
        )
    return SceneCameraManifest(
        scene_name=scene_root.name,
        camera_name=cam_name,
        metadata_path=metadata_path,
        frame_names_path=all_path,
        valid_frame_names_path=valid_path,
        reference_camera=camera_label(
            str(payload.get("reference_camera_label", payload.get("reference_camera_id", cam_name)))
        ),
        pose_stats=payload.get("pose_stats"),
        scene_flow_rgb_hi=(
            None
            if payload.get("scene_flow_rgb_hi") is None
            else tuple(float(value) for value in payload["scene_flow_rgb_hi"])
        ),
        records=records,
    )


def _load_evaluation_units(scene_root: Path) -> tuple[str, tuple[EvaluationUnit, ...]]:
    metadata_path = scene_root / "metadata" / SCENE_EVAL_UNITS_FILENAME
    if not metadata_path.is_file():
        raise WorldTreeBenchmarkError(f"Missing evaluation-unit manifest: {metadata_path}")
    payload = load_json(metadata_path)
    units_payload = payload.get("evaluation_units", payload.get("units"))
    if not isinstance(units_payload, list) or not units_payload:
        raise WorldTreeBenchmarkError(
            f"Expected a non-empty 'evaluation_units' list in {metadata_path}"
        )

    top_reference = payload.get("reference_camera", payload.get("reference_camera_label"))
    if top_reference is None:
        top_reference = units_payload[0].get(
            "reference_camera", units_payload[0].get("source_camera")
        )
    reference_camera = camera_label(str(top_reference))

    units: list[EvaluationUnit] = []
    seen_ids: set[str] = set()
    for item in units_payload:
        if not isinstance(item, dict):
            raise WorldTreeBenchmarkError(f"Invalid evaluation-unit payload in {metadata_path}")
        source_camera = camera_label(str(item.get("source_camera", item.get("source_cam"))))
        target_camera = camera_label(str(item.get("target_camera", item.get("target_cam"))))
        unit_reference = camera_label(str(item.get("reference_camera", reference_camera)))
        unit_id = str(item.get("unit_id", f"{source_camera}_to_{target_camera}")).strip()
        if unit_id == "":
            raise WorldTreeBenchmarkError(
                f"Evaluation unit in {metadata_path} is missing unit_id."
            )
        if unit_id in seen_ids:
            raise WorldTreeBenchmarkError(
                f"Duplicate evaluation unit id '{unit_id}' in {metadata_path}."
            )
        seen_ids.add(unit_id)
        units.append(
            EvaluationUnit(
                unit_id=unit_id,
                source_camera=source_camera,
                target_camera=target_camera,
                reference_camera=unit_reference,
            )
        )
    return reference_camera, tuple(units)


def _check_scene_modalities(
    scene_root: Path,
    camera_manifests: dict[str, SceneCameraManifest],
    available_optional_modalities: set[str],
) -> None:
    for modality in REQUIRED_MODALITIES:
        for cam_name, manifest in camera_manifests.items():
            camera_dir = _expect_modality_dir(scene_root, modality, cam_name)
            expected_frames = manifest.valid_frame_names if modality == "gt" else manifest.frame_names
            missing = _missing_frames(camera_dir, expected_frames, ".png")
            if missing:
                preview = ", ".join(missing[:5])
                raise WorldTreeBenchmarkError(
                    f"Missing {modality} frames under {camera_dir}: {preview}"
                )

    if "scene_flow_rgb" in available_optional_modalities:
        for cam_name, manifest in camera_manifests.items():
            flow_dir = _expect_modality_dir(scene_root, "scene_flow_rgb", cam_name)
            missing = _missing_frames(flow_dir, manifest.frame_names, ".png")
            if missing:
                preview = ", ".join(missing[:5])
                raise WorldTreeBenchmarkError(
                    f"Missing scene_flow_rgb frames under {flow_dir}: {preview}"
                )

    if "scene_coords" in available_optional_modalities:
        for cam_name, manifest in camera_manifests.items():
            coord_dir = _expect_modality_dir(scene_root, "scene_coords", cam_name)
            missing = _missing_frames(coord_dir, manifest.frame_names, ".npz")
            if missing:
                preview = ", ".join(missing[:5])
                raise WorldTreeBenchmarkError(
                    f"Missing scene_coords tensors under {coord_dir}: {preview}"
                )

    if "scene_flow" in available_optional_modalities:
        for cam_name, manifest in camera_manifests.items():
            flow_dir = _expect_modality_dir(scene_root, "scene_flow", cam_name)
            missing = _missing_frames(flow_dir, manifest.frame_names, ".npz")
            if missing:
                preview = ", ".join(missing[:5])
                raise WorldTreeBenchmarkError(
                    f"Missing scene_flow tensors under {flow_dir}: {preview}"
                )

    if "test_dynamic" in available_optional_modalities:
        for cam_name, manifest in camera_manifests.items():
            dynamic_dir = _expect_modality_dir(scene_root, "test_dynamic", cam_name)
            expected_frames = tuple(
                record.frame_name for record in manifest.records if record.has_test_dynamic
            )
            missing = _missing_frames(dynamic_dir, expected_frames, ".png")
            if missing:
                preview = ", ".join(missing[:5])
                raise WorldTreeBenchmarkError(
                    f"Missing test_dynamic masks under {dynamic_dir}: {preview}"
                )


def _validate_scene_evaluation_units(scene_layout: SceneLayout) -> None:
    camera_manifests = scene_layout.camera_manifests
    for unit in scene_layout.evaluation_units:
        if unit.reference_camera != scene_layout.reference_camera:
            raise WorldTreeBenchmarkError(
                f"Evaluation unit {unit.unit_id} uses reference camera {unit.reference_camera}, "
                f"expected {scene_layout.reference_camera}."
            )
        if unit.source_camera not in camera_manifests:
            raise WorldTreeBenchmarkError(
                f"Evaluation unit {unit.unit_id} references missing source camera {unit.source_camera}."
            )
        if unit.target_camera not in camera_manifests:
            raise WorldTreeBenchmarkError(
                f"Evaluation unit {unit.unit_id} references missing target camera {unit.target_camera}."
            )
        source_manifest = camera_manifests[unit.source_camera]
        target_manifest = camera_manifests[unit.target_camera]
        if target_manifest.reference_camera != unit.reference_camera:
            raise WorldTreeBenchmarkError(
                f"Target camera {unit.target_camera} declares reference camera "
                f"{target_manifest.reference_camera}, expected {unit.reference_camera}."
            )
        missing_source_refs = [
            name
            for name in (record.source_input_frame_name for record in target_manifest.records)
            if name not in set(source_manifest.frame_names)
        ]
        if missing_source_refs:
            preview = ", ".join(missing_source_refs[:5])
            raise WorldTreeBenchmarkError(
                f"Evaluation unit {unit.unit_id} references source frames absent from "
                f"{source_manifest.metadata_path}: {preview}"
            )
        if len(source_manifest.records) != len(target_manifest.records):
            raise WorldTreeBenchmarkError(
                f"Evaluation unit {unit.unit_id} requires aligned source/target timelines, "
                f"but {unit.source_camera} has {len(source_manifest.records)} frames and "
                f"{unit.target_camera} has {len(target_manifest.records)}."
            )
        if source_manifest.time_ids != target_manifest.time_ids:
            raise WorldTreeBenchmarkError(
                f"Evaluation unit {unit.unit_id} requires matching time ids between "
                f"{unit.source_camera} and {unit.target_camera}."
            )


def validate_scene_layout(
    scene_root: str | Path,
    *,
    strict_files: bool = True,
    required_optional_modalities: Sequence[str] | None = None,
) -> SceneLayout:
    scene_root = Path(scene_root).expanduser().resolve()
    if not scene_root.is_dir():
        raise WorldTreeBenchmarkError(f"Scene root is not a directory: {scene_root}")

    for required_dir in ("cameras", "metadata", "frame_names"):
        if not (scene_root / required_dir).is_dir():
            raise WorldTreeBenchmarkError(f"Missing required directory: {scene_root / required_dir}")

    reference_camera, evaluation_units = _load_evaluation_units(scene_root)
    camera_names = sorted(
        {
            reference_camera,
            *[unit.source_camera for unit in evaluation_units],
            *[unit.target_camera for unit in evaluation_units],
        }
    )
    camera_manifests = {
        cam_name: _load_scene_camera_manifest(scene_root, cam_name) for cam_name in camera_names
    }
    for cam_name, manifest in camera_manifests.items():
        _ensure_camera_npz(
            scene_root / "cameras" / f"{cam_name}_c2w.npz",
            expected_length=len(manifest.records),
            camera_name=cam_name,
        )

    missing_required_modalities = [
        modality for modality in REQUIRED_MODALITIES if not (scene_root / modality).is_dir()
    ]
    if missing_required_modalities:
        all_missing = len(missing_required_modalities) == len(REQUIRED_MODALITIES)
        if all_missing:
            raise WorldTreeBenchmarkError(
                f"Scene root looks metadata-only: {scene_root}. "
                "Found metadata/frame_names/cameras but none of the required renderings "
                f"({', '.join(REQUIRED_MODALITIES)})."
            )
        raise WorldTreeBenchmarkError(
            f"Missing required modality directories under {scene_root}: "
            f"{', '.join(missing_required_modalities)}"
        )

    available_optional_modalities = {
        modality for modality in OPTIONAL_MODALITIES if (scene_root / modality).is_dir()
    }
    required_optional_modalities = set(required_optional_modalities or ())
    missing_optional = sorted(required_optional_modalities.difference(available_optional_modalities))
    if missing_optional:
        raise WorldTreeBenchmarkError(
            f"Scene {scene_root.name} is missing required optional modalities: "
            f"{', '.join(missing_optional)}"
        )

    if strict_files:
        _check_scene_modalities(
            scene_root=scene_root,
            camera_manifests=camera_manifests,
            available_optional_modalities=available_optional_modalities,
        )

    scene_layout = SceneLayout(
        scene_name=scene_root.name,
        scene_root=scene_root,
        reference_camera=reference_camera,
        camera_manifests=camera_manifests,
        evaluation_units=evaluation_units,
        available_optional_modalities=tuple(sorted(available_optional_modalities)),
    )
    _validate_scene_evaluation_units(scene_layout)
    return scene_layout


def discover_scene_layouts(
    benchmark_root: str | Path,
    *,
    scene_names: Sequence[str] | None = None,
    strict_files: bool = True,
    required_optional_modalities: Sequence[str] | None = None,
) -> list[SceneLayout]:
    benchmark_root = Path(benchmark_root).expanduser().resolve()
    if not benchmark_root.is_dir():
        raise WorldTreeBenchmarkError(f"Benchmark root is not a directory: {benchmark_root}")

    requested_scenes = {str(name).strip() for name in scene_names or () if str(name).strip()}
    layouts: list[SceneLayout] = []
    for scene_root in sorted(benchmark_root.iterdir()):
        if not scene_root.is_dir():
            continue
        if requested_scenes and scene_root.name not in requested_scenes:
            continue
        metadata_dir = scene_root / "metadata"
        if not metadata_dir.is_dir():
            continue
        if not (metadata_dir / SCENE_EVAL_UNITS_FILENAME).is_file():
            continue
        layouts.append(
            validate_scene_layout(
                scene_root,
                strict_files=strict_files,
                required_optional_modalities=required_optional_modalities,
            )
        )
    if requested_scenes:
        found_scenes = {layout.scene_name for layout in layouts}
        missing_scenes = sorted(requested_scenes.difference(found_scenes))
        if missing_scenes:
            raise WorldTreeBenchmarkError(
                f"Requested scenes not found under {benchmark_root}: {', '.join(missing_scenes)}"
            )
    return layouts
