from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v3 as iio
import numpy as np

from gen4dgs.formats.worldtree_benchmark import (
    PREFERRED_NPZ_KEYS,
    camera_label,
    strip_camera_prefix,
)

from .export_worldtree_layout import (
    EvaluationUnitPlan,
    ExportFrameRecord,
    RenderedFrame,
    SceneExportPlan,
)
from .stationary_pose_sampler import (
    StationaryPose,
    select_stationary_anchors,
    stationary_pose_from_payload,
    stationary_pose_to_payload,
)
from .worldtree_render_bridge import (
    DEFAULT_WORLDTREE_ROOT,
    WorldTreeRenderConfig,
    WorldTreeStationaryRenderer,
    resolve_worldtree_saved_dir,
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _infer_image_size(pair_root: Path, source_cam: str) -> tuple[int, int]:
    first_frame = sorted((pair_root / "rgb" / source_cam).glob("*.png"))
    if not first_frame:
        raise FileNotFoundError(f"No source RGB frames found under {pair_root / 'rgb' / source_cam}")
    frame = np.asarray(iio.imread(first_frame[0]))
    if frame.ndim == 2:
        height, width = frame.shape
    else:
        height, width = frame.shape[:2]
    return int(width), int(height)


def _load_npz_array(path: Path, kind: str) -> np.ndarray:
    preferred_keys = PREFERRED_NPZ_KEYS[kind]
    with np.load(path) as data:
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])
        if len(data.files) == 0:
            raise ValueError(f"Empty npz file: {path}")
        return np.asarray(data[data.files[0]])


def _anchor_index_from_label(label: str | None) -> int | None:
    if label is None:
        return None
    normalized = str(label).strip().lower().replace("_", "")
    if normalized in {"left", "first", "anchor0", "0"}:
        return 0
    if normalized in {"right", "second", "anchor1", "1"}:
        return 1
    if normalized.startswith("anchor") and normalized[6:].isdigit():
        return int(normalized[6:])
    return None


def default_target_camera_sides(
    num_anchors: int = 2,
    *,
    start_camera_id: int = 100,
) -> dict[str, str]:
    if num_anchors <= 0:
        raise ValueError(f"num_anchors must be positive, got {num_anchors}")
    return {
        camera_label(f"cam{start_camera_id + rank}"): f"anchor{rank}"
        for rank in range(int(num_anchors))
    }


class TrajectoryStationaryAdapter:
    """Export stationary target cameras chosen from the full cam0 trajectory.

    The source branch reuses real modality exports from an existing WorldTree pair.
    The target branch now calls WorldTree's real renderer with a fixed 6DoF pose
    chosen by ordered greedy anchor selection on the solved reference trajectory.
    Legacy left/right labels are preserved as compatibility aliases for the
    first two anchor ranks.
    """

    DEFAULT_TARGET_CAMERA_SIDES = default_target_camera_sides()

    def __init__(
        self,
        *,
        scene_root: str | Path,
        template_pair_root: str | Path,
        scene_name: str = "block",
        source_cam: str = "cam0",
        clip_start: int = 0,
        selection_axis: str = "world_x",
        num_anchors: int | None = None,
        width: int | None = None,
        height: int | None = None,
        target_camera_sides: Mapping[str, str] | None = None,
        fixed_target_poses: Mapping[str, StationaryPose | Mapping[str, Any]] | None = None,
        worldtree_root: str | Path = DEFAULT_WORLDTREE_ROOT,
        worldtree_saved_dir: str | Path | None = None,
        worldtree_device: str | None = None,
        worldtree_flow_step: int = 1,
        worldtree_scene_space: str = "world",
        worldtree_bg_color: Sequence[float] = (1.0, 1.0, 1.0),
        worldtree_alpha_threshold: float = 1e-3,
    ) -> None:
        self.scene_root = Path(scene_root).expanduser().resolve()
        self.template_pair_root = Path(template_pair_root).expanduser().resolve()
        self.scene_name = str(scene_name)
        self.source_cam = camera_label(source_cam)
        self.clip_start = max(0, int(clip_start))
        self.selection_axis = str(selection_axis)
        explicit_num_anchors = None if num_anchors is None else int(num_anchors)
        if explicit_num_anchors is not None and explicit_num_anchors <= 0:
            raise ValueError(f"num_anchors must be positive, got {explicit_num_anchors}")
        if not self.template_pair_root.is_dir():
            raise FileNotFoundError(f"Template pair root not found: {self.template_pair_root}")

        self.source_payload = _load_json(
            self.template_pair_root / "metadata" / f"{self.source_cam}_frames.json"
        )
        self.source_c2w = _load_npz_array(
            self.template_pair_root / "cameras" / f"{self.source_cam}_c2w.npz",
            kind="camera",
        ).astype(np.float32)
        resolved_target_camera_sides = (
            default_target_camera_sides(explicit_num_anchors or 2)
            if target_camera_sides is None
            else target_camera_sides
        )
        self.target_camera_sides = {
            camera_label(name): str(side).strip().lower()
            for name, side in resolved_target_camera_sides.items()
        }
        required_anchor_count = 0
        for side in self.target_camera_sides.values():
            anchor_index = _anchor_index_from_label(side)
            if anchor_index is None:
                raise ValueError(
                    f"Unsupported stationary anchor label {side!r}; expected anchorN, left, or right."
                )
            required_anchor_count = max(required_anchor_count, int(anchor_index) + 1)
        if explicit_num_anchors is not None and explicit_num_anchors < required_anchor_count:
            raise ValueError(
                f"num_anchors={explicit_num_anchors} is smaller than the configured "
                f"target anchor requirement {required_anchor_count}."
            )
        self.num_anchors = required_anchor_count if explicit_num_anchors is None else explicit_num_anchors

        inferred_width, inferred_height = _infer_image_size(self.template_pair_root, self.source_cam)
        self.width = inferred_width if width is None else int(width)
        self.height = inferred_height if height is None else int(height)

        self.worldtree_saved_dir = resolve_worldtree_saved_dir(
            template_pair_root=self.template_pair_root,
            saved_dir=worldtree_saved_dir,
        )
        self.worldtree_renderer = WorldTreeStationaryRenderer(
            WorldTreeRenderConfig(
                worldtree_root=Path(worldtree_root),
                saved_dir=self.worldtree_saved_dir,
                scene_root=self.scene_root,
                source_cam=self.source_cam,
                device=worldtree_device,
                flow_step=worldtree_flow_step,
                scene_space=worldtree_scene_space,
                bg_color=tuple(float(value) for value in worldtree_bg_color),
                alpha_threshold=float(worldtree_alpha_threshold),
            )
        )
        self.trajectory = self.worldtree_renderer.load_source_camera_trajectory()
        self.stationary_selection = select_stationary_anchors(
            self.trajectory,
            axis_name=self.selection_axis,
            num_anchors=self.num_anchors,
        )
        self.fixed_target_poses = self._normalize_fixed_target_poses(fixed_target_poses)

    def build_scene_plan(
        self,
        reference_camera: str,
        *,
        target_cameras: Sequence[str] | None = None,
        max_frames: int | None = None,
    ) -> SceneExportPlan:
        reference_camera = camera_label(reference_camera)
        if reference_camera != self.source_cam:
            raise ValueError(
                f"Stationary adapter is bound to reference camera {self.source_cam}, got {reference_camera}"
            )

        selected_targets = self._selected_target_cameras(target_cameras)
        source_records = tuple(self._build_source_records(max_frames=max_frames))
        camera_records: dict[str, tuple[ExportFrameRecord, ...]] = {
            self.source_cam: source_records,
        }
        camera_pose_stats: dict[str, Mapping[str, Any] | None] = {
            self.source_cam: {
                "camera_id": strip_camera_prefix(self.source_cam),
                "clip_start": self.clip_start,
                "clip_num_frames": len(source_records),
                "sampling_num_frames": self.trajectory.num_frames,
                "trajectory_pose_source_kind": self.trajectory.pose_source_kind,
                "trajectory_pose_source_path": self.trajectory.pose_source_path,
            }
        }
        camera_scene_flow_rgb_hi: dict[str, Sequence[float] | None] = {
            self.source_cam: self._scene_flow_hi(self.source_payload),
        }
        evaluation_units: list[EvaluationUnitPlan] = []
        target_summaries: list[dict[str, Any]] = []

        for target_cam in selected_targets:
            selected_pose = self._selected_target_pose(target_cam)
            pose_origin = self.selected_target_pose_origin(target_cam)
            target_records = tuple(
                self._build_target_records(
                    target_cam=target_cam,
                    source_records=source_records,
                    selected_pose=selected_pose,
                )
            )
            camera_records[target_cam] = target_records
            camera_pose_stats[target_cam] = {
                "camera_id": strip_camera_prefix(target_cam),
                "selection_side": selected_pose.side,
                "selection_anchor_label": selected_pose.side,
                "selection_anchor_rank": int(selected_pose.selection_rank),
                "selection_strategy": str(selected_pose.selection_strategy),
                "selection_axis": selected_pose.axis_name,
                "selected_full_sequence_index": selected_pose.selected_index,
                "selected_full_sequence_time_id": selected_pose.selected_time_id,
                "selected_full_sequence_frame_name": selected_pose.selected_frame_name,
                "selected_axis_value": selected_pose.axis_value,
                "selected_center_world": selected_pose.center.tolist(),
                "selection_score": (
                    None
                    if selected_pose.selection_score is None
                    else float(selected_pose.selection_score)
                ),
                "clip_start": self.clip_start,
                "clip_num_frames": len(target_records),
                "worldtree_saved_dir": str(self.worldtree_saved_dir),
                "reference_pose_source_kind": pose_origin["pose_source_kind"],
                "reference_pose_source_path": pose_origin["pose_source_path"],
                "fixed_pose_override": bool(pose_origin["fixed_pose_override"]),
            }
            camera_scene_flow_rgb_hi[target_cam] = None
            evaluation_units.append(
                EvaluationUnitPlan(
                    unit_id=f"{self.source_cam}_to_{target_cam}",
                    source_camera=self.source_cam,
                    target_camera=target_cam,
                    reference_camera=self.source_cam,
                )
            )
            target_summaries.append(
                {
                    "target_camera": target_cam,
                    "selection_side": selected_pose.side,
                    "selection_anchor_label": selected_pose.side,
                    "selection_anchor_rank": int(selected_pose.selection_rank),
                    "selection_strategy": str(selected_pose.selection_strategy),
                    "selected_full_sequence_index": selected_pose.selected_index,
                    "selected_full_sequence_time_id": selected_pose.selected_time_id,
                    "selected_full_sequence_frame_name": selected_pose.selected_frame_name,
                    "selected_axis_value": selected_pose.axis_value,
                    "selected_center_world": selected_pose.center.tolist(),
                }
            )

        scene_summary_extra = {
            "adapter": "trajectory_stationary_adapter",
            "selection_strategy": self.stationary_selection.selection_strategy,
            "selection_axis": self.selection_axis,
            "configured_num_anchors": self.num_anchors,
            "selection_anchor_count": len(self.stationary_selection.anchors),
            "clip_start": self.clip_start,
            "clip_num_frames": len(source_records),
            "sampling_camera": self.source_cam,
            "sampling_num_frames": self.trajectory.num_frames,
            "reference_pose_source_kind": self.trajectory.pose_source_kind,
            "reference_pose_source_path": self.trajectory.pose_source_path,
            "worldtree_saved_dir": str(self.worldtree_saved_dir),
            "worldtree_workspace": str(self.worldtree_renderer.workspace_path),
            "target_cameras": list(selected_targets),
            "fixed_target_poses": {
                target_cam: stationary_pose_to_payload(
                    self._selected_target_pose(target_cam),
                    pose_source_kind=self.selected_target_pose_origin(target_cam)["pose_source_kind"],
                    pose_source_path=self.selected_target_pose_origin(target_cam)["pose_source_path"],
                )
                for target_cam in selected_targets
            },
            "stationary_targets": target_summaries,
        }
        return SceneExportPlan(
            scene_name=self.scene_name,
            reference_camera=self.source_cam,
            camera_records=camera_records,
            evaluation_units=tuple(evaluation_units),
            camera_pose_stats=camera_pose_stats,
            camera_scene_flow_rgb_hi=camera_scene_flow_rgb_hi,
            scene_summary_extra=scene_summary_extra,
        )

    def render_camera_frames(
        self,
        camera_name: str,
        records: Sequence[ExportFrameRecord],
        outputs: Sequence[str],
    ):
        camera_name = camera_label(camera_name)
        outputs_set = set(outputs)
        if camera_name == self.source_cam:
            yield from self._render_source_frames(records=records, outputs_set=outputs_set)
            return

        selected_pose = self._selected_target_pose(camera_name)
        yield from self.worldtree_renderer.render_stationary_frames(
            target_cam=camera_name,
            records=records,
            selected_pose=selected_pose,
            outputs=outputs,
            target_img_wh=(self.width, self.height),
        )

    def selected_target_pose(self, target_cam: str) -> StationaryPose:
        return self._selected_target_pose(target_cam)

    def selected_target_pose_origin(self, target_cam: str) -> dict[str, Any]:
        target_cam = camera_label(target_cam)
        if target_cam in self.fixed_target_poses:
            fixed_payload = self.fixed_target_poses[target_cam]
            return {
                "pose_source_kind": str(
                    fixed_payload.get("pose_source_kind", self.trajectory.pose_source_kind)
                ),
                "pose_source_path": (
                    None
                    if fixed_payload.get("pose_source_path") is None
                    else str(fixed_payload["pose_source_path"])
                ),
                "fixed_pose_override": True,
            }
        return {
            "pose_source_kind": self.trajectory.pose_source_kind,
            "pose_source_path": self.trajectory.pose_source_path,
            "fixed_pose_override": False,
        }

    def _build_source_records(self, *, max_frames: int | None) -> list[ExportFrameRecord]:
        frames = list(self.source_payload.get("frames", []))
        frames = frames[self.clip_start:]
        if max_frames is not None:
            frames = frames[: int(max_frames)]
        return [self._record_from_payload(frame) for frame in frames]

    def _build_target_records(
        self,
        *,
        target_cam: str,
        source_records: Sequence[ExportFrameRecord],
        selected_pose: StationaryPose,
    ) -> list[ExportFrameRecord]:
        target_prefix = strip_camera_prefix(target_cam)
        records = []
        for source_record in source_records:
            suffix = source_record.frame_name.split("_", 1)[1]
            target_frame_name = f"{target_prefix}_{suffix}"
            records.append(
                ExportFrameRecord(
                    frame_idx=int(source_record.frame_idx),
                    time_id=int(source_record.time_id),
                    frame_name=target_frame_name,
                    has_gt=False,
                    has_test_dynamic=False,
                    has_dataset_frame_at_time=False,
                    has_exact_tto_pose=True,
                    render_pose_source_kind=(
                        "stationary_"
                        f"{selected_pose.selection_strategy}_anchor{selected_pose.selection_rank}"
                    ),
                    render_pose_source_time_id=int(selected_pose.selected_time_id),
                    render_pose_source_frame_name=selected_pose.selected_frame_name,
                    render_pose_archive_path=str(self.worldtree_saved_dir),
                    rgb_source_kind="render_stationary",
                    scene_flow_valid=True,
                    reference_frame_name=source_record.frame_name,
                    source_frame_name=source_record.frame_name,
                )
            )
        return records

    def _selected_target_pose(self, target_cam: str) -> StationaryPose:
        target_cam = camera_label(target_cam)
        fixed_payload = self.fixed_target_poses.get(target_cam)
        if fixed_payload is not None:
            return stationary_pose_from_payload(dict(fixed_payload))
        side = self.target_camera_sides.get(target_cam)
        anchor_index = _anchor_index_from_label(side)
        if anchor_index is not None and anchor_index < len(self.stationary_selection.anchors):
            return self.stationary_selection.anchor(anchor_index)
        raise KeyError(
            f"No stationary anchor configured for target camera {target_cam}. "
            f"Known targets: {sorted(self.target_camera_sides)}"
        )

    def _selected_target_cameras(self, target_cameras: Sequence[str] | None) -> tuple[str, ...]:
        if target_cameras is None:
            return tuple(self.target_camera_sides)
        selected = []
        seen = set()
        for target_cam in target_cameras:
            normalized = camera_label(target_cam)
            if normalized in seen:
                continue
            if normalized not in self.target_camera_sides:
                raise KeyError(
                    f"No stationary anchor configured for target camera {normalized}. "
                    f"Known targets: {sorted(self.target_camera_sides)}"
                )
            selected.append(normalized)
            seen.add(normalized)
        if not selected:
            raise ValueError("At least one target camera must be selected.")
        return tuple(selected)

    def _record_from_payload(self, frame: dict[str, Any]) -> ExportFrameRecord:
        return ExportFrameRecord(
            frame_idx=int(frame["frame_idx"]),
            time_id=int(frame.get("time_id", frame["frame_idx"])),
            frame_name=str(frame["frame_name"]),
            has_gt=bool(frame.get("has_gt", False)),
            has_test_dynamic=bool(frame.get("has_test_dynamic", False)),
            has_dataset_frame_at_time=bool(frame.get("has_dataset_frame_at_time", False)),
            has_exact_tto_pose=bool(frame.get("has_exact_tto_pose", False)),
            render_pose_source_kind=str(frame.get("render_pose_source_kind", "source")),
            render_pose_source_time_id=(
                None
                if frame.get("render_pose_source_time_id") is None
                else int(frame["render_pose_source_time_id"])
            ),
            render_pose_source_frame_name=(
                None
                if frame.get("render_pose_source_frame_name") is None
                else str(frame["render_pose_source_frame_name"])
            ),
            render_pose_archive_path=(
                None
                if frame.get("render_pose_archive_path") is None
                else str(frame["render_pose_archive_path"])
            ),
            rgb_source_kind=str(frame.get("rgb_source_kind", "source")),
            scene_flow_valid=bool(frame.get("scene_flow_valid", True)),
            reference_frame_name=(
                None if frame.get("reference_frame_name") is None else str(frame["reference_frame_name"])
            ),
            source_frame_name=(
                None if frame.get("source_frame_name") is None else str(frame["source_frame_name"])
            ),
        )

    def _render_source_frames(
        self,
        *,
        records: Sequence[ExportFrameRecord],
        outputs_set: set[str],
    ):
        for record in records:
            images: dict[str, np.ndarray] = {}
            tensors: dict[str, np.ndarray] = {}
            for modality in outputs_set:
                if modality in {"scene_coords", "scene_flow"}:
                    tensors[modality] = self._load_template_tensor(
                        modality=modality,
                        camera_name=self.source_cam,
                        frame_name=record.frame_name,
                    )
                elif modality == "test_dynamic":
                    if record.has_test_dynamic:
                        images[modality] = self._load_template_image(
                            modality=modality,
                            camera_name=self.source_cam,
                            frame_name=record.frame_name,
                        )
                else:
                    if modality == "gt" and not record.has_gt:
                        continue
                    images[modality] = self._load_template_image(
                        modality=modality,
                        camera_name=self.source_cam,
                        frame_name=record.frame_name,
                    )
            yield RenderedFrame(
                frame_name=record.frame_name,
                c2w=np.asarray(self.source_c2w[record.frame_idx], dtype=np.float32),
                images=images,
                tensors=tensors,
            )

    def _load_template_image(self, *, modality: str, camera_name: str, frame_name: str) -> np.ndarray:
        path = self.template_pair_root / modality / camera_name / f"{frame_name}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing template image: {path}")
        return np.asarray(iio.imread(path))

    def _load_template_tensor(self, *, modality: str, camera_name: str, frame_name: str) -> np.ndarray:
        kind = "scene_coords" if modality == "scene_coords" else "scene_flow"
        path = self.template_pair_root / modality / camera_name / f"{frame_name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing template tensor: {path}")
        return _load_npz_array(path, kind=kind)

    def _scene_flow_hi(self, payload: dict[str, Any]) -> tuple[float, ...] | None:
        values = payload.get("scene_flow_rgb_hi")
        if values is None:
            return None
        return tuple(float(value) for value in values)

    def _normalize_fixed_target_poses(
        self,
        fixed_target_poses: Mapping[str, StationaryPose | Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if fixed_target_poses is None:
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for target_cam, payload in fixed_target_poses.items():
            normalized_cam = camera_label(target_cam)
            if isinstance(payload, StationaryPose):
                payload_dict = stationary_pose_to_payload(
                    payload,
                    pose_source_kind=self.trajectory.pose_source_kind,
                    pose_source_path=self.trajectory.pose_source_path,
                )
            else:
                payload_dict = dict(payload)
                pose = stationary_pose_from_payload(payload_dict)
                payload_dict = stationary_pose_to_payload(
                    pose,
                    pose_source_kind=str(
                        payload_dict.get("pose_source_kind", self.trajectory.pose_source_kind)
                    ),
                    pose_source_path=(
                        None
                        if payload_dict.get("pose_source_path") is None
                        else str(payload_dict["pose_source_path"])
                    ),
                )
            normalized[normalized_cam] = payload_dict
        return normalized


def build_trajectory_stationary_adapter(**kwargs: Any) -> TrajectoryStationaryAdapter:
    return TrajectoryStationaryAdapter(**kwargs)
