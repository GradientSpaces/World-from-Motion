from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from gen4dgs.formats.worldtree_benchmark import camera_label, strip_camera_prefix

from .export_worldtree_layout import (
    EvaluationUnitPlan,
    ExportFrameRecord,
    SceneExportPlan,
)
from .stationary_pose_sampler import (
    StationaryPose,
    select_stationary_anchors,
    stationary_pose_from_payload,
    stationary_pose_to_payload,
)
from .trajectory_stationary_adapter import default_target_camera_sides
from .worldtree_render_bridge import (
    DEFAULT_WORLDTREE_ROOT,
    WorldTreeRenderConfig,
    WorldTreeStationaryRenderer,
)


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


class TrajectoryFullseqStationaryAdapter:
    """Export full-sequence source and stationary targets without a template pair."""

    DEFAULT_TARGET_CAMERA_SIDES = default_target_camera_sides()

    def __init__(
        self,
        *,
        scene_root: str | Path,
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
        worldtree_process_height: int | None = None,
        worldtree_fit_cfg_path: str | Path | None = None,
        worldtree_use_fachain: bool | None = None,
        worldtree_bg_color: Sequence[float] = (1.0, 1.0, 1.0),
        worldtree_alpha_threshold: float = 1e-3,
        reference_rgb_source: str = "gt",
    ) -> None:
        self.scene_root = Path(scene_root).expanduser().resolve()
        self.scene_name = str(scene_name)
        self.source_cam = camera_label(source_cam)
        self.clip_start = max(0, int(clip_start))
        self.selection_axis = str(selection_axis)
        explicit_num_anchors = None if num_anchors is None else int(num_anchors)
        if explicit_num_anchors is not None and explicit_num_anchors <= 0:
            raise ValueError(f"num_anchors must be positive, got {explicit_num_anchors}")
        if not self.scene_root.is_dir():
            raise FileNotFoundError(f"Scene root not found: {self.scene_root}")
        if worldtree_saved_dir is None:
            raise ValueError("worldtree_saved_dir is required for full-sequence stationary export.")
        self.worldtree_saved_dir = Path(worldtree_saved_dir).expanduser().resolve()
        if not self.worldtree_saved_dir.is_dir():
            raise FileNotFoundError(f"WorldTree saved_dir not found: {self.worldtree_saved_dir}")
        self.reference_rgb_source = str(reference_rgb_source).strip().lower()
        if self.reference_rgb_source not in {"gt", "render"}:
            raise ValueError(
                "reference_rgb_source must be either 'gt' or 'render', "
                f"got {reference_rgb_source!r}."
            )

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

        self.worldtree_renderer = WorldTreeStationaryRenderer(
            WorldTreeRenderConfig(
                worldtree_root=Path(worldtree_root),
                saved_dir=self.worldtree_saved_dir,
                scene_root=self.scene_root,
                source_cam=self.source_cam,
                device=worldtree_device,
                flow_step=worldtree_flow_step,
                scene_space=worldtree_scene_space,
                process_height=worldtree_process_height,
                fit_cfg_path=(
                    None
                    if worldtree_fit_cfg_path is None
                    else Path(worldtree_fit_cfg_path)
                ),
                use_fachain=worldtree_use_fachain,
                bg_color=tuple(float(value) for value in worldtree_bg_color),
                alpha_threshold=float(worldtree_alpha_threshold),
            )
        )
        inferred_width, inferred_height = self.worldtree_renderer.source_img_wh
        self.width = inferred_width if width is None else int(width)
        self.height = inferred_height if height is None else int(height)
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
        source_records = tuple(
            self.worldtree_renderer.build_source_records(
                clip_start=self.clip_start,
                max_frames=max_frames,
                reference_rgb_source=self.reference_rgb_source,
            )
        )
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
                **self._worldtree_render_summary_fields(),
            }
        }
        camera_scene_flow_rgb_hi: dict[str, Sequence[float] | None] = {
            self.source_cam: None,
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
                **self._worldtree_render_summary_fields(),
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
            "adapter": "trajectory_fullseq_stationary_adapter",
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
            "reference_rgb_source": self.reference_rgb_source,
            **self._worldtree_render_summary_fields(),
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

    def _worldtree_render_summary_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if self.worldtree_renderer.process_height is not None:
            fields["process_height"] = int(self.worldtree_renderer.process_height)
        if self.worldtree_renderer.min_interval is not None:
            fields["min_interval"] = int(self.worldtree_renderer.min_interval)
        if self.worldtree_renderer.use_fachain is not None:
            fields["use_fachain"] = bool(self.worldtree_renderer.use_fachain)
        if self.worldtree_renderer.fachain_ref_time_mask is not None:
            fields["fachain_ref_time_mask"] = bool(
                self.worldtree_renderer.fachain_ref_time_mask
            )
        if self.worldtree_renderer.extend_length is not None:
            fields["extend_length"] = int(self.worldtree_renderer.extend_length)
        if self.worldtree_renderer.fit_cfg_path is not None:
            fields["worldtree_fit_cfg_path"] = str(self.worldtree_renderer.fit_cfg_path)
        return fields

    def render_camera_frames(
        self,
        camera_name: str,
        records: Sequence[ExportFrameRecord],
        outputs: Sequence[str],
    ):
        camera_name = camera_label(camera_name)
        if camera_name == self.source_cam:
            yield from self.worldtree_renderer.render_source_frames(
                records=records,
                outputs=outputs,
                reference_rgb_source=self.reference_rgb_source,
            )
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


def build_trajectory_fullseq_stationary_adapter(**kwargs: Any) -> TrajectoryFullseqStationaryAdapter:
    return TrajectoryFullseqStationaryAdapter(**kwargs)
