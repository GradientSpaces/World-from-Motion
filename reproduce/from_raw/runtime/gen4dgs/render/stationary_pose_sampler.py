from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from gen4dgs.formats.worldtree_benchmark import camera_label, strip_camera_prefix

GREEDY_ANCHOR_SELECTION_STRATEGY = "greedy_mean_then_farthest_point"
_AXIS_TO_INDEX = {
    "world_x": 0,
    "world_y": 1,
    "world_z": 2,
}


@dataclass(frozen=True)
class TrajectoryPose:
    frame_name: str
    time_id: int
    c2w: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]


@dataclass(frozen=True)
class CameraTrajectory:
    scene_root: Path
    camera_name: str
    poses: tuple[TrajectoryPose, ...]
    pose_source_kind: str = "camera_json"
    pose_source_path: str | None = None

    @property
    def num_frames(self) -> int:
        return len(self.poses)

    @property
    def frame_names(self) -> tuple[str, ...]:
        return tuple(pose.frame_name for pose in self.poses)

    @property
    def time_ids(self) -> tuple[int, ...]:
        return tuple(pose.time_id for pose in self.poses)

    @property
    def c2w(self) -> np.ndarray:
        return np.stack([pose.c2w for pose in self.poses], axis=0).astype(np.float32)

    @property
    def centers(self) -> np.ndarray:
        return np.stack([pose.center for pose in self.poses], axis=0).astype(np.float32)


@dataclass(frozen=True)
class StationaryPose:
    side: str
    source_camera_name: str
    selected_index: int
    selected_time_id: int
    selected_frame_name: str
    c2w: np.ndarray
    center: np.ndarray
    axis_name: str
    axis_value: float
    selection_rank: int = 0
    selection_strategy: str = GREEDY_ANCHOR_SELECTION_STRATEGY
    selection_score: float | None = None


@dataclass(frozen=True)
class StationaryPoseSelection:
    trajectory: CameraTrajectory
    anchors: tuple[StationaryPose, ...]

    @property
    def left(self) -> StationaryPose:
        return self.anchor(0)

    @property
    def right(self) -> StationaryPose:
        return self.anchor(1)

    @property
    def selection_strategy(self) -> str:
        if not self.anchors:
            return GREEDY_ANCHOR_SELECTION_STRATEGY
        return str(self.anchors[0].selection_strategy)

    def anchor(self, rank: int) -> StationaryPose:
        if rank < 0 or rank >= len(self.anchors):
            raise IndexError(f"Requested anchor rank {rank}, but only {len(self.anchors)} anchors exist.")
        return self.anchors[int(rank)]


def _load_camera_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _camera_dir(scene_root: Path) -> Path:
    scene_root = scene_root.expanduser().resolve()
    cameras_dir = scene_root / "cameras"
    if cameras_dir.is_dir():
        return cameras_dir
    if scene_root.is_dir():
        return scene_root
    raise FileNotFoundError(f"Camera scene root does not exist: {scene_root}")


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    text = str(path.expanduser().resolve())
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:  # pragma: no cover - defensive
            pass


def _time_id_from_name(frame_name: str) -> int:
    prefix, separator, suffix = frame_name.partition("_")
    if separator == "" or not suffix.isdigit():
        raise ValueError(f"Unexpected camera frame name: {frame_name}")
    return int(suffix)


def load_camera_trajectory(
    scene_root: str | Path,
    *,
    camera_name: str = "cam0",
) -> CameraTrajectory:
    """Load a DyCheck / Nerfies-style camera trajectory.

    Matches the pose convention used in WorldTree's `load_iphone_cameras()`:
    position is world-space translation and orientation is transposed into a
    world-from-camera rotation.
    """

    scene_root = Path(scene_root).expanduser().resolve()
    cameras_dir = _camera_dir(scene_root)
    camera_name = camera_label(camera_name)
    camera_prefix = strip_camera_prefix(camera_name)

    json_paths = sorted(
        cameras_dir.glob(f"{camera_prefix}_*.json"),
        key=lambda path: _time_id_from_name(path.stem),
    )
    if not json_paths:
        raise FileNotFoundError(
            f"No camera JSON files found for {camera_name} under {cameras_dir}"
        )

    poses = []
    for path in json_paths:
        payload = _load_camera_json(path)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.asarray(payload["position"], dtype=np.float32)
        c2w[:3, :3] = np.asarray(payload["orientation"], dtype=np.float32).T
        poses.append(
            TrajectoryPose(
                frame_name=path.stem,
                time_id=_time_id_from_name(path.stem),
                c2w=c2w,
            )
        )
    return CameraTrajectory(
        scene_root=scene_root,
        camera_name=camera_name,
        poses=tuple(poses),
        pose_source_kind="camera_json",
        pose_source_path=str(cameras_dir),
    )


def load_photometric_camera_trajectory(
    scene_root: str | Path,
    *,
    saved_dir: str | Path,
    worldtree_root: str | Path,
    camera_name: str = "cam0",
    device: str | None = None,
) -> CameraTrajectory:
    """Load a solved camera trajectory from WorldTree's photometric checkpoint.

    Frame names and time ids come from the DyCheck camera JSONs under `scene_root`,
    while the actual poses are read from `photometric_cam.pth`.
    """

    reference_trajectory = load_camera_trajectory(scene_root, camera_name=camera_name)
    saved_dir = Path(saved_dir).expanduser().resolve()
    worldtree_root = Path(worldtree_root).expanduser().resolve()
    cam_path = saved_dir / "photometric_cam.pth"
    if not worldtree_root.is_dir():
        raise FileNotFoundError(f"WorldTree root not found: {worldtree_root}")
    if not saved_dir.is_dir():
        raise FileNotFoundError(f"WorldTree saved_dir not found: {saved_dir}")
    if not cam_path.is_file():
        raise FileNotFoundError(f"Missing camera checkpoint: {cam_path}")

    with _prepend_sys_path(worldtree_root):
        camera_module = importlib.import_module("lib_moca.camera")
    ckpt = torch.load(cam_path, map_location="cpu")
    cams = camera_module.MonocularCameras.load_from_ckpt(ckpt)
    torch_device = torch.device("cpu" if device is None or str(device).strip() == "" else str(device))
    cams = cams.to(torch_device)

    poses = []
    with torch.inference_mode():
        for pose in reference_trajectory.poses:
            c2w = torch.linalg.inv(cams.T_cw(int(pose.time_id))).detach().cpu().numpy()
            poses.append(
                TrajectoryPose(
                    frame_name=pose.frame_name,
                    time_id=pose.time_id,
                    c2w=np.asarray(c2w, dtype=np.float32),
                )
            )
    return CameraTrajectory(
        scene_root=reference_trajectory.scene_root,
        camera_name=reference_trajectory.camera_name,
        poses=tuple(poses),
        pose_source_kind="photometric_cam",
        pose_source_path=str(cam_path),
    )


def _stationary_pose(
    trajectory: CameraTrajectory,
    *,
    side: str,
    axis_name: str,
    index: int,
    selection_rank: int = 0,
    selection_strategy: str = GREEDY_ANCHOR_SELECTION_STRATEGY,
    selection_score: float | None = None,
) -> StationaryPose:
    pose = trajectory.poses[int(index)]
    center = np.asarray(pose.center, dtype=np.float32)
    axis_idx = _AXIS_TO_INDEX.get(str(axis_name).strip().lower(), 0)
    return StationaryPose(
        side=side,
        source_camera_name=trajectory.camera_name,
        selected_index=int(index),
        selected_time_id=int(pose.time_id),
        selected_frame_name=pose.frame_name,
        c2w=np.asarray(pose.c2w, dtype=np.float32),
        center=center,
        axis_name=axis_name,
        axis_value=float(center[axis_idx]),
        selection_rank=int(selection_rank),
        selection_strategy=str(selection_strategy),
        selection_score=(
            None if selection_score is None else float(selection_score)
        ),
    )


def _validate_trajectory_centers(trajectory: CameraTrajectory) -> np.ndarray:
    centers = trajectory.centers
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 camera centers, got {centers.shape} for {trajectory.camera_name}"
        )
    return np.asarray(centers, dtype=np.float32)


def _anchor_label(rank: int) -> str:
    return f"anchor{rank}"


def select_stationary_anchors(
    trajectory: CameraTrajectory,
    *,
    axis_name: str = "world_x",
    num_anchors: int = 2,
) -> StationaryPoseSelection:
    return select_stationary_extrema(
        trajectory,
        axis_name=axis_name,
        num_anchors=num_anchors,
    )


def select_stationary_extrema(
    trajectory: CameraTrajectory,
    *,
    axis_name: str = "world_x",
    num_anchors: int = 2,
    initial_anchors: Sequence[StationaryPose] | None = None,
) -> StationaryPoseSelection:
    if num_anchors <= 0:
        raise ValueError(f"Expected a positive number of anchors, got {num_anchors}")
    centers = _validate_trajectory_centers(trajectory)
    if len(trajectory.poses) < num_anchors:
        raise ValueError(
            f"Cannot select {num_anchors} anchors from {len(trajectory.poses)} poses."
        )
    seeded_anchors = tuple(initial_anchors or ())
    if len(seeded_anchors) > num_anchors:
        raise ValueError(
            f"Received {len(seeded_anchors)} seeded anchors but num_anchors={num_anchors}."
        )
    mean_center = np.mean(centers, axis=0, dtype=np.float32)
    selected_indices: list[int] = []
    selected_scores: list[float | None] = []
    for anchor in seeded_anchors:
        if camera_label(anchor.source_camera_name) != trajectory.camera_name:
            raise ValueError(
                "Seeded anchors must come from the same source camera as the trajectory."
            )
        if str(anchor.axis_name).strip().lower() != str(axis_name).strip().lower():
            raise ValueError(
                f"Seeded anchor axis {anchor.axis_name!r} does not match requested axis {axis_name!r}."
            )
        selected_index = int(anchor.selected_index)
        if selected_index < 0 or selected_index >= len(trajectory.poses):
            raise ValueError(
                f"Seeded anchor index {selected_index} is outside the trajectory range."
            )
        if selected_index in selected_indices:
            raise ValueError(f"Duplicate seeded anchor index {selected_index} is not allowed.")
        selected_indices.append(selected_index)
        selected_scores.append(
            None if anchor.selection_score is None else float(anchor.selection_score)
        )

    for rank in range(len(seeded_anchors), num_anchors):
        if not selected_indices:
            distances = np.linalg.norm(centers - mean_center[None, :], axis=1)
        else:
            selected_centers = centers[np.asarray(selected_indices, dtype=np.int64)]
            deltas = centers[:, None, :] - selected_centers[None, :, :]
            distances = np.linalg.norm(deltas, axis=2).min(axis=1)
            distances = distances.astype(np.float32, copy=False)
            distances[np.asarray(selected_indices, dtype=np.int64)] = -np.inf
        best_idx = int(np.argmax(distances))
        selected_indices.append(best_idx)
        selected_scores.append(float(distances[best_idx]))
    anchors = list(seeded_anchors)
    anchors.extend(
        _stationary_pose(
            trajectory,
            side=_anchor_label(rank),
            axis_name=axis_name,
            index=index,
            selection_rank=rank,
            selection_strategy=GREEDY_ANCHOR_SELECTION_STRATEGY,
            selection_score=selected_scores[rank],
        )
        for rank, index in enumerate(selected_indices[len(seeded_anchors) :], start=len(seeded_anchors))
    )
    return StationaryPoseSelection(
        trajectory=trajectory,
        anchors=tuple(anchors),
    )


def select_stationary_poses(
    scene_root: str | Path,
    *,
    camera_name: str = "cam0",
    axis_name: str = "world_x",
    num_anchors: int = 2,
    trajectory_source: str = "camera_json",
    saved_dir: str | Path | None = None,
    worldtree_root: str | Path | None = None,
    device: str | None = None,
) -> StationaryPoseSelection:
    source = str(trajectory_source).strip().lower()
    if source == "camera_json":
        trajectory = load_camera_trajectory(scene_root, camera_name=camera_name)
    elif source == "photometric_cam":
        if saved_dir is None:
            raise ValueError("saved_dir is required when trajectory_source='photometric_cam'")
        if worldtree_root is None:
            raise ValueError("worldtree_root is required when trajectory_source='photometric_cam'")
        trajectory = load_photometric_camera_trajectory(
            scene_root,
            saved_dir=saved_dir,
            worldtree_root=worldtree_root,
            camera_name=camera_name,
            device=device,
        )
    else:
        raise ValueError(
            f"Unsupported trajectory_source {trajectory_source!r}; "
            "expected 'camera_json' or 'photometric_cam'"
        )
    return select_stationary_extrema(
        trajectory,
        axis_name=axis_name,
        num_anchors=num_anchors,
    )


def stationary_pose_to_payload(
    pose: StationaryPose,
    *,
    pose_source_kind: str | None = None,
    pose_source_path: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "side": pose.side,
        "source_camera_name": pose.source_camera_name,
        "selected_index": int(pose.selected_index),
        "selected_time_id": int(pose.selected_time_id),
        "selected_frame_name": pose.selected_frame_name,
        "c2w": np.asarray(pose.c2w, dtype=np.float32).tolist(),
        "center": np.asarray(pose.center, dtype=np.float32).tolist(),
        "axis_name": pose.axis_name,
        "axis_value": float(pose.axis_value),
        "selection_rank": int(pose.selection_rank),
        "selection_strategy": str(pose.selection_strategy),
    }
    if pose.selection_score is not None:
        payload["selection_score"] = float(pose.selection_score)
    if pose_source_kind is not None:
        payload["pose_source_kind"] = str(pose_source_kind)
    if pose_source_path is not None:
        payload["pose_source_path"] = str(pose_source_path)
    return payload


def stationary_pose_from_payload(payload: dict[str, object]) -> StationaryPose:
    return StationaryPose(
        side=str(payload["side"]),
        source_camera_name=camera_label(str(payload["source_camera_name"])),
        selected_index=int(payload["selected_index"]),
        selected_time_id=int(payload["selected_time_id"]),
        selected_frame_name=str(payload["selected_frame_name"]),
        c2w=np.asarray(payload["c2w"], dtype=np.float32),
        center=np.asarray(payload["center"], dtype=np.float32),
        axis_name=str(payload["axis_name"]),
        axis_value=float(payload["axis_value"]),
        selection_rank=int(payload.get("selection_rank", 0)),
        selection_strategy=str(
            payload.get("selection_strategy", GREEDY_ANCHOR_SELECTION_STRATEGY)
        ),
        selection_score=(
            None
            if payload.get("selection_score") is None
            else float(payload["selection_score"])
        ),
    )


def slice_trajectory(
    trajectory: CameraTrajectory,
    *,
    clip_start: int = 0,
    max_frames: int | None = None,
) -> tuple[TrajectoryPose, ...]:
    start = max(0, int(clip_start))
    stop = trajectory.num_frames if max_frames is None else start + max(0, int(max_frames))
    return trajectory.poses[start:stop]
