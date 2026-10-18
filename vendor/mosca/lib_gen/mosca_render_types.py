from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseAssignment:
    T_cw: np.ndarray
    source_kind: str
    source_time_id: int | None
    source_frame_name: str | None
    source_camera_json_path: str | None
    pose_archive_path: str
    has_exact_tto_pose: bool


@dataclass(frozen=True)
class CameraRenderSpec:
    cam_id: str
    time_id: int
    frame_name: str
    camera_json_path: str | None
    image_path: str | None
    K_raw: np.ndarray
    img_wh: tuple[int, int]
    T_cw: np.ndarray
    has_dataset_frame_at_time: bool
    pose_assignment: PoseAssignment
