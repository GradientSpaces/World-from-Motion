"""
AgiBotWorld dataset helpers for MoSca.

Loads camera intrinsics/extrinsics and converts to MoSca format (T_wi).
Supports computing world-frame camera poses by combining:
  - Per-frame cam-to-base extrinsic (from aligned JSON)
  - Robot base-to-world pose (from proprio_stats.h5)
"""

import json
import os
import os.path as osp
import numpy as np
import torch
import logging


def load_agibot_intrinsics(intrinsic_json_path):
    """Load camera intrinsics from AgiBotWorld JSON.

    Returns:
        dict with keys: fx, fy, ppx, ppy, distortion_model, k1, k2, k3, p1, p2
    """
    with open(intrinsic_json_path, "r") as f:
        data = json.load(f)
    return data["intrinsic"]


def load_agibot_extrinsics_aligned(extrinsic_json_path):
    """Load per-frame aligned extrinsics (camera-to-base) from AgiBotWorld JSON.

    Returns:
        R_list: (T, 3, 3) rotation matrices
        t_list: (T, 3) translation vectors
    """
    with open(extrinsic_json_path, "r") as f:
        data = json.load(f)

    R_list = []
    t_list = []
    for entry in data:
        R = np.array(entry["extrinsic"]["rotation_matrix"])
        t = np.array(entry["extrinsic"]["translation_vector"])
        R_list.append(R)
        t_list.append(t)

    return np.array(R_list), np.array(t_list)


def load_agibot_robot_base_pose(h5_path):
    """Load robot base pose (position + orientation) from proprio_stats.h5.

    Returns:
        positions: (T, 3) xyz in meters
        orientations: (T, 4) quaternion xyzw
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        positions = f["state/robot/position"][:]
        orientations = f["state/robot/orientation"][:]
    return positions, orientations


def quat_xyzw_to_rotation_matrix(q):
    """Convert quaternion (xyzw) to 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(q).as_matrix()


def compute_agibot_cam_to_world(
    cam2base_R, cam2base_t, robot_positions, robot_orientations
):
    """Compute per-frame camera-to-world transforms.

    T_world_cam = T_world_base @ T_base_cam

    Args:
        cam2base_R: (T, 3, 3) camera-to-base rotation
        cam2base_t: (T, 3) camera-to-base translation
        robot_positions: (T, 3) robot base position in world
        robot_orientations: (T, 4) robot base orientation (xyzw quaternion)

    Returns:
        T_wi: (T, 4, 4) camera-to-world transforms
    """
    T = len(cam2base_R)
    T_wi = np.zeros((T, 4, 4))

    for i in range(T):
        # T_base_cam (cam in base frame)
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = cam2base_R[i]
        T_base_cam[:3, 3] = cam2base_t[i]

        # T_world_base (base in world frame)
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = quat_xyzw_to_rotation_matrix(robot_orientations[i])
        T_world_base[:3, 3] = robot_positions[i]

        # T_world_cam = T_world_base @ T_base_cam
        T_wi[i] = T_world_base @ T_base_cam

    return T_wi


def load_agibot_gt_poses(
    camera_param_dir, proprio_h5_path=None, cam_name="head", num_frames=None,
    frame_step=1,
):
    """Load GT camera poses for AgiBotWorld data.

    Args:
        camera_param_dir: path to parameters/camera/ directory
        proprio_h5_path: path to proprio_stats.h5 (optional, for mobile base)
        cam_name: camera name (default: "head")
        num_frames: number of video frames (to handle frame count mismatch)

    Returns:
        gt_training_cam_T_wi: (T, 4, 4) camera-to-world transforms
        gt_training_fov: field of view in degrees
        gt_training_cxcy_ratio: list of [cx_ratio, cy_ratio]
    """
    # Load intrinsics
    intrinsic = load_agibot_intrinsics(
        osp.join(camera_param_dir, f"{cam_name}_intrinsic_params.json")
    )
    fx, fy = intrinsic["fx"], intrinsic["fy"]
    ppx, ppy = intrinsic["ppx"], intrinsic["ppy"]

    # Image size from rs_camera_info.json
    rs_info_path = osp.join(camera_param_dir, "rs_camera_info.json")
    if osp.exists(rs_info_path):
        with open(rs_info_path, "r") as f:
            rs_info = json.load(f)
        # Find the entry for this camera
        for dev in rs_info.values():
            if dev.get("name") == cam_name:
                W, H = int(dev["width"]), int(dev["height"])
                break
        else:
            W, H = 640, 480
    else:
        W, H = 640, 480

    # Build K matrix directly (avoids FOV roundtrip errors)
    # NOTE: AgiBotWorld depth maps are from the RealSense D455 depth sensor (not
    # aligned to color). The intrinsics (fx, fy, ppx, ppy) correspond to the depth
    # sensor, which has different FOV than the color sensor. fx != fy because the
    # native depth sensor resolution (1280x800) is non-uniformly downsampled to
    # 640x480. We use the depth intrinsics as-is for correct 3D geometry.
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = ppx
    K[1, 2] = ppy

    # Load aligned extrinsics (cam-to-base per frame)
    ext_path = osp.join(
        camera_param_dir, f"{cam_name}_extrinsic_params_aligned.json"
    )
    cam2base_R, cam2base_t = load_agibot_extrinsics_aligned(ext_path)

    # Subsample extrinsics to match video frame_step
    if frame_step > 1:
        indices = list(range(0, len(cam2base_R), frame_step))
        cam2base_R = cam2base_R[indices]
        cam2base_t = cam2base_t[indices]

    # Optionally combine with robot base pose
    if proprio_h5_path is not None and osp.exists(proprio_h5_path):
        robot_pos, robot_ori = load_agibot_robot_base_pose(proprio_h5_path)
        if frame_step > 1:
            indices = list(range(0, len(robot_pos), frame_step))
            robot_pos = robot_pos[indices]
            robot_ori = robot_ori[indices]
        # Align frame counts
        n_ext = len(cam2base_R)
        n_robot = len(robot_pos)
        n = min(n_ext, n_robot)
        T_wi = compute_agibot_cam_to_world(
            cam2base_R[:n], cam2base_t[:n], robot_pos[:n], robot_ori[:n]
        )
    else:
        # Use cam-to-base as cam-to-world (static base assumption)
        n = len(cam2base_R)
        T_wi = np.zeros((n, 4, 4))
        for i in range(n):
            T_wi[i] = np.eye(4)
            T_wi[i, :3, :3] = cam2base_R[i]
            T_wi[i, :3, 3] = cam2base_t[i]

    if num_frames is not None and num_frames != len(T_wi):
        logging.warning(
            f"Frame count mismatch: {num_frames} video frames vs {len(T_wi)} extrinsic frames. "
            f"Truncating to {min(num_frames, len(T_wi))}."
        )
        n = min(num_frames, len(T_wi))
        T_wi = T_wi[:n]

    T_wi = torch.from_numpy(T_wi).float()
    K = torch.from_numpy(K).float()

    return T_wi, K, (W, H), intrinsic
