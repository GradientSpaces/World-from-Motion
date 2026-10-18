import json
import os
import os.path as osp
import re
import shutil

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
CAMERA_ID_PATTERN = re.compile(r"cam(\d+)$")


def parse_frame_index_from_name(filename: str, fallback: int | None = None) -> int:
    base = osp.splitext(osp.basename(filename))[0]
    for prefix in ("frame_", "frame"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    try:
        return int(base)
    except ValueError:
        if fallback is None:
            raise
        return fallback


def load_image_sequence(image_dir: str) -> tuple[list[str], list[int]]:
    image_names = sorted(
        [
            fn
            for fn in os.listdir(image_dir)
            if fn.lower().endswith(IMAGE_EXTENSIONS)
        ]
    )
    if not image_names:
        raise ValueError(f"No images found in {image_dir}")
    raw_frame_indices = [
        parse_frame_index_from_name(fn, idx) for idx, fn in enumerate(image_names)
    ]
    return image_names, raw_frame_indices


def get_image_size(image_dir: str) -> tuple[int, int]:
    image_names, _ = load_image_sequence(image_dir)
    first_image = cv2.imread(osp.join(image_dir, image_names[0]), cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise ValueError(f"Failed to read image {osp.join(image_dir, image_names[0])}")
    height, width = first_image.shape[:2]
    return width, height


def reset_output_dir(path: str) -> None:
    if osp.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def extract_video_frames_to_dir(
    video_path: str,
    output_dir: str,
    filename_template: str = "frame_{:06d}.png",
    overwrite: bool = True,
) -> tuple[list[str], list[int]]:
    if overwrite:
        reset_output_dir(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    written_filenames: list[str] = []
    raw_frame_indices: list[int] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        filename = filename_template.format(frame_index)
        output_path = osp.join(output_dir, filename)
        if not cv2.imwrite(output_path, frame):
            raise RuntimeError(f"Failed to write frame {frame_index} to {output_path}")
        written_filenames.append(filename)
        raw_frame_indices.append(frame_index)
        frame_index += 1

    cap.release()
    if not written_filenames:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    return written_filenames, raw_frame_indices


def parse_intrinsics_str(intrinsics_str: str) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = [float(x.strip()) for x in intrinsics_str.split(",")]
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsics_arr = np.array([fx, fy, cx, cy], dtype=np.float32)
    return K, intrinsics_arr


def load_multicamvideo_pose_data(pose_file: str) -> dict:
    with open(pose_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_camera_ids_from_pose_data(pose_data: dict) -> list[str]:
    frame_keys = sorted(
        [k for k in pose_data.keys() if k.startswith("frame")],
        key=lambda x: int(x[5:]),
    )
    if not frame_keys:
        raise ValueError(f"No frame entries found in pose data")
    return sorted([k for k in pose_data[frame_keys[0]].keys() if k.startswith("cam")])


def parse_multicamvideo_matrix_string(matrix_str: str) -> np.ndarray:
    pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    numbers = re.findall(pattern, matrix_str)
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers in matrix string, got {len(numbers)}")
    return np.array([float(n) for n in numbers], dtype=np.float32).reshape(4, 4)


def convert_multicamvideo_unreal_to_opencv_c2w(
    unreal_matrix: np.ndarray,
    unit_scale: float = 100.0,
) -> np.ndarray:
    c2w = unreal_matrix.T.copy()

    # Camera axes: Unreal [forward, right, up] -> OpenCV [right, down, forward].
    c2w = c2w[:, [1, 2, 0, 3]]
    c2w[:, 1] *= -1.0

    # Apply the same basis transform to world axes.
    c2w = c2w[[1, 2, 0, 3], :]
    c2w[1, :] *= -1.0

    c2w[:3, 3] /= unit_scale
    return c2w.astype(np.float32)


def load_multicamvideo_c2ws(
    pose_file: str,
    camera_id: str,
    frame_indices: list[int],
    unit_scale: float = 100.0,
) -> np.ndarray:
    pose_data = load_multicamvideo_pose_data(pose_file)
    c2ws = []
    for frame_idx in frame_indices:
        frame_key = f"frame{frame_idx}"
        matrix_str = pose_data[frame_key][camera_id]
        unreal_matrix = parse_multicamvideo_matrix_string(matrix_str)
        c2ws.append(
            convert_multicamvideo_unreal_to_opencv_c2w(
                unreal_matrix, unit_scale=unit_scale
            )
        )
    return np.stack(c2ws, axis=0).astype(np.float32)


def load_multicamvideo_w2cs(
    pose_file: str,
    camera_id: str,
    frame_indices: list[int],
    unit_scale: float = 100.0,
) -> np.ndarray:
    c2ws = load_multicamvideo_c2ws(
        pose_file=pose_file,
        camera_id=camera_id,
        frame_indices=frame_indices,
        unit_scale=unit_scale,
    )
    return np.linalg.inv(c2ws).astype(np.float32)


def discover_multicamvideo_videos(video_dir: str) -> dict[str, str]:
    discovered = {}
    for filename in sorted(os.listdir(video_dir)):
        stem, ext = osp.splitext(filename)
        if ext.lower() not in VIDEO_EXTENSIONS:
            continue
        if CAMERA_ID_PATTERN.fullmatch(stem) is None:
            continue
        discovered[stem] = osp.join(video_dir, filename)
    return discovered


def camera_id_to_worldtree_id(camera_id: str, is_training_camera: bool) -> int:
    if is_training_camera:
        return 0

    match = CAMERA_ID_PATTERN.fullmatch(camera_id)
    if match is None:
        raise ValueError(f"Invalid MultiCamVideo camera id: {camera_id}")

    # `load_iphone_cameras()` groups files via `startswith(f\"{cam_id}\")`, so we use
    # fixed-width non-overlapping ids for test cameras to avoid collisions like 1 vs 10.
    return 1000 + int(match.group(1))


def build_worldtree_camera_json(
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_size_wh: tuple[int, int],
) -> dict:
    width, height = image_size_wh
    return {
        "orientation": c2w[:3, :3].T.astype(float).tolist(),
        "position": c2w[:3, 3].astype(float).tolist(),
        "focal_length": float(0.5 * (fx + fy)),
        "principal_point": [float(cx), float(cy)],
        "image_size": [int(width), int(height)],
    }
