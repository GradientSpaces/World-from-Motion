from __future__ import annotations

import json
import os.path as osp
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = osp.abspath(osp.dirname(__file__))
PAIR_TOKEN_SPLIT = re.compile(r"[-,:/]")
CAMERA_FILE_PATTERN = re.compile(r"(\d+)_(\d+)\.json$")


@dataclass(frozen=True)
class FrameSpec:
    cam_id: str
    time_id: int
    frame_name: str
    camera_json_path: str
    image_path: str | None
    K_raw: np.ndarray
    img_wh: tuple[int, int]


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


def camera_label(cam_id: str | int) -> str:
    token = str(cam_id)
    if token.isdigit():
        return f"cam{int(token)}"
    return token if token.startswith("cam") else f"cam{token}"


def normalize_camera_token(token: str) -> str:
    token = token.strip().lower()
    if token.startswith("cam"):
        token = token[3:]
    if token.isdigit():
        return str(int(token))
    return token


def parse_pair_token(token: str) -> tuple[str, str]:
    parts = [p for p in PAIR_TOKEN_SPLIT.split(token.replace(" ", "")) if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid pair token '{token}'.")
    a = normalize_camera_token(parts[0])
    b = normalize_camera_token(parts[1])
    if a == b:
        raise ValueError(f"Pair token '{token}' uses the same camera twice.")
    return a, b


def parse_pairs(
    pair_tokens: list[str] | None,
    reference_camera: str,
    test_camera_ids: list[str],
) -> list[tuple[str, str]]:
    if pair_tokens is None:
        return [(reference_camera, cam_id) for cam_id in test_camera_ids]
    parsed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in pair_tokens:
        pair = parse_pair_token(token)
        if pair in seen:
            continue
        parsed.append(pair)
        seen.add(pair)
    return parsed


def resolve_repo_relative(path: str) -> str:
    if osp.isabs(path):
        return path
    return osp.normpath(osp.join(REPO_ROOT, path))


def read_commandline_tokens(path: str) -> list[str]:
    line = Path(path).read_text(encoding="utf-8").strip()
    if not line:
        raise ValueError(f"Command line file is empty: {path}")
    return shlex.split(line)


def get_flag_value(tokens: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def infer_cfg_path(saved_dir: str, cfg_override: str | None = None) -> str:
    if cfg_override is not None:
        resolved = resolve_repo_relative(cfg_override)
        return resolved if osp.exists(resolved) else cfg_override
    args_path = osp.join(saved_dir, "fit_commandline_args.txt")
    if not osp.isfile(args_path):
        raise FileNotFoundError(
            f"Missing --cfg override and fit_commandline_args.txt was not found in {saved_dir}"
        )
    tokens = read_commandline_tokens(args_path)
    cfg_path = get_flag_value(tokens, "--cfg")
    if cfg_path is None:
        raise ValueError(f"Could not infer --cfg from {args_path}")
    resolved = resolve_repo_relative(cfg_path)
    return resolved if osp.exists(resolved) else cfg_path


def infer_workspace_path(saved_dir: str, ws_override: str | None = None) -> str:
    if ws_override is not None:
        return osp.abspath(ws_override)
    args_path = osp.join(saved_dir, "fit_commandline_args.txt")
    if not osp.isfile(args_path):
        raise FileNotFoundError(
            f"Missing --ws override and fit_commandline_args.txt was not found in {saved_dir}"
        )
    tokens = read_commandline_tokens(args_path)
    ws_path = get_flag_value(tokens, "--ws")
    if ws_path is None:
        raise ValueError(f"Could not infer --ws from {args_path}")
    resolved = resolve_repo_relative(ws_path)
    return osp.abspath(resolved if osp.exists(resolved) else ws_path)


def load_camera_json(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        camera_data = json.load(f)

    focal = float(camera_data["focal_length"])
    principal_point = camera_data["principal_point"]
    cx = float(principal_point[0])
    cy = float(principal_point[1])
    K = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    image_size = camera_data.get("image_size")
    if image_size is not None and len(image_size) == 2:
        img_wh = (int(image_size[0]), int(image_size[1]))
    else:
        img_wh = (int(round(cx * 2.0)), int(round(cy * 2.0)))
    return K, img_wh


def resolve_frame_intrinsics(
    K_raw: np.ndarray,
    declared_img_wh: tuple[int, int],
    image_path: str | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if image_path is None or not osp.isfile(image_path):
        return K_raw, declared_img_wh

    with Image.open(image_path) as img:
        actual_img_wh = (int(img.width), int(img.height))

    if actual_img_wh == declared_img_wh:
        return K_raw, declared_img_wh

    declared_w, declared_h = declared_img_wh
    if declared_w <= 0 or declared_h <= 0:
        return K_raw, actual_img_wh

    scale_x = actual_img_wh[0] / float(declared_w)
    scale_y = actual_img_wh[1] / float(declared_h)
    K_scaled = K_raw.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    return K_scaled, actual_img_wh


def discover_camera_frames(camera_dir: str, image_dir: str) -> dict[str, list[FrameSpec]]:
    grouped: dict[str, list[FrameSpec]] = {}
    camera_root = Path(camera_dir)
    if not camera_root.is_dir():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
    for path in sorted(camera_root.glob("*.json")):
        match = CAMERA_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        cam_id = str(int(match.group(1)))
        time_id = int(match.group(2))
        frame_name = path.stem
        image_path = Path(image_dir) / f"{frame_name}.png"
        K_raw, img_wh = load_camera_json(str(path))
        image_path_str = str(image_path) if image_path.exists() else None
        K_raw, img_wh = resolve_frame_intrinsics(K_raw, img_wh, image_path_str)
        grouped.setdefault(cam_id, []).append(
            FrameSpec(
                cam_id=cam_id,
                time_id=time_id,
                frame_name=frame_name,
                camera_json_path=str(path),
                image_path=image_path_str,
                K_raw=K_raw,
                img_wh=img_wh,
            )
        )
    for entries in grouped.values():
        entries.sort(key=lambda spec: spec.time_id)
    return dict(sorted(grouped.items(), key=lambda kv: int(kv[0])))


def build_time_lookup(entries: list[FrameSpec]) -> dict[int, FrameSpec]:
    lookup: dict[int, FrameSpec] = {}
    for spec in entries:
        if spec.time_id not in lookup:
            lookup[spec.time_id] = spec
    return lookup


def make_synthetic_frame_name(cam_id: str, time_id: int) -> str:
    return f"{cam_id}_{time_id:05d}"


def decode_frame_names(arr: np.ndarray) -> list[str]:
    decoded = []
    for value in arr.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def align_pose_archive_to_specs(
    archive_path: str,
    expected_specs: list[FrameSpec],
    short_pose_alignment: str,
) -> tuple[list[tuple[FrameSpec, np.ndarray]], str]:
    with np.load(archive_path) as archive:
        poses = archive["poses"].astype(np.float32)
        archive_files = set(archive.files)
        frame_names = (
            decode_frame_names(archive["frame_names"])
            if "frame_names" in archive_files
            else None
        )

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected pose shape in {archive_path}: {poses.shape}")
    if len(poses) == 0:
        raise ValueError(f"No poses found in {archive_path}")
    if not expected_specs:
        raise ValueError(
            f"No expected dataset frames available to align poses from {archive_path}"
        )

    if frame_names is not None:
        if len(frame_names) != len(poses):
            raise ValueError(
                f"Archive {archive_path} has {len(poses)} poses but "
                f"{len(frame_names)} frame_names."
            )
        frame_lookup = {spec.frame_name: spec for spec in expected_specs}
        aligned: list[tuple[FrameSpec, np.ndarray]] = []
        for frame_name, pose in zip(frame_names, poses):
            if frame_name not in frame_lookup:
                raise KeyError(
                    f"Frame name '{frame_name}' from {archive_path} not found in "
                    "expected test specs."
                )
            aligned.append((frame_lookup[frame_name], pose))
        return aligned, "frame_names"

    if len(poses) == len(expected_specs):
        return list(zip(expected_specs, poses)), "exact_length"

    if len(poses) > len(expected_specs):
        raise ValueError(
            f"Pose archive {archive_path} has more poses ({len(poses)}) than expected "
            f"specs ({len(expected_specs)})."
        )

    if short_pose_alignment == "error":
        raise ValueError(
            f"Pose archive {archive_path} has only {len(poses)} poses for "
            f"{len(expected_specs)} expected specs."
        )

    if short_pose_alignment == "suffix":
        aligned_specs = expected_specs[-len(poses) :]
    elif short_pose_alignment == "prefix":
        aligned_specs = expected_specs[: len(poses)]
    else:
        raise ValueError(f"Unknown short_pose_alignment={short_pose_alignment}")

    return list(zip(aligned_specs, poses)), short_pose_alignment


def expand_tto_camera_specs(
    *,
    cam_id: str,
    canonical_entries: list[FrameSpec],
    test_lookup: dict[int, FrameSpec],
    template_spec: FrameSpec,
    pose_archive_path: str,
    short_pose_alignment: str,
    initial_pose_fill: str,
) -> tuple[list[CameraRenderSpec], dict]:
    expected_specs = [
        test_lookup[spec.time_id]
        for spec in canonical_entries
        if spec.time_id in test_lookup
    ]
    aligned_pose_pairs, alignment_mode = align_pose_archive_to_specs(
        pose_archive_path, expected_specs, short_pose_alignment
    )
    pose_by_time = {spec.time_id: (spec, pose) for spec, pose in aligned_pose_pairs}
    exact_pose_times = sorted(pose_by_time.keys())
    first_exact_spec, first_exact_pose = pose_by_time[exact_pose_times[0]]
    first_assignment = PoseAssignment(
        T_cw=first_exact_pose,
        source_kind="backfill_first_available",
        source_time_id=first_exact_spec.time_id,
        source_frame_name=first_exact_spec.frame_name,
        source_camera_json_path=first_exact_spec.camera_json_path,
        pose_archive_path=pose_archive_path,
        has_exact_tto_pose=False,
    )

    specs: list[CameraRenderSpec] = []
    num_exact = 0
    num_forward_filled = 0
    num_backfilled = 0
    last_assignment: PoseAssignment | None = None

    for canonical_spec in canonical_entries:
        actual_spec = test_lookup.get(canonical_spec.time_id)
        if canonical_spec.time_id in pose_by_time:
            matched_spec, pose = pose_by_time[canonical_spec.time_id]
            assignment = PoseAssignment(
                T_cw=pose.astype(np.float32),
                source_kind="exact_tto",
                source_time_id=matched_spec.time_id,
                source_frame_name=matched_spec.frame_name,
                source_camera_json_path=matched_spec.camera_json_path,
                pose_archive_path=pose_archive_path,
                has_exact_tto_pose=True,
            )
            num_exact += 1
            last_assignment = assignment
        elif last_assignment is not None:
            assignment = PoseAssignment(
                T_cw=last_assignment.T_cw,
                source_kind="carry_forward",
                source_time_id=last_assignment.source_time_id,
                source_frame_name=last_assignment.source_frame_name,
                source_camera_json_path=last_assignment.source_camera_json_path,
                pose_archive_path=pose_archive_path,
                has_exact_tto_pose=False,
            )
            num_forward_filled += 1
        elif initial_pose_fill == "first_available":
            assignment = first_assignment
            num_backfilled += 1
        else:
            raise ValueError(
                f"Camera {camera_label(cam_id)} has no prior pose for time_id="
                f"{canonical_spec.time_id} and initial fill mode is "
                f"'{initial_pose_fill}'."
            )

        frame_spec = actual_spec if actual_spec is not None else template_spec
        frame_name = (
            actual_spec.frame_name
            if actual_spec is not None
            else make_synthetic_frame_name(cam_id, canonical_spec.time_id)
        )
        specs.append(
            CameraRenderSpec(
                cam_id=cam_id,
                time_id=canonical_spec.time_id,
                frame_name=frame_name,
                camera_json_path=(
                    actual_spec.camera_json_path
                    if actual_spec is not None
                    else template_spec.camera_json_path
                ),
                image_path=actual_spec.image_path if actual_spec is not None else None,
                K_raw=frame_spec.K_raw.copy(),
                img_wh=frame_spec.img_wh,
                T_cw=assignment.T_cw.copy(),
                has_dataset_frame_at_time=actual_spec is not None,
                pose_assignment=assignment,
            )
        )

    stats = {
        "camera_id": cam_id,
        "alignment_mode": alignment_mode,
        "pose_archive_path": pose_archive_path,
        "num_pose_rows": len(aligned_pose_pairs),
        "num_expected_dataset_frames": len(expected_specs),
        "num_exact_pose_frames": num_exact,
        "num_forward_filled_frames": num_forward_filled,
        "num_backfilled_frames": num_backfilled,
    }
    return specs, stats
