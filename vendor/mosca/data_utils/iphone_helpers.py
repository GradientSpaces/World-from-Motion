# deal with iphone dataset
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any, Mapping

import logging
import numpy as np
import torch
from tqdm import tqdm


def normalize_iphone_focal_length(
    focal_length: float | list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[float, float]:
    focal = np.asarray(focal_length, dtype=np.float64).reshape(-1)
    if focal.size == 0:
        raise ValueError("Empty focal_length payload")
    if focal.size == 1:
        value = float(focal[0])
        return value, value
    if focal.size == 2:
        return float(focal[0]), float(focal[1])
    raise ValueError(f"Unsupported focal_length payload shape: {focal_length}")


def iphone_focal_payload(
    focal_x: float,
    focal_y: float,
    *,
    collapse_tol: float = 1e-6,
) -> float | list[float]:
    if abs(float(focal_x) - float(focal_y)) <= collapse_tol:
        return float(focal_x)
    return [float(focal_x), float(focal_y)]


def normalize_iphone_fovdeg(
    fovdeg: float | list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[float, float]:
    fov = np.asarray(fovdeg, dtype=np.float64).reshape(-1)
    if fov.size == 0:
        raise ValueError("Empty fovdeg payload")
    if fov.size == 1:
        value = float(fov[0])
        return value, value
    if fov.size == 2:
        return float(fov[0]), float(fov[1])
    raise ValueError(f"Unsupported fovdeg payload shape: {fovdeg}")


def iphone_rel_focal_from_fovdeg(
    fovdeg: float | list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    fov_x_deg, fov_y_deg = normalize_iphone_fovdeg(fovdeg)
    return np.array(
        [
            1.0 / np.tan(np.deg2rad(fov_x_deg) / 2.0),
            1.0 / np.tan(np.deg2rad(fov_y_deg) / 2.0),
        ],
        dtype=np.float32,
    )


def load_iphone_camera_json(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iphone_intrinsics_from_camera_json(
    camera_json: Mapping[str, Any],
) -> tuple[tuple[float, float], list[float], list[float]]:
    focal_length = normalize_iphone_focal_length(camera_json["focal_length"])
    image_size = [float(x) for x in camera_json["image_size"]]
    principal_point = [float(x) for x in camera_json["principal_point"]]
    return focal_length, image_size, principal_point


def iphone_camera_json_from_T_wi(
    T_wi: np.ndarray,
    focal_length: float | list[float] | tuple[float, float] | np.ndarray,
    image_size: list[float],
    principal_point: list[float],
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a DyCheck/Nerfies-style camera JSON from a world-from-camera matrix.

    This matches ``load_iphone_cameras``: the on-disk ``orientation`` is stored such
    that ``np.asarray(orientation).T`` is the 3x3 world-from-camera rotation.
    """
    T_wi = np.asarray(T_wi, dtype=np.float64)
    if T_wi.shape != (4, 4):
        raise ValueError(f"Expected T_wi (4,4), got {T_wi.shape}")
    R_wc = T_wi[:3, :3]
    t_wc = T_wi[:3, 3]
    orientation = R_wc.T
    focal_x, focal_y = normalize_iphone_focal_length(focal_length)
    payload: dict[str, Any] = {
        "focal_length": iphone_focal_payload(focal_x, focal_y),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "orientation": orientation.astype(float).tolist(),
        "pixel_aspect_ratio": 1.0,
        "position": t_wc.astype(float).tolist(),
        "principal_point": [float(principal_point[0]), float(principal_point[1])],
        "radial_distortion": [0.0, 0.0, 0.0],
        "skew": 0.0,
        "tangential_distortion": [0.0, 0.0],
    }
    if template is not None:
        for key in (
            "pixel_aspect_ratio",
            "radial_distortion",
            "skew",
            "tangential_distortion",
        ):
            if key in template:
                payload[key] = template[key]
    return payload


def write_iphone_camera_json(
    path: str | os.PathLike,
    T_wi: np.ndarray,
    focal_length: float | list[float] | tuple[float, float] | np.ndarray,
    image_size: list[float],
    principal_point: list[float],
    template: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = iphone_camera_json_from_T_wi(
        T_wi, focal_length, image_size, principal_point, template=template
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def load_iphone_multiview_manifest(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_iphone_multiview_manifest(
    manifest: Mapping[str, Any],
    selected_cameras: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if selected_cameras is None:
        return dict(manifest)

    anchor_camera = str(manifest["anchor_camera"])
    selected_aux_cameras = []
    for camera_name in selected_cameras:
        camera_name = str(camera_name)
        if camera_name == anchor_camera or camera_name in selected_aux_cameras:
            continue
        selected_aux_cameras.append(camera_name)

    manifest_cameras = {
        str(camera["camera_name"]): dict(camera) for camera in manifest["cameras"]
    }
    if anchor_camera not in manifest_cameras:
        raise KeyError(f"Anchor camera {anchor_camera} missing from multiview manifest")
    missing = [
        camera_name
        for camera_name in selected_aux_cameras
        if camera_name not in manifest_cameras
    ]
    if missing:
        raise KeyError(f"Unknown selected multiview cameras: {missing}")

    filtered_manifest = dict(manifest)
    filtered_manifest["cameras"] = [manifest_cameras[anchor_camera]] + [
        manifest_cameras[camera_name] for camera_name in selected_aux_cameras
    ]
    filtered_manifest["selected_auxiliary_cameras"] = selected_aux_cameras
    return filtered_manifest


def load_iphone_multiview_reference_poses(
    workspace: str | os.PathLike, manifest: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    ws = Path(workspace)
    out: dict[str, np.ndarray] = {}
    for camera in manifest["cameras"]:
        name = camera["camera_name"]
        rel = camera["reference_c2w_path"]
        data = np.load(ws / rel)
        if "c2w" not in data:
            raise KeyError(f"Missing c2w in {ws / rel}")
        out[name] = np.asarray(data["c2w"], dtype=np.float32)
    return out


def load_iphone_cameras(dir_path):
    ######################################
    # ! dycheck use Nerfies convention, which is opencv format
    # https://github.com/google/nerfies#datasets
    # https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md
    ######################################

    all_fns = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    all_fns.sort()
    all_cam_ids = np.array([int(f.split("_")[0].split(".")[0]) for f in all_fns])
    all_cam_ids = np.unique(all_cam_ids).tolist()
    logging.info(f"Loading iphone cameras from {dir_path} with camid={all_cam_ids}")
    ret = {}
    for cam_id in all_cam_ids:
        cam_fns = [f for f in all_fns if f.startswith(f"{cam_id}")]
        cam_fns.sort()
        T_wi_list, fovdeg_list, t_list = [], [], []
        cxcy_ratio_list = []
        # ! iphone dataset only has one fov
        for fn in tqdm(cam_fns):
            time_ind = int(fn.split("_")[1].split(".")[0])  # ! time start from 0, not 1
            t_list.append(time_ind)

            with open(osp.join(dir_path, fn), "r") as f:
                cam_data = json.load(f)

            # ! for shape of motion camera
            if "image_size" not in cam_data:
                cam_data["image_size"] = [720, 960]  # W,H

            focal_x, focal_y = normalize_iphone_focal_length(cam_data["focal_length"])
            T_wi = np.eye(4)
            t_wi = np.asarray(cam_data["position"])
            R_wi = np.asarray(cam_data["orientation"]).T
            T_wi[:3, :3] = R_wi
            T_wi[:3, 3] = t_wi
            # world = R.T @ local
            # https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/camera.py#L263
            # the optical axis is the last row
            T_wi_list.append(torch.from_numpy(T_wi))

            # ! get the fov in current code format, the focal with the shortest side
            short_size = min(cam_data["image_size"])  # -1, +1
            fovdeg = [
                float(np.rad2deg(2 * np.arctan(short_size / (2 * focal_x)))),
                float(np.rad2deg(2 * np.arctan(short_size / (2 * focal_y)))),
            ]
            if abs(fovdeg[0] - fovdeg[1]) <= 1e-6:
                fovdeg = fovdeg[0]

            # also load camera center
            principal_point = np.asarray(cam_data["principal_point"])
            image_size = np.asarray(cam_data["image_size"])
            cx_ratio = principal_point[0] / image_size[0]
            cy_ratio = principal_point[1] / image_size[1]
            cxcy_ratio_list.append([cx_ratio, cy_ratio])

            fovdeg_list.append(fovdeg)
        ret[cam_id] = {
            "T_wi": torch.stack(T_wi_list, 0).float(),
            "fovdeg": fovdeg_list,
            "cxcy_ratio": cxcy_ratio_list,
            "t": t_list,
            "filenames": [f[:-5] for f in cam_fns],
        }
    return ret


def load_iphone_gt_poses(src, t_subsample):
    gt_training_cams = load_iphone_cameras(osp.join(src, "cameras"))
    assert (
        len(gt_training_cams) == 1
    ), "Only support Mono camera for now in Iphone dataset"
    # subsample the gt camera
    gt_training_cams = {k: v[::t_subsample] for k, v in gt_training_cams[0].items()}
    gt_testing_cams = load_iphone_cameras(osp.join(src, "test_cameras"))
    gt_training_fov = gt_training_cams["fovdeg"][0]
    logging.info(
        f"Iphone gt training camera fov is {gt_training_cams['fovdeg'][0]} deg."
    )
    gt_training_cam_T_wi = gt_training_cams["T_wi"]
    # ! this bound which test frames can be retrieved
    gt_training_tids = gt_training_cams["t"]
    gt_training_cxcy_ratio = gt_training_cams["cxcy_ratio"]

    gt_testing_cam_T_wi_list, gt_testing_tids_list = [], []
    gt_testing_fov_list, gt_testing_fns_list = [], []
    gt_testing_cxcy_ratio_list = []
    for it in gt_testing_cams.values():
        sample_index = [i for i, t in enumerate(it["t"]) if t in gt_training_tids]
        gt_testing_tids = [gt_training_tids.index(it["t"][i]) for i in sample_index]
        gt_testing_tids_list.append(gt_testing_tids)
        gt_testing_cam_T_wi = it["T_wi"][sample_index]
        gt_testing_cam_T_wi_list.append(gt_testing_cam_T_wi)
        gt_testing_fns_list.append([it["filenames"][i] for i in sample_index])
        # ! assume all cam stays the same across time, use the first one
        gt_testing_fov_list.append(it["fovdeg"][0])
        gt_testing_cxcy_ratio_list.append(it["cxcy_ratio"][0])
    # todo: can use gt camera for evaluation
    return (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
    )
