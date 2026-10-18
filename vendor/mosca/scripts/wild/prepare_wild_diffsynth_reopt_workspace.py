#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TEMPLATE_CFG = REPO_ROOT / "profile" / "demo" / "demo_fit.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a wild multiview reoptimization workspace from a rendered "
            "WorldTree/DiffSynth benchmark and its DiffSynth predictions."
        )
    )
    parser.add_argument("--source-ws", "--workspace", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--inference-root", required=True, type=Path)
    parser.add_argument("--out-ws", required=True, type=Path)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--target-cameras", nargs="+", default=None)
    parser.add_argument("--num-fps-cameras", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--pred-subdir", default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--cfg-out", type=Path, default=None)
    parser.add_argument("--template-cfg", type=Path, default=DEFAULT_TEMPLATE_CFG)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--reopt-mode",
        choices=("wild", "iphone"),
        default="wild",
        help=(
            "Dataset mode written into the generated reopt config. Use 'iphone' "
            "for iphone_extra/DyCheck-style sensor-depth scenes; the anchor camera "
            "is still initialized from the benchmark cam0 recon."
        ),
    )
    parser.add_argument(
        "--tap-mode",
        default=None,
        help="Override TAP mode in the generated fit config.",
    )
    parser.add_argument(
        "--no-anchor-camera-init",
        action="store_true",
        help=(
            "Do not initialize the anchor camera from the benchmark/rendered camera "
            "trajectory; let static BA initialize from tracks/depth."
        ),
    )
    parser.add_argument(
        "--init-camera-recon-path",
        default=None,
        help=(
            "Workspace-relative recon .npy to use for anchor camera initialization "
            "instead of the generated benchmark cam0 recon."
        ),
    )
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--photo-total-steps", type=int, default=None)
    parser.add_argument(
        "--allow-camera-ba",
        action="store_true",
        help=(
            "Do not force camera BA/photometric camera learning rates to zero; "
            "preserve the template/default MoSca behavior."
        ),
    )
    parser.add_argument(
        "--enable-multiview-depth",
        action="store_true",
        help="Enable depth-aware multiview reoptimization for auxiliary cameras.",
    )
    parser.add_argument(
        "--anchor-depth-dirname",
        default=None,
        help=(
            "Override the anchor Saved2D depth directory in the generated fit config. "
            "Useful when DA3 is run jointly on anchor and auxiliary cameras."
        ),
    )
    parser.add_argument(
        "--enable-multiview-depth-init",
        action="store_true",
        help="Use auxiliary-camera depth during static MoCa bundle adjustment.",
    )
    parser.add_argument(
        "--enable-multiview-track-init",
        action="store_true",
        help="Use auxiliary-camera tracks during static MoCa scaffold initialization.",
    )
    parser.add_argument(
        "--enable-multiview-track-loss",
        action="store_true",
        help="Use auxiliary-camera tracks during photometric reoptimization.",
    )
    parser.add_argument(
        "--multiview-track-camera-weight",
        type=float,
        default=0.5,
        help="Per-auxiliary-camera weight for multiview track losses.",
    )
    parser.add_argument(
        "--multiview-use-masks",
        action="store_true",
        help="Apply auxiliary dynamic masks to multiview RGB/depth photometric losses.",
    )
    parser.add_argument("--multiview-lambda-dep", type=float, default=0.05)
    parser.add_argument("--multiview-depth-camera-weight", type=float, default=0.25)
    parser.add_argument("--copy-assets", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the manifest/config without materializing the output workspace.",
    )
    parser.add_argument(
        "--skip-rgb-check",
        action="store_true",
        help="Do not require DiffSynth prediction images to exist before preparing.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def resolve_scene_root(root: Path, scene_name: str) -> Path:
    root = resolve_repo_path(root)
    if (root / "metadata" / "scene_summary.json").is_file():
        return root
    scene_root = root / scene_name
    if (scene_root / "metadata" / "scene_summary.json").is_file():
        return scene_root
    raise FileNotFoundError(
        f"Could not find metadata/scene_summary.json under {root} or {scene_root}"
    )


def resolve_inference_scene_root(root: Path, scene_name: str) -> Path:
    root = resolve_repo_path(root)
    if (root / scene_name).is_dir():
        return root / scene_name
    return root


def load_camera_frame_names(scene_root: Path, camera_name: str) -> list[str]:
    metadata_path = scene_root / "metadata" / f"{camera_name}_frames.json"
    payload = load_json(metadata_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Missing non-empty frames list in {metadata_path}")
    return [str(frame["frame_name"]) for frame in frames]


def select_target_cameras(
    summary: dict[str, Any],
    requested: Sequence[str] | None,
    num_fps_cameras: int | None,
) -> list[str]:
    if requested:
        cameras = [str(camera) for camera in requested]
    else:
        cameras = [str(camera) for camera in summary.get("target_cameras", [])]
        if num_fps_cameras is not None:
            cameras = cameras[: int(num_fps_cameras)]
    if not cameras:
        raise ValueError("No target cameras selected.")
    if len(set(cameras)) != len(cameras):
        raise ValueError(f"Duplicate target cameras: {cameras}")
    return cameras


def validate_inputs(
    *,
    scene_root: Path,
    inference_scene_root: Path,
    frame_names: Sequence[str],
    target_cameras: Sequence[str],
    pred_subdir: str,
    skip_rgb_check: bool,
) -> None:
    cameras = ["cam0", *target_cameras]
    for camera in cameras:
        camera_path = scene_root / "cameras" / f"{camera}_c2w.npz"
        if not camera_path.is_file():
            raise FileNotFoundError(f"Missing camera trajectory: {camera_path}")
        rgb_dir = scene_root / "rgb" / camera
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"Missing benchmark RGB directory: {rgb_dir}")
    if skip_rgb_check:
        return
    missing: list[Path] = []
    for camera in target_cameras:
        pred_dir = inference_scene_root / pred_subdir / camera
        for frame_name in frame_names:
            pred_path = pred_dir / f"{frame_name}.png"
            if not pred_path.is_file():
                missing.append(pred_path)
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break
    if missing:
        preview = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing DiffSynth prediction frames:\n{preview}")


def build_multiview_manifest(
    *,
    scene_name: str,
    source_ws: Path,
    scene_root: Path,
    inference_scene_root: Path,
    summary: dict[str, Any],
    frame_names: Sequence[str],
    target_cameras: Sequence[str],
    pred_subdir: str,
    anchor_weight: float,
    aux_weight: float,
    anchor_depth_dirname: str | None,
) -> dict[str, Any]:
    time_ids = [int(value) for value in summary.get("time_ids", [])][: len(frame_names)]
    if len(time_ids) != len(frame_names):
        time_ids = list(range(len(frame_names)))

    def camera_entry(camera_name: str, weight: float) -> dict[str, Any]:
        return {
            "camera_name": camera_name,
            "rgb_dir": f"multiview/rgb/{camera_name}",
            "camera_json_dir": f"multiview/camera_json/{camera_name}",
            "reference_c2w_path": f"multiview/cameras/{camera_name}_c2w.npz",
            "frame_names": list(frame_names),
            "weight": float(weight),
        }

    return {
        "scene_name": scene_name,
        "source_workspace": str(source_ws),
        "benchmark_root": str(scene_root),
        "inference_root": str(inference_scene_root),
        "pred_rgb_subdir": pred_subdir,
        "anchor_camera": "cam0",
        "anchor_rgb_camera": "cam0",
        "skip_source_depth_dirs": anchor_depth_dirname is not None,
        "frame_stride": int(summary.get("frame_stride", 1)),
        "num_frames": len(frame_names),
        "frame_names": list(frame_names),
        "time_ids": time_ids,
        "selected_auxiliary_cameras": list(target_cameras),
        "cameras": [
            camera_entry("cam0", anchor_weight),
            *[camera_entry(camera, aux_weight) for camera in target_cameras],
        ],
    }


def write_reopt_config(
    *,
    template_cfg: Path,
    cfg_out: Path,
    source_ws: Path,
    scene_name: str,
    target_cameras: Sequence[str],
    exp_name: str,
    tap_mode: str | None,
    anchor_weight: float,
    aux_weight: float,
    anchor_recon_path: str | None,
    init_anchor_camera: bool,
    anchor_depth_dirname: str | None,
    photo_total_steps: int | None,
    allow_camera_ba: bool,
    enable_multiview_depth: bool,
    enable_multiview_depth_init: bool,
    enable_multiview_track_init: bool,
    enable_multiview_track_loss: bool,
    multiview_use_masks: bool,
    multiview_lambda_dep: float,
    multiview_depth_camera_weight: float,
    multiview_track_camera_weight: float,
    reopt_mode: str = "wild",
) -> None:
    template_cfg = resolve_repo_path(template_cfg)
    if not template_cfg.is_file():
        raise FileNotFoundError(f"Template config not found: {template_cfg}")
    cfg = OmegaConf.load(template_cfg)
    cfg.exp_name = exp_name
    cfg.mode = str(reopt_mode)
    if anchor_depth_dirname is not None:
        cfg.depth_dirname = str(anchor_depth_dirname)
    elif reopt_mode == "iphone" and (source_ws / "sensor_depth").is_dir():
        # iphone_extra scenes ship metric ARKit sensor depth; prefer it explicitly
        # so the reopt anchor depth matches the original iphone fit.
        cfg.depth_dirname = "sensor_depth"
    elif (source_ws / "depthcrafter_depth_sharp").is_dir():
        cfg.depth_dirname = "depthcrafter_depth_sharp"
    elif (source_ws / "depthcrafter_depth").is_dir():
        cfg.depth_dirname = "depthcrafter_depth"
    elif (source_ws / "vggt_omega_depth").is_dir():
        cfg.depth_dirname = "vggt_omega_depth"
    elif (source_ws / "depthanything3_depth").is_dir():
        cfg.depth_dirname = "depthanything3_depth"
    elif (source_ws / "sensor_depth").is_dir():
        cfg.depth_dirname = "sensor_depth"
    elif not hasattr(cfg, "depth_dirname"):
        cfg.depth_dirname = "depthanything3_depth"
    cfg.anchor_rgb_dirname = "images"
    cfg.tap_mode = str(
        tap_mode
        or infer_tap_mode(source_ws, str(getattr(cfg, "tap_mode", "spatracker")))
    )
    flow_dirname = infer_flow_dirname(source_ws)
    if flow_dirname is not None:
        cfg.flow_dirname = flow_dirname
    elif float(getattr(cfg, "photo_track_flow_chance", 0.0)) > 0.0:
        print(
            f"Warning: no flow_waft/flow_raft found under {source_ws}; "
            "disable photometric flow-backed track loss.",
            file=sys.stderr,
        )
        cfg.photo_track_flow_chance = 0.0
    if init_anchor_camera and anchor_recon_path is not None:
        cfg.init_da3_camera = True
        cfg.da3_recon_path = anchor_recon_path
        cfg.rescale_gt_cam_transl = False
    else:
        cfg.init_da3_camera = False
    if not allow_camera_ba:
        cfg.ba_lr_cam_q = 0.0
        cfg.ba_lr_cam_t = 0.0
        cfg.ba_lr_cam_f = 0.0
    cfg.use_multiview_photometric = True
    cfg.multiview_manifest = "multiview_manifest.json"
    cfg.multiview_selected_cameras = list(target_cameras)
    cfg.multiview_tie_anchor_intrinsics = True
    cfg.multiview_use_masks = bool(multiview_use_masks)
    if multiview_use_masks:
        cfg.epi_th = 0.5
        cfg.ba_epi_th = 0.5
        cfg.dynamic_gs_mask_source = "sam3"
        cfg.dynamic_gs_sam3_epi_th = 0.5
        cfg.dynamic_gs_sam3_dilate_ksize = 5
    cfg.multiview_log_interval = int(getattr(cfg, "multiview_log_interval", 50))
    cfg.multiview_rgb_ssim_lambda = float(
        getattr(cfg, "multiview_rgb_ssim_lambda", 0.1)
    )
    cfg.multiview_camera_weights = {
        "cam0": float(anchor_weight),
        **{camera: float(aux_weight) for camera in target_cameras},
    }
    cfg.multiview_lpips_camera_weights = {
        "cam0": 1.0,
        **{camera: 1.0 for camera in target_cameras},
    }
    if not allow_camera_ba:
        cfg.photo_lr_cam_q = 0.0
        cfg.photo_lr_cam_t = 0.0
        cfg.photo_lr_cam_f = 0.0
        cfg.photo_warm_lr_cam_q = 0.0
        cfg.photo_warm_lr_cam_t = 0.0
        cfg.photo_warm_lr_cam_f = 0.0
    if photo_total_steps is not None:
        cfg.photo_total_steps = int(photo_total_steps)
    cfg.use_multiview_depth_init = bool(enable_multiview_depth_init)
    cfg.use_multiview_depth_gs_init = bool(enable_multiview_depth)
    cfg.use_multiview_depth_sup = bool(enable_multiview_depth)
    if enable_multiview_depth or enable_multiview_depth_init:
        cfg.multiview_lambda_dep = float(multiview_lambda_dep)
        cfg.ba_multiview_lambda_depth = float(multiview_lambda_dep)
        cfg.multiview_dep_st_invariant = True
        cfg.multiview_depth_camera_weights = {
            camera: float(multiview_depth_camera_weight) for camera in target_cameras
        }
    cfg.use_multiview_track_init = bool(enable_multiview_track_init)
    cfg.use_multiview_track_loss = bool(enable_multiview_track_loss)
    if enable_multiview_track_init or enable_multiview_track_loss:
        cfg.multiview_track_camera_weights = {
            camera: float(multiview_track_camera_weight) for camera in target_cameras
        }
    cfg.skip_final_eval = True
    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, cfg_out)


def infer_tap_mode(source_ws: Path, fallback: str) -> str:
    for tap_path in sorted(source_ws.glob("*uniform*_tap.npz")):
        stem = tap_path.stem
        if stem.endswith("_tap"):
            stem = stem[: -len("_tap")]
        token = stem.rsplit("_", 1)[-1]
        if token:
            return token
    return fallback


def infer_flow_dirname(source_ws: Path) -> str | None:
    for flow_dirname in ("flow_waft", "flow_raft"):
        flow_dir = source_ws / flow_dirname
        if flow_dir.is_dir() and any(flow_dir.glob("*.npz")):
            return flow_dirname
    return None


def run_workspace_prepare(
    *,
    scene_name: str,
    out_ws: Path,
    manifest_path: Path,
    num_frames: int,
    force: bool,
    copy_assets: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "prepare_iphone_multiview_workspace.py"),
        "--scene-name",
        scene_name,
        "--out-ws",
        str(out_ws),
        "--manifest",
        str(manifest_path),
        "--num-frames",
        str(num_frames),
    ]
    if force:
        cmd.append("--force")
    if copy_assets:
        cmd.append("--copy-assets")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def frame_token(path: Path) -> str:
    return path.name.split(".", 1)[0]


def prune_directory_to_frames(directory: Path, frame_names: Sequence[str]) -> None:
    if not directory.is_dir():
        return
    keep = {str(frame_name) for frame_name in frame_names}
    for child in directory.iterdir():
        if child.is_file() and frame_token(child) not in keep:
            child.unlink()


def prune_flow_directory_to_frames(directory: Path, frame_names: Sequence[str]) -> None:
    if not directory.is_dir():
        return
    keep = {str(frame_name) for frame_name in frame_names}
    for child in directory.iterdir():
        if not child.is_file() or child.suffix != ".npz":
            continue
        try:
            src_name, dst_name = child.stem.split("_to_", 1)
        except ValueError:
            child.unlink()
            continue
        src_stem = Path(src_name).stem
        dst_stem = Path(dst_name).stem
        if src_stem not in keep or dst_stem not in keep:
            child.unlink()


def trim_tap_tracks(out_ws: Path, time_ids: Sequence[int]) -> None:
    indices = np.asarray([int(time_id) for time_id in time_ids], dtype=np.int64)
    if indices.size == 0:
        return
    for tap_path in sorted(out_ws.glob("*tap.npz")):
        with np.load(tap_path, allow_pickle=True) as data:
            payload: dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                if key in {"tracks", "visibility"} and value.ndim > 0:
                    if value.shape[0] == len(indices):
                        payload[key] = value
                    elif int(indices.max()) < value.shape[0]:
                        payload[key] = value[indices]
                    else:
                        raise ValueError(
                            f"Cannot slice {tap_path.name}:{key} with max index "
                            f"{int(indices.max())} for shape {value.shape}"
                        )
                else:
                    payload[key] = value
        np.savez_compressed(tap_path, **payload)


def trim_anchor_workspace(
    *,
    out_ws: Path,
    frame_names: Sequence[str],
    time_ids: Sequence[int],
) -> None:
    prune_directory_to_frames(out_ws / "images", frame_names)
    prune_directory_to_frames(out_ws / "depthcrafter_depth", frame_names)
    prune_directory_to_frames(out_ws / "depthcrafter_depth_sharp", frame_names)
    prune_directory_to_frames(out_ws / "vggt_omega_depth", frame_names)
    prune_directory_to_frames(out_ws / "depthanything3_depth", frame_names)
    prune_directory_to_frames(out_ws / "sensor_depth", frame_names)
    prune_directory_to_frames(out_ws / "epi" / "error", frame_names)
    prune_flow_directory_to_frames(out_ws / "flow_raft", frame_names)
    prune_flow_directory_to_frames(out_ws / "flow_waft", frame_names)
    trim_tap_tracks(out_ws, time_ids)


def load_saved_camera_intrinsics(summary: dict[str, Any]) -> np.ndarray:
    saved_dir = Path(str(summary.get("saved_dir", ""))).expanduser()
    cam_path = saved_dir / "photometric_cam.pth"
    if cam_path.is_file():
        import torch
        from lib_moca.camera import MonocularCameras

        cams = MonocularCameras.load_from_ckpt(torch.load(cam_path, map_location="cpu"))
        return cams.default_K.detach().cpu().numpy().astype(np.float32)

    width, height = (int(value) for value in summary.get("img_wh", [512, 384]))
    focal = float(min(width, height))
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def write_anchor_camera_recon(
    *,
    out_ws: Path,
    scene_root: Path,
    summary: dict[str, Any],
    num_frames: int,
) -> str:
    c2w_path = scene_root / "cameras" / "cam0_c2w.npz"
    with np.load(c2w_path) as data:
        if "c2w" in data:
            c2w = np.asarray(data["c2w"], dtype=np.float32)
        else:
            c2w = np.asarray(data[data.files[0]], dtype=np.float32)
    c2w = c2w[:num_frames]
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected cam0 c2w shape in {c2w_path}: {c2w.shape}")
    k = load_saved_camera_intrinsics(summary)
    intrinsics = np.asarray([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float32)
    recon_name = "anchor_cam0_recon.npy"
    np.save(out_ws / recon_name, {"traj_c2w": c2w, "intrinsics": intrinsics})
    return recon_name


def main() -> None:
    args = parse_args()
    source_ws = resolve_repo_path(args.source_ws)
    if not source_ws.is_dir():
        raise FileNotFoundError(f"Source workspace not found: {source_ws}")
    scene_name = str(args.scene_name or source_ws.name)
    scene_root = resolve_scene_root(args.benchmark_root, scene_name)
    inference_scene_root = resolve_inference_scene_root(args.inference_root, scene_name)
    out_ws = resolve_repo_path(args.out_ws)
    summary = load_json(scene_root / "metadata" / "scene_summary.json")
    target_cameras = select_target_cameras(
        summary, args.target_cameras, args.num_fps_cameras
    )
    pred_subdir = str(args.pred_subdir or f"pred_rgb_first{args.num_frames or 73}")
    frame_names = load_camera_frame_names(scene_root, "cam0")
    if args.num_frames is not None:
        frame_names = frame_names[: int(args.num_frames)]
    validate_inputs(
        scene_root=scene_root,
        inference_scene_root=inference_scene_root,
        frame_names=frame_names,
        target_cameras=target_cameras,
        pred_subdir=pred_subdir,
        skip_rgb_check=bool(args.skip_rgb_check),
    )

    manifest = build_multiview_manifest(
        scene_name=scene_name,
        source_ws=source_ws,
        scene_root=scene_root,
        inference_scene_root=inference_scene_root,
        summary=summary,
        frame_names=frame_names,
        target_cameras=target_cameras,
        pred_subdir=pred_subdir,
        anchor_weight=float(args.anchor_weight),
        aux_weight=float(args.aux_weight),
        anchor_depth_dirname=args.anchor_depth_dirname,
    )
    manifest_out = (
        resolve_repo_path(args.manifest_out)
        if args.manifest_out is not None
        else out_ws.parent / f"{out_ws.name}_multiview_manifest.json"
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.manifest_only:
        run_workspace_prepare(
            scene_name=scene_name,
            out_ws=out_ws,
            manifest_path=manifest_out,
            num_frames=len(frame_names),
            force=bool(args.force),
            copy_assets=bool(args.copy_assets),
        )
        trim_anchor_workspace(
            out_ws=out_ws,
            frame_names=frame_names,
            time_ids=manifest["time_ids"],
        )
        anchor_recon_path = write_anchor_camera_recon(
            out_ws=out_ws,
            scene_root=scene_root,
            summary=summary,
            num_frames=len(frame_names),
        )
    else:
        anchor_recon_path = None

    cfg_out = (
        resolve_repo_path(args.cfg_out)
        if args.cfg_out is not None
        else out_ws / "reopt_fit.yaml"
    )
    exp_name = str(
        args.exp_name
        or f"{scene_name}_fps{len(target_cameras)}_first{len(frame_names)}_reopt"
    )
    init_camera_recon_path = (
        str(args.init_camera_recon_path)
        if args.init_camera_recon_path is not None
        else anchor_recon_path
    )
    write_reopt_config(
        template_cfg=args.template_cfg,
        cfg_out=cfg_out,
        source_ws=source_ws,
        scene_name=scene_name,
        target_cameras=target_cameras,
        exp_name=exp_name,
        tap_mode=args.tap_mode,
        anchor_weight=float(args.anchor_weight),
        aux_weight=float(args.aux_weight),
        anchor_recon_path=init_camera_recon_path,
        init_anchor_camera=not bool(args.no_anchor_camera_init),
        anchor_depth_dirname=args.anchor_depth_dirname,
        photo_total_steps=args.photo_total_steps,
        allow_camera_ba=bool(args.allow_camera_ba),
        enable_multiview_depth=bool(args.enable_multiview_depth),
        enable_multiview_depth_init=bool(args.enable_multiview_depth_init),
        enable_multiview_track_init=bool(args.enable_multiview_track_init),
        enable_multiview_track_loss=bool(args.enable_multiview_track_loss),
        multiview_use_masks=bool(args.multiview_use_masks),
        multiview_lambda_dep=float(args.multiview_lambda_dep),
        multiview_depth_camera_weight=float(args.multiview_depth_camera_weight),
        multiview_track_camera_weight=float(args.multiview_track_camera_weight),
        reopt_mode=str(args.reopt_mode),
    )

    print(f"Manifest: {manifest_out}")
    print(f"Workspace: {out_ws}")
    print(f"Config: {cfg_out}")
    print(f"Target cameras: {' '.join(target_cameras)}")


if __name__ == "__main__":
    main()
