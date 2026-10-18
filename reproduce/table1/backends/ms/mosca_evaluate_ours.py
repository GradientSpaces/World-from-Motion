from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from data_utils.iphone_helpers import load_iphone_gt_poses
from eval_utils.campose_alignment import align_ate_c2b_use_a2b
from eval_utils.eval_dyncheck import eval_dycheck
from lib_render.render_helper import render
from mosca_viz import load_model_cfg
from wt_compat_common import camera_label, infer_cfg_path, infer_workspace_path


logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


@dataclass(frozen=True)
class TtoSettings:
    eval_tto_mode: str
    static_camera_mode: str
    loss_mask_mode: str
    loss_mask_dir: str | None
    loss_valid_mask_dir: str | None
    joint_track_batch_size: int
    tto_steps: int
    decay_start: int
    lr_p: float
    lr_q: float
    lr_final: float
    initialize_from_previous_step_factor: int
    initialize_from_previous_lr_factor: float
    fg_mask_th: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MoSca iPhone TTO with a WorldTree-style control surface and save "
            "tto_test_pose_<idx>.npz archives for later benchmark export."
        )
    )
    parser.add_argument("--saved_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory that receives tto_test/ previews and tto_test_pose_*.npz. "
            "Defaults to saved_dir."
        ),
    )
    parser.add_argument(
        "--ws",
        type=str,
        default=None,
        help="Workspace override. If omitted, infer it from fit_commandline_args.txt.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config override. If omitted, infer it from fit_commandline_args.txt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--test_camera_index",
        type=int,
        default=None,
        help="Optional test camera index to process. Omit to process all test cameras.",
    )
    parser.add_argument(
        "--test_camera_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional dataset camera IDs to process, e.g. 1 2 5 or cam01 cam02. "
            "Mutually exclusive with --test_camera_index."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--skip_test_gen", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--eval_non_masked", action="store_true")
    parser.add_argument(
        "--save_preview_mp4",
        action="store_true",
        default=True,
        help="Save per-camera TTO preview mp4 files. Enabled by default.",
    )
    parser.add_argument(
        "--no_save_preview_mp4",
        action="store_false",
        dest="save_preview_mp4",
        help="Do not save preview mp4 files.",
    )
    parser.add_argument(
        "--eval_tto_mode",
        type=str,
        default=None,
        choices=["per_frame", "per_test_track_joint"],
        help="WorldTree-style TTO mode override.",
    )
    parser.add_argument(
        "--eval_tto_static_camera_mode",
        type=str,
        default=None,
        choices=["none", "global_first_frame", "interval_first_frame"],
        help="Static-camera mode override for per-frame TTO.",
    )
    parser.add_argument(
        "--eval_tto_static_camera_after_first",
        action="store_true",
        default=None,
        help=(
            "Shorthand for global_first_frame when "
            "--eval_tto_static_camera_mode is not set."
        ),
    )
    parser.add_argument(
        "--no_eval_tto_static_camera_after_first",
        action="store_false",
        dest="eval_tto_static_camera_after_first",
        help="Disable the static-camera-after-first shorthand.",
    )
    parser.add_argument(
        "--eval_tto_loss_mask_mode",
        type=str,
        default=None,
        choices=["none", "direct", "static_from_dynamic"],
        help="WorldTree-style TTO loss mask mode override.",
    )
    parser.add_argument(
        "--eval_tto_loss_mask_dirname",
        type=str,
        default=None,
        help="Mask directory under the workspace when eval_tto_loss_mask_mode=direct.",
    )
    parser.add_argument(
        "--eval_tto_dynamic_mask_dirname",
        type=str,
        default=None,
        help=(
            "Dynamic mask directory under the workspace when "
            "eval_tto_loss_mask_mode=static_from_dynamic."
        ),
    )
    parser.add_argument(
        "--eval_tto_valid_mask_dirname",
        type=str,
        default=None,
        help="Optional valid-mask directory under the workspace for TTO loss masking.",
    )
    parser.add_argument(
        "--eval_tto_joint_track_batch_size",
        type=int,
        default=None,
        help="Batch size for per_test_track_joint TTO. -1 means full track.",
    )
    parser.add_argument("--tto_steps", type=int, default=None)
    parser.add_argument("--tto_decay_start", type=int, default=None)
    parser.add_argument("--tto_lr_p", type=float, default=None)
    parser.add_argument("--tto_lr_q", type=float, default=None)
    parser.add_argument("--tto_lr_final", type=float, default=None)
    parser.add_argument(
        "--tto_initialize_from_previous_step_factor",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tto_initialize_from_previous_lr_factor",
        type=float,
        default=None,
    )
    parser.add_argument("--tto_fg_mask_th", type=float, default=None)
    return parser.parse_args()


def _normalize_camera_id(camera_id: str | int) -> int:
    token = str(camera_id).strip().lower()
    if token.startswith("cam"):
        token = token[3:]
    if not token.isdigit():
        raise ValueError(
            f"Camera ID must be an integer or cam-prefixed integer, got {camera_id!r}"
        )
    return int(token)


def discover_test_camera_ids(ws: str) -> list[int]:
    camera_dir = osp.join(ws, "test_cameras")
    camera_ids = {
        int(name.split("_", 1)[0])
        for name in os.listdir(camera_dir)
        if name.endswith(".json") and name.split("_", 1)[0].isdigit()
    }
    return sorted(camera_ids)


def resolve_test_camera_selection(
    args: argparse.Namespace,
    ws: str,
    num_test_cameras: int,
) -> tuple[list[int], list[int]]:
    test_camera_ids = discover_test_camera_ids(ws)
    if len(test_camera_ids) != num_test_cameras:
        raise ValueError(
            f"Discovered {len(test_camera_ids)} test camera IDs under {ws}/test_cameras "
            f"but loaded {num_test_cameras} test pose groups."
        )

    if args.test_camera_index is not None and args.test_camera_ids is not None:
        raise ValueError("--test_camera_index and --test_camera_ids are mutually exclusive.")

    if args.test_camera_index is not None:
        if args.test_camera_index < 0 or args.test_camera_index >= num_test_cameras:
            raise ValueError(
                f"Invalid test_camera_index={args.test_camera_index}; expected 0 <= index < "
                f"{num_test_cameras}"
            )
        test_indices = [args.test_camera_index]
    elif args.test_camera_ids is not None:
        camera_index_by_id = {camera_id: idx for idx, camera_id in enumerate(test_camera_ids)}
        requested_ids = [_normalize_camera_id(camera_id) for camera_id in args.test_camera_ids]
        requested_ids = [
            camera_id + 1000
            if camera_id not in camera_index_by_id
            and camera_id < 1000
            and camera_id + 1000 in camera_index_by_id
            else camera_id
            for camera_id in requested_ids
        ]
        missing_ids = [camera_id for camera_id in requested_ids if camera_id not in camera_index_by_id]
        if missing_ids:
            raise ValueError(
                "Requested test camera IDs are missing from the workspace: "
                f"{[camera_label(camera_id) for camera_id in missing_ids]}; "
                f"available: {[camera_label(camera_id) for camera_id in test_camera_ids]}"
            )
        test_indices = [camera_index_by_id[camera_id] for camera_id in requested_ids]
    else:
        test_indices = list(range(num_test_cameras))

    selected_camera_ids = [test_camera_ids[idx] for idx in test_indices]
    return test_indices, selected_camera_ids


def log(message: str) -> None:
    print(message, flush=True)


def slice_sequence(seq, start: int, end: int, step: int):
    if step <= 0:
        raise ValueError("--step must be positive")
    if end < 0:
        end = len(seq)
    start = max(start, 0)
    end = min(end, len(seq))
    if start >= end:
        raise ValueError(f"Empty frame range: start={start}, end={end}")
    return seq[start:end:step]


def _ensure_npz_path(path: str | None) -> str | None:
    if path is None:
        return None
    if path.endswith(".npz"):
        return path
    return f"{path}.npz"


def _build_eval_cam_K(
    H: int,
    W: int,
    focal: float,
    cxcy_ratio,
    device: torch.device,
) -> torch.Tensor:
    L = min(H, W)
    fx = focal * L / 2.0
    fy = focal * L / 2.0
    cx = W * cxcy_ratio[0]
    cy = H * cxcy_ratio[1]
    cam_K = torch.eye(3, device=device)
    cam_K[0, 0] = float(fx)
    cam_K[1, 1] = float(fy)
    cam_K[0, 2] = float(cx)
    cam_K[1, 2] = float(cy)
    return cam_K


def _load_binary_mask_png(mask_path: str) -> np.ndarray:
    mask = imageio.imread(mask_path)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 127).astype(np.float32)


def _load_tto_loss_mask(
    *,
    frame_name: str,
    mask_mode: str,
    mask_dir: str | None = None,
    valid_mask_dir: str | None = None,
) -> np.ndarray | None:
    if mask_mode == "none" or mask_dir is None:
        return None

    mask_path = osp.join(mask_dir, f"{frame_name}.png")
    if not osp.exists(mask_path):
        raise FileNotFoundError(f"TTO loss mask not found: {mask_path}")

    primary_mask = _load_binary_mask_png(mask_path)
    if mask_mode == "direct":
        loss_mask = primary_mask
    elif mask_mode == "static_from_dynamic":
        loss_mask = 1.0 - primary_mask
    else:
        raise ValueError(f"Unknown TTO loss mask mode: {mask_mode}")

    if valid_mask_dir is not None:
        valid_mask_path = osp.join(valid_mask_dir, f"{frame_name}.png")
        if not osp.exists(valid_mask_path):
            raise FileNotFoundError(f"TTO valid mask not found: {valid_mask_path}")
        loss_mask = loss_mask * _load_binary_mask_png(valid_mask_path)

    return loss_mask.astype(np.float32)


def _save_rgb_preview(path: str, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8))


def _save_pose_archive(
    path: str | None,
    poses: list[np.ndarray],
    frame_names: list[str],
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "poses": np.asarray(poses, dtype=np.float32),
        "frame_names": np.asarray(frame_names),
    }
    if extra_arrays is not None:
        payload.update(extra_arrays)
    np.savez(path, **payload)


def _coerce_u8_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [
        np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        if frame.dtype != np.uint8
        else frame
        for frame in frames
    ]


def _resolve_mask_dirs(
    cfg,
    args: argparse.Namespace,
    ws: str,
    loss_mask_mode: str,
) -> tuple[str | None, str | None]:
    if loss_mask_mode == "none":
        return None, None

    def _pick_dirname(arg_name: str, cfg_name: str, default=None):
        value = getattr(args, arg_name)
        if value is not None:
            return value
        return getattr(cfg, cfg_name, default)

    if loss_mask_mode == "direct":
        mask_dirname = _pick_dirname(
            "eval_tto_loss_mask_dirname",
            "eval_tto_loss_mask_dirname",
            None,
        )
        if mask_dirname is None:
            raise ValueError(
                "eval_tto_loss_mask_dirname must be set when "
                "eval_tto_loss_mask_mode=direct"
            )
        mask_dir = osp.join(ws, mask_dirname)
        if not osp.isdir(mask_dir):
            raise FileNotFoundError(f"TTO loss mask directory not found: {mask_dir}")
        return mask_dir, None

    if loss_mask_mode == "static_from_dynamic":
        dynamic_dirname = _pick_dirname(
            "eval_tto_dynamic_mask_dirname",
            "eval_tto_dynamic_mask_dirname",
            "test_dynamic",
        )
        valid_dirname = _pick_dirname(
            "eval_tto_valid_mask_dirname",
            "eval_tto_valid_mask_dirname",
            "test_covisible",
        )
        mask_dir = osp.join(ws, dynamic_dirname)
        if not osp.isdir(mask_dir):
            raise FileNotFoundError(f"TTO loss mask directory not found: {mask_dir}")
        valid_dir = osp.join(ws, valid_dirname) if valid_dirname else None
        if valid_dir is not None and not osp.isdir(valid_dir):
            raise FileNotFoundError(f"TTO valid mask directory not found: {valid_dir}")
        return mask_dir, valid_dir

    raise ValueError(f"Unknown eval_tto_loss_mask_mode={loss_mask_mode}")


def resolve_tto_settings(
    cfg,
    args: argparse.Namespace,
    ws: str,
) -> TtoSettings:
    def _pick(arg_name: str, cfg_name: str, default):
        value = getattr(args, arg_name)
        if value is not None:
            return value
        return getattr(cfg, cfg_name, default)

    eval_tto_mode = _pick("eval_tto_mode", "eval_tto_mode", "per_frame")
    explicit_static_mode = args.eval_tto_static_camera_mode
    if explicit_static_mode is not None:
        static_camera_mode = explicit_static_mode
    else:
        static_camera_mode = getattr(cfg, "eval_tto_static_camera_mode", None)
    static_after_first = args.eval_tto_static_camera_after_first
    if static_after_first is None:
        static_after_first = getattr(cfg, "eval_tto_static_camera_after_first", False)
    if static_camera_mode is None:
        static_camera_mode = "global_first_frame" if static_after_first else "none"
    if static_camera_mode == "interval_first_frame":
        log(
            "Single-run MoSca evaluator aliases eval_tto_static_camera_mode="
            "interval_first_frame to global_first_frame."
        )
        static_camera_mode = "global_first_frame"

    loss_mask_mode = _pick(
        "eval_tto_loss_mask_mode",
        "eval_tto_loss_mask_mode",
        "none",
    )
    joint_track_batch_size = _pick(
        "eval_tto_joint_track_batch_size",
        "eval_tto_joint_track_batch_size",
        -1,
    )
    tto_steps = int(_pick("tto_steps", "tto_steps", 30))
    decay_start = int(_pick("tto_decay_start", "tto_decay_start", 15))
    lr_p = float(_pick("tto_lr_p", "tto_lr_p", 0.003))
    lr_q = float(_pick("tto_lr_q", "tto_lr_q", 0.003))
    lr_final = float(_pick("tto_lr_final", "tto_lr_final", 0.0001))
    init_step_factor = int(
        _pick(
            "tto_initialize_from_previous_step_factor",
            "tto_initialize_from_previous_step_factor",
            10,
        )
    )
    init_lr_factor = float(
        _pick(
            "tto_initialize_from_previous_lr_factor",
            "tto_initialize_from_previous_lr_factor",
            0.1,
        )
    )
    fg_mask_th = float(_pick("tto_fg_mask_th", "tto_fg_mask_th", 0.1))

    if eval_tto_mode == "per_test_track_joint":
        if static_camera_mode != "none":
            raise ValueError(
                "eval_tto_static_camera_mode is only supported for per_frame TTO "
                "in the single-run MoSca evaluator."
            )
        if loss_mask_mode != "none":
            raise ValueError(
                "eval_tto_loss_mask_mode is only supported for per_frame TTO in "
                "the single-run MoSca evaluator."
            )

    loss_mask_dir, loss_valid_mask_dir = _resolve_mask_dirs(
        cfg, args, ws, loss_mask_mode
    )
    return TtoSettings(
        eval_tto_mode=eval_tto_mode,
        static_camera_mode=static_camera_mode,
        loss_mask_mode=loss_mask_mode,
        loss_mask_dir=loss_mask_dir,
        loss_valid_mask_dir=loss_valid_mask_dir,
        joint_track_batch_size=joint_track_batch_size,
        tto_steps=tto_steps,
        decay_start=decay_start,
        lr_p=lr_p,
        lr_q=lr_q,
        lr_final=lr_final,
        initialize_from_previous_step_factor=init_step_factor,
        initialize_from_previous_lr_factor=init_lr_factor,
        fg_mask_th=fg_mask_th,
    )


def render_test_tto_single_run(
    *,
    H: int,
    W: int,
    cams,
    s_model,
    d_model,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    gt_rgb_dir: str,
    save_pose_fn: str | None,
    static_camera_pose_fn: str | None,
    fn_list: list[str],
    focal: float,
    cxcy_ratio,
    settings: TtoSettings,
    save_dir: str | None,
) -> list[np.ndarray]:
    if len(test_camera_tid) == 0:
        raise ValueError("test_camera_tid must not be empty")

    device = s_model.device
    s_model.eval()
    d_model.eval()

    with torch.no_grad():
        solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=test_camera_T_wi,
        )

    cam_K = _build_eval_cam_K(H, W, focal, cxcy_ratio, device)
    pose_save_path = _ensure_npz_path(save_pose_fn)
    static_camera_pose_path = _ensure_npz_path(static_camera_pose_fn)
    fixed_T_cw = None
    if (
        settings.static_camera_mode == "global_first_frame"
        and static_camera_pose_path is not None
        and osp.exists(static_camera_pose_path)
    ):
        fixed_pose_data = np.load(static_camera_pose_path)
        fixed_T_cw = torch.from_numpy(fixed_pose_data["pose"]).float().to(device)

    def _render_pred_rgb(working_t: int, T_cw: torch.Tensor):
        with torch.no_grad():
            render_dict = render([s_model(), d_model(working_t)], H, W, cam_K, T_cw=T_cw)
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            rendered_mask = render_dict["alpha"].squeeze(-1).squeeze(0) > settings.fg_mask_th
        return pred_rgb, rendered_mask

    def _build_optim_mask(
        rendered_mask: torch.Tensor,
        gt_mask_bool: torch.Tensor,
        frame_name: str,
    ) -> torch.Tensor:
        if settings.loss_mask_mode == "none":
            optim_mask = rendered_mask
        else:
            optim_mask = torch.logical_and(rendered_mask, gt_mask_bool)
        if int(optim_mask.sum().item()) == 0:
            raise RuntimeError(
                f"TTO loss mask for frame {frame_name} has zero valid pixels after "
                "intersecting with the rendered foreground"
            )
        return optim_mask

    test_ret: list[np.ndarray] = []
    solved_pose_list: list[np.ndarray] = []
    for idx in range(len(test_camera_tid)):
        if idx == 0:
            step_factor = settings.initialize_from_previous_step_factor
            lr_factor = 1.0
        else:
            step_factor = 1
            lr_factor = settings.initialize_from_previous_lr_factor

        frame_name = fn_list[idx]
        working_t = int(test_camera_tid[idx])
        gt_rgb = imageio.imread(osp.join(gt_rgb_dir, f"{frame_name}.png")) / 255.0
        gt_rgb = torch.tensor(gt_rgb[..., :3], device=device).float()
        gt_mask = _load_tto_loss_mask(
            frame_name=frame_name,
            mask_mode=settings.loss_mask_mode,
            mask_dir=settings.loss_mask_dir,
            valid_mask_dir=settings.loss_valid_mask_dir,
        )
        if gt_mask is None:
            gt_mask = np.ones_like(gt_rgb[..., 0].detach().cpu().numpy())
        gt_mask_tensor = torch.tensor(gt_mask, device=device).float()
        gt_mask_bool = gt_mask_tensor > 0.5
        if int(gt_mask_bool.sum().item()) == 0:
            raise RuntimeError(
                f"TTO loss mask for frame {frame_name} is empty before rendering"
            )

        if settings.static_camera_mode != "none" and fixed_T_cw is not None:
            pred_rgb, rendered_mask = _render_pred_rgb(working_t, fixed_T_cw)
            optim_mask = _build_optim_mask(rendered_mask, gt_mask_bool, frame_name)
            mse = ((pred_rgb - gt_rgb) ** 2)[optim_mask].mean()
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            rgb = np.clip(pred_rgb.detach().cpu().numpy(), 0.0, 1.0)
            test_ret.append(rgb)
            solved_pose_list.append(fixed_T_cw.detach().cpu().numpy())
            if save_dir is not None:
                _save_rgb_preview(osp.join(save_dir, f"{frame_name}.png"), rgb)
            log(f"TTO_STATIC {frame_name}: {psnr.item():.3f}")
            continue

        T_cw_init = torch.linalg.inv(aligned_test_camera_T_wi[idx]).to(device)
        T_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        t_init = torch.nn.Parameter(T_cw_init[:3, 3].detach())
        q_init = torch.nn.Parameter(matrix_to_quaternion(T_cw_init[:3, :3]).detach())
        optimizer = torch.optim.Adam(
            [
                {"params": t_init, "lr": settings.lr_p * lr_factor},
                {"params": q_init, "lr": settings.lr_q * lr_factor},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, settings.tto_steps * step_factor - settings.decay_start),
            eta_min=settings.lr_final * lr_factor,
        )

        loss_list = []
        for step in range(settings.tto_steps * step_factor):
            optimizer.zero_grad()
            _T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
            T_cw = torch.cat([_T_cw, T_bottom[None]], 0)
            render_dict = render([s_model(), d_model(working_t)], H, W, cam_K, T_cw=T_cw)
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            rendered_mask = render_dict["alpha"].squeeze(-1).squeeze(0) > settings.fg_mask_th
            optim_mask = _build_optim_mask(rendered_mask, gt_mask_bool, frame_name)
            mse = ((pred_rgb - gt_rgb) ** 2)[optim_mask].mean()
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            loss = -psnr
            loss.backward()
            optimizer.step()
            if step >= settings.decay_start:
                scheduler.step()
            loss_list.append(loss.item())

        solved_T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
        solved_T_cw = torch.cat([solved_T_cw, T_bottom[None]], 0)
        if settings.static_camera_mode != "none" and fixed_T_cw is None:
            fixed_T_cw = solved_T_cw.detach().clone()
            if static_camera_pose_path is not None:
                Path(static_camera_pose_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    static_camera_pose_path,
                    pose=fixed_T_cw.detach().cpu().numpy(),
                    source_fn=frame_name,
                )
        render_pose = fixed_T_cw if fixed_T_cw is not None else solved_T_cw
        solved_pose_list.append(render_pose.detach().cpu().numpy())
        pred_rgb, _ = _render_pred_rgb(working_t, render_pose)
        rgb = np.clip(pred_rgb.detach().cpu().numpy(), 0.0, 1.0)
        test_ret.append(rgb)
        if save_dir is not None:
            _save_rgb_preview(osp.join(save_dir, f"{frame_name}.png"), rgb)
        if loss_list:
            log(f"TTO {frame_name}: {loss_list[0]:.3f}->{loss_list[-1]:.3f}")
        if (
            settings.static_camera_mode == "none"
            and idx < len(test_camera_tid) - 1
        ):
            aligned_test_camera_T_wi[idx + 1] = torch.linalg.inv(solved_T_cw).to(
                aligned_test_camera_T_wi
            )

    extra_arrays = {}
    if fixed_T_cw is not None:
        extra_arrays["shared_pose"] = fixed_T_cw.detach().cpu().numpy().astype(np.float32)
    _save_pose_archive(pose_save_path, solved_pose_list, fn_list, extra_arrays)
    return test_ret


def render_test_tto_per_test_track_joint_single_run(
    *,
    H: int,
    W: int,
    cams,
    s_model,
    d_model,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    gt_rgb_dir: str,
    save_pose_fn: str | None,
    fn_list: list[str],
    focal: float,
    cxcy_ratio,
    settings: TtoSettings,
    save_dir: str | None,
) -> list[np.ndarray]:
    if len(test_camera_tid) == 0:
        raise ValueError("test_camera_tid must not be empty")

    device = s_model.device
    with torch.no_grad():
        solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi.detach().cpu().clone(),
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=test_camera_T_wi.detach().cpu().clone(),
        )

    cam_K = _build_eval_cam_K(H, W, focal, cxcy_ratio, device)
    pose_save_path = _ensure_npz_path(save_pose_fn)

    def _compute_psnr_loss(pred_rgb, gt_rgb, rendered_mask):
        sq_err = (pred_rgb - gt_rgb) ** 2
        if rendered_mask.any().item():
            mse = sq_err[rendered_mask].mean()
        else:
            mse = sq_err.mean()
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return -psnr

    samples = []
    for sample_idx, global_t in enumerate(test_camera_tid):
        working_t = int(global_t)
        if working_t < 0 or working_t >= cams.T:
            raise ValueError(
                f"Test frame {fn_list[sample_idx]} mapped to invalid model time {working_t}"
            )
        gt_rgb = imageio.imread(osp.join(gt_rgb_dir, f"{fn_list[sample_idx]}.png")) / 255.0
        gt_rgb = torch.tensor(gt_rgb[..., :3], device=device).float()
        samples.append(
            {
                "working_t": working_t,
                "fn": fn_list[sample_idx],
                "gt_rgb": gt_rgb,
            }
        )

    T_cw_init = torch.linalg.inv(aligned_test_camera_T_wi[0]).to(device)
    T_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    t_init = torch.nn.Parameter(T_cw_init[:3, 3].detach())
    q_init = torch.nn.Parameter(matrix_to_quaternion(T_cw_init[:3, :3]).detach())
    optimizer = torch.optim.Adam(
        [
            {"params": t_init, "lr": settings.lr_p},
            {"params": q_init, "lr": settings.lr_q},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, settings.tto_steps - settings.decay_start),
        eta_min=settings.lr_final,
    )

    effective_batch_size = len(samples)
    if settings.joint_track_batch_size > 0:
        effective_batch_size = min(settings.joint_track_batch_size, len(samples))

    loss_list = []
    for step in range(settings.tto_steps):
        optimizer.zero_grad()
        if effective_batch_size >= len(samples):
            step_indices = list(range(len(samples)))
        else:
            step_indices = torch.randperm(len(samples))[:effective_batch_size].tolist()
        step_loss_total = 0.0
        batch_norm = float(len(step_indices))
        for sample_idx in step_indices:
            sample = samples[sample_idx]
            T_cw = torch.cat(
                [
                    torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1),
                    T_bottom[None],
                ],
                0,
            )
            render_dict = render([s_model(), d_model(sample["working_t"])], H, W, cam_K, T_cw=T_cw)
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            rendered_mask = render_dict["alpha"].squeeze(-1).squeeze(0) > settings.fg_mask_th
            rgb_loss = _compute_psnr_loss(pred_rgb, sample["gt_rgb"], rendered_mask)
            (rgb_loss / batch_norm).backward()
            step_loss_total += rgb_loss.detach().item()
        optimizer.step()
        if step >= settings.decay_start:
            scheduler.step()
        loss_list.append(step_loss_total / batch_norm)

    solved_T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
    solved_T_cw = torch.cat([solved_T_cw, T_bottom[None]], 0)

    test_ret = []
    solved_pose = solved_T_cw.detach().cpu().numpy().astype(np.float32)
    for sample in samples:
        with torch.no_grad():
            render_dict = render(
                [s_model(), d_model(sample["working_t"])],
                H,
                W,
                cam_K,
                T_cw=solved_T_cw,
            )
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
        rgb = np.clip(pred_rgb.detach().cpu().numpy(), 0.0, 1.0)
        test_ret.append(rgb)
        if save_dir is not None:
            _save_rgb_preview(osp.join(save_dir, f"{sample['fn']}.png"), rgb)

    repeated_poses = np.repeat(solved_pose[None], len(samples), axis=0)
    _save_pose_archive(
        pose_save_path,
        list(repeated_poses),
        fn_list,
        {"shared_pose": solved_pose},
    )
    if loss_list:
        log(
            f"TTO_SHARED_TRACK {fn_list[0]}->{fn_list[-1]} "
            f"({len(samples)} frames): {loss_list[0]:.3f}->{loss_list[-1]:.3f}"
        )
    return test_ret


def main() -> None:
    args = parse_args()
    saved_dir = osp.abspath(args.saved_dir.rstrip("/"))
    cfg_path = infer_cfg_path(saved_dir, args.cfg)
    ws = infer_workspace_path(saved_dir, args.ws)
    output_dir = osp.abspath(args.output_dir) if args.output_dir is not None else saved_dir
    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg, d_model, s_model, cams = load_model_cfg(cfg_path, saved_dir, device=device)

    dataset_mode = getattr(cfg, "mode", getattr(cfg, "dataset_mode", "iphone"))
    if dataset_mode not in {"iphone", "iphone_ours_v3"}:
        raise ValueError(
            f"mosca_evaluate_ours.py only supports iPhone-style datasets, got {dataset_mode}"
        )

    settings = resolve_tto_settings(cfg, args, ws)
    test_preview_dir = osp.join(output_dir, "tto_test")
    if not args.skip_test_gen:
        shutil.rmtree(test_preview_dir, ignore_errors=True)
        Path(test_preview_dir).mkdir(parents=True, exist_ok=True)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        _gt_training_fov,
        gt_testing_fov_list,
        _,
        gt_testing_cxcy_ratio_list,
    ) = load_iphone_gt_poses(ws, getattr(cfg, "t_subsample", 1))
    gt_dir = osp.join(ws, "test_images")

    test_indices, selected_camera_ids = resolve_test_camera_selection(
        args,
        ws,
        len(gt_testing_cam_T_wi_list),
    )

    sample_fn = [
        f for f in os.listdir(gt_dir)
        if f.endswith(".png") or f.endswith(".jpg")
    ][0]
    sample = imageio.imread(osp.join(gt_dir, sample_fn))
    H, W = sample.shape[:2]

    log(f"saved_dir: {saved_dir}")
    log(f"output_dir: {output_dir}")
    log(f"ws: {ws}")
    log(f"cfg_path: {cfg_path}")
    log(f"device: {device}")
    log(f"eval_tto_mode: {settings.eval_tto_mode}")
    log(f"static_camera_mode: {settings.static_camera_mode}")
    log(f"loss_mask_mode: {settings.loss_mask_mode}")
    log(f"test_camera_indices: {test_indices}")
    log(f"test_cameras: {[camera_label(camera_id) for camera_id in selected_camera_ids]}")

    if not args.skip_test_gen:
        for test_i in test_indices:
            time_ids = slice_sequence(
                gt_testing_tids_list[test_i],
                args.start,
                args.end,
                args.step,
            )
            test_camera_T_wi = slice_sequence(
                gt_testing_cam_T_wi_list[test_i],
                args.start,
                args.end,
                args.step,
            )
            frame_names = slice_sequence(
                gt_testing_fns_list[test_i],
                args.start,
                args.end,
                args.step,
            )
            testing_fov = gt_testing_fov_list[test_i]
            testing_focal = 1.0 / np.tan(np.deg2rad(testing_fov) / 2.0)
            cxcy_ratio = gt_testing_cxcy_ratio_list[test_i]
            pose_path = osp.join(output_dir, f"tto_test_pose_{test_i}")
            static_pose_path = osp.join(output_dir, f"tto_static_test_pose_{test_i}")

            if settings.eval_tto_mode == "per_test_track_joint":
                preview_frames = render_test_tto_per_test_track_joint_single_run(
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=test_camera_T_wi,
                    test_camera_tid=time_ids,
                    gt_rgb_dir=gt_dir,
                    save_pose_fn=pose_path,
                    fn_list=frame_names,
                    focal=testing_focal,
                    cxcy_ratio=cxcy_ratio,
                    settings=settings,
                    save_dir=test_preview_dir,
                )
            else:
                preview_frames = render_test_tto_single_run(
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=test_camera_T_wi,
                    test_camera_tid=time_ids,
                    gt_rgb_dir=gt_dir,
                    save_pose_fn=pose_path,
                    static_camera_pose_fn=static_pose_path,
                    fn_list=frame_names,
                    focal=testing_focal,
                    cxcy_ratio=cxcy_ratio,
                    settings=settings,
                    save_dir=test_preview_dir,
                )

            if args.save_preview_mp4:
                imageio.mimsave(
                    osp.join(output_dir, f"tto_test_cam{test_i}.mp4"),
                    _coerce_u8_frames(preview_frames),
                )

    if args.skip_metrics:
        log("Skipping DyCheck metrics because --skip_metrics was set.")
        return

    pred_dir = osp.join(output_dir, "tto_test")
    if not osp.isdir(pred_dir):
        log("Skipping DyCheck metrics because the tto_test preview directory is missing.")
        return

    strict_eval_all_gt = (
        args.test_camera_index is None
        and args.test_camera_ids is None
        and args.start == 0
        and args.end == -1
        and args.step == 1
    )
    eval_dycheck(
        save_dir=output_dir,
        gt_rgb_dir=gt_dir,
        gt_mask_dir=osp.join(ws, "test_covisible"),
        pred_dir=pred_dir,
        save_prefix="tto_",
        strict_eval_all_gt_flag=strict_eval_all_gt,
        eval_non_masked=args.eval_non_masked,
        dynamic_mask_dir=osp.join(ws, "test_dynamic"),
    )


if __name__ == "__main__":
    main()
