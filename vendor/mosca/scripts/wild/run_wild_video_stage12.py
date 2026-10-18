#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def force_float32_precision() -> None:
    import torch

    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_autocast_enabled(False)
    except TypeError:
        pass
    try:
        torch.set_autocast_enabled("cuda", False)
    except (AttributeError, TypeError, RuntimeError):
        pass


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def list_image_filenames(image_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        return []
    return sorted(
        fn.name for fn in image_dir.iterdir() if fn.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_frame(frame: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    import cv2

    if target_height <= 0 and target_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if target_height <= 0:
        target_height = int(round(height * target_width / width))
    if target_width <= 0:
        target_width = int(round(width * target_height / height))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def extract_video_frames(
    video_path: Path,
    image_dir: Path,
    *,
    stride: int,
    max_frames: int | None,
    require_max_frames: bool,
    pad_short_to_max_frames: bool,
    target_height: int,
    target_width: int,
    overwrite: bool,
) -> tuple[list[str], list[int], float]:
    import cv2
    import imageio.v2 as imageio

    existing = list_image_filenames(image_dir)
    if existing and not overwrite:
        if require_max_frames and max_frames is not None and len(existing) != int(max_frames):
            raise RuntimeError(
                f"Existing frame directory {image_dir} has {len(existing)} frames; "
                f"expected exactly {int(max_frames)}. Pass --overwrite to restage."
            )
        return existing, list(range(len(existing))), 0.0
    if image_dir.exists() and overwrite:
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if (
        require_max_frames
        and max_frames is not None
        and total_frames > 0
        and not pad_short_to_max_frames
    ):
        needed_source = int(max_frames) * int(stride)
        if total_frames < needed_source:
            raise RuntimeError(
                f"{video_path} has {total_frames} source frames, but this run requires "
                f"at least {needed_source} source frames for {int(max_frames)} frames "
                f"at stride {int(stride)}."
            )

    frame_names: list[str] = []
    source_indices: list[int] = []
    source_index = 0
    output_index = 0
    while source_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = resize_frame(frame_rgb, target_height, target_width)
        frame_name = f"{output_index:05d}.png"
        imageio.imwrite(image_dir / frame_name, frame_rgb)
        frame_names.append(frame_name)
        source_indices.append(source_index)
        output_index += 1
        if max_frames is not None and output_index >= max_frames:
            break
        source_index += stride
    cap.release()

    if not frame_names:
        raise RuntimeError(f"No frames extracted from {video_path}")
    if require_max_frames and max_frames is not None and len(frame_names) != int(max_frames):
        if pad_short_to_max_frames and len(frame_names) < int(max_frames):
            last_frame = imageio.imread(image_dir / frame_names[-1])
            last_source_index = source_indices[-1]
            while len(frame_names) < int(max_frames):
                frame_name = f"{len(frame_names):05d}.png"
                imageio.imwrite(image_dir / frame_name, last_frame)
                frame_names.append(frame_name)
                source_indices.append(last_source_index)
            return frame_names, source_indices, fps
        needed_source = int(max_frames) * int(stride)
        raise RuntimeError(
            f"Extracted {len(frame_names)} frames from {video_path}, expected exactly "
            f"{int(max_frames)} with stride {int(stride)}. Need at least "
            f"{needed_source} readable source frames."
        )
    return frame_names, source_indices, fps


def extract_image_folder_frames(
    source_dir: Path,
    image_dir: Path,
    *,
    stride: int,
    max_frames: int | None,
    require_max_frames: bool,
    pad_short_to_max_frames: bool,
    target_height: int,
    target_width: int,
    overwrite: bool,
) -> tuple[list[str], list[int], float]:
    import imageio.v2 as imageio

    existing = list_image_filenames(image_dir)
    if existing and not overwrite:
        if require_max_frames and max_frames is not None and len(existing) != int(max_frames):
            raise RuntimeError(
                f"Existing frame directory {image_dir} has {len(existing)} frames; "
                f"expected exactly {int(max_frames)}. Pass --overwrite to restage."
            )
        return existing, list(range(len(existing))), 0.0
    if image_dir.exists() and overwrite:
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    source_names = list_image_filenames(source_dir)
    if not source_names:
        raise RuntimeError(f"No images found in {source_dir}")
    selected_indices = list(range(0, len(source_names), stride))
    if max_frames is not None:
        selected_indices = selected_indices[: int(max_frames)]
    if not selected_indices:
        raise RuntimeError(f"No frames selected from {source_dir}")
    if (
        require_max_frames
        and max_frames is not None
        and len(selected_indices) < int(max_frames)
    ):
        if not pad_short_to_max_frames:
            raise RuntimeError(
                f"Selected {len(selected_indices)} frames from {source_dir}, expected "
                f"exactly {int(max_frames)} with stride {int(stride)}."
            )
        selected_indices.extend(
            [selected_indices[-1]] * (int(max_frames) - len(selected_indices))
        )

    frame_names: list[str] = []
    source_indices: list[int] = []
    for output_index, source_index in enumerate(selected_indices):
        frame = imageio.imread(source_dir / source_names[source_index])
        frame = resize_frame(frame, target_height, target_width)
        frame_name = f"{output_index:05d}.png"
        imageio.imwrite(image_dir / frame_name, frame)
        frame_names.append(frame_name)
        source_indices.append(source_index)

    return frame_names, source_indices, 0.0


def write_manifest(
    ws: Path,
    *,
    video_path: Path,
    frame_names: list[str],
    source_indices: list[int],
    fps: float,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "source_video": str(video_path),
        "source_type": "image_folder" if video_path.is_dir() else "video",
        "workspace": str(ws),
        "rgb_dirname": args.rgb_dirname,
        "num_frames": len(frame_names),
        "frame_names": frame_names,
        "source_frame_indices": source_indices,
        "source_fps": fps,
        "frame_stride": args.frame_stride,
        "source_crop_frames": None if not source_indices else int(max(source_indices)) + 1,
        "depth_mode": args.depth_mode,
        "target_height": args.target_height,
        "target_width": args.target_width,
    }
    with open(ws / "wild_stage12_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_da3(
    ws: Path,
    *,
    rgb_dirname: str,
    device: str,
    weights: str,
) -> None:
    import torch

    force_float32_precision()
    from lib_prior.depth_models.depthanything3_wrapper import (
        depthanything3_process_folder,
        get_depthanything3_model,
    )
    from mosca_precompute import load_imgs_from_dir

    img_list, img_fns = load_imgs_from_dir(str(ws), rgb_dirname=rgb_dirname)
    model = get_depthanything3_model(device=device, weights=weights)
    try:
        depthanything3_process_folder(
            model=model,
            img_list=img_list,
            fn_list=img_fns,
            dst=str(ws),
        )
    finally:
        del model
        torch.cuda.empty_cache()


def run_vggt_omega(ws: Path, args: argparse.Namespace) -> None:
    import torch

    force_float32_precision()
    from lib_prior.depth_models.vggt_omega_wrapper import (
        get_vggt_omega_model,
        vggt_omega_process_folder,
    )
    from mosca_precompute import load_imgs_from_dir

    img_list, img_fns = load_imgs_from_dir(str(ws), rgb_dirname=args.rgb_dirname)
    model = get_vggt_omega_model(
        device=args.vggt_omega_device,
        checkpoint_path=args.vggt_omega_checkpoint,
    )
    try:
        vggt_omega_process_folder(
            model=model,
            img_list=img_list,
            fn_list=img_fns,
            dst=str(ws),
            image_resolution=int(args.vggt_omega_image_resolution),
            preprocess_mode=str(args.vggt_omega_preprocess_mode),
            output_prefix="vggt_omega",
            save_depth_confidence=not args.skip_vggt_omega_confidence,
            save_pointmaps=bool(args.save_vggt_omega_pointmaps),
        )
    finally:
        del model
        torch.cuda.empty_cache()


def default_recon_name(depth_mode: str) -> str:
    if depth_mode == "vggt_omega":
        return "vggt_omega_recon.npy"
    if depth_mode == "depthanything3":
        return "depthanything3_recon.npy"
    raise ValueError(f"Unsupported depth mode: {depth_mode}")


def build_prep_cfg(args: argparse.Namespace):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.prep_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.prep_overrides))
    cfg.dep_mode = args.depth_mode
    cfg.tap_mode = "cowtracker"
    cfg.dynamic_mask_mode = args.dynamic_mask_mode
    cfg.compute_flow = bool(args.compute_flow or args.dynamic_mask_mode == "raft_epi")
    cfg.flow_mode = args.flow_mode if (cfg.compute_flow or args.flow_mode == "waft") else "none"
    cfg.rgb_dirname = args.rgb_dirname
    if args.dynamic_mask_mode == "sam3":
        if not args.sam3_text:
            raise ValueError("--sam3_text is required when --dynamic_mask_mode=sam3")
        cfg.sam3_text = args.sam3_text
    if args.sam3_gpus is not None:
        cfg.sam3_gpus = args.sam3_gpus
    if args.sam3_checkpoint is not None:
        cfg.sam3_checkpoint = args.sam3_checkpoint
    if args.sam3_bpe is not None:
        cfg.sam3_bpe = args.sam3_bpe
    if args.sam3_lib_prior_root is not None:
        cfg.sam3_lib_prior_root = args.sam3_lib_prior_root
    if args.sam3_fps is not None:
        cfg.sam3_fps = args.sam3_fps
    if args.cowtracker_checkpoint is not None:
        cfg.cowtracker_checkpoint = args.cowtracker_checkpoint
    if args.waft_checkpoint is not None:
        cfg.flow_checkpoint = args.waft_checkpoint
    return cfg


def run_precompute(ws: Path, args: argparse.Namespace) -> None:
    import gc
    import torch

    from mosca_precompute import get_moca_processor, load_imgs_from_dir, preprocess

    force_float32_precision()
    cfg = build_prep_cfg(args)
    img_list, img_fns = load_imgs_from_dir(str(ws), rgb_dirname=args.rgb_dirname)
    processor = get_moca_processor(cfg)
    try:
        preprocess(
            img_list=img_list,
            img_fns=img_fns,
            ws=str(ws),
            moca_processor=processor,
            pre_cfg=cfg,
            resample_for_dynamic=not args.skip_dynamic_resample,
        )
    finally:
        del processor
        gc.collect()
        torch.cuda.empty_cache()


def run_sam3_only(ws: Path, args: argparse.Namespace) -> None:
    from lib_prior.moca_processor import normalize_sam3_prompts
    from lib_prior.preprocessor_utils import seed_everything
    from lib_prior.sam3_wrapper import generate_sam3_epi, save_sam3_epi_artifacts
    from mosca_precompute import load_imgs_from_dir
    from wfm.masks import union_mask_sequences

    force_float32_precision()
    cfg = build_prep_cfg(args)
    if cfg.dynamic_mask_mode != "sam3":
        raise ValueError("--rerun_sam3_only requires --dynamic_mask_mode=sam3")
    prompt_list = normalize_sam3_prompts(cfg.sam3_text)
    if len(prompt_list) == 0:
        raise ValueError("SAM3 text prompt is required when rerunning SAM3 masks")

    img_list, img_fns = load_imgs_from_dir(str(ws), rgb_dirname=args.rgb_dirname)
    seed_everything(12345)

    combined_mask_list = None
    for prompt in prompt_list:
        print(f"Running SAM3-only prompt: {prompt}")
        mask_list = generate_sam3_epi(
            save_dir=str(ws),
            img_name_list=img_fns,
            img_list=img_list,
            text_prompt=prompt,
            gpus=getattr(cfg, "sam3_gpus", None),
            checkpoint_path=getattr(cfg, "sam3_checkpoint", None),
            bpe_path=getattr(cfg, "sam3_bpe", None),
            sam3_lib_prior_root=getattr(cfg, "sam3_lib_prior_root", None),
            fps=getattr(cfg, "sam3_fps", 10.0),
            save_mask_png=len(prompt_list) == 1,
            seed_frame_indices=getattr(cfg, "sam3_seed_frame_indices", None),
        )
        combined_mask_list = union_mask_sequences(combined_mask_list, mask_list)
    if len(prompt_list) > 1:
        save_sam3_epi_artifacts(
            save_dir=str(ws),
            img_name_list=img_fns,
            img_list=img_list,
            mask_list=combined_mask_list,
            fps=getattr(cfg, "sam3_fps", 10.0),
        )


def build_fit_cfg(args: argparse.Namespace):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.fit_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.fit_overrides))
    cfg.mode = getattr(cfg, "mode", "wild")
    cfg.depth_dirname = args.depth_dirname
    cfg.tap_mode = "cowtracker"
    cfg.init_da3_camera = True
    cfg.da3_recon_path = args.init_recon_path or default_recon_name(args.depth_mode)
    if bool(args.compute_flow or args.dynamic_mask_mode == "raft_epi" or args.flow_mode == "waft"):
        cfg.flow_dirname = f"flow_{args.flow_mode}"
    if not args.preserve_fit_camera_optimization:
        cfg.ba_total_steps = int(args.ba_total_steps)
        cfg.ba_lr_cam_f = 0.0
        cfg.ba_lr_cam_q = 0.0
        cfg.ba_lr_cam_t = 0.0
        cfg.ba_lr_dep_c = 0.0
        cfg.ba_lr_dep_s = 0.0
        cfg.photo_lr_cam_f = 0.0
        cfg.photo_lr_cam_q = 0.0
        cfg.photo_lr_cam_t = 0.0
        cfg.photo_warm_lr_cam_f = 0.0
        cfg.photo_warm_lr_cam_q = 0.0
        cfg.photo_warm_lr_cam_t = 0.0
    return cfg


def run_reconstruct(ws: Path, args: argparse.Namespace) -> str:
    from lite_moca_reconstruct import static_reconstruct
    from mosca_reconstruct import (
        photometric_reconstruct,
        photometric_warmup,
        scaffold_reconstruct,
    )
    from recon_utils import setup_recon_ws

    force_float32_precision()
    cfg = build_fit_cfg(args)
    logdir = setup_recon_ws(str(ws), fit_cfg=cfg)
    static_reconstruct(str(ws), logdir, cfg)
    photometric_warmup(str(ws), logdir, cfg)
    scaffold_reconstruct(str(ws), logdir, cfg)
    photometric_reconstruct(str(ws), logdir, cfg)
    return logdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wild-video/image-folder depth, flow, CoWTracker precompute and MoSca reconstruction."
    )
    parser.add_argument("--video", required=True, help="Input wild/generated MP4 or image folder.")
    parser.add_argument("--ws", required=True, help="Output MoSca workspace.")
    parser.add_argument("--sam3_text", default=None, help="SAM3 dynamic object prompt.")
    parser.add_argument(
        "--dynamic_mask_mode",
        default="sam3",
        choices=["sam3", "raft_epi"],
        help="Dynamic mask source for precompute.",
    )
    parser.add_argument("--prep_cfg", default="profile/wild/wild_da3_sam3_prep.yaml")
    parser.add_argument("--fit_cfg", default="profile/wild/wild_da3_fixed_fit.yaml")
    parser.add_argument("--rgb_dirname", default="images")
    parser.add_argument(
        "--depth_mode",
        default="depthanything3",
        choices=["depthanything3", "vggt_omega"],
        help="Depth/pose prior to materialize before MoSca precompute.",
    )
    parser.add_argument("--depth_dirname", default=None)
    parser.add_argument(
        "--init_recon_path",
        default=None,
        help="Workspace-relative camera recon .npy used to initialize MoSca.",
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--require_max_frames",
        action="store_true",
        help="Require exactly --max_frames extracted frames; useful for 73-frame diffusion runs.",
    )
    parser.add_argument(
        "--pad_short_to_max_frames",
        action="store_true",
        help="When a source has fewer than --max_frames selected frames, repeat the last frame.",
    )
    parser.add_argument("--target_height", type=int, default=-1)
    parser.add_argument("--target_width", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--da3_device", default="cuda")
    parser.add_argument(
        "--da3_weights",
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )
    parser.add_argument("--vggt_omega_device", default="cuda")
    parser.add_argument("--vggt_omega_checkpoint", default=None)
    parser.add_argument("--vggt_omega_image_resolution", type=int, default=512)
    parser.add_argument(
        "--vggt_omega_preprocess_mode",
        choices=["balanced", "max_size"],
        default="balanced",
    )
    parser.add_argument("--skip_vggt_omega_confidence", action="store_true")
    parser.add_argument("--save_vggt_omega_pointmaps", action="store_true")
    parser.add_argument("--compute_flow", type=parse_bool, default=False)
    parser.add_argument(
        "--flow_mode",
        choices=["raft", "waft"],
        default="raft",
        help="Optical-flow backend used when flow generation is enabled.",
    )
    parser.add_argument("--sam3_gpus", default=None)
    parser.add_argument("--sam3_checkpoint", default=None)
    parser.add_argument("--sam3_bpe", default=None)
    parser.add_argument("--sam3_lib_prior_root", default=None)
    parser.add_argument("--sam3_fps", type=float, default=None)
    parser.add_argument("--cowtracker_checkpoint", default=None)
    parser.add_argument("--waft_checkpoint", default=None)
    parser.add_argument("--skip_da3", action="store_true")
    parser.add_argument("--skip_precompute", action="store_true")
    parser.add_argument(
        "--rerun_sam3_only",
        action="store_true",
        help="Regenerate SAM3 masks while reusing existing depth and TAP artifacts.",
    )
    parser.add_argument("--skip_dynamic_resample", action="store_true")
    parser.add_argument("--skip_reconstruct", action="store_true")
    parser.add_argument("--ba_total_steps", type=int, default=0)
    parser.add_argument(
        "--preserve_fit_camera_optimization",
        action="store_true",
        help=(
            "Do not force fixed-camera BA/photo LR overrides; use the fit config's "
            "camera/depth optimization settings."
        ),
    )
    parser.add_argument(
        "--prep_overrides",
        nargs="*",
        default=[],
        help="Extra OmegaConf dotlist overrides for precompute.",
    )
    parser.add_argument(
        "--fit_overrides",
        nargs="*",
        default=[],
        help="Extra OmegaConf dotlist overrides for reconstruction.",
    )
    return parser.parse_args()


def main() -> None:
    force_float32_precision()
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    ws = Path(args.ws).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input source not found: {video_path}")
    if not (video_path.is_file() or video_path.is_dir()):
        raise FileNotFoundError(f"Input source must be a video file or image folder: {video_path}")
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max_frames must be positive")
    if args.depth_dirname is None:
        args.depth_dirname = (
            "vggt_omega_depth"
            if args.depth_mode == "vggt_omega"
            else "depthanything3_depth"
        )
    ws.mkdir(parents=True, exist_ok=True)

    if video_path.is_dir():
        frame_names, source_indices, fps = extract_image_folder_frames(
            video_path,
            ws / args.rgb_dirname,
            stride=args.frame_stride,
            max_frames=args.max_frames,
            require_max_frames=args.require_max_frames,
            pad_short_to_max_frames=args.pad_short_to_max_frames,
            target_height=args.target_height,
            target_width=args.target_width,
            overwrite=args.overwrite,
        )
    else:
        frame_names, source_indices, fps = extract_video_frames(
            video_path,
            ws / args.rgb_dirname,
            stride=args.frame_stride,
            max_frames=args.max_frames,
            require_max_frames=args.require_max_frames,
            pad_short_to_max_frames=args.pad_short_to_max_frames,
            target_height=args.target_height,
            target_width=args.target_width,
            overwrite=args.overwrite,
        )
    write_manifest(
        ws,
        video_path=video_path,
        frame_names=frame_names,
        source_indices=source_indices,
        fps=fps,
        args=args,
    )

    if not args.skip_da3:
        if args.depth_mode == "depthanything3":
            run_da3(
                ws,
                rgb_dirname=args.rgb_dirname,
                device=args.da3_device,
                weights=args.da3_weights,
            )
        elif args.depth_mode == "vggt_omega":
            run_vggt_omega(ws, args)
        else:
            raise ValueError(f"Unsupported depth mode: {args.depth_mode}")
    force_float32_precision()

    if args.rerun_sam3_only:
        run_sam3_only(ws, args)
    elif not args.skip_precompute:
        run_precompute(ws, args)
    force_float32_precision()

    if not args.skip_reconstruct:
        logdir = run_reconstruct(ws, args)
        print(f"Reconstruction logdir: {logdir}")

    print(f"Wild stage 1-2 workspace ready: {ws}")


if __name__ == "__main__":
    main()
