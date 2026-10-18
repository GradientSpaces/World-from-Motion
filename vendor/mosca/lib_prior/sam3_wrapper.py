import logging
import ast
import os
import os.path as osp
import sys
from importlib.util import module_from_spec, spec_from_file_location
from typing import Iterable, Optional

import cv2
import imageio
import numpy as np

DEFAULT_SAM3_LIB_PRIOR_ROOT = os.environ.get("WFM_SAM3_LIB_PRIOR_ROOT")


def _ensure_sam3_import_paths(sam3_lib_prior_root: Optional[str] = None):
    sam3_lib_prior_root = sam3_lib_prior_root or DEFAULT_SAM3_LIB_PRIOR_ROOT
    if not sam3_lib_prior_root:
        raise FileNotFoundError(
            "SAM3 source root is not configured; pass sam3_lib_prior_root from the WFM runtime"
        )
    sam3_pkg_root = osp.join(sam3_lib_prior_root, "sam3")
    if not osp.exists(sam3_pkg_root):
        raise FileNotFoundError(
            f"SAM3 package root not found at {sam3_pkg_root}. "
            "Pass sam3_lib_prior_root or update DEFAULT_SAM3_LIB_PRIOR_ROOT."
        )
    for path in [sam3_lib_prior_root, sam3_pkg_root]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return sam3_lib_prior_root


def _parse_gpu_ids(gpus):
    if gpus is None:
        return None
    if isinstance(gpus, (int, np.integer)):
        return [int(gpus)]
    if isinstance(gpus, float) and float(gpus).is_integer():
        return [int(gpus)]
    if isinstance(gpus, str):
        if gpus.strip() == "":
            return None
        return [int(it) for it in gpus.split(",") if it.strip() != ""]
    if isinstance(gpus, Iterable):
        return [int(it) for it in gpus]
    raise TypeError(f"Unsupported gpus specification: {type(gpus)}")


def _resolve_bpe_path(bpe_path: Optional[str], sam3_lib_prior_root: Optional[str] = None):
    if bpe_path:
        return bpe_path
    sam3_lib_prior_root = _ensure_sam3_import_paths(sam3_lib_prior_root)
    default_bpe = osp.join(
        sam3_lib_prior_root, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
    )
    if osp.exists(default_bpe):
        return default_bpe
    logging.warning("SAM3 BPE file not found under SoM assets; using model default")
    return None


def _load_run_video_text_prompt(sam3_lib_prior_root: Optional[str] = None):
    sam3_lib_prior_root = _ensure_sam3_import_paths(sam3_lib_prior_root)
    runner_path = osp.join(
        sam3_lib_prior_root, "sam3", "scripts", "run_text_prompt_video.py"
    )
    if not osp.exists(runner_path):
        raise FileNotFoundError(f"SAM3 runner script not found at {runner_path}")
    module_name = "mosca_sam3_run_text_prompt_video"
    spec = spec_from_file_location(module_name, runner_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_video_text_prompt


def _load_sam3_video_predictor_builder(sam3_lib_prior_root: Optional[str] = None):
    _ensure_sam3_import_paths(sam3_lib_prior_root)
    from sam3.model_builder import build_sam3_video_predictor

    return build_sam3_video_predictor


def _patch_sam3_nms_for_triton_compat(sam3_lib_prior_root: Optional[str] = None):
    _ensure_sam3_import_paths(sam3_lib_prior_root)
    try:
        import triton.language as tl
    except Exception as exc:
        logging.warning(f"Failed to import Triton for SAM3 NMS probing: {exc}")
        return

    if hasattr(tl, "cast"):
        return

    from sam3.perflib import nms as sam3_nms

    def _generic_nms_safe(ious, scores, iou_threshold=0.5):
        return sam3_nms.generic_nms_cpu(ious, scores, iou_threshold)

    sam3_nms.generic_nms = _generic_nms_safe
    logging.warning(
        "Detected Triton without tl.cast; forcing SAM3 generic NMS to CPU fallback"
    )


def _write_video_from_images(img_list, output_path, fps):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, list(img_list), fps=fps, macro_block_size=1)


def _decode_mask_video(mask_video_path, img_list):
    cap = cv2.VideoCapture(mask_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open SAM3 mask video: {mask_video_path}")

    target_h, target_w = img_list[0].shape[:2]
    mask_list = []
    while True:
        ret, frame = cap.read()
        if not ret or len(mask_list) >= len(img_list):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[:2] != (target_h, target_w):
            gray = cv2.resize(
                gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
        mask_list.append((gray > 127).astype(np.uint8))
    cap.release()

    empty = np.zeros((target_h, target_w), dtype=np.uint8)
    while len(mask_list) < len(img_list):
        mask_list.append(empty.copy())
    return mask_list


def _normalize_seed_frame_indices(seed_frame_indices, num_frames):
    if seed_frame_indices is None:
        return [0]
    if isinstance(seed_frame_indices, str):
        stripped = seed_frame_indices.strip()
        if not stripped:
            return [0]
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in stripped.split(",")]
    else:
        parsed = seed_frame_indices
    if isinstance(parsed, (int, np.integer)):
        parsed = [int(parsed)]
    indices = []
    for item in parsed:
        index = int(item)
        if index < 0 or index >= num_frames:
            logging.warning(
                "Ignoring SAM3 seed frame %d outside valid range [0, %d)",
                index,
                num_frames,
            )
            continue
        indices.append(index)
    indices = sorted(set(indices))
    return indices or [0]


def _run_sam3_text_prompt_video(
    video_path,
    text_prompt,
    gpus=None,
    checkpoint_path=None,
    bpe_path=None,
    sam3_lib_prior_root=None,
    seed_frame_indices=None,
    num_frames=None,
):
    if checkpoint_path is None:
        raise FileNotFoundError(
            "SAM3 checkpoint must be supplied from the verified WFM model bundle; "
            "public runtime downloads are disabled."
        )
    build_sam3_video_predictor = _load_sam3_video_predictor_builder(
        sam3_lib_prior_root
    )
    predictor = build_sam3_video_predictor(
        gpus_to_use=_parse_gpu_ids(gpus),
        checkpoint_path=checkpoint_path,
        bpe_path=_resolve_bpe_path(bpe_path, sam3_lib_prior_root=sam3_lib_prior_root),
    )
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    outputs_per_frame = {}
    try:
        seed_indices = _normalize_seed_frame_indices(
            seed_frame_indices, num_frames=num_frames
        )
        any_positive_seed = False
        for frame_index in seed_indices:
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=text_prompt,
                )
            )
            prompt_outputs = response.get("outputs", {})
            outputs_per_frame[frame_index] = prompt_outputs
            if _outputs_have_any_positive_mask(prompt_outputs):
                any_positive_seed = True
            else:
                logging.warning(
                    "SAM3 prompt %r produced an empty seed mask on frame %d",
                    text_prompt,
                    frame_index,
                )

        if any_positive_seed:
            for response in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]
        else:
            logging.warning(
                "SAM3 prompt %r produced empty seed masks on all seed frames %s; "
                "skipping propagation",
                text_prompt,
                seed_indices,
            )
    finally:
        try:
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
        except Exception as exc:
            logging.warning(f"Failed to close SAM3 session cleanly: {exc}")
        predictor.shutdown()
    return outputs_per_frame


def _outputs_to_mask_list(outputs_per_frame, img_list):
    target_h, target_w = img_list[0].shape[:2]
    mask_list = []
    for frame_idx in range(len(img_list)):
        frame_outputs = outputs_per_frame.get(frame_idx)
        frame_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        if frame_outputs is not None:
            for mask in frame_outputs.get("out_binary_masks", []):
                mask = np.asarray(mask)
                if mask.ndim > 2:
                    mask = np.squeeze(mask)
                if mask.shape != (target_h, target_w):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                frame_mask = np.maximum(
                    frame_mask, (mask > 0.5).astype(np.uint8)
                )
        mask_list.append(frame_mask)
    return mask_list


def _outputs_have_any_positive_mask(outputs):
    if not outputs:
        return False
    for mask in outputs.get("out_binary_masks", []):
        mask = np.asarray(mask)
        if mask.size == 0:
            continue
        if np.any(mask > 0.5):
            return True
    return False


def _to_uint8_rgb(frame):
    frame = np.asarray(frame)[..., :3]
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0)
    return frame.astype(np.uint8)


def _write_mask_preview_videos(save_dir, img_list, mask_list, fps):
    overlay_path = osp.join(save_dir, "sam3_overlay.mp4")
    mask_video_path = osp.join(save_dir, "sam3_mask.mp4")

    overlay_frames = []
    mask_frames = []
    color = np.array([255, 64, 64], dtype=np.float32)
    alpha = 0.5
    for img, mask in zip(img_list, mask_list):
        img_u8 = _to_uint8_rgb(img)
        mask_bool = mask.astype(bool)
        overlay = img_u8.astype(np.float32).copy()
        if mask_bool.any():
            overlay[mask_bool] = alpha * color + (1.0 - alpha) * overlay[mask_bool]
        overlay_frames.append(overlay.astype(np.uint8))
        mask_u8 = (mask.astype(np.uint8) * 255)[..., None]
        mask_frames.append(np.repeat(mask_u8, 3, axis=-1))

    imageio.mimsave(overlay_path, overlay_frames, fps=fps, macro_block_size=1)
    imageio.mimsave(mask_video_path, mask_frames, fps=fps, macro_block_size=1)


def _write_sam3_epi_artifacts(save_dir, img_name_list, mask_list, save_mask_png=True):
    epi_error_dir = osp.join(save_dir, "epi", "error")
    os.makedirs(epi_error_dir, exist_ok=True)
    for fn in os.listdir(epi_error_dir):
        if fn.endswith(".npy"):
            os.remove(osp.join(epi_error_dir, fn))

    mask_dir = osp.join(save_dir, "masks")
    if save_mask_png:
        os.makedirs(mask_dir, exist_ok=True)

    for img_name, mask in zip(img_name_list, mask_list):
        save_name = osp.basename(img_name)
        np.save(osp.join(epi_error_dir, save_name + ".npy"), mask.astype(np.float32))
        if save_mask_png:
            imageio.imwrite(
                osp.join(mask_dir, osp.splitext(save_name)[0] + ".png"),
                (mask * 255).astype(np.uint8),
            )


def save_sam3_epi_artifacts(
    save_dir,
    img_name_list,
    img_list,
    mask_list,
    fps=10.0,
    save_mask_png=True,
):
    _write_mask_preview_videos(save_dir, img_list, mask_list, fps=fps)
    _write_sam3_epi_artifacts(
        save_dir=save_dir,
        img_name_list=img_name_list,
        mask_list=mask_list,
        save_mask_png=save_mask_png,
    )


def generate_sam3_epi(
    save_dir,
    img_name_list,
    img_list,
    text_prompt,
    gpus=None,
    checkpoint_path=None,
    bpe_path=None,
    sam3_lib_prior_root=None,
    fps=10.0,
    save_mask_png=True,
    seed_frame_indices=None,
):
    if not text_prompt:
        raise ValueError("SAM3 text prompt is required when dynamic_mask_mode=sam3")

    _patch_sam3_nms_for_triton_compat(sam3_lib_prior_root)

    video_path = osp.join(save_dir, "rgb.mp4")

    logging.info("Preparing RGB video for SAM3 inference")
    _write_video_from_images(img_list, video_path, fps=fps)

    logging.info("Running SAM3 text-prompt video inference")
    outputs_per_frame = _run_sam3_text_prompt_video(
        video_path=video_path,
        text_prompt=text_prompt,
        gpus=gpus,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        sam3_lib_prior_root=sam3_lib_prior_root,
        seed_frame_indices=seed_frame_indices,
        num_frames=len(img_list),
    )

    logging.info("Converting SAM3 masks to epi/error shims")
    mask_list = _outputs_to_mask_list(outputs_per_frame, img_list)
    save_sam3_epi_artifacts(
        save_dir=save_dir,
        img_name_list=img_name_list,
        img_list=img_list,
        mask_list=mask_list,
        fps=fps,
        save_mask_png=save_mask_png,
    )
    return mask_list
