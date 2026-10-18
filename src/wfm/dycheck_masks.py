# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lossless RGB transport for the independently validated paper mask recipe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def write_lossless_rgb(img_list, video, fps):
    """Encode FFV1 and verify every decoded RGB byte before SAM3 sees the video."""
    import cv2
    import imageio_ffmpeg
    import numpy as np

    video = Path(video)
    if video.exists():
        raise FileExistsError(video)
    if len(img_list) == 0:
        raise ValueError("Expected at least one RGB frame")
    h, w, channels = img_list[0].shape
    if channels != 3:
        raise ValueError("Expected RGB frames")
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    command = [encoder, "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "ffv1",
               "-pix_fmt", "bgr0", "-threads", "2", str(video)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        try:
            for frame in img_list:
                if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
                    raise ValueError("Expected equally sized uint8 RGB frames")
                proc.stdin.write(frame.tobytes())
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("FFV1 encoding failed; partial video retained")
    cap = cv2.VideoCapture(str(video))
    try:
        for frame in img_list:
            ok, decoded = cap.read()
            if not ok or not np.array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), frame):
                raise ValueError("Lossless transport changed or lost an RGB frame")
        if cap.read()[0]:
            raise ValueError("Lossless transport contains extra RGB frames")
    finally:
        cap.release()
    def digest(path):
        result = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()
    return dict(codec="ffv1", pixel_format="bgr0", encoder=encoder,
                encoder_sha256=digest(encoder), command=command,
                video_sha256=digest(video), frames=len(img_list), decoded_exact_rgb=True)


def generate_lossless_sam3_epi(save_dir, img_name_list, img_list, text_prompt,
        gpus=None, checkpoint_path=None, bpe_path=None, sam3_lib_prior_root=None,
        fps=10.0, save_mask_png=True, seed_frame_indices=None):
    """Keep SAM3 inference and seed selection unchanged; only preserve input RGB."""
    from lib_prior import sam3_wrapper as backend

    video = Path(save_dir) / "rgb-lossless.mkv"
    transport = write_lossless_rgb(img_list, video, fps)
    transport.update(seed_indices=[0] if seed_frame_indices is None else seed_frame_indices,
                     sam3_preprocessing="unchanged OpenCV video loader")
    (Path(save_dir) / "rgb-transport.json").write_text(json.dumps(transport, indent=2) + "\n")
    backend._patch_sam3_nms_for_triton_compat(sam3_lib_prior_root)
    outputs = backend._run_sam3_text_prompt_video(
        video_path=str(video), text_prompt=text_prompt, gpus=gpus,
        checkpoint_path=checkpoint_path, bpe_path=bpe_path,
        sam3_lib_prior_root=sam3_lib_prior_root, seed_frame_indices=seed_frame_indices,
        num_frames=len(img_list))
    masks = backend._outputs_to_mask_list(outputs, img_list)
    backend.save_sam3_epi_artifacts(save_dir, img_name_list, img_list, masks,
                                   fps=fps, save_mask_png=save_mask_png)
    return masks
