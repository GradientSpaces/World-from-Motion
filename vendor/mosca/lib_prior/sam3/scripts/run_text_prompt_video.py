#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import save_masklet_video


def parse_gpus(gpus_str):
    if gpus_str is None or gpus_str == "":
        return None
    return [int(x) for x in gpus_str.split(",") if x.strip() != ""]


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not video_frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    if fps <= 0:
        fps = 10
    return video_frames, fps


def save_binary_mask_video(outputs_per_frame, video_size, out_path, fps=10):
    height, width = video_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Use unique temp file in same directory as output to avoid conflicts
    out_path = Path(out_path)
    temp_path = out_path.parent / f"temp_mask_{out_path.stem}_{os.getpid()}.mp4"
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height), True)

    frame_count = 0
    for frame_idx in sorted(outputs_per_frame.keys()):
        outputs = outputs_per_frame[frame_idx]
        frame_mask = np.zeros((height, width), dtype=np.uint8)
        for mask in outputs["out_binary_masks"]:
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            frame_mask = np.maximum(frame_mask, (mask > 0.5).astype(np.uint8) * 255)
        frame_bgr = cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)
        frame_count += 1

    writer.release()
    
    # Check if temp file was created with content
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write mask video frames (wrote {frame_count} frames)")
    
    # Run ffmpeg with error checking
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(temp_path), str(out_path)],
        capture_output=True, text=True
    )
    temp_path.unlink(missing_ok=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to convert mask video: {result.stderr}")
    
    # Verify output was created
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create mask video at {out_path}")


def run_video_text_prompt(
    video_path,
    text_prompt,
    output_path,
    gpus_to_use=None,
    checkpoint_path=None,
    bpe_path=None,
    draw_boxes=True,
    mask_output_path=None,
):
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
    )

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )

    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    video_frames, fps = load_video_frames(video_path)
    save_masklet_video(
        video_frames,
        outputs_per_frame,
        output_path,
        fps=fps,
        # draw_boxes=draw_boxes,
    )
    if mask_output_path is not None:
        height, width = video_frames[0].shape[:2]
        save_binary_mask_video(
            outputs_per_frame,
            (height, width),
            mask_output_path,
            fps=fps,
        )

    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    predictor.shutdown()


def collect_mp4s(input_path):
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.glob("*.mp4"))
    if path.is_file() and path.suffix.lower() == ".mp4":
        return [path]
    raise ValueError("Input must be an .mp4 file or a directory of .mp4 files.")


def resolve_output_path(output_arg, video_path, is_batch):
    output_path = Path(output_arg)
    if is_batch:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{video_path.stem}_overlay.mp4"
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{video_path.stem}_overlay.mp4"
    return output_path


def resolve_mask_output_path(mask_arg, output_path, video_path, is_batch):
    if mask_arg:
        mask_path = Path(mask_arg)
        if is_batch or mask_path.is_dir():
            mask_path.mkdir(parents=True, exist_ok=True)
            return mask_path / f"{video_path.stem}_mask.mp4"
        return mask_path
    output_path = Path(output_path)
    if is_batch or output_path.is_dir():
        return output_path / f"{video_path.stem}_mask.mp4"
    return output_path.with_name(f"{output_path.stem}_mask{output_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 video inference from a text prompt on an MP4."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to an input .mp4 or a directory of .mp4 files",
    )
    parser.add_argument("--text", required=True, help="Text prompt (e.g. 'person')")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .mp4 path or output directory for batch input",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help='Comma-separated GPU ids (e.g. "0,1"). Defaults to current GPU.',
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to SAM3 checkpoint (defaults to HF download).",
    )
    parser.add_argument(
        "--bpe",
        default=None,
        help="Optional path to BPE tokenizer file.",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Only overlay masks (no bounding boxes or labels).",
    )
    parser.add_argument(
        "--mask-video-output",
        default=None,
        help="Optional output .mp4 path or directory for binary mask videos.",
    )
    args = parser.parse_args()

    videos = collect_mp4s(args.video)
    if not videos:
        raise RuntimeError(f"No .mp4 files found in: {args.video}")

    gpus_to_use = parse_gpus(args.gpus)
    is_batch = len(videos) > 1
    for video_path in videos:
        output_path = resolve_output_path(args.output, video_path, is_batch=is_batch)
        mask_output_path = resolve_mask_output_path(
            args.mask_video_output, output_path, video_path, is_batch=is_batch
        )
        run_video_text_prompt(
            str(video_path),
            args.text,
            str(output_path),
            gpus_to_use=gpus_to_use,
            checkpoint_path=args.checkpoint,
            bpe_path=args.bpe,
            draw_boxes=not args.mask_only,
            mask_output_path=str(mask_output_path),
        )


if __name__ == "__main__":
    main()
