#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal CoWTracker inference demo.

Usage:
    python demo.py --video input.mp4 --output output.mp4
    python demo.py --video input.mp4 --output output.mp4 --checkpoint ~/run168/cow_tracker_model.pth
"""

import argparse
import os

import mediapy
import numpy as np
import torch

from cowtracker import CoWTracker
from cowtracker.utils.visualization import paint_point_track

inf_dtype = torch.float16
def preprocess_video(video_path, max_frames=200, target_size=(336, 560)):
    """Load and preprocess video.

    Args:
        video_path: Path to input video
        max_frames: Maximum number of frames to process
        target_size: Target size (H, W) for inference

    Returns:
        Tuple of (video_array, fps)
    """
    video_arr = mediapy.read_video(video_path)
    video_fps = video_arr.metadata.fps
    num_frames = video_arr.shape[0]

    # Truncate if too long
    if num_frames > max_frames:
        print(f"Video is too long. Truncating to first {max_frames} frames.")
        video_arr = video_arr[:max_frames]

    # Resize to target size
    video_arr = mediapy.resize_video(video_arr, target_size)

    return np.array(video_arr), video_fps


def run_inference(model, video):
    """Run tracking inference on video.

    Args:
        model: CoWTracker model
        video: Video array [T, H, W, C] in uint8

    Returns:
        Tuple of (tracks, visibilities, confidences)
            - tracks: [T, H, W, 2]
            - visibilities: [T, H, W]
            - confidences: [T, H, W]
    """
    device = next(model.parameters()).device

    # Convert to tensor [T, C, H, W]
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(device)
    T, C, H, W = video_tensor.shape
    print(f"Video size: {H}x{W}")

    torch.cuda.empty_cache()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=inf_dtype):
            predictions = model.forward(video=video_tensor, queries=None)

            tracks = predictions["track"][0].cpu()
            visibility = predictions["vis"][0].cpu()
            confidence = predictions["conf"][0].cpu()

    visconf = visibility * confidence
    return tracks, visconf > 0.1, visconf


def create_visualization(video, tracks, visibilities, rate=8, fps=30, show_bkg=True):
    """Create visualization video.

    Args:
        video: Video array [T, H, W, C]
        tracks: Tracks [T, H, W, 2]
        visibilities: Visibility mask [T, H, W]
        rate: Subsampling rate for points
        fps: Output video fps
        show_bkg: Whether to show background

    Returns:
        Painted video frames [T, H, W, C]
    """
    T, H, W, _ = video.shape

    # Subsample tracks for visualization
    tracks_np = tracks.permute(1, 2, 0, 3).reshape(-1, T, 2).numpy()  # [HW, T, 2]
    vis_np = visibilities.permute(1, 2, 0).reshape(-1, T).numpy()  # [HW, T]

    # Subsample
    tracks_sub = tracks_np.reshape(H, W, T, 2)[::rate, ::rate].reshape(-1, T, 2)
    vis_sub = vis_np.reshape(H, W, T)[::rate, ::rate].reshape(-1, T)

    # Paint tracks
    painted_video = paint_point_track(
        video, tracks_sub, vis_sub, rate=rate, show_bkg=show_bkg
    )

    return painted_video


def main():
    parser = argparse.ArgumentParser(description="CoWTracker Inference Demo")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--rate", type=int, default=8, help="Subsampling rate for visualization"
    )
    parser.add_argument(
        "--max_frames", type=int, default=200, help="Maximum number of frames"
    )
    parser.add_argument("--no_bkg", action="store_true", help="Hide video and show only tracks on black background")
    parser.add_argument(
        "--coord_convention",
        type=str,
        choices=["pixel_center", "corner_aligned"],
        default="pixel_center",
        help="Coordinate convention: 'pixel_center' means (0,0) is the center of the first pixel; "
             "'corner_aligned' means (0,0) is the top-left corner (pixel center at 0.5,0.5). "
             "Use 'corner_aligned' for compatibility with COLMAP, graphics pipelines, etc.",
    )
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base_name}_tracked.mp4"

    print("=" * 60)
    print("CoWTracker Inference Demo")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    model = CoWTracker.from_checkpoint(
        args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=inf_dtype if torch.cuda.is_available() else torch.float32,
    )

    # Load video
    print("\n[2/4] Loading video...")
    video, fps = preprocess_video(args.video, max_frames=args.max_frames)
    print(f"Video shape: {video.shape}, FPS: {fps}")

    # Run inference
    print("\n[3/4] Running inference...")
    tracks, visibilities, confidences = run_inference(model, video)
    print(f"Tracks shape: {tracks.shape}")

    # Apply coordinate convention offset if needed
    if args.coord_convention == "corner_aligned":
        tracks = tracks + 0.5

    # Create visualization
    print("\n[4/4] Creating visualization...")
    painted_video = create_visualization(
        video, tracks, visibilities, rate=args.rate, fps=fps, show_bkg=not args.no_bkg
    )

    # Save output
    mediapy.write_video(args.output, painted_video, fps=fps)
    print(f"\nSaved output to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

