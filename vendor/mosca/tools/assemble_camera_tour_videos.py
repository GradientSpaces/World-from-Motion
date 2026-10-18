#!/usr/bin/env python3
"""Assemble the camera-tour predictions into final per-scene videos.

v1 order: seg1 + seg2 + seg3 -> <scene>_tour.mp4 (+ per-segment mp4s).
v2 order: sweep_intro + segA + segB + segC + sweep_outro -> <scene>_tour_v2.mp4
          (+ per-segment mp4s). Sweep clips (rendered from a recon at a different
          resolution) are resized to the predicted segment frame size before concat.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

# v1 segments (id, camera name, mp4 basename) -- all DiffSynth-predicted segments.
V1_SEGMENTS = (
    ("seg1", "cam_fixcam_first", "01_fixcam_first"),
    ("seg2", "cam_sweep_orig", "02_sweep_orig"),
    ("seg3", "cam_sweep_orig2new", "03_sweep_orig2new"),
)
# v2 plan entries: (kind, dir-id, camera/subdir name, mp4 basename). kind is
# "pred" (DiffSynth inference seg) or "sweep" (recon depth-sweep clip).
V2_PLAN = (
    ("sweep", "sweep_intro", "cam_sweep_intro", "00_sweep_intro"),
    ("pred", "segA", "cam_fwd_orig", "01_fwd_orig"),
    ("pred", "segB", "cam_bullet", "02_bullet"),
    ("pred", "segC", "cam_rev_new", "03_rev_new"),
    ("sweep", "sweep_outro", "cam_sweep_outro", "04_sweep_outro"),
)
# v3: forward / bullet / reverse, no depth sweeps (overwrites the v2 tour output).
V3_SEGMENTS = (
    ("segA", "cam_fwd_orig", "01_fwd_orig"),
    ("segB", "cam_bullet", "02_bullet"),
    ("segC", "cam_rev_new", "03_rev_new"),
)
# Sweep mp4s left over from a previous v2 run that v3 should remove.
V3_STALE_MP4S = ("00_sweep_intro.mp4", "04_sweep_outro.mp4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refined-root", default="data/release_ws/refined", type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", choices=("v1", "v2", "v3"), default="v1")
    p.add_argument("--output-subdir", default="camera_tour",
                   help="Per-scene working subdir (e.g. camera_tour or camera_tour_v2).")
    p.add_argument("--pred-subdir", default="pred_rgb_first73")
    p.add_argument("--videos-dir", default="data/release_ws/videos", type=Path)
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def pred_cam_dir(refined_root: Path, scene: str, output_subdir: str, seg_id: str,
                 cam: str, pred_subdir: str) -> Path:
    return (refined_root / scene / output_subdir / f"{seg_id}_inference"
            / scene / pred_subdir / cam)


def sweep_cam_dir(refined_root: Path, scene: str, output_subdir: str,
                  sweep_id: str, cam: str) -> Path:
    return refined_root / scene / output_subdir / sweep_id / cam


def load_frames(cam_dir: Path, scene: str, label: str) -> list[Path]:
    frames = sorted(cam_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"[{scene}] no PNG frames for {label} in {cam_dir}")
    return frames


def resized(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if frame.shape[:2] == target_hw:
        return frame
    th, tw = target_hw
    img = Image.fromarray(frame).resize((tw, th), Image.BILINEAR)
    return np.asarray(img)


def encode(frames: list[Path], dst_mp4: Path, fps: int,
           target_hw: tuple[int, int] | None) -> None:
    writer = imageio.get_writer(dst_mp4, fps=fps, macro_block_size=None)
    try:
        for f in frames:
            arr = imageio.imread(f)
            if target_hw is not None:
                arr = resized(arr, target_hw)
            writer.append_data(arr)
    finally:
        writer.close()


def assemble_v1(args: argparse.Namespace, out_dir: Path) -> None:
    concat_frames: list[Path] = []
    for seg_id, cam, mp4_base in V1_SEGMENTS:
        cam_dir = pred_cam_dir(args.refined_root, args.scene, args.output_subdir,
                               seg_id, cam, args.pred_subdir)
        frames = load_frames(cam_dir, args.scene, seg_id)
        concat_frames.extend(frames)
        src_mp4 = cam_dir.parent / f"{cam}.mp4"
        dst_mp4 = out_dir / f"{mp4_base}.mp4"
        if src_mp4.exists():
            shutil.copy2(src_mp4, dst_mp4)
            logging.info(f"[{args.scene}] {seg_id}: copied {src_mp4} -> {dst_mp4}")
        else:
            encode(frames, dst_mp4, args.fps, None)
            logging.info(f"[{args.scene}] {seg_id}: encoded {len(frames)} frames -> {dst_mp4}")

    tour_mp4 = out_dir / f"{args.scene}_tour.mp4"
    encode(concat_frames, tour_mp4, args.fps, None)
    logging.info(f"[{args.scene}] tour: {len(concat_frames)} frames @ {args.fps} fps -> {tour_mp4}")


def assemble_v2(args: argparse.Namespace, out_dir: Path) -> None:
    # Resolve frame lists for every entry first.
    entries: list[tuple[str, str, list[Path]]] = []  # (mp4_base, kind, frames)
    target_hw: tuple[int, int] | None = None
    for kind, dir_id, cam, mp4_base in V2_PLAN:
        if kind == "pred":
            cam_dir = pred_cam_dir(args.refined_root, args.scene, args.output_subdir,
                                   dir_id, cam, args.pred_subdir)
        else:
            cam_dir = sweep_cam_dir(args.refined_root, args.scene, args.output_subdir,
                                    dir_id, cam)
        frames = load_frames(cam_dir, args.scene, dir_id)
        entries.append((mp4_base, kind, frames))
        if target_hw is None and kind == "pred":
            target_hw = tuple(imageio.imread(frames[0]).shape[:2])  # match predicted size
    if target_hw is None:
        raise RuntimeError(f"[{args.scene}] no predicted segments found to size against")

    concat_frames: list[Path] = []
    for mp4_base, kind, frames in entries:
        dst_mp4 = out_dir / f"{mp4_base}.mp4"
        encode(frames, dst_mp4, args.fps, target_hw)
        concat_frames.extend(frames)
        logging.info(f"[{args.scene}] {mp4_base} ({kind}): {len(frames)} frames -> {dst_mp4}")

    tour_mp4 = out_dir / f"{args.scene}_tour_v2.mp4"
    writer = imageio.get_writer(tour_mp4, fps=args.fps, macro_block_size=None)
    try:
        for f in concat_frames:
            writer.append_data(resized(imageio.imread(f), target_hw))
    finally:
        writer.close()
    logging.info(f"[{args.scene}] tour_v2: {len(concat_frames)} frames @ {args.fps} fps -> {tour_mp4}")


def assemble_v3(args: argparse.Namespace, out_dir: Path) -> None:
    concat_frames: list[Path] = []
    for seg_id, cam, mp4_base in V3_SEGMENTS:
        cam_dir = pred_cam_dir(args.refined_root, args.scene, args.output_subdir,
                               seg_id, cam, args.pred_subdir)
        frames = load_frames(cam_dir, args.scene, seg_id)
        concat_frames.extend(frames)
        src_mp4 = cam_dir.parent / f"{cam}.mp4"
        dst_mp4 = out_dir / f"{mp4_base}.mp4"
        if src_mp4.exists():
            shutil.copy2(src_mp4, dst_mp4)
            logging.info(f"[{args.scene}] {seg_id}: copied {src_mp4} -> {dst_mp4}")
        else:
            encode(frames, dst_mp4, args.fps, None)
            logging.info(f"[{args.scene}] {seg_id}: encoded {len(frames)} frames -> {dst_mp4}")

    tour_mp4 = out_dir / f"{args.scene}_tour_v2.mp4"
    encode(concat_frames, tour_mp4, args.fps, None)
    logging.info(f"[{args.scene}] tour_v2: {len(concat_frames)} frames @ {args.fps} fps -> {tour_mp4}")
    # Remove stale depth-sweep mp4s from a previous v2 run.
    for stale in V3_STALE_MP4S:
        p = out_dir / stale
        if p.exists():
            p.unlink()
            logging.info(f"[{args.scene}] removed stale {p}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = args.videos_dir / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.variant == "v1":
        assemble_v1(args, out_dir)
    elif args.variant == "v3":
        assemble_v3(args, out_dir)
    else:
        assemble_v2(args, out_dir)


if __name__ == "__main__":
    main()
