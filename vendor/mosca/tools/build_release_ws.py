#!/usr/bin/env python3
"""Consolidate per-scene release artifacts into data/release_ws/original/<scene>/.

Per scene (manifest row: scene_key, orig_saved_dir, new_traj_full_npz, pose_npz,
diffsynth_pred_dir, cam_name, input_T), this gathers:
  recon_logs/<saved_dir_name>/  - the original reconstruction logs (checkpoints + viz)
  new_trajectory/               - full new-traj npz + sampled 73-frame pose npz + meta.json
  diffsynth_new_traj/frames/    - the new-traj DiffSynth prediction frames
  diffsynth_new_traj/<scene>_new_traj_diffsynth.mp4

Precompute prior files are intentionally NOT copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out-root", default="data/release_ws/original", type=Path)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--symlink-logs", action="store_true", help="Symlink recon logs instead of copying.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", nargs="+", default=None, help="Restrict to these scene keys.")
    return p.parse_args()


def dir_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e6


def main() -> None:
    args = parse_args()
    rows = [l.rstrip("\n").split("\t") for l in args.manifest.read_text().splitlines() if l.strip()]
    for row in rows:
        if len(row) != 7:
            raise ValueError(f"Bad manifest row (expected 7 cols): {row}")
        scene, saved_dir, new_full, pose_npz, pred_dir, cam, input_T = row
        if args.only and scene not in args.only:
            continue
        saved_dir, new_full, pose_npz, pred_dir = map(Path, (saved_dir, new_full, pose_npz, pred_dir))
        for pth in (saved_dir, pose_npz, pred_dir):
            if not pth.exists():
                raise FileNotFoundError(f"{scene}: missing {pth}")
        pred_frames = sorted(pred_dir.glob("*.png"))
        out = args.out_root / scene
        logs_dst = out / "recon_logs" / saved_dir.name
        nt_dst = out / "new_trajectory"
        ds_dst = out / "diffsynth_new_traj"

        if args.dry_run:
            print(
                f"[dry] {scene}: logs<-{saved_dir.name}({dir_size_mb(saved_dir):.0f}MB, "
                f"{'symlink' if args.symlink_logs else 'copy'}) "
                f"new_traj(full={new_full.name}, pose={pose_npz.name}) "
                f"diffsynth({len(pred_frames)} frames, cam={cam}) -> {out}"
            )
            continue

        if out.exists() and args.overwrite:
            shutil.rmtree(out)
        (out / "recon_logs").mkdir(parents=True, exist_ok=True)
        nt_dst.joinpath("cameras").mkdir(parents=True, exist_ok=True)
        ds_dst.joinpath("frames").mkdir(parents=True, exist_ok=True)

        # recon logs
        if logs_dst.exists():
            if logs_dst.is_symlink() or logs_dst.is_file():
                logs_dst.unlink()
            else:
                shutil.rmtree(logs_dst)
        if args.symlink_logs:
            logs_dst.symlink_to(saved_dir.resolve())
        else:
            shutil.copytree(saved_dir, logs_dst)

        # new trajectory
        shutil.copy2(new_full, nt_dst / new_full.name)
        if pose_npz.resolve() != (nt_dst / "cameras" / pose_npz.name).resolve():
            shutil.copy2(pose_npz, nt_dst / "cameras" / pose_npz.name)

        # diffsynth frames + mp4
        for f in pred_frames:
            shutil.copy2(f, ds_dst / "frames" / f.name)
        mp4 = ds_dst / f"{scene}_new_traj_diffsynth.mp4"
        imageio.mimsave(mp4, [imageio.imread(f) for f in pred_frames], fps=args.fps,
                        quality=8, macro_block_size=1)

        meta = {
            "scene": scene,
            "input_T": int(input_T),
            "new_traj_camera": cam,
            "new_traj_full_npz": str(new_full),
            "new_traj_sampled_pose_npz": str(pose_npz),
            "diffsynth_pred_dir": str(pred_dir),
            "num_diffsynth_frames": len(pred_frames),
            "orig_saved_dir": str(saved_dir),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"OK {scene}: logs={logs_dst.name} new_traj+pose copied diffsynth={len(pred_frames)}f -> {out}")


if __name__ == "__main__":
    main()
