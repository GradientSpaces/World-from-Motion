#!/usr/bin/env python3
"""Materialize a DyCheck-style MoSca workspace with multiview assets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from data_utils.iphone_helpers import (
    load_iphone_camera_json,
    load_iphone_multiview_manifest,
    write_iphone_camera_json,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a monocular iPhone workspace and wire multiview RGB, masks, "
        "reference poses, per-frame camera JSONs, and optional DA3 depth metadata."
    )
    parser.add_argument("--scene-name", type=str, required=True)
    parser.add_argument("--out-ws", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Explicit multiview_manifest.json path (defaults to <out-ws>/ when present, "
        "else data/iphone_mv/<scene>_multiview/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output workspace before rebuilding.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Optional cap on the number of synchronized frames (per camera).",
    )
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copy multiview RGB/masks instead of symlinking (slower, portable).",
    )
    return parser.parse_args()


def _default_manifest_path(scene_name: str, repo_root: Path) -> Path:
    return repo_root / "data" / "iphone_mv" / f"{scene_name}_multiview" / "multiview_manifest.json"


def _resolve_manifest_path(
    scene_name: str,
    repo_root: Path,
    out_ws: Path,
    explicit_manifest: str | None,
) -> Path:
    if explicit_manifest is not None:
        manifest_path = Path(explicit_manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path
        return manifest_path.resolve()
    staged_manifest_path = out_ws / "multiview_manifest.json"
    if staged_manifest_path.is_file():
        return staged_manifest_path
    return _default_manifest_path(scene_name, repo_root)


def _is_seed_manifest_directory(out_ws: Path, manifest_path: Path) -> bool:
    if not out_ws.is_dir():
        return False
    if manifest_path != out_ws / "multiview_manifest.json":
        return False
    allowed_names = {"multiview_manifest.json"}
    child_names = {child.name for child in out_ws.iterdir() if not child.name.startswith(".")}
    return len(child_names) > 0 and child_names <= allowed_names


def _resolve_rgb_src(manifest: dict, camera_name: str, frame_stem: str) -> Path:
    benchmark_root = Path(manifest["benchmark_root"])
    inference_root = Path(manifest["inference_root"])
    anchor = manifest.get("anchor_camera", "cam0")
    if camera_name == anchor:
        source_ws = Path(manifest["source_workspace"])
        source_rgb_dir = source_ws / manifest.get("anchor_saved2d_rgb_dir", "images")
        for suffix in (".png", ".jpg", ".jpeg"):
            source_rgb = source_rgb_dir / f"{frame_stem}{suffix}"
            if source_rgb.is_file():
                return source_rgb
        return benchmark_root / "rgb" / camera_name / f"{frame_stem}.png"
    pred_subdir = str(manifest.get("pred_rgb_subdir", "pred_rgb"))
    for pred in (
        inference_root / pred_subdir / camera_name / f"{frame_stem}.png",
        inference_root / "pred_rgb" / camera_name / f"{frame_stem}.png",
        inference_root / camera_name / f"{frame_stem}.png",
    ):
        if pred.is_file():
            return pred
    fallback = benchmark_root / "rgb" / camera_name / f"{frame_stem}.png"
    return fallback


def _resolve_mask_src(manifest: dict, camera_name: str, frame_stem: str) -> Path:
    benchmark_root = Path(manifest["benchmark_root"])
    return benchmark_root / "test_dynamic" / camera_name / f"{frame_stem}.png"


def _resolve_legacy_mask_src(
    repo_root: Path, manifest: dict, camera_name: str, frame_stem: str
) -> Path | None:
    anchor = manifest.get("anchor_camera", "cam0")
    if camera_name != anchor:
        return None
    legacy_mask = (
        repo_root
        / "data"
        / "iphone_mv"
        / f"{manifest['scene_name']}_multiview"
        / "multiview"
        / "test_dynamic"
        / camera_name
        / f"{frame_stem}.png"
    )
    return legacy_mask if legacy_mask.is_file() else None


def _resolve_c2w_src(manifest: dict, camera_name: str) -> Path:
    benchmark_root = Path(manifest["benchmark_root"])
    return benchmark_root / "cameras" / f"{camera_name}_c2w.npz"


def _resolve_available_mask_sources(
    manifest: dict, repo_root: Path, camera_name: str, frame_names: list[str]
) -> list[tuple[str, Path]] | None:
    mask_sources = []
    missing_frame_names = []
    for frame_stem in frame_names:
        mask_src = _resolve_mask_src(manifest, camera_name, frame_stem)
        if not mask_src.is_file():
            legacy_mask_src = _resolve_legacy_mask_src(
                repo_root, manifest, camera_name, frame_stem
            )
            if legacy_mask_src is not None:
                mask_src = legacy_mask_src
        if not mask_src.is_file():
            missing_frame_names.append(frame_stem)
            continue
        mask_sources.append((frame_stem, mask_src))
    if not mask_sources:
        return None
    if missing_frame_names:
        raise FileNotFoundError(
            f"Incomplete multiview mask set for {camera_name}: "
            f"missing {missing_frame_names[:5]}"
        )
    return mask_sources


def _remove_path(dst: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _remove_path(dst)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())


def _load_anchor_intrinsic_template(source_ws: Path) -> tuple[float, list[float], list[float], dict]:
    cam_dir = source_ws / "cameras"
    if cam_dir.is_dir():
        json_files = sorted(p for p in cam_dir.iterdir() if p.suffix == ".json")
        if json_files:
            template = load_iphone_camera_json(json_files[0])
            focal = float(template["focal_length"])
            image_size = [float(x) for x in template["image_size"]]
            principal_point = [float(x) for x in template["principal_point"]]
            return focal, image_size, principal_point, template

    image_files = sorted(
        path
        for path in (source_ws / "images").iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(
            f"No camera JSON templates under {cam_dir} and no images under {source_ws / 'images'}"
        )
    image = imageio.imread(image_files[0])
    height, width = image.shape[:2]
    focal = float(min(width, height))
    image_size = [float(width), float(height)]
    principal_point = [float(width) * 0.5, float(height) * 0.5]
    template = {
        "focal_length": focal,
        "image_size": [int(width), int(height)],
        "principal_point": principal_point,
    }
    return focal, image_size, principal_point, template


def _resolve_multiview_depth_root(manifest: dict, manifest_root: Path) -> Path | None:
    explicit_root = manifest.get("multiview_depth_root")
    candidates = []
    if explicit_root is not None:
        explicit_root = Path(explicit_root)
        if not explicit_root.is_absolute():
            explicit_root = manifest_root / explicit_root
        candidates.append(explicit_root)
    candidates.append(manifest_root / "DA3_depth")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _materialize_workspace(
    manifest: dict,
    manifest_root: Path,
    repo_root: Path,
    out_ws: Path,
    *,
    num_frames: int | None,
    copy_assets: bool,
) -> None:
    source_ws = Path(manifest["source_workspace"])
    if not source_ws.is_dir():
        raise FileNotFoundError(f"source_workspace missing: {source_ws}")

    source_has_camera_jsons = (source_ws / "cameras").is_dir()

    def ignore_source_artifacts(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != source_ws.resolve():
            return set()
        if not source_has_camera_jsons:
            keep_names = {"cameras", "epi", "flow_raft", "flow_waft", "images"}
            if not bool(manifest.get("skip_source_depth_dirs", False)):
                keep_names.update(
                    {
                        "depthcrafter_depth",
                        "depthcrafter_depth_sharp",
                        "vggt_omega_depth",
                        "depthanything3_depth",
                        "sensor_depth",
                    }
                )
            return {
                name
                for name in names
                if name not in keep_names and not name.endswith("_tap.npz")
            }
        generated_prefixes = (
            "camera_ui_gbuffers",
            "diffsynth_",
            "render_from_",
        )
        ignored = {
            name
            for name in names
            if name == "logs" or name.startswith(generated_prefixes)
        }
        ignored.update(
            name
            for name in names
            if Path(name).suffix.lower() in {".gif", ".jpg", ".mp4"}
        )
        return ignored

    shutil.copytree(
        source_ws,
        out_ws,
        symlinks=True,
        ignore=ignore_source_artifacts,
        copy_function=shutil.copy,
    )

    focal, image_size, principal_point, json_template = _load_anchor_intrinsic_template(
        source_ws
    )
    multiview_depth_root = _resolve_multiview_depth_root(manifest, manifest_root)
    default_depth_npz_key = str(manifest.get("depth_npz_key", "dep"))
    default_depth_boundary_th = float(manifest.get("multiview_depth_boundary_th", 1.0))
    default_tap_root = Path(manifest.get("multiview_tap_root", "multiview/tap_priors"))

    manifest_out = json.loads(json.dumps(manifest))
    cameras_out = []
    for camera in manifest["cameras"]:
        names = list(camera["frame_names"])
        if num_frames is not None:
            names = names[:num_frames]
        cameras_out.append({**camera, "frame_names": names})
    manifest_out["cameras"] = cameras_out
    if num_frames is not None:
        manifest_out["num_frames"] = len(cameras_out[0]["frame_names"])

    for camera in manifest_out["cameras"]:
        camera_name = camera["camera_name"]
        rgb_dst_dir = out_ws / camera["rgb_dir"]
        camera.setdefault("camera_json_dir", f"multiview/camera_json/{camera_name}")
        cam_json_dst_dir = out_ws / camera["camera_json_dir"]
        c2w_dst = out_ws / camera["reference_c2w_path"]

        rgb_dst_dir.mkdir(parents=True, exist_ok=True)
        cam_json_dst_dir.mkdir(parents=True, exist_ok=True)
        c2w_dst.parent.mkdir(parents=True, exist_ok=True)

        c2w_src = _resolve_c2w_src(manifest, camera_name)
        if not c2w_src.is_file():
            raise FileNotFoundError(f"Missing reference trajectory {c2w_src}")
        data = np.load(c2w_src)
        c2w = np.asarray(data["c2w"], dtype=np.float32)
        if num_frames is not None:
            c2w = c2w[:num_frames]
        np.savez(c2w_dst, c2w=c2w)

        tap_prior_dir = camera.get("tap_prior_dir")
        if tap_prior_dir is None:
            candidate_tap_dir = default_tap_root / camera_name
            if (out_ws / candidate_tap_dir).is_dir():
                camera["tap_prior_dir"] = candidate_tap_dir.as_posix()
        elif not (out_ws / tap_prior_dir).is_dir():
            camera.pop("tap_prior_dir", None)

        mask_src_by_frame = {}
        mask_dir = camera.get("mask_dir")
        if mask_dir is not None:
            mask_sources = _resolve_available_mask_sources(
                manifest,
                repo_root,
                camera_name,
                camera["frame_names"],
            )
            if mask_sources is None:
                print(
                    f"No multiview masks found for {camera_name}; omit explicit mask_dir",
                    file=sys.stderr,
                )
                camera.pop("mask_dir", None)
            else:
                mask_dst_dir = out_ws / mask_dir
                mask_dst_dir.mkdir(parents=True, exist_ok=True)
                mask_src_by_frame = {frame_stem: mask_src for frame_stem, mask_src in mask_sources}

        if multiview_depth_root is not None:
            depth_src_dir = multiview_depth_root / camera_name
            if depth_src_dir.is_dir():
                depth_rel = Path(camera.get("depth_dir", Path("DA3_depth") / camera_name))
                _link_or_copy(depth_src_dir, out_ws / depth_rel, copy=copy_assets)
                camera["depth_dir"] = depth_rel.as_posix()
                camera.setdefault("depth_npz_key", default_depth_npz_key)
                camera.setdefault("depth_boundary_th", default_depth_boundary_th)

        for frame_idx, frame_stem in enumerate(camera["frame_names"]):
            rgb_src = _resolve_rgb_src(manifest, camera_name, frame_stem)
            if not rgb_src.is_file():
                raise FileNotFoundError(
                    f"Missing RGB for {camera_name} frame {frame_stem}: {rgb_src}"
                )
            _link_or_copy(
                rgb_src,
                rgb_dst_dir / f"{frame_stem}.png",
                copy=copy_assets,
            )

            mask_src = mask_src_by_frame.get(frame_stem)
            if mask_src is not None:
                _link_or_copy(
                    mask_src,
                    (out_ws / camera["mask_dir"]) / f"{frame_stem}.png",
                    copy=copy_assets,
                )

            T_wc_pose = np.asarray(c2w[frame_idx], dtype=np.float64)
            write_iphone_camera_json(
                cam_json_dst_dir / f"{frame_stem}.json",
                T_wc_pose,
                focal,
                image_size,
                principal_point,
                template=json_template,
            )

    manifest_path = out_ws / "multiview_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_out, handle, indent=2)

    summary = {
        "scene_name": manifest_out["scene_name"],
        "workspace": str(out_ws.resolve()),
        "num_frames": manifest_out.get("num_frames"),
        "anchor_rgb_dir": manifest_out.get("anchor_saved2d_rgb_dir", "images"),
        "anchor_rgb_override_dir": manifest_out["cameras"][0]["rgb_dir"],
        "aux_rgb_dirs": [c["rgb_dir"] for c in manifest_out["cameras"][1:]],
        "depth_dirs": {
            c["camera_name"]: c.get("depth_dir") for c in manifest_out["cameras"]
        },
        "tap_prior_dirs": {
            c["camera_name"]: c.get("tap_prior_dir") for c in manifest_out["cameras"]
        },
        "camera_weights": {
            c["camera_name"]: float(c.get("weight", 1.0))
            for c in manifest_out["cameras"]
        },
    }
    with open(out_ws / "multiview_workspace_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    out_ws = Path(args.out_ws).expanduser().resolve()
    manifest_path = _resolve_manifest_path(
        args.scene_name,
        repo_root,
        out_ws,
        args.manifest,
    )
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        raise SystemExit(1)

    manifest = load_iphone_multiview_manifest(manifest_path)
    if manifest.get("scene_name") != args.scene_name:
        print(
            f"Warning: manifest scene_name={manifest.get('scene_name')!r} "
            f"!= --scene-name={args.scene_name!r}",
            file=sys.stderr,
        )

    marker = out_ws / "multiview_workspace_summary.json"
    if out_ws.exists() or out_ws.is_symlink():
        if marker.is_file() and not args.force:
            print(f"Workspace already exists, skipping prepare: {out_ws}")
            return
        if _is_seed_manifest_directory(out_ws, manifest_path):
            _remove_path(out_ws)
        elif args.force:
            _remove_path(out_ws)
        else:
            print(
                f"Refusing to clobber existing directory without multiview summary: {out_ws}\n"
                "Use --force to replace it.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    out_ws.parent.mkdir(parents=True, exist_ok=True)
    _materialize_workspace(
        manifest,
        manifest_path.parent,
        repo_root,
        out_ws,
        num_frames=args.num_frames,
        copy_assets=args.copy_assets,
    )
    print(f"Prepared multiview workspace at {out_ws}")


if __name__ == "__main__":
    main()
