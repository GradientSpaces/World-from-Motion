from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import ProductionConfig
from .errors import ConfigurationError
from .video import probe_video


NATIVE_CHECKPOINTS = (
    "photometric_cam.pth",
    "photometric_s_model_native_add3.pth",
    "photometric_d_model_native_add3.pth",
)


def validate_output(
    output: Path, profile: ProductionConfig, *, allow_running_status: bool = False
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    failures: list[str] = []
    manifest_path = output / "run_manifest.json"
    status_path = output / "status.json"
    if not manifest_path.is_file():
        failures.append(f"missing {manifest_path}")
        manifest: dict[str, Any] = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_frames = int(manifest.get("input", {}).get("selected_frames", 0))
    expected_fps = float(manifest.get("input", {}).get("fps", 0))
    if not status_path.is_file():
        failures.append(f"missing {status_path}")
    else:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        accepted_states = {"completed", "running"} if allow_running_status else {"completed"}
        if status.get("state") not in accepted_states:
            failures.append(f"run state is {status.get('state')!r}, not completed")

    native = output / "reconstruction" / "native"
    for name in NATIVE_CHECKPOINTS:
        if not (native / name).is_file():
            failures.append(f"missing native checkpoint {name}")

    distillation = manifest.get("distillation", {})
    if bool(distillation.get("enabled")):
        initial_native = output / "reconstruction" / "initial" / "native"
        for name in NATIVE_CHECKPOINTS:
            if not (initial_native / name).is_file():
                failures.append(f"missing initial native checkpoint {name}")
        summary_path = output / "diagnostics" / "distillation_summary.json"
        history_path = output / "diagnostics" / "distillation_history.json"
        if not summary_path.is_file():
            failures.append("missing distillation_summary.json")
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_steps = int(distillation.get("steps", 0))
            if int(summary.get("num_frames", 0)) != expected_frames:
                failures.append(
                    f"distillation frames={summary.get('num_frames')}, "
                    f"expected={expected_frames}"
                )
            if int(summary.get("num_steps", 0)) != expected_steps:
                failures.append(
                    f"distillation steps={summary.get('num_steps')}, "
                    f"expected={expected_steps}"
                )
            if summary.get("target_cameras") != list(profile.target_cameras):
                failures.append("distillation target cameras do not match the profile")
            final_loss = summary.get("final_metrics", {}).get("loss_total")
            if not isinstance(final_loss, (int, float)) or not math.isfinite(final_loss):
                failures.append("distillation final loss is missing or non-finite")
            for key in (
                "source_native",
                "distilled_native",
                "benchmark_root",
                "prediction_root",
            ):
                if Path(str(summary.get(key, "/"))).is_absolute():
                    failures.append(f"distillation {key} is absolute; output is not relocatable")
        if not history_path.is_file():
            failures.append("missing distillation_history.json")
        else:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if len(history) != int(distillation.get("steps", 0)):
                failures.append("distillation history length does not match configured steps")

    brush = output / "reconstruction" / "brush"
    for name in ("scene_dynamic.ply", "initial_camera.json", "release_manifest.json"):
        if not (brush / name).is_file():
            failures.append(f"missing Brush artifact {name}")
    release_manifest = brush / "release_manifest.json"
    if release_manifest.is_file():
        release = json.loads(release_manifest.read_text(encoding="utf-8"))
        if Path(str(release.get("saved_dir", "/"))).is_absolute():
            failures.append("Brush saved_dir is absolute; output is not relocatable")
        if Path(str(release.get("ply_path", "/"))).is_absolute():
            failures.append("Brush ply_path is absolute; output is not relocatable")
        if int(release.get("total_splats", 0)) <= 0:
            failures.append("Brush export has no splats")
        if expected_frames and int(release.get("frames", 0)) != expected_frames:
            failures.append(
                f"Brush export frames={release.get('frames')}, expected={expected_frames}"
            )
        if expected_frames and release.get("source_frame_indices") != list(range(expected_frames)):
            failures.append("Brush export does not represent every selected source frame")
        release_checks = release.get("validation", {})
        if not bool(release_checks.get("byte_size_match")):
            failures.append("Brush export byte-size validation failed")
        if float(release_checks.get("max_roundtrip_error", float("inf"))) > float(
            release_checks.get("max_roundtrip_bound", float("-inf"))
        ) + 1e-5:
            failures.append("Brush export quantization round-trip validation failed")

    video_info = {}
    for camera in profile.target_cameras:
        video = output / "novel_views" / f"{camera}.mp4"
        if not video.is_file():
            failures.append(f"missing novel video {video.name}")
            continue
        info = probe_video(video)
        video_info[camera] = info.__dict__
        if expected_frames and info.frames != expected_frames:
            failures.append(f"{camera} frames={info.frames}, expected={expected_frames}")
        if expected_fps and abs(info.fps - expected_fps) > 0.05:
            failures.append(f"{camera} fps={info.fps}, expected={expected_fps}")
    preview = output / "novel_views" / "all_views_2x2.mp4"
    if not preview.is_file():
        failures.append("missing all_views_2x2.mp4")
    else:
        preview_info = probe_video(preview)
        video_info["all_views_2x2"] = preview_info.__dict__
        if expected_frames and preview_info.frames != expected_frames:
            failures.append(
                f"all_views_2x2 frames={preview_info.frames}, expected={expected_frames}"
            )
        if expected_fps and abs(preview_info.fps - expected_fps) > 0.05:
            failures.append(f"all_views_2x2 fps={preview_info.fps}, expected={expected_fps}")

    result = {"ok": not failures, "failures": failures, "videos": video_info}
    if failures:
        raise ConfigurationError("Output validation failed: " + "; ".join(failures))
    return result
