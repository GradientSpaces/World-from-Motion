from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactManager
from .benchmark import prepare_inference_benchmark
from .config import ProductionConfig, repository_root, vendor_root
from .doctor import run_doctor
from .errors import ConfigurationError
from .journal import StageJournal
from .metrics import RuntimeMonitor
from .util import normalize_classes, run_command, sanitize_scene_name, sha256_file, sha256_json, utc_now, write_json_atomic
from .validation import validate_output
from .video import encode_png_sequence, make_grid, probe_video


@dataclass(frozen=True)
class RunOptions:
    video: Path
    dynamic_classes: tuple[str, ...]
    output: Path
    seed: int
    max_frames: int | None = None
    distillation_steps: int | None = None
    device: str = "cuda"
    resume: bool = True
    keep_workdir: bool = True
    dry_run: bool = False


def _latest_saved_dir(workspace: Path) -> Path:
    candidates = sorted((workspace / "logs").glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
    required = (
        "photometric_cam.pth",
        "photometric_s_model_native_add3.pth",
        "photometric_d_model_native_add3.pth",
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(f"No complete MoSca photometric checkpoint under {workspace / 'logs'}")


def _copy_tree_atomic(source: Path, destination: Path) -> None:
    if destination.exists():
        required = (
            "photometric_cam.pth",
            "photometric_s_model_native_add3.pth",
            "photometric_d_model_native_add3.pth",
        )
        if all((destination / name).is_file() for name in required):
            return
        destination.rename(destination.with_name(f".{destination.name}.invalid-{time.time_ns()}"))
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        temporary.rename(temporary.with_name(f".{destination.name}.invalid-{time.time_ns()}"))
    shutil.copytree(source, temporary)
    temporary.replace(destination)


def _native_valid(path: Path) -> bool:
    return all((path / name).is_file() for name in (
        "photometric_cam.pth",
        "photometric_s_model_native_add3.pth",
        "photometric_d_model_native_add3.pth",
    ))


def _brush_valid(path: Path, expected_frames: int | None = None) -> bool:
    try:
        manifest = json.loads((path / "release_manifest.json").read_text(encoding="utf-8"))
        ply = path / "scene_dynamic.ply"
        return (
            ply.is_file()
            and (path / "initial_camera.json").is_file()
            and int(manifest["total_splats"]) > 0
            and int(manifest["frames"]) > 0
            and (expected_frames is None or int(manifest["frames"]) == expected_frames)
            and int(manifest["actual_bytes"]) == ply.stat().st_size
            and bool(manifest["validation"]["byte_size_match"])
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _make_brush_metadata_portable(native: Path, brush: Path) -> None:
    """Replace exporter-local absolute paths with output-relative references."""
    relative_native = Path(os.path.relpath(native, brush)).as_posix()
    release_path = brush / "release_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["saved_dir"] = relative_native
    release["ply_path"] = str(release["ply_file"])
    write_json_atomic(release_path, release)

    camera_path = brush / "initial_camera.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    camera["saved_dir"] = relative_native
    write_json_atomic(camera_path, camera)


def _benchmark_valid(root: Path, scene_name: str, cameras: tuple[str, ...]) -> bool:
    scene = root / scene_name
    required_modalities = ("rgb", "depth", "normal", "alpha", "scene_coords", "scene_flow")
    return all(
        (scene / "metadata" / f"{camera}_frames.json").is_file()
        and all(any((scene / modality / camera).glob("*")) for modality in required_modalities)
        for camera in cameras
    )


def _inference_valid(root: Path, scene_name: str, cameras: tuple[str, ...], frames: int) -> bool:
    pred = root / scene_name / "pred_rgb"
    return all(len(list((pred / camera).glob("*.png"))) >= frames for camera in cameras)


def _encoded_videos_valid(root: Path, cameras: tuple[str, ...], frames: int, fps: float) -> bool:
    try:
        for camera in cameras:
            info = probe_video(root / f"{camera}.mp4")
            if info.frames != frames or abs(info.fps - fps) > 0.05:
                return False
        return (root / "all_views_2x2.mp4").is_file()
    except Exception:
        return False


def _distillation_valid(
    root: Path,
    *,
    expected_frames: int,
    expected_steps: int,
    expected_cameras: tuple[str, ...],
) -> bool:
    try:
        summary = json.loads(
            (root / "distillation_summary.json").read_text(encoding="utf-8")
        )
        history = json.loads(
            (root / "distillation_history.json").read_text(encoding="utf-8")
        )
        return (
            _native_valid(root / "native")
            and int(summary["num_frames"]) == expected_frames
            and int(summary["num_steps"]) == expected_steps
            and tuple(summary["target_cameras"]) == expected_cameras
            and len(history) == expected_steps
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _copy_distillation_diagnostics(source: Path, diagnostics: Path) -> None:
    diagnostics.mkdir(parents=True, exist_ok=True)
    summary = json.loads(
        (source / "distillation_summary.json").read_text(encoding="utf-8")
    )
    run_root = diagnostics.parent
    summary.update(
        {
            "source_native": Path(
                os.path.relpath(run_root / "reconstruction" / "initial" / "native", diagnostics)
            ).as_posix(),
            "distilled_native": Path(
                os.path.relpath(run_root / "reconstruction" / "native", diagnostics)
            ).as_posix(),
            "benchmark_root": Path(
                os.path.relpath(run_root / "work" / "benchmark", diagnostics)
            ).as_posix(),
            "prediction_root": Path(
                os.path.relpath(run_root / "work" / "inference", diagnostics)
            ).as_posix(),
        }
    )
    write_json_atomic(diagnostics / "distillation_summary.json", summary)
    shutil.copy2(
        source / "distillation_history.json",
        diagnostics / "distillation_history.json",
    )
    source_debug = source / "debug"
    destination_debug = diagnostics / "distillation_debug"
    if source_debug.is_dir():
        if destination_debug.exists():
            destination_debug.rename(
                destination_debug.with_name(
                    f".{destination_debug.name}.invalid-{time.time_ns()}"
                )
            )
        shutil.copytree(source_debug, destination_debug)


def _exported_distillation_valid(
    native: Path,
    diagnostics: Path,
    *,
    expected_frames: int,
    expected_steps: int,
    expected_cameras: tuple[str, ...],
) -> bool:
    try:
        summary = json.loads(
            (diagnostics / "distillation_summary.json").read_text(encoding="utf-8")
        )
        history = json.loads(
            (diagnostics / "distillation_history.json").read_text(encoding="utf-8")
        )
        return (
            _native_valid(native)
            and int(summary["num_frames"]) == expected_frames
            and int(summary["num_steps"]) == expected_steps
            and tuple(summary["target_cameras"]) == expected_cameras
            and len(history) == expected_steps
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _installed_distributions() -> dict[str, str]:
    try:
        from importlib.metadata import distributions

        return {
            str(dist.metadata["Name"]): str(dist.version)
            for dist in distributions()
            if dist.metadata.get("Name")
        }
    except Exception:
        return {}


def _portable_platform() -> str:
    """OS, architecture and C library, without the host's kernel release."""
    libc, version = platform.libc_ver()
    parts = [platform.system(), platform.machine()]
    return "-".join(parts + ([f"with-{libc}{version}"] if libc else []))


class PipelineRunner:
    def __init__(self, profile: ProductionConfig, artifacts: ArtifactManager | None = None) -> None:
        self.profile = profile
        self.artifacts = artifacts or ArtifactManager()
        self.repo = repository_root()
        self.vendor = vendor_root()
        self.mosca = self.vendor / "mosca"

    def _environment(self, weights: dict[str, Path]) -> dict[str, str]:
        env = os.environ.copy()
        python_paths = [str(self.repo / "src"), str(self.mosca), str(self.vendor / "diffsynth")]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(python_paths),
                "TOKENIZERS_PARALLELISM": "false",
                "WANDB_MODE": "disabled",
                "WANDB_DISABLED": "true",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                "WFM_WAFT_CHECKPOINT": str(weights["waft_checkpoint"]),
                "WFM_WAFT_DEPTH_CHECKPOINT": str(weights["waft_depth_anything_checkpoint"]),
                "WFM_COWTRACKER_CHECKPOINT": str(weights["cowtracker_checkpoint"]),
            }
        )
        return env

    def dry_run_plan(self, options: RunOptions) -> dict[str, Any]:
        classes = normalize_classes(options.dynamic_classes)
        distillation_steps = int(
            self.profile.distillation_steps
            if options.distillation_steps is None
            else options.distillation_steps
        )
        return {
            "video": str(options.video.expanduser().resolve()),
            "dynamic_classes": list(classes),
            "output": str(options.output.expanduser().resolve()),
            "profile": self.profile.raw,
            "distillation_steps": distillation_steps,
            "stages": [
                "preflight_and_artifacts",
                "reconstruct",
                "export_initial_native",
                "render_gbuffers",
                "prepare_inference_benchmark",
                "diffsynth_inference",
                "distill_4dgs",
                "export_native",
                "export_brush",
                "encode_videos",
                "validate",
            ],
        }

    def run(self, options: RunOptions) -> dict[str, Any]:
        video = options.video.expanduser().resolve()
        output = options.output.expanduser().resolve()
        classes = normalize_classes(options.dynamic_classes)
        if options.dry_run:
            return self.dry_run_plan(options)
        if not video.is_file():
            raise ConfigurationError(f"Input video does not exist: {video}")
        if options.max_frames is not None and options.max_frames <= 0:
            raise ConfigurationError("--max-frames must be positive")
        distillation_steps = int(
            self.profile.distillation_steps
            if options.distillation_steps is None
            else options.distillation_steps
        )
        if distillation_steps <= 0:
            raise ConfigurationError("--distillation-steps must be positive")
        info = probe_video(video)
        selected_frames = min(info.frames, options.max_frames) if options.max_frames else info.frames
        input_hash = sha256_file(video)
        fingerprint_payload = {
            "video_sha256": input_hash,
            "dynamic_classes": classes,
            "profile": self.profile.raw,
            "seed": options.seed,
            "max_frames": options.max_frames,
            "distillation_steps": distillation_steps,
            "wfm_version": __version__,
        }
        output.mkdir(parents=True, exist_ok=True)
        for directory in ("diagnostics", "logs", "novel_views", "reconstruction", "work"):
            (output / directory).mkdir(exist_ok=True)
        preflight = run_doctor(
            self.profile,
            output=output,
            artifact_manager=self.artifacts,
            require_artifacts=False,
        )
        artifact_validation = self.artifacts.ensure_validated()
        source_provenance_path = self.vendor / "PROVENANCE.json"
        requirements_path = self.repo / "requirements.lock"
        lock_path = self.repo / "config" / "artifacts.lock.json"
        fingerprint_payload.update(
            {
                "artifact_version": self.artifacts.lock.version,
                "model_manifest_sha256": artifact_validation["manifest_sha256"],
                "artifact_lock_sha256": sha256_file(lock_path),
                "source_provenance_sha256": sha256_file(source_provenance_path),
                "requirements_lock_sha256": sha256_file(requirements_path),
            }
        )
        fingerprint = sha256_json(fingerprint_payload)
        journal = StageJournal(
            output / "status.json",
            fingerprint,
            resume=options.resume,
            provenance=fingerprint_payload,
        )
        completed_manifest = output / "run_manifest.json"
        if journal.payload.get("state") == "completed" and completed_manifest.is_file():
            try:
                validate_output(output, self.profile)
            except ConfigurationError:
                pass
            else:
                return json.loads(completed_manifest.read_text(encoding="utf-8"))
        monitor = RuntimeMonitor(output)
        monitor.start()

        with journal.stage(
            "preflight_and_artifacts",
            validator=lambda: (output / "diagnostics" / "preflight.json").is_file()
            and bool(artifact_validation),
        ) as execute:
            if execute:
                write_json_atomic(output / "diagnostics" / "preflight.json", preflight)
        weights = self.artifacts.paths()
        env = self._environment(weights)
        workspace = output / "work" / "workspace"
        scene_name = sanitize_scene_name(video.stem)
        benchmark = output / "work" / "benchmark"
        inference_benchmark = output / "work" / "benchmark_inference"
        inference = output / "work" / "inference"
        distillation = output / "work" / "distillation"

        with journal.stage(
            "reconstruct", validator=lambda: _native_valid(_latest_saved_dir(workspace))
        ) as execute:
            if execute:
                command = [
                    sys.executable,
                    "scripts/wild/run_wild_video_stage12.py",
                    "--video",
                    str(video),
                    "--ws",
                    str(workspace),
                    "--sam3_text",
                    repr(list(classes)),
                    "--dynamic_mask_mode",
                    "sam3",
                    "--prep_cfg",
                    "profile/wild/wild_vggt_omega_sam3_prep.yaml",
                    "--fit_cfg",
                    "profile/wild/wild_vggt_omega_ba_fit.yaml",
                    "--depth_mode",
                    "vggt_omega",
                    "--depth_dirname",
                    "vggt_omega_depth",
                    "--init_recon_path",
                    "vggt_omega_recon.npy",
                    "--flow_mode",
                    "waft",
                    "--compute_flow",
                    "true",
                    "--target_width",
                    str(self.profile.frame_width),
                    "--target_height",
                    str(self.profile.frame_height),
                    "--frame_stride",
                    str(self.profile.frame_stride),
                    "--vggt_omega_checkpoint",
                    str(weights["vggt_omega_checkpoint"]),
                    "--sam3_checkpoint",
                    str(weights["sam3_checkpoint"]),
                    "--sam3_bpe",
                    str(weights["sam3_bpe"]),
                    "--sam3_lib_prior_root",
                    str(self.mosca / "lib_prior"),
                    "--cowtracker_checkpoint",
                    str(weights["cowtracker_checkpoint"]),
                    "--waft_checkpoint",
                    str(weights["waft_checkpoint"]),
                    "--preserve_fit_camera_optimization",
                ]
                if options.max_frames:
                    command.extend(["--max_frames", str(options.max_frames)])
                run_command(command, cwd=self.mosca, env=env, log_path=output / "logs/reconstruct.log")

        saved_dir = _latest_saved_dir(workspace)
        stage_manifest_path = workspace / "wild_stage12_manifest.json"
        if stage_manifest_path.is_file():
            stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
            selected_frames = int(stage_manifest.get("num_frames", selected_frames))
            if float(stage_manifest.get("source_fps", 0)) > 0:
                info = type(info)(
                    width=info.width,
                    height=info.height,
                    fps=float(stage_manifest["source_fps"]),
                    frames=info.frames,
                    duration=info.duration,
                )
        initial_native = output / "reconstruction" / "initial" / "native"
        with journal.stage(
            "export_initial_native", validator=lambda: _native_valid(initial_native)
        ) as execute:
            if execute:
                _copy_tree_atomic(saved_dir, initial_native)

        with journal.stage(
            "render_gbuffers",
            validator=lambda: _benchmark_valid(
                benchmark, scene_name, tuple(self.profile.target_cameras)
            ),
        ) as execute:
            if execute:
                run_command(
                    [
                        sys.executable,
                        "scripts/wild/render_wild_fixed_camera_gbuffers.py",
                        "--ws",
                        str(workspace),
                        "--saved-dir",
                        str(initial_native),
                        "--output-root",
                        str(benchmark),
                        "--scene-name",
                        scene_name,
                        "--frame-stride",
                        "1",
                        "--outputs",
                        "rgb",
                        "depth",
                        "normal",
                        "alpha",
                        "scene_coords",
                        "scene_flow",
                        "scene_flow_rgb",
                        "--device",
                        options.device,
                        "--fixed-frame-preset",
                        "camera_fps",
                        "--num-fps-cameras",
                        str(self.profile.num_fps_cameras),
                        "--append-reverse-trajectory",
                        "--reverse-camera-name",
                        "cam_reverse",
                        "--overwrite",
                    ],
                    cwd=self.mosca,
                    env=env,
                    log_path=output / "logs/render_gbuffers.log",
                )

        benchmark_for_inference = benchmark
        padded_frames = selected_frames
        with journal.stage(
            "prepare_inference_benchmark",
            validator=lambda: selected_frames >= int(self.profile.diffsynth_chunk_size)
            or _benchmark_valid(
                inference_benchmark, scene_name, tuple(self.profile.target_cameras)
            ),
        ) as execute:
            if execute:
                benchmark_for_inference, _, padded_frames = prepare_inference_benchmark(
                    benchmark,
                    inference_benchmark,
                    scene_name,
                    int(self.profile.diffsynth_chunk_size),
                )
            elif inference_benchmark.exists():
                benchmark_for_inference = inference_benchmark
                padded_frames = max(selected_frames, int(self.profile.diffsynth_chunk_size))

        with journal.stage(
            "diffsynth_inference",
            validator=lambda: _inference_valid(
                inference, scene_name, tuple(self.profile.target_cameras), padded_frames
            ),
        ) as execute:
            if execute:
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "wfm.diffsynth_entry",
                        "--benchmark-root",
                        str(benchmark_for_inference),
                        "--output-root",
                        str(inference),
                        "--scene-name",
                        scene_name,
                        "--wan-root",
                        str(weights["wan_base"]),
                        "--tokenizer-root",
                        str(weights["wan_tokenizer"]),
                        "--checkpoint",
                        str(weights["diffsynth_checkpoint"]),
                        "--seed",
                        str(options.seed),
                        "--device",
                        options.device,
                    ],
                    cwd=self.repo,
                    env=env,
                    log_path=output / "logs/diffsynth.log",
                )

        distill_command = [
            sys.executable,
            "-m",
            "wfm.distill_entry",
            "--benchmark-root",
            str(benchmark),
            "--prediction-root",
            str(inference),
            "--output-root",
            str(distillation),
            "--scene-name",
            scene_name,
            "--source-native",
            str(initial_native),
            "--mosca-root",
            str(self.mosca),
            "--reference-camera",
            str(self.profile.reference_camera),
            "--device",
            options.device,
            "--seed",
            str(options.seed),
            "--steps",
            str(distillation_steps),
            "--source-weight",
            str(self.profile.distillation_source_weight),
            "--virtual-weight",
            str(self.profile.distillation_virtual_weight),
            "--ssim-lambda",
            str(self.profile.distillation_ssim_lambda),
            "--lr-scale",
            str(self.profile.distillation_lr_scale),
            "--grad-clip-norm",
            str(self.profile.distillation_grad_clip_norm),
            "--debug-every",
            str(self.profile.distillation_debug_every),
        ]
        if not bool(self.profile.distillation_use_alpha_mask):
            distill_command.append("--no-use-alpha-mask")
        for camera in self.profile.target_cameras:
            distill_command.extend(["--target-camera", str(camera)])
        with journal.stage(
            "distill_4dgs",
            validator=lambda: _distillation_valid(
                distillation,
                expected_frames=selected_frames,
                expected_steps=distillation_steps,
                expected_cameras=tuple(self.profile.target_cameras),
            ),
        ) as execute:
            if execute:
                run_command(
                    distill_command,
                    cwd=self.repo,
                    env=env,
                    log_path=output / "logs/distill_4dgs.log",
                )

        native = output / "reconstruction" / "native"
        diagnostics = output / "diagnostics"
        with journal.stage(
            "export_native",
            validator=lambda: _exported_distillation_valid(
                native,
                diagnostics,
                expected_frames=selected_frames,
                expected_steps=distillation_steps,
                expected_cameras=tuple(self.profile.target_cameras),
            ),
        ) as execute:
            if execute:
                _copy_tree_atomic(distillation / "native", native)
                _copy_distillation_diagnostics(distillation, diagnostics)

        brush = output / "reconstruction" / "brush"
        with journal.stage(
            "export_brush", validator=lambda: _brush_valid(brush, selected_frames)
        ) as execute:
            if execute:
                run_command(
                    [
                        sys.executable,
                        "tools/export_mosca_brush_ply.py",
                        "--saved-dir",
                        str(native),
                        "--backend",
                        str(self.profile.brush_backend),
                        "--output-dir",
                        str(brush),
                        "--ply-name",
                        "scene_dynamic.ply",
                        "--include-static",
                        "--all-frames",
                        "--overwrite",
                    ],
                    cwd=self.mosca,
                    env=env,
                    log_path=output / "logs/export_brush.log",
                )
        # The vendored exporter records its invocation paths. Normalize them
        # even when this stage is resumed so the delivered run is relocatable.
        _make_brush_metadata_portable(native, brush)

        novel_views = output / "novel_views"
        with journal.stage(
            "encode_videos",
            validator=lambda: _encoded_videos_valid(
                novel_views, tuple(self.profile.target_cameras), selected_frames, info.fps
            ),
        ) as execute:
            if execute:
                pred_root = inference / scene_name / "pred_rgb"
                videos = []
                for camera in self.profile.target_cameras:
                    destination = novel_views / f"{camera}.mp4"
                    encode_png_sequence(pred_root / camera, destination, info.fps, selected_frames)
                    videos.append(destination)
                make_grid(videos, novel_views / "all_views_2x2.mp4")

        run_manifest = {
            "schema_version": 1,
            "wfm_version": __version__,
            "created_at": utc_now(),
            "run_fingerprint": fingerprint,
            "input": {
                "path": str(video),
                "sha256": input_hash,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "source_frames": info.frames,
                "selected_frames": selected_frames,
                "dynamic_classes": list(classes),
            },
            "production_profile": self.profile.raw,
            "model_bundle": {
                "version": self.artifacts.lock.version,
                "ngc_model_ref": self.artifacts.lock.model_ref,
                "manifest_sha256": artifact_validation["manifest_sha256"],
                "files": json.loads(
                    self.artifacts.manifest_path.read_text(encoding="utf-8")
                )["files"],
            },
            "software": {
                "python": sys.version,
                "platform": _portable_platform(),
                "image_reference": os.environ.get("WFM_IMAGE_REFERENCE"),
                "image_digest": os.environ.get("WFM_IMAGE_DIGEST"),
                "wfm_source_revision": os.environ.get("WFM_SOURCE_REVISION"),
                "requirements_lock_sha256": sha256_file(requirements_path),
                "source_revisions": json.loads(
                    source_provenance_path.read_text(encoding="utf-8")
                ),
                "installed_distributions": _installed_distributions(),
            },
            "scene_name": scene_name,
            "inference_frames": padded_frames,
            "distillation": {
                "enabled": True,
                "steps": distillation_steps,
                "reference_camera": self.profile.reference_camera,
                "target_cameras": list(self.profile.target_cameras),
                "initial_native": "reconstruction/initial/native",
                "distilled_native": "reconstruction/native",
                "summary": "diagnostics/distillation_summary.json",
                "history": "diagnostics/distillation_history.json",
            },
            "outputs": {
                "native": "reconstruction/native",
                "brush": "reconstruction/brush/scene_dynamic.ply",
                "novel_views": [f"novel_views/{camera}.mp4" for camera in self.profile.target_cameras],
                "preview": "novel_views/all_views_2x2.mp4",
                "runtime_metrics": "diagnostics/runtime_metrics.json",
            },
        }
        if stage_manifest_path.is_file():
            shutil.copy2(stage_manifest_path, output / "diagnostics" / "wild_stage12_manifest.json")
        benchmark_summary = benchmark / scene_name / "metadata" / "scene_summary.json"
        if benchmark_summary.is_file():
            shutil.copy2(benchmark_summary, output / "diagnostics" / "scene_summary.json")
        write_json_atomic(output / "run_manifest.json", run_manifest)
        with journal.stage("validate") as execute:
            if execute:
                validate_output(output, self.profile, allow_running_status=True)
        journal.finish()
        monitor.finish()
        return run_manifest
