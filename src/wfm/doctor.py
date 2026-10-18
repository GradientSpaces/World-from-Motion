from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactManager
from .config import ProductionConfig
from .errors import ConfigurationError


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(
    config: ProductionConfig,
    *,
    output: Path,
    artifact_manager: ArtifactManager | None = None,
    require_artifacts: bool = True,
    require_gpu: bool = True,
) -> dict[str, Any]:
    checks: list[Check] = []
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        checks.append(Check(executable, path is not None, path or "not found"))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            encoders = subprocess.check_output(
                [ffmpeg, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True
            )
            has_h264 = "libx264" in encoders
            checks.append(Check("h264_encoder", has_h264, "libx264" if has_h264 else "missing"))
            filters = subprocess.check_output(
                [ffmpeg, "-hide_banner", "-filters"], stderr=subprocess.STDOUT, text=True
            )
            has_drawtext = " drawtext " in filters
            checks.append(
                Check("preview_label_filter", has_drawtext, "drawtext" if has_drawtext else "missing")
            )
        except subprocess.CalledProcessError as exc:
            checks.append(Check("h264_encoder", False, f"ffmpeg probe failed: {exc}"))

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output).free / (1024**3)
    minimum_disk = float(config.minimum_free_disk_gib)
    checks.append(Check("free_disk", free_gib >= minimum_disk, f"{free_gib:.1f} GiB"))

    manager = artifact_manager or ArtifactManager()
    model_storage = manager.bundle_root.parent
    model_storage.mkdir(parents=True, exist_ok=True)
    model_free_gib = shutil.disk_usage(model_storage).free / (1024**3)
    checks.append(
        Check("model_cache_disk", model_free_gib >= minimum_disk, f"{model_free_gib:.1f} GiB")
    )

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        checks.append(Check("cuda", cuda_available or not require_gpu, str(cuda_available)))
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            capability = f"{props.major}.{props.minor}"
            vram_gib = props.total_memory / (1024**3)
            checks.append(
                Check(
                    "compute_capability",
                    capability in set(config.supported_compute_capabilities),
                    capability,
                )
            )
            checks.append(
                Check("vram", vram_gib >= float(config.minimum_vram_gib), f"{vram_gib:.1f} GiB")
            )
            supported_names = tuple(str(item) for item in config.supported_gpu_name_patterns)
            checks.append(
                Check(
                    "gpu_model",
                    any(pattern in gpu_name for pattern in supported_names),
                    gpu_name,
                )
            )
    except ImportError:
        checks.append(Check("torch", not require_gpu, "not installed"))

    if require_artifacts:
        try:
            result = manager.validate(full_hash=True)
        except Exception as exc:
            checks.append(Check("model_artifacts", False, str(exc)))
        else:
            checks.append(Check("model_artifacts", True, json.dumps(result, sort_keys=True)))

    if require_gpu and not any(check.name == "cuda" and check.ok for check in checks):
        checks.append(Check("gpu_required", False, "A CUDA GPU is required"))

    payload = {
        "ok": all(check.ok for check in checks),
        "checks": [check.__dict__ for check in checks],
    }
    if not payload["ok"]:
        failures = "; ".join(f"{check.name}: {check.detail}" for check in checks if not check.ok)
        raise ConfigurationError(f"Preflight failed: {failures}")
    return payload
