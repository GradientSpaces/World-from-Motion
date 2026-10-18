from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


def repository_root() -> Path:
    override = os.environ.get("WFM_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProductionConfig:
    raw: dict[str, Any]

    @property
    def profile_name(self) -> str:
        return str(self.raw["profile_name"])

    def __getattr__(self, name: str) -> Any:
        try:
            return self.raw[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()


def load_production_config(path: Path | None = None) -> ProductionConfig:
    config_path = path or repository_root() / "config" / "production.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot load production profile {config_path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ConfigurationError(f"Unsupported profile schema in {config_path}")
    if int(payload.get("diffsynth_chunk_size", 0)) <= 0:
        raise ConfigurationError("diffsynth_chunk_size must be positive")
    if int(payload.get("distillation_steps", 0)) <= 0:
        raise ConfigurationError("distillation_steps must be positive")
    if float(payload.get("distillation_source_weight", 0)) <= 0:
        raise ConfigurationError("distillation_source_weight must be positive")
    if float(payload.get("distillation_virtual_weight", 0)) <= 0:
        raise ConfigurationError("distillation_virtual_weight must be positive")
    ssim_lambda = float(payload.get("distillation_ssim_lambda", -1))
    if not 0 <= ssim_lambda <= 1:
        raise ConfigurationError("distillation_ssim_lambda must be in [0, 1]")
    if float(payload.get("distillation_lr_scale", 0)) <= 0:
        raise ConfigurationError("distillation_lr_scale must be positive")
    cameras = payload.get("target_cameras")
    if cameras != ["cam_fps_00", "cam_fps_01", "cam_fps_02", "cam_reverse"]:
        raise ConfigurationError("Production target camera contract was modified")
    return ProductionConfig(payload)


def vendor_root() -> Path:
    return Path(os.environ.get("WFM_VENDOR_ROOT", repository_root() / "vendor")).expanduser().resolve()
