from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import repository_root
from .errors import ArtifactError
from .util import sha256_file


@dataclass(frozen=True)
class ArtifactLock:
    payload: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.payload["bundle_version"])

    @property
    def model_ref(self) -> str:
        override = os.environ.get("WFM_NGC_MODEL_REF")
        return override or str(self.payload["ngc_model_ref"])

    @property
    def required_paths(self) -> dict[str, str]:
        return dict(self.payload["required_paths"])


def load_artifact_lock(path: Path | None = None) -> ArtifactLock:
    lock_path = path or repository_root() / "config" / "artifacts.lock.json"
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot load artifact lock {lock_path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ArtifactError(f"Unsupported artifact lock schema in {lock_path}")
    return ArtifactLock(payload)


class ArtifactManager:
    def __init__(self, lock: ArtifactLock | None = None, cache_root: Path | None = None) -> None:
        self.lock = lock or load_artifact_lock()
        default_cache = Path(os.environ.get("WFM_CACHE_DIR", "~/.cache/world-from-motion"))
        self.cache_root = (cache_root or default_cache).expanduser().resolve()

    @property
    def bundle_root(self) -> Path:
        override = os.environ.get("WFM_MODEL_ROOT")
        if override:
            return Path(override).expanduser().resolve()
        return self.cache_root / "models" / self.lock.version

    @property
    def manifest_path(self) -> Path:
        return self.bundle_root / str(self.lock.payload.get("manifest", "manifest.json"))

    def paths(self) -> dict[str, Path]:
        return {name: self.bundle_root / relative for name, relative in self.lock.required_paths.items()}

    def validate(self, *, full_hash: bool = True) -> dict[str, Any]:
        root = self.bundle_root
        manifest_path = self.manifest_path
        if not manifest_path.is_file():
            raise ArtifactError(f"Model manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Invalid model manifest {manifest_path}: {exc}") from exc
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ArtifactError(f"Model manifest has no files: {manifest_path}")
        if str(manifest.get("bundle_version")) != self.lock.version:
            raise ArtifactError(
                f"Model bundle version {manifest.get('bundle_version')!r} does not match "
                f"locked version {self.lock.version!r}"
            )
        checked = 0
        recorded_paths: set[str] = set()
        for record in files:
            relative = str(record["path"])
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ArtifactError(f"Model manifest path escapes bundle root: {relative}") from exc
            if not path.is_file():
                raise ArtifactError(f"Model file is missing: {path}")
            actual_size = path.stat().st_size
            if actual_size != int(record["size"]):
                raise ArtifactError(
                    f"Size mismatch for {path}: {actual_size} != {record['size']}"
                )
            if full_hash and sha256_file(path) != str(record["sha256"]):
                raise ArtifactError(f"SHA-256 mismatch for {path}")
            recorded_paths.add(relative)
            checked += 1
        for name, path in self.paths().items():
            if not path.exists():
                raise ArtifactError(f"Required artifact {name} is missing: {path}")
            relative = path.relative_to(root).as_posix()
            covered = relative in recorded_paths or any(
                item.startswith(relative.rstrip("/") + "/") for item in recorded_paths
            )
            if not covered:
                raise ArtifactError(f"Required artifact {name} is not covered by the manifest: {path}")
        return {
            "root": str(root),
            "version": self.lock.version,
            "files_checked": checked,
            "manifest_sha256": sha256_file(manifest_path),
        }

    def ensure(self) -> Path:
        self.ensure_validated()
        return self.bundle_root

    def ensure_validated(self) -> dict[str, Any]:
        try:
            return self.validate(full_hash=True)
        except ArtifactError:
            if self.bundle_root.exists():
                raise
        self.pull()
        return self.validate(full_hash=True)

    def pull(self) -> Path:
        model_ref = self.lock.model_ref
        if "<org>" in model_ref or "<team>" in model_ref:
            raise ArtifactError(
                "NGC namespace is not configured; set WFM_NGC_MODEL_REF to "
                "<org>/<team>/world-from-motion-weights:<version>"
            )
        if shutil.which("ngc") is None:
            raise ArtifactError("The NGC CLI is required to download the private model bundle")
        target = self.bundle_root
        if target.exists():
            raise ArtifactError(f"Refusing to replace existing model cache: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="wfm-ngc-", dir=target.parent) as temp_text:
            temp = Path(temp_text)
            subprocess.run(
                ["ngc", "registry", "model", "download-version", model_ref, "--dest", str(temp)],
                check=True,
            )
            candidates = [path.parent for path in temp.rglob("manifest.json")]
            if len(candidates) != 1:
                raise ArtifactError(
                    f"Expected one downloaded model manifest, found {len(candidates)} under {temp}"
                )
            shutil.move(str(candidates[0]), str(target))
        return target
