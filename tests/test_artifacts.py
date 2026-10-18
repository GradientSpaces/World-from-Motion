from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wfm.artifacts import ArtifactLock, ArtifactManager
from wfm.errors import ArtifactError


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ArtifactManager, Path]:
    root = tmp_path / "bundle"
    root.mkdir()
    model = root / "model.bin"
    model.write_bytes(b"weights")
    manifest = {
        "bundle_version": "test",
        "files": [
            {
                "path": "model.bin",
                "size": model.stat().st_size,
                "sha256": hashlib.sha256(b"weights").hexdigest(),
            }
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    lock = ArtifactLock(
        {
            "bundle_version": "test",
            "ngc_model_ref": "org/team/model:test",
            "manifest": "manifest.json",
            "required_paths": {"model": "model.bin"},
        }
    )
    monkeypatch.setenv("WFM_MODEL_ROOT", str(root))
    return ArtifactManager(lock=lock, cache_root=tmp_path), model


def test_artifact_manifest_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = _manager(tmp_path, monkeypatch)
    assert manager.validate()["files_checked"] == 1


def test_artifact_corruption_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, model = _manager(tmp_path, monkeypatch)
    model.write_bytes(b"corrupt")
    with pytest.raises(ArtifactError):
        manager.validate()
