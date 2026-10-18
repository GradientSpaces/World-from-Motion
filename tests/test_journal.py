from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfm.errors import ConfigurationError
from wfm.journal import StageJournal


def test_journal_resumes_completed_stage(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    journal = StageJournal(path, "same")
    with journal.stage("one") as execute:
        assert execute
    resumed = StageJournal(path, "same")
    with resumed.stage("one") as execute:
        assert not execute


def test_journal_rejects_changed_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    StageJournal(path, "first")
    with pytest.raises(ConfigurationError):
        StageJournal(path, "second")


def test_journal_records_failure(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    journal = StageJournal(path, "same")
    with pytest.raises(RuntimeError):
        with journal.stage("broken"):
            raise RuntimeError("expected")
    payload = json.loads(path.read_text())
    assert payload["state"] == "failed"
    assert payload["stages"]["broken"]["error"] == "expected"


def test_journal_clears_failure_diagnostics_after_success(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    journal = StageJournal(path, "same")
    with pytest.raises(RuntimeError):
        with journal.stage("flaky"):
            raise RuntimeError("first attempt")

    resumed = StageJournal(path, "same")
    with resumed.stage("flaky") as execute:
        assert execute

    record = json.loads(path.read_text())["stages"]["flaky"]
    assert record["state"] == "completed"
    assert record["error"] is None
    assert "traceback" not in record


def test_journal_reruns_invalid_stage_and_invalidates_downstream(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    journal = StageJournal(path, "same", provenance={"model_manifest_sha256": "abc"})
    with journal.stage("one"):
        pass
    with journal.stage("two"):
        pass

    resumed = StageJournal(path, "same")
    with resumed.stage("one", validator=lambda: False) as execute:
        assert execute
    payload = json.loads(path.read_text())
    assert payload["provenance"]["model_manifest_sha256"] == "abc"
    assert payload["stages"]["two"]["state"] == "invalidated"
