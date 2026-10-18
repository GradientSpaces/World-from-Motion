# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Input isolation and stage-boundary tests for independent reproduction."""
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reproduce/from_raw"
sys.path.insert(0, str(HERE))
import stages
import download

spec = importlib.util.spec_from_file_location("from_raw_run", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("outside_workbook", [False, True])
def test_collect_selected_retries_preserves_workbook_containment(tmp_path, outside_workbook):
    collection = tmp_path / "selected"
    collection.mkdir()
    for method in ("ms", "wt"):
        for scene in runner.SCENES:
            run = tmp_path / "attempts" / f"{method}_{scene}"
            evaluation = run / "evaluate"
            evaluation.mkdir(parents=True)
            (run / "execution.json").write_text(json.dumps(dict(
                status="complete", stages=[dict(stage=s, status="complete", returncode=0)
                                            for s in runner.STAGES])))
            for name in ("static.xlsx", "per_frame.xlsx"):
                (evaluation / name).write_bytes(b"collector checks containment; scoring checks workbook contents")
            static = "static.xlsx"
            if outside_workbook and method == "ms" and scene == "apple":
                outside = tmp_path / "unrelated.xlsx"
                outside.write_bytes(b"must not be accepted as this run's output")
                static = str(outside)
            entry = dict(method=method, scene=scene, checkpoint="../fit",
                         workspace="../workspace", metric_workspace="../workspace",
                         static_metrics=static, per_frame_metrics="per_frame.xlsx")
            (evaluation / "runs.json").write_text(json.dumps(dict(schema_version=1, runs=[entry])))
            (collection / f"{method}_{scene}").symlink_to(run, target_is_directory=True)
    result = subprocess.run([sys.executable, str(HERE / "collect.py"), str(collection)],
                            text=True, capture_output=True)
    if outside_workbook:
        assert result.returncode != 0
        assert "Missing fresh workbook" in result.stderr
        assert not (collection / "runs.json").exists()
    else:
        assert result.returncode == 0, result.stderr
        rows = json.loads((collection / "runs.json").read_text())["runs"]
        assert len(rows) == 14
        assert all(Path(row["static_metrics"]).is_relative_to(tmp_path / "attempts") for row in rows)


def test_archive_discards_tracks_and_checkpoints(tmp_path):
    archive = tmp_path / "public.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for kind in (*stages.OBSERVATIONS, "tracks", "logs"):
            zf.writestr(f"released_data/dycheck/iphone/wheel/{kind}/example", b"data")
    out = tmp_path / "observations"
    download.extract_observations(archive, out, ["wheel"])
    assert set(p.name for p in (out / "wheel").iterdir()) == set(stages.OBSERVATIONS)
    assert json.loads((out / "source.json").read_text())["prepared_priors_imported"] is False


def test_archive_rejects_missing_masks_before_creating_output(tmp_path):
    archive = tmp_path / "public.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("released_data/dycheck/iphone/wheel/images/0_00000.png", b"data")
    out = tmp_path / "observations"
    with pytest.raises(ValueError, match="lacks"):
        download.extract_observations(archive, out, ["wheel"])
    assert not out.exists()


def test_download_rejects_existing_corrupted_weight(tmp_path):
    (tmp_path / "model").write_bytes(b"broken")
    with pytest.raises(ValueError, match="mismatch"):
        download.fetch(tmp_path, dict(path="model", bytes=6, sha256="0"*64))


def test_unpublished_checkpoint_fails_before_network_or_output(tmp_path, monkeypatch):
    def unexpected_network(*args, **kwargs):
        pytest.fail("Missing checkpoint location must fail before downloading")
    monkeypatch.setattr(download, "fetch", unexpected_network)
    monkeypatch.setattr(sys, "argv", ["download.py", "--root", str(tmp_path / "inputs")])
    with pytest.raises(SystemExit) as exc:
        download.main()
    assert exc.value.code == 2
    assert not (tmp_path / "inputs").exists()


def test_verified_local_checkpoint_needs_no_url(tmp_path):
    payload = b"local checkpoint fixture"
    target = tmp_path / "models/model.safetensors"
    target.parent.mkdir()
    target.write_bytes(payload)
    spec = dict(path="models/model.safetensors", url=None, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())
    assert download.fetch(tmp_path, spec) == target


def test_missing_checkpoint_url_does_not_create_partial_files(tmp_path):
    with pytest.raises(ValueError, match="hosting is pending"):
        download.fetch(tmp_path, dict(path="models/model.safetensors", url=None))
    assert not (tmp_path / "models").exists()


def test_http_download_resumes_and_retries_short_ranges(tmp_path, monkeypatch):
    payload = b"abcdefghij"
    partial = tmp_path / "model.part"
    partial.write_bytes(payload[:3])
    requests = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self, data, content_range):
            super().__init__(data)
            self.headers = {"Content-Range": content_range}

    def open_range(request, timeout):
        extent = request.get_header("Range")
        requests.append(extent)
        start, end = map(int, extent.removeprefix("bytes=").split("-"))
        data = payload[start:end + 1]
        if len(requests) == 1:
            data = data[:1]
        return Response(data, f"bytes {start}-{end}/{len(payload)}")

    monkeypatch.setattr(download.urllib.request, "urlopen", open_range)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    download.fetch_https("https://example.test/model", partial, len(payload), chunk_bytes=4)
    download.verify(partial, {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    assert requests == ["bytes=3-6", "bytes=3-6", "bytes=7-9"]


def test_http_download_rejects_wrong_range_without_appending(tmp_path, monkeypatch):
    partial = tmp_path / "model.part"
    partial.write_bytes(b"abc")
    response = io.BytesIO(b"defg")
    response.status = 206
    response.headers = {"Content-Range": "bytes 0-3/10"}
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError, match="range"):
        download.fetch_https("https://example.test/model", partial, 10, chunk_bytes=4)
    assert partial.read_bytes() == b"abc"


def test_http_download_never_promotes_wrong_content(tmp_path, monkeypatch):
    response = io.BytesIO(b"bad")
    response.status = 200
    response.headers = {}
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: response)
    spec = {"path": "model", "url": "https://example.test/model", "bytes": 3,
            "sha256": hashlib.sha256(b"good"[:3]).hexdigest()}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.fetch(tmp_path, spec)
    assert not (tmp_path / "model").exists()
    assert (tmp_path / "model.part").read_bytes() == b"bad"


def test_http_download_exhausted_retries_preserve_prefix(tmp_path, monkeypatch):
    partial = tmp_path / "model.part"
    partial.write_bytes(b"abc")
    calls = []

    def incomplete(*args, **kwargs):
        calls.append(True)
        response = io.BytesIO(b"d")
        response.status = 206
        response.headers = {"Content-Range": "bytes 3-6/10"}
        return response

    monkeypatch.setattr(download.urllib.request, "urlopen", incomplete)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    with pytest.raises(OSError, match="Incomplete HTTP range"):
        download.fetch_https("https://example.test/model", partial, 10, chunk_bytes=4, attempts=2)
    assert len(calls) == 2
    assert partial.read_bytes() == b"abc"


def test_archive_rejects_duplicate_observation_destinations(tmp_path):
    archive = tmp_path / "public.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for prefix in ("one", "two"):
            zf.writestr(f"{prefix}/iphone/wheel/images/0_00000.png", b"data")
    with pytest.raises(ValueError, match="Duplicate"):
        download.extract_observations(archive, tmp_path / "out", ["wheel"])


def test_no_reconstruction_or_teacher_is_an_input(tmp_path):
    args = SimpleNamespace(method="wt", scene="wheel", inputs=tmp_path / "inputs",
                           output=tmp_path / "output", cpu_threads=8)
    plan = dict(runner.commands(args))
    assert tuple(plan) == runner.STAGES
    assert str(args.output / "runs.json") in plan["fit"]
    assert str(args.output / "fit/runs.json") in plan["evaluate"]
    assert all("retained_runs" not in token for cmd in plan.values() for token in cmd)


def test_failure_stops_before_next_stage(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    (inputs / "observations/wheel").mkdir(parents=True)
    (inputs / "models").mkdir()
    args = SimpleNamespace(method="ms", scene="wheel", inputs=inputs,
                           output=tmp_path / "output", cpu_threads=1, plan=False)
    stages_seen = []
    def fake_run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="test-commit\n")
        if "stages.py" in command[1]:
            stages_seen.append(command[2])
            return SimpleNamespace(returncode=3)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="prepare failed"):
        runner.run(args)
    assert stages_seen == ["prepare"]
    report = json.loads((args.output / "execution.json").read_text())
    assert report["status"] == "failed"
    assert report["stages"][0]["returncode"] == 3


def test_existing_output_is_never_reused(tmp_path):
    args = SimpleNamespace(method="wt", scene="wheel", inputs=tmp_path / "inputs",
                           output=tmp_path, cpu_threads=1, plan=False)
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_initial_only_stops_before_teacher_generation(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    (inputs / "observations/wheel").mkdir(parents=True)
    (inputs / "models").mkdir()
    args = SimpleNamespace(method="ms", scene="wheel", inputs=inputs,
                           output=tmp_path / "output", cpu_threads=1,
                           plan=False, initial_only=True)
    stages_seen = []

    def fake_run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="test-commit\n")
        if command[1].endswith("stages.py"):
            stages_seen.append(command[2])
        elif command[1].endswith("run.py"):
            pytest.fail("Initial-only execution must not fit or evaluate distillation")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.run(args)
    assert stages_seen == ["prepare", "priors", "masks", "initial"]
    report = json.loads((args.output / "execution.json").read_text())
    assert report["status"] == "complete"
    assert report["scope"] == "initial_only"
    assert "metrics_manifest" not in report
    assert all(stage["status"] == "complete" for stage in report["stages"])
