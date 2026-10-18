# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Input isolation and stage-boundary tests for independent reproduction."""
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import subprocess
from types import ModuleType, SimpleNamespace
from urllib.parse import urlsplit
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reproduce/from_raw"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import stages
import download
from PIL import Image
from wfm.released_masks import build_bundle, write_archive

spec = importlib.util.spec_from_file_location("from_raw_run", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

LOCK = json.loads((HERE / "assets.lock.json").read_text())
PUBLIC_HOSTS = {"huggingface.co", "dl.dropboxusercontent.com", "storage.googleapis.com"}


@pytest.fixture(autouse=True)
def no_download_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("WFM_ASSET_MIRROR", raising=False)


@pytest.fixture
def fake_hub(monkeypatch):
    """Stand-in for huggingface_hub that serves `payload` and records each request."""
    hub = SimpleNamespace(calls=[], payload=None)

    def hf_hub_download(repo_id, filename, *, repo_type, revision, local_dir):
        hub.calls.append(dict(repo_id=repo_id, filename=filename, repo_type=repo_type,
                              revision=revision, local_dir=Path(local_dir)))
        (Path(local_dir) / ".cache/huggingface").mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / filename).write_bytes(hub.payload)
        return str(Path(local_dir) / filename)

    module = ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return hub


def hub_spec(payload, **overrides):
    return dict(path="models/model.safetensors", hf_repo="org/model", hf_file="model.safetensors",
                repo_type="model", revision="a" * 40, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), group="models") | overrides


@pytest.fixture
def mask_release(tmp_path):
    """A two-frame bundle built by the release packager, and its lock entry."""
    workspace = tmp_path / "workspace"
    (workspace / "test_images").mkdir(parents=True)
    (workspace / "test_dynamic").mkdir()
    for index in range(2):
        Image.new("RGB", (4, 3), (index, 10, 20)).save(workspace / f"test_images/1_{index:05d}.png")
        mask = Image.new("L", (4, 3), 0)
        mask.putpixel((index, 1), 255)
        mask.save(workspace / f"test_dynamic/1_{index:05d}.png")
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({"runs": [{"method": "ms", "scene": "wheel", "workspace": "workspace"}]}))
    bundle = tmp_path / "built" / "fixture-masks"
    report = build_bundle(runs, tmp_path, bundle, "fixture-masks")
    archive = tmp_path / "built" / "fixture-masks.zip"
    packed = write_archive(bundle, archive)
    pin = dict(path="downloads/fixture-masks.zip", hf_repo="org/masks", hf_file="fixture-masks.zip",
               repo_type="dataset", revision="b" * 40, bytes=packed["bytes"],
               sha256=packed["archive_sha256"], bundle="evaluation-masks/fixture-masks",
               manifest_sha256=report["manifest_sha256"], files=report["files"])
    return SimpleNamespace(bundle=bundle, archive=archive, pin=pin)


def repack(mask_release, tmp_path, edit):
    """Rebuild the fixture archive after `edit(bundle_copy)`, pinning the new archive hash."""
    copy = tmp_path / "edited" / "fixture-masks"
    shutil.copytree(mask_release.bundle, copy)
    edit(copy)
    archive = tmp_path / "edited.zip"
    packed = write_archive(copy, archive)
    return archive.read_bytes(), dict(mask_release.pin, bytes=packed["bytes"],
                                      sha256=packed["archive_sha256"])


def use_lock(monkeypatch, tmp_path, artifacts, wfm):
    here = tmp_path / "lock"
    here.mkdir()
    (here / "assets.lock.json").write_text(json.dumps(dict(schema_version=2, artifacts=artifacts, wfm=wfm)))
    monkeypatch.setattr(download, "HERE", here)


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
    use_lock(monkeypatch, tmp_path, [hub_spec(b"model")],
             hub_spec(b"wfm", path="models/wfm.safetensors", revision=download.UNPUBLISHED, pending="placeholder"))
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
    with pytest.raises(ValueError, match="No published download"):
        download.fetch(tmp_path, dict(path="models/model.safetensors", url=None))
    assert not (tmp_path / "models").exists()


def test_lock_uses_only_public_https_sources():
    assert "s3://" not in (HERE / "assets.lock.json").read_text()
    for item in [*LOCK["artifacts"], LOCK["wfm"]]:
        assert not {"endpoint", "profile"} & set(item), item["path"]
        if item.get("url"):
            url = urlsplit(item["url"])
            assert url.scheme == "https" and url.hostname in PUBLIC_HOSTS, item["url"]


def test_lock_pins_every_hub_source_to_a_commit():
    download.check_lock(LOCK)
    for item in download.pinned(LOCK):
        on_hub = item.get("hf_repo") or "huggingface.co" in (item.get("url") or "")
        if on_hub or "revision" in item:
            assert re.fullmatch(r"[0-9a-f]{40}", item["revision"]), item["path"]
        if "huggingface.co" in (item.get("url") or ""):
            assert f"/resolve/{item['revision']}/" in item["url"], item["path"]
    wfm, masks = LOCK["wfm"], LOCK["masks"]
    assert wfm["repo_type"] == "model" and wfm["hf_file"] == Path(wfm["path"]).name
    assert masks["repo_type"] == "dataset" and masks["hf_file"] == Path(masks["path"]).name
    assert Path(masks["path"]).stem == Path(masks["bundle"]).name
    # The placeholder note and the null revision are replaced together on publication.
    for item in (wfm, masks):
        assert ("pending" in item) == (item["revision"] == download.UNPUBLISHED)


def test_lock_mask_release_covers_both_methods_and_all_scenes():
    masks = LOCK["masks"]
    assert re.fullmatch(r"[0-9a-f]{64}", masks["manifest_sha256"])
    per_method = dict(apple=532, block=667, **{"paper-windmill": 419, "space-out": 681},
                      spin=722, teddy=688, wheel=219)
    assert masks["files"] == 2 * sum(per_method.values())
    assert "sam3" in masks["bundle"]


def test_sam3_is_its_own_download_group():
    groups = {item["path"]: item["group"] for item in LOCK["artifacts"]}
    assert groups["models/sam3/sam3.pt"] == "sam3"
    assert sum(group == "sam3" for group in groups.values()) == 1


def test_lock_checkpoint_identity_matches_sampling_config_and_notice():
    config = json.loads((ROOT / "config/dycheck_sampling.json").read_text())
    wfm = LOCK["wfm"]
    assert Path(wfm["path"]).name == config["checkpoint"]
    assert wfm["sha256"] == config["checkpoint_sha256"]
    assert wfm["bytes"] == 16591588312
    assert wfm["sha256"] in (ROOT / "NOTICE").read_text()


@pytest.mark.parametrize("revision", ["main", "v1.0", "refs/pr/1", "3c879f3", "A" * 40, None])
def test_mutable_revisions_are_rejected_before_network(tmp_path, fake_hub, revision):
    item = hub_spec(b"weights")
    if revision is None:
        del item["revision"]
    else:
        item["revision"] = revision
    with pytest.raises(ValueError, match="40-hex"):
        download.check_lock(dict(artifacts=[item], wfm=LOCK["wfm"]))
    with pytest.raises(ValueError, match="40-hex"):
        download.fetch(tmp_path, item)
    assert not fake_hub.calls


def test_hub_download_is_staged_beside_target_and_records_pin(tmp_path, fake_hub):
    payload = b"published weights"
    fake_hub.payload = payload
    records = []
    target = download.fetch(tmp_path, hub_spec(payload), records=records)
    assert target.read_bytes() == payload
    (call,) = fake_hub.calls
    assert call["local_dir"].parent == target.parent
    assert (call["repo_type"], call["revision"]) == ("model", "a" * 40)
    assert not call["local_dir"].exists()
    assert records == [dict(path="models/model.safetensors", bytes=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest(), source="huggingface",
                            repo="org/model@" + "a" * 40, repo_type="model")]


def test_tampered_hub_download_is_rejected_and_discarded(tmp_path, fake_hub):
    fake_hub.payload = b"published weightZ"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.fetch(tmp_path, hub_spec(b"published weights"))
    assert not (tmp_path / "models/model.safetensors").exists()
    assert not fake_hub.calls[0]["local_dir"].exists()


def test_mirror_supplies_verified_copy_offline(tmp_path, fake_hub):
    payload = b"mirrored weights"
    mirror = tmp_path / "mirror"
    (mirror / "models").mkdir(parents=True)
    (mirror / "models/model.safetensors").write_bytes(payload)
    records = []
    target = download.fetch(tmp_path / "root", hub_spec(payload), mirror=mirror, offline=True,
                            records=records)
    assert target.read_bytes() == payload
    assert (mirror / "models/model.safetensors").read_bytes() == payload
    assert not fake_hub.calls
    assert records[0]["source"] == "mirror" and records[0]["repo"] == "org/model@" + "a" * 40


def test_tampered_mirror_file_is_rejected(tmp_path, fake_hub):
    mirror = tmp_path / "mirror"
    (mirror / "models").mkdir(parents=True)
    (mirror / "models/model.safetensors").write_bytes(b"mirrored weightZ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.fetch(tmp_path / "root", hub_spec(b"mirrored weights"), mirror=mirror)
    assert not list((tmp_path / "root/models").iterdir())
    assert not fake_hub.calls


def test_offline_fetch_never_uses_network(tmp_path, fake_hub, monkeypatch):
    monkeypatch.setattr(download.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("offline fetch opened a URL"))
    with pytest.raises(ValueError, match="Offline"):
        download.fetch(tmp_path, hub_spec(b"weights"), offline=True)
    with pytest.raises(ValueError, match="Offline"):
        download.fetch(tmp_path, dict(path="model", url="https://example.test/model", bytes=1,
                                      sha256="0" * 64), offline=True)
    assert not fake_hub.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("use_environment", [False, True])
def test_offline_run_fails_before_any_download(tmp_path, fake_hub, monkeypatch, use_environment):
    argv = ["download.py", "--root", str(tmp_path / "inputs"), "--group", "models"]
    if use_environment:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        argv.append("--offline")
    monkeypatch.setattr(download, "fetch", lambda *a, **k: pytest.fail("offline run fetched"))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        download.main()
    assert exc.value.code == 2
    assert not (tmp_path / "inputs").exists()


def test_offline_mirror_run_records_each_source(tmp_path, fake_hub, monkeypatch):
    model, weights = b"pretrained prior", b"wfm weights"
    prior = hub_spec(model, path="models/prior.pth")
    wfm = hub_spec(weights, path="models/wfm-step-12500.safetensors", revision=download.UNPUBLISHED,
                   pending="placeholder")
    use_lock(monkeypatch, tmp_path, [prior], wfm)
    mirror = tmp_path / "mirror"
    for item, payload in ((prior, model), (wfm, weights)):
        (mirror / item["path"]).parent.mkdir(parents=True, exist_ok=True)
        (mirror / item["path"]).write_bytes(payload)
    monkeypatch.setenv("WFM_ASSET_MIRROR", str(mirror))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    root = tmp_path / "inputs"
    for group in ("models", "wfm"):
        monkeypatch.setattr(sys, "argv", ["download.py", "--root", str(root), "--group", group])
        download.main()
    assert not fake_hub.calls
    models = json.loads((root / "downloads-models.json").read_text())
    assert models["offline"] is True
    assert models["assets"] == [dict(path="models/prior.pth", bytes=len(model),
                                     sha256=hashlib.sha256(model).hexdigest(), source="mirror",
                                     mirror=str(mirror.resolve()), repo="org/model@" + "a" * 40,
                                     repo_type="model")]
    (record,) = json.loads((root / "downloads-wfm.json").read_text())["assets"]
    assert record["source"] == "mirror" and "repo" not in record
    assert (root / wfm["path"]).read_bytes() == weights


def test_checkpoint_url_and_preplaced_file_work_while_unpublished(tmp_path, fake_hub, monkeypatch):
    weights = b"wfm weights"
    wfm = hub_spec(weights, path="models/wfm-step-12500.safetensors", revision=download.UNPUBLISHED,
                   pending="placeholder")
    use_lock(monkeypatch, tmp_path, [], wfm)
    monkeypatch.setattr(download, "fetch_https", lambda source, partial, size: partial.write_bytes(weights))
    root = tmp_path / "inputs"
    monkeypatch.setattr(sys, "argv", ["download.py", "--root", str(root), "--group", "wfm", "--wfm-url",
                                      "https://example.test/wfm-step-12500.safetensors?signature=secret"])
    download.main()
    (record,) = json.loads((root / "downloads-wfm.json").read_text())["assets"]
    assert record["source"] == "url"
    assert record["url"] == "https://example.test/wfm-step-12500.safetensors"
    monkeypatch.setattr(download, "fetch_https", lambda *a: pytest.fail("verified file re-downloaded"))
    monkeypatch.setattr(sys, "argv", ["download.py", "--root", str(root), "--group", "wfm"])
    download.main()
    (record,) = json.loads((root / "downloads-wfm.json").read_text())["assets"]
    assert record["source"] == "existing"
    assert not fake_hub.calls


def test_generate_rejects_wrong_checkpoint_before_loading(tmp_path):
    (tmp_path / "wfm-step-12500.safetensors").write_bytes(b"not the released checkpoint")
    args = SimpleNamespace(models=tmp_path, output=tmp_path / "output", method="ms", scene="wheel")
    with pytest.raises(ValueError, match="Wrong WfM checkpoint"):
        stages.generate(args)


def test_wfm_checkpoint_requires_pinned_hash(tmp_path):
    payload = b"weights"
    path = tmp_path / "wfm-step-12500.safetensors"
    path.write_bytes(payload)
    pin = dict(path="models/wfm-step-12500.safetensors", bytes=len(payload),
               sha256=hashlib.sha256(payload).hexdigest())
    assert stages.wfm_checkpoint(tmp_path, pin) == path
    path.write_bytes(b"weightZ")
    with pytest.raises(ValueError, match="Wrong WfM checkpoint"):
        stages.wfm_checkpoint(tmp_path, pin)


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


def test_mask_archive_from_hub_is_unpacked_verified_and_recorded(tmp_path, fake_hub, mask_release):
    fake_hub.payload = mask_release.archive.read_bytes()
    records = []
    target = download.fetch_masks(tmp_path / "root", mask_release.pin, records=records)
    (call,) = fake_hub.calls
    assert (call["repo_type"], call["revision"], call["filename"]) == ("dataset", "b" * 40, "fixture-masks.zip")
    assert target == tmp_path / "root/evaluation-masks/fixture-masks"
    assert sorted(p.name for p in target.rglob("*.png")) == ["1_00000.png", "1_00001.png"]
    assert (target / "manifest.json").read_bytes() == (mask_release.bundle / "manifest.json").read_bytes()
    assert not list(target.parent.glob(".*"))
    assert records[0]["repo"] == "org/masks@" + "b" * 40 and records[0]["source"] == "huggingface"
    # A verified bundle is reused without the network; a later edit is caught.
    assert download.fetch_masks(tmp_path / "root", mask_release.pin, offline=True) == target
    (target / "ms/wheel/test_dynamic/1_00000.png").write_bytes(b"edited")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.fetch_masks(tmp_path / "root", mask_release.pin, offline=True)


def test_tampered_mask_archive_is_rejected_before_unpacking(tmp_path, fake_hub, mask_release):
    data = bytearray(mask_release.archive.read_bytes())
    data[-30] ^= 1
    fake_hub.payload = bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.fetch_masks(tmp_path / "root", mask_release.pin)
    assert not (tmp_path / "root/evaluation-masks").exists()


def corrupt_mask(bundle):
    (bundle / "ms/wheel/test_dynamic/1_00000.png").write_bytes(b"corrupt")


def edit_manifest(bundle):
    (bundle / "manifest.json").write_text((bundle / "manifest.json").read_text() + " ")


def add_unlisted(bundle):
    (bundle / "ms/wheel/test_dynamic/payload.py").write_text("unlisted")


def drop_mask(bundle):
    (bundle / "ms/wheel/test_dynamic/1_00001.png").unlink()


@pytest.mark.parametrize("edit", [corrupt_mask, edit_manifest, add_unlisted, drop_mask])
def test_archive_with_matching_hash_but_bad_bundle_is_rejected(tmp_path, fake_hub, mask_release, edit):
    fake_hub.payload, pin = repack(mask_release, tmp_path, edit)
    with pytest.raises((ValueError, FileNotFoundError)):
        download.fetch_masks(tmp_path / "root", pin)
    masks_dir = tmp_path / "root/evaluation-masks"
    assert not (masks_dir / "fixture-masks").exists() and not list(masks_dir.iterdir())


@pytest.mark.parametrize("member", ["../escape.png", "/abs.png", "fixture-masks/../../escape.png",
                                    "other/manifest.json"])
def test_unsafe_mask_archive_members_are_rejected(tmp_path, fake_hub, mask_release, member):
    archive = tmp_path / "unsafe.zip"
    shutil.copy(mask_release.archive, archive)
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr(member, b"x")
    fake_hub.payload = archive.read_bytes()
    pin = dict(mask_release.pin, bytes=archive.stat().st_size,
               sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="archive"):
        download.fetch_masks(tmp_path / "root", pin)
    assert not (tmp_path / "escape.png").exists() and not (tmp_path / "root/escape.png").exists()


def test_mask_archive_from_mirror_works_offline(tmp_path, fake_hub, mask_release):
    mirror = tmp_path / "mirror"
    (mirror / "downloads").mkdir(parents=True)
    shutil.copy(mask_release.archive, mirror / mask_release.pin["path"])
    records = []
    target = download.fetch_masks(tmp_path / "root", mask_release.pin, mirror=mirror, offline=True,
                                  records=records)
    assert (target / "manifest.json").is_file()
    assert records[0]["source"] == "mirror" and not fake_hub.calls


def test_unpublished_mask_archive_needs_mirror_or_copy(tmp_path, fake_hub, mask_release):
    pin = dict(mask_release.pin, revision=download.UNPUBLISHED)
    with pytest.raises(ValueError, match="No published download"):
        download.fetch_masks(tmp_path / "root", pin)
    with pytest.raises(ValueError, match="Offline"):
        download.fetch_masks(tmp_path / "root", mask_release.pin, offline=True)
    assert not fake_hub.calls


def test_masks_group_downloads_the_pinned_archive(tmp_path, fake_hub, mask_release, monkeypatch):
    use_lock(monkeypatch, tmp_path, [], dict(hub_spec(b"wfm"), path="models/wfm.safetensors"))
    lock = json.loads((download.HERE / "assets.lock.json").read_text())
    (download.HERE / "assets.lock.json").write_text(json.dumps(dict(lock, masks=mask_release.pin)))
    fake_hub.payload = mask_release.archive.read_bytes()
    root = tmp_path / "inputs"
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", ["download.py", "--root", str(root), "--group", "masks"])
        download.main()
    assert [call["repo_type"] for call in fake_hub.calls] == ["dataset"]
    (record,) = json.loads((root / "downloads-masks.json").read_text())["assets"]
    assert record["source"] == "existing" and (root / mask_release.pin["bundle"]).is_dir()


def test_released_evaluation_masks_use_the_lock_pin(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "run", seen.append)
    monkeypatch.setattr(sys, "argv", ["run.py", "--inputs", str(tmp_path / "inputs"), "--output",
                                      str(tmp_path / "output"), "--method", "wt", "--scene", "wheel",
                                      "--evaluation-masks", "released"])
    runner.main()
    (args,) = seen
    assert args.evaluation_mask_bundle == (tmp_path / "inputs").resolve() / LOCK["masks"]["bundle"]
    assert args.evaluation_mask_manifest_sha256 == LOCK["masks"]["manifest_sha256"]
    masks = dict(runner.commands(args))["masks"]
    assert masks[masks.index("--evaluation-mask-manifest-sha256") + 1] == LOCK["masks"]["manifest_sha256"]
