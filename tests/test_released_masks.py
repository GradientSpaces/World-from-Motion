# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Released masks must bind exact mask bytes to the correct held-out RGB."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reproduce/from_raw"))
from wfm.released_masks import build_bundle, install_bundle, verify_bundle
import stages

spec = importlib.util.spec_from_file_location("mask_runner", ROOT / "reproduce/from_raw/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def release(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for root in (source, target):
        (root / "test_images").mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (4, 3), (index, 10, 20)).save(root / "test_images" / f"1_{index:05d}.png")
    (source / "test_dynamic").mkdir()
    for index in range(2):
        mask = Image.new("L", (4, 3), 0)
        mask.putpixel((index, 1), 255)
        mask.save(source / "test_dynamic" / f"1_{index:05d}.png")
    manifest = tmp_path / "runs.json"
    manifest.write_text(json.dumps({"runs": [{"method": "ms", "scene": "wheel", "workspace": "source"}]}))
    bundle = tmp_path / "bundle"
    report = build_bundle(manifest, tmp_path, bundle, "fixture")
    return source, target, bundle, report["manifest_sha256"]


def test_exact_roundtrip_and_no_private_paths(release):
    source, target, bundle, sha = release
    report, _ = verify_bundle(bundle, sha, "ms", "wheel", target)
    assert not (target / "test_dynamic").exists()
    assert report["frames"] == 2
    report = install_bundle(bundle, sha, "ms", "wheel", target)
    assert report["reference_rgb_verified"]
    for original in (source / "test_dynamic").iterdir():
        assert original.read_bytes() == (target / "test_dynamic" / original.name).read_bytes()
    assert str(source.parent) not in (bundle / "manifest.json").read_text()
    assert not list(bundle.rglob("test_images"))
    with pytest.raises(FileExistsError):
        install_bundle(bundle, sha, "ms", "wheel", target)


@pytest.mark.parametrize("damage", ["mask", "manifest", "rgb", "missing", "duplicate", "traversal", "method", "dimensions"])
def test_reject_bad_bundle_before_creating_masks(release, damage):
    source, target, bundle, sha = release
    manifest_path = bundle / "manifest.json"
    method = "ms"
    if damage == "mask":
        (bundle / "ms/wheel/test_dynamic/1_00000.png").write_bytes(b"corrupt")
    elif damage == "manifest":
        manifest_path.write_text(manifest_path.read_text() + " ")
    elif damage in ("rgb", "dimensions"):
        Image.new("RGB", (5, 3) if damage == "dimensions" else (4, 3), (9, 9, 9)).save(target / "test_images/1_00000.png")
    elif damage == "method":
        method = "wt"
    else:
        manifest = json.loads(manifest_path.read_text())
        frames = manifest["sets"][0]["frames"]
        if damage == "missing":
            frames.pop()
        elif damage == "duplicate":
            frames.append(frames[0])
        else:
            frames[0]["name"] = "../escape.png"
        manifest_path.write_text(json.dumps(manifest))
        sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        install_bundle(bundle, sha, method, "wheel", target)
    assert not (target / "test_dynamic").exists()


def test_repack_refuses_existing_bundle(release):
    source, target, bundle, sha = release
    with pytest.raises(FileExistsError):
        build_bundle(source.parent / "runs.json", source.parent, bundle, "fixture")


def test_mask_stage_bypasses_sam3(release, monkeypatch):
    source, target, bundle, sha = release
    # If the legacy branch imports the SAM3 wrapper this test must fail.
    monkeypatch.setitem(sys.modules, "extract_test_masks", None)
    args = SimpleNamespace(output=target.parent / "run", method="ms", scene="wheel",
                           evaluation_mask_bundle=bundle, evaluation_mask_manifest_sha256=sha)
    args.output.mkdir()
    (args.output / "workspace").symlink_to(target, target_is_directory=True)
    stages.masks(args)
    assert json.loads((args.output / "evaluation_masks.json").read_text())["source"] == "released_bundle"


def test_runner_only_passes_bundle_to_masks(release):
    source, target, bundle, sha = release
    args = SimpleNamespace(output=target, inputs=source, method="ms", scene="wheel", cpu_threads=1,
                           evaluation_mask_bundle=bundle, evaluation_mask_manifest_sha256=sha)
    plan = dict(runner.commands(args))
    for stage, command in plan.items():
        assert ("--evaluation-mask-bundle" in command) == (stage == "masks")
    args.evaluation_mask_manifest_sha256 = None
    with pytest.raises(ValueError, match="Provide both"):
        runner.commands(args)


def test_runner_rejects_bad_pin_before_creating_run_or_starting_stages(release, monkeypatch):
    source, target, bundle, sha = release
    inputs = source.parent / "inputs"
    (inputs / "observations").mkdir(parents=True)
    (inputs / "observations/wheel").symlink_to(target, target_is_directory=True)
    (inputs / "models").mkdir()
    output = source.parent / "fresh-run"
    args = SimpleNamespace(output=output, inputs=inputs, method="ms", scene="wheel", cpu_threads=1,
                           plan=False, evaluation_mask_bundle=bundle,
                           evaluation_mask_manifest_sha256="0" * 64)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("No stage may start"))
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        runner.run(args)
    assert not output.exists()


@pytest.mark.parametrize("difference", [None, "mask", "rgb"])
def test_distinct_alignment_and_scoring_workspaces_must_agree(release, difference):
    source, target, bundle, sha = release
    # 'target' has matching RGB, but create independently stored scoring masks.
    (target / "test_dynamic").mkdir()
    for mask in (source / "test_dynamic").iterdir():
        (target / "test_dynamic" / mask.name).write_bytes(mask.read_bytes())
    if difference == "mask":
        Image.new("L", (4, 3), 255).save(target / "test_dynamic/1_00000.png")
    elif difference == "rgb":
        Image.new("RGB", (4, 3), (99, 0, 0)).save(target / "test_images/1_00000.png")
    manifest = source.parent / "roles.json"
    manifest.write_text(json.dumps({"runs": [{"method": "wt", "scene": "wheel",
        "workspace": "source", "metric_workspace": "target"}]}))
    output = source.parent / "role-bundle"
    if difference:
        with pytest.raises(ValueError, match="Alignment/scoring mask or RGB identities differ"):
            build_bundle(manifest, source.parent, output, "role-fixture")
        assert not output.exists()
    else:
        build_bundle(manifest, source.parent, output, "role-fixture")
        assert json.loads((output / "manifest.json").read_text())["sets"][0]["metric_workspace_identity_checked"]
