# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression checks for paper mask transport and empty-track rejection."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reproduce/from_raw"))
from wfm import dycheck_masks
import stages


def test_ffv1_preserves_rgb_bytes_and_frame_count(tmp_path):
    # Odd dimensions and sharp chroma changes expose padding/chroma subsampling.
    frames = [np.random.default_rng(i).integers(0, 256, (17, 19, 3), dtype=np.uint8)
              for i in range(3)]
    report = dycheck_masks.write_lossless_rgb(frames, tmp_path / "rgb.mkv", 10)
    assert report["decoded_exact_rgb"] and report["frames"] == 3
    assert report["codec"] == "ffv1" and len(report["video_sha256"]) == 64
    with pytest.raises(FileExistsError):
        dycheck_masks.write_lossless_rgb(frames, tmp_path / "rgb.mkv", 10)


def load_extractor(monkeypatch, backend):
    prior = ModuleType("lib_prior")
    prior.sam3_wrapper = backend
    monkeypatch.setitem(sys.modules, "lib_prior", prior)
    monkeypatch.setitem(sys.modules, "lib_prior.sam3_wrapper", backend)
    spec = importlib.util.spec_from_file_location(
        "mask_extractor_test", ROOT / "reproduce/from_raw/runtime/extract_test_masks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_routes_lossless_with_unchanged_seed_and_model(tmp_path, monkeypatch):
    backend = ModuleType("lib_prior.sam3_wrapper")
    calls = []
    def legacy(**kwargs):
        pytest.fail("Paper must not use compressed RGB transport")
    backend.generate_sam3_epi = legacy
    backend.save_sam3_epi_artifacts = lambda *args, **kwargs: None
    backend._patch_sam3_nms_for_triton_compat = lambda root: None
    def infer(**kwargs):
        calls.append(kwargs)
        return "outputs"
    backend._run_sam3_text_prompt_video = infer
    backend._outputs_to_mask_list = lambda outputs, frames: [np.ones((3, 4), dtype=np.uint8)]
    implementation = load_extractor(monkeypatch, backend)
    monkeypatch.setattr(dycheck_masks, "write_lossless_rgb", lambda *args: {"decoded_exact_rgb": True})
    options = SimpleNamespace(force=False, gpus="0", sam3_checkpoint="model.pt",
                              sam3_bpe="bpe.gz", sam3_lib_prior_root="prior", fps=10)
    prompts = json.loads((ROOT / "reproduce/from_raw/configs/mask_prompts.json").read_text())
    result, stats = implementation.run_track_prompt_union(
        "paper-windmill", "2", str(tmp_path), ["2_00000.png"],
        [np.zeros((3, 4, 3), dtype=np.uint8)], prompts["paper-windmill"], options)
    assert stats["nonzero_frames"] == 1 and result[0]["used_in_union"]
    assert len(calls) == 1 and calls[0]["text_prompt"] == "pinwheel"
    assert calls[0]["seed_frame_indices"] is None
    assert calls[0]["checkpoint_path"] == "model.pt"
    assert calls[0]["video_path"].endswith("rgb-lossless.mkv")


def test_wheel_preserves_legacy_transport(tmp_path, monkeypatch):
    backend = ModuleType("lib_prior.sam3_wrapper")
    calls = []
    def legacy(**kwargs):
        calls.append(kwargs)
        return [np.ones((3, 4), dtype=np.uint8)]
    backend.generate_sam3_epi = legacy
    backend.save_sam3_epi_artifacts = lambda *args, **kwargs: None
    implementation = load_extractor(monkeypatch, backend)
    options = SimpleNamespace(force=False, gpus="0", sam3_checkpoint="model.pt",
                              sam3_bpe="bpe.gz", sam3_lib_prior_root="prior", fps=10)
    implementation.run_track_prompt_union("wheel", "1", str(tmp_path), ["1_00000.png"],
        [np.zeros((3, 4, 3), dtype=np.uint8)], ["steering wheel", "arm"], options)
    assert [call["text_prompt"] for call in calls] == ["steering wheel", "arm"]


def test_changed_rgb_is_rejected(tmp_path, monkeypatch):
    import cv2
    class CorruptDecoder:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        def release(self):
            pass
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: CorruptDecoder())
    with pytest.raises(ValueError, match="changed or lost"):
        dycheck_masks.write_lossless_rgb([np.full((4, 4, 3), 255, dtype=np.uint8)],
                                       tmp_path / "rgb.mkv", 10)


@pytest.mark.parametrize("second_track_nonempty", [False, True])
def test_mask_stage_checks_each_track_before_accepting(tmp_path, monkeypatch, second_track_nonempty):
    workspace = tmp_path / "workspace"
    images = workspace / "test_images"
    images.mkdir(parents=True)
    for name in ("1_00000.png", "2_00000.png"):
        Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(images / name)
    def process_scene(root, scene, options):
        assert options.rgb_transport == "ffv1"
        assert options.prompt_override_list == ["pinwheel"]
        out = Path(root) / options.output_dirname / "masks"
        out.mkdir(parents=True)
        Image.fromarray(np.ones((3, 4), dtype=np.uint8)).save(out / "1_00000.png")
        Image.fromarray(np.full((3, 4), int(second_track_nonempty), dtype=np.uint8)).save(out / "2_00000.png")
    monkeypatch.setitem(sys.modules, "extract_test_masks", SimpleNamespace(process_scene=process_scene))
    checkpoint = tmp_path / "sam3/sam3.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test-model")
    args = SimpleNamespace(scene="paper-windmill", output=tmp_path, models=tmp_path)
    if not second_track_nonempty:
        with pytest.raises(ValueError, match="test tracks"):
            stages.masks(args)
        assert not (workspace / "test_dynamic").exists()
        assert not (tmp_path / "evaluation_masks.json").exists()
    else:
        stages.masks(args)
        record = json.loads((tmp_path / "evaluation_masks.json").read_text())
        assert record["track_nonzero_pixels"] == {"1": 12, "2": 12}
        assert record["rgb_transport"] == "ffv1"
