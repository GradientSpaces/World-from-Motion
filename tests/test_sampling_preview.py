# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib.util
import json
from pathlib import Path

import pytest


def fixture(tmp_path, failure=None):
    path = Path(__file__).resolve().parents[1] / "reproduce/sampling/preview.py"
    spec = importlib.util.spec_from_file_location("sampling_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"; run.mkdir()
    source = tmp_path / "source"
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(run / "score-execution.json", dict(status="complete", returncode=0,
          source_run=str(source), scene="wheel", method="wt"))
    write(run / "source-unchanged.json", {})
    write(run / "camera-audit.json", dict(targets=["1"], full_sequence_frames=2, evaluated_frames=1))
    write(run / "conditioning/wheel/metadata/cam1_frames.json", dict(num_frames=2,
          frames=[dict(frame_name="1_00000"), dict(frame_name="1_00001")]))
    for name in ("1_00000", "1_00001"):
        if failure != "missing_generated" or name != "1_00001":
            write(run / "samples/wheel/pred_rgb/cam1" / (name + ".png"), name)
    if failure != "missing_gt":
        write(source / "workspace/test_images/1_00000.png", "gt")
    write(run / "predictions/1_00000.png", "score")
    write(run / "alignment/tto_test/1_00000.png", "initial")
    write(run / "prediction-audit.json", dict(frames=[dict(frame="1_00000.png",
          generated_sha256=module.digest(run / "samples/wheel/pred_rgb/cam1/1_00000.png"),
          scored_sha256=module.digest(run / "predictions/1_00000.png"))]))
    return module, run


def test_preview_validates_complete_sequences(tmp_path):
    module, run = fixture(tmp_path)
    _, gt, sequences = module.validate_inputs(run)
    assert len(gt) == 1 and len(sequences["1"]) == 2


@pytest.mark.parametrize("failure", ["missing_gt", "missing_generated"])
def test_preview_rejects_incomplete_inputs_before_creating_output(tmp_path, failure):
    module, run = fixture(tmp_path, failure)
    output = tmp_path / "preview"
    with pytest.raises(ValueError, match="coverage mismatch"):
        module.preview(run, output, 10.)
    assert not output.exists()


def test_sample_labels_remain_correct_across_frames(tmp_path, monkeypatch):
    module, run = fixture(tmp_path)
    stage = dict(scene="wheel", method="wt", config=dict(guidance_mode="apg", guidance_scale=3.0))
    gt = []
    generated = []
    for i in range(2):
        name = f"1_{i:05d}.png"
        for folder in (tmp_path / "gt", run / "alignment/tto_test", run / "predictions", tmp_path / "generated"):
            folder.mkdir(parents=True, exist_ok=True)
            module.Image.new("RGB", (32, 32)).save(folder / name)
        gt.append(tmp_path / "gt" / name)
        generated.append(tmp_path / "generated" / name)
    monkeypatch.setattr(module, "validate_inputs", lambda _: (stage, gt, {"1": generated}))
    class Writer:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def append_data(self, frame): pass
    monkeypatch.setattr(module.imageio, "get_writer", lambda *a, **kw: Writer())
    monkeypatch.setattr(module, "digest", lambda _: "mock-video-hash")
    labels = []
    original = module.ImageDraw.ImageDraw.text
    def text(self, xy, value, *args, **kwargs):
        labels.append(value)
        return original(self, xy, value, *args, **kwargs)
    monkeypatch.setattr(module.ImageDraw.ImageDraw, "text", text)
    module.preview(run, tmp_path / "preview", 10.)
    assert labels.count("WfM sample, APG 3.0") == 2
    assert labels.count("Held-out reference") == 2
