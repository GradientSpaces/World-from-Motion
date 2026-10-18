from __future__ import annotations

import json
from pathlib import Path

from wfm.config import load_production_config
from wfm.pipeline import _make_brush_metadata_portable
from wfm.validation import NATIVE_CHECKPOINTS, validate_output
from wfm.video import VideoInfo


def test_stable_output_contract_validates_without_model_imports(
    tmp_path: Path, monkeypatch
) -> None:
    frames = 5
    fps = 24.0
    profile = load_production_config()
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "input": {"selected_frames": frames, "fps": fps},
                "distillation": {"enabled": True, "steps": 2},
            }
        )
    )
    (tmp_path / "status.json").write_text(json.dumps({"state": "completed"}))
    native = tmp_path / "reconstruction" / "native"
    native.mkdir(parents=True)
    for name in NATIVE_CHECKPOINTS:
        (native / name).touch()
    initial = tmp_path / "reconstruction" / "initial" / "native"
    initial.mkdir(parents=True)
    for name in NATIVE_CHECKPOINTS:
        (initial / name).touch()
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "distillation_summary.json").write_text(
        json.dumps(
            {
                "num_frames": frames,
                "num_steps": 2,
                "target_cameras": list(profile.target_cameras),
                "final_metrics": {"loss_total": 1.0},
                "source_native": "../reconstruction/initial/native",
                "distilled_native": "../reconstruction/native",
                "benchmark_root": "../work/benchmark",
                "prediction_root": "../work/inference",
            }
        )
    )
    (diagnostics / "distillation_history.json").write_text(json.dumps([{}, {}]))
    brush = tmp_path / "reconstruction" / "brush"
    brush.mkdir()
    (brush / "scene_dynamic.ply").touch()
    (brush / "initial_camera.json").write_text("{}")
    (brush / "release_manifest.json").write_text(
        json.dumps(
            {
                "saved_dir": "../native",
                "ply_path": "scene_dynamic.ply",
                "total_splats": 10,
                "frames": frames,
                "source_frame_indices": list(range(frames)),
                "validation": {
                    "byte_size_match": True,
                    "max_roundtrip_error": 0.01,
                    "max_roundtrip_bound": 0.02,
                },
            }
        )
    )
    novel = tmp_path / "novel_views"
    novel.mkdir()
    for camera in profile.target_cameras:
        (novel / f"{camera}.mp4").touch()
    (novel / "all_views_2x2.mp4").touch()
    monkeypatch.setattr(
        "wfm.validation.probe_video",
        lambda _: VideoInfo(width=768, height=432, fps=fps, frames=frames, duration=frames / fps),
    )

    assert validate_output(tmp_path, profile)["ok"]


def test_brush_metadata_is_made_relocatable(tmp_path: Path) -> None:
    native = tmp_path / "reconstruction" / "native"
    brush = tmp_path / "reconstruction" / "brush"
    native.mkdir(parents=True)
    brush.mkdir()
    (brush / "release_manifest.json").write_text(
        json.dumps(
            {
                "saved_dir": "/old/run/reconstruction/native",
                "ply_file": "scene_dynamic.ply",
                "ply_path": "/old/run/reconstruction/brush/scene_dynamic.ply",
            }
        )
    )
    (brush / "initial_camera.json").write_text(
        json.dumps({"saved_dir": "/old/run/reconstruction/native"})
    )

    _make_brush_metadata_portable(native, brush)

    release = json.loads((brush / "release_manifest.json").read_text())
    camera = json.loads((brush / "initial_camera.json").read_text())
    assert release["saved_dir"] == "../native"
    assert release["ply_path"] == "scene_dynamic.ply"
    assert camera["saved_dir"] == "../native"
