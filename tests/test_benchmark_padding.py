from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wfm.benchmark import prepare_inference_benchmark


def _make_scene(root: Path, frames: int = 2) -> None:
    scene = root / "scene"
    cameras = ("cam0", "cam_fps_00")
    for camera in cameras:
        records = []
        for index in range(frames):
            records.append(
                {
                    "frame_idx": index,
                    "time_id": index,
                    "frame_name": f"{index:05d}",
                    "reference_frame_name": f"{index:05d}",
                    "source_frame_name": f"{index:05d}",
                    "has_gt": camera == "cam0",
                    "scene_flow_valid": True,
                }
            )
        metadata = {
            "num_frames": frames,
            "num_valid_gt": frames if camera == "cam0" else 0,
            "frames": records,
        }
        path = scene / "metadata" / f"{camera}_frames.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata))
        frame_names = scene / "frame_names"
        frame_names.mkdir(exist_ok=True)
        (frame_names / f"{camera}_all.txt").write_text("".join(f"{i:05d}\n" for i in range(frames)))
        valid = range(frames) if camera == "cam0" else ()
        (frame_names / f"{camera}_valid.txt").write_text("".join(f"{i:05d}\n" for i in valid))
        for modality in ("rgb", "depth", "normal", "alpha", "gt", "scene_flow_rgb"):
            directory = scene / modality / camera
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(frames):
                (directory / f"{index:05d}.png").write_bytes(f"{camera}-{index}".encode())
        camera_dir = scene / "cameras"
        camera_dir.mkdir(exist_ok=True)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
        np.savez_compressed(camera_dir / f"{camera}_c2w.npz", c2w=poses)


def test_short_benchmark_is_padded_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "padded"
    _make_scene(source)
    root, original, padded = prepare_inference_benchmark(source, destination, "scene", 5)
    assert root == destination
    assert (original, padded) == (2, 5)
    payload = json.loads((destination / "scene/metadata/cam_fps_00_frames.json").read_text())
    assert len(payload["frames"]) == 5
    assert payload["frames"][-1]["frame_name"] == "00004"
    assert payload["frames"][-1]["reference_frame_name"] == "00001"
    with np.load(destination / "scene/cameras/cam_fps_00_c2w.npz") as data:
        assert data["c2w"].shape == (5, 4, 4)
    original_payload = json.loads((source / "scene/metadata/cam_fps_00_frames.json").read_text())
    assert len(original_payload["frames"]) == 2


def test_long_benchmark_is_not_cloned(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_scene(source, frames=3)
    root, original, padded = prepare_inference_benchmark(source, tmp_path / "unused", "scene", 3)
    assert root == source
    assert (original, padded) == (3, 3)


def test_incomplete_padded_scene_is_quarantined_and_rebuilt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "padded"
    _make_scene(source)
    incomplete = destination / "scene"
    incomplete.mkdir(parents=True)
    (incomplete / "partial.txt").write_text("interrupted")

    prepare_inference_benchmark(source, destination, "scene", 5)

    payload = json.loads((destination / "scene/metadata/cam0_frames.json").read_text())
    assert len(payload["frames"]) == 5
    quarantined = list(destination.glob(".scene.invalid-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "partial.txt").read_text() == "interrupted"
