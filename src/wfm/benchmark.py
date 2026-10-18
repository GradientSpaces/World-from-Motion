from __future__ import annotations

import json
import shutil
import time
from copy import deepcopy
from pathlib import Path


def _copy_file(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def prepare_inference_benchmark(
    benchmark_root: Path,
    destination_root: Path,
    scene_name: str,
    minimum_frames: int,
) -> tuple[Path, int, int]:
    """Clone and last-frame-pad a WorldTree scene only when DiffSynth needs it."""
    scene = benchmark_root / scene_name
    metadata_files = sorted((scene / "metadata").glob("*_frames.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No camera metadata under {scene}")
    sample = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    original_frames = len(sample["frames"])
    if original_frames >= minimum_frames:
        return benchmark_root, original_frames, original_frames

    destination_scene = destination_root / scene_name
    if destination_scene.exists():
        destination_metadata = sorted((destination_scene / "metadata").glob("*_frames.json"))
        try:
            complete = bool(destination_metadata) and all(
                len(json.loads(path.read_text(encoding="utf-8"))["frames"]) == minimum_frames
                for path in destination_metadata
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            complete = False
        if complete:
            return destination_root, original_frames, minimum_frames
        destination_scene.rename(
            destination_scene.with_name(f".{scene_name}.invalid-{time.time_ns()}")
        )
    destination_root.mkdir(parents=True, exist_ok=True)
    working_scene = destination_root / f".{scene_name}.partial"
    if working_scene.exists():
        working_scene.rename(working_scene.with_name(f".{scene_name}.invalid-{time.time_ns()}"))
    shutil.copytree(scene, working_scene, copy_function=shutil.copy2)

    for metadata_path in sorted((working_scene / "metadata").glob("*_frames.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = payload["frames"]
        last = frames[-1]
        for index in range(len(frames), minimum_frames):
            record = deepcopy(last)
            record.update(
                {
                    "frame_idx": index,
                    "time_id": int(last.get("time_id", original_frames - 1)),
                    "frame_name": f"{index:05d}",
                    "reference_frame_name": str(last.get("reference_frame_name", last["frame_name"])),
                    "source_frame_name": str(last.get("source_frame_name", last["frame_name"])),
                    "has_gt": False,
                    "has_test_dynamic": False,
                    "has_dataset_frame_at_time": False,
                    "scene_flow_valid": False,
                }
            )
            frames.append(record)
        payload["num_frames"] = minimum_frames
        payload["num_valid_gt"] = sum(bool(record.get("has_gt")) for record in frames)
        metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        camera = metadata_path.name.removesuffix("_frames.json")
        all_path = working_scene / "frame_names" / f"{camera}_all.txt"
        valid_path = working_scene / "frame_names" / f"{camera}_valid.txt"
        all_path.write_text("".join(f"{record['frame_name']}\n" for record in frames), encoding="utf-8")
        valid_path.write_text(
            "".join(f"{record['frame_name']}\n" for record in frames if record.get("has_gt")),
            encoding="utf-8",
        )

    modalities = (
        "rgb",
        "depth",
        "normal",
        "alpha",
        "gt",
        "scene_coords",
        "scene_flow",
        "scene_flow_rgb",
        "test_dynamic",
    )
    for modality in modalities:
        modality_root = working_scene / modality
        if not modality_root.is_dir():
            continue
        for camera_dir in (path for path in modality_root.iterdir() if path.is_dir()):
            existing = sorted(path for path in camera_dir.iterdir() if path.is_file())
            if not existing:
                continue
            last = existing[-1]
            for index in range(original_frames, minimum_frames):
                _copy_file(last, camera_dir / f"{index:05d}{last.suffix}")

    import numpy as np

    for camera_path in sorted((working_scene / "cameras").glob("*_c2w.npz")):
        with np.load(camera_path) as data:
            payload = {key: data[key] for key in data.files}
        for key, value in payload.items():
            if value.ndim >= 1 and value.shape[0] == original_frames:
                repeats = np.repeat(value[-1:], minimum_frames - original_frames, axis=0)
                payload[key] = np.concatenate([value, repeats], axis=0)
        np.savez_compressed(camera_path, **payload)

    summary_path = working_scene / "metadata" / "scene_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["num_frames"] = minimum_frames
        summary["inference_padding"] = {
            "original_frames": original_frames,
            "padded_frames": minimum_frames,
            "mode": "repeat_last",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    working_scene.replace(destination_scene)
    return destination_root, original_frames, minimum_frames
