# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Make labeled video previews from completed, scored sample outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_inputs(run):
    stage = json.loads((run / "score-execution.json").read_text())
    if stage["status"] != "complete" or stage.get("returncode") != 0:
        raise ValueError("Score stage has not completed")
    if not (run / "source-unchanged.json").is_file():
        raise ValueError("Missing initial-checkpoint verification")
    source = Path(stage["source_run"])
    scene, method = stage["scene"], stage["method"]
    gt_paths = sorted((source / "workspace/test_images").glob("*.png"))
    audit = json.loads((run / "prediction-audit.json").read_text())["frames"]
    camera_audit = json.loads((run / "camera-audit.json").read_text())
    expected_gt = [r["frame"] for r in audit]
    if (not gt_paths or len(set(expected_gt)) != len(expected_gt)
            or sorted(expected_gt) != [p.name for p in gt_paths]
            or len(gt_paths) != camera_audit["evaluated_frames"]):
        raise ValueError("Held-out frame coverage mismatch")
    if sorted(p.name for p in (run / "predictions").glob("*.png")) != sorted(expected_gt):
        raise ValueError("Scored frame coverage mismatch")
    cameras = sorted({p.stem.split("_")[0] for p in gt_paths}, key=int)
    if cameras != sorted(camera_audit["targets"], key=int):
        raise ValueError("Target camera coverage mismatch")
    sequences = {}
    for camera in cameras:
        metadata = json.loads((run / "conditioning" / scene / "metadata" /
                               f"cam{camera}_frames.json").read_text())
        expected = [f["frame_name"] + ".png" for f in metadata["frames"]]
        generated = sorted((run / "samples" / scene / "pred_rgb" / f"cam{camera}").glob("*.png"))
        if (not generated or len(expected) != camera_audit["full_sequence_frames"]
                or metadata["num_frames"] != len(expected)
                or len(set(expected)) != len(expected)
                or sorted(expected) != [p.name for p in generated]):
            raise ValueError("Full generated sequence coverage mismatch")
        sequences[camera] = generated
    for record in audit:
        name = record["frame"]
        generated = run / "samples" / scene / "pred_rgb" / f"cam{name.split('_')[0]}" / name
        if (digest(generated) != record["generated_sha256"]
                or digest(run / "predictions" / name) != record["scored_sha256"]):
            raise ValueError("Prediction hash mismatch")
        if not (run / "alignment/tto_test" / name).is_file():
            raise ValueError("Missing initial comparison frame")
    return stage, gt_paths, sequences


def preview(run, output, fps):
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Positive finite playback frame rate required")
    stage, gt_paths, sequences = validate_inputs(run)
    scene, method = stage["scene"], stage["method"]
    sample_label = f"WfM sample, {stage['config']['guidance_mode'].upper()} {stage['config']['guidance_scale']:.1f}"
    output.mkdir(parents=True)
    records = []
    for camera, generated in sequences.items():
        movie = output / f"{method}-{scene}-cam{camera}-generated.mp4"
        with imageio.get_writer(movie, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-threads", "2"]) as writer:
            for path in generated:
                with Image.open(path) as image:
                    writer.append_data(np.asarray(image.convert("RGB")))
        held_out = [p for p in gt_paths if p.stem.split("_")[0] == camera]
        compare = output / f"{method}-{scene}-cam{camera}-comparison.mp4"
        with imageio.get_writer(compare, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-threads", "2"]) as writer:
            for index, gt in enumerate(held_out):
                with Image.open(gt) as image:
                    reference = image.convert("RGB")
                paths = [run / "alignment/tto_test" / gt.name, run / "predictions" / gt.name]
                panels = []
                for path in paths:
                    with Image.open(path) as image:
                        panel = image.convert("RGB")
                    if panel.size != reference.size:
                        raise ValueError("Comparison panel geometry mismatch")
                    panels.append(panel)
                panels.append(reference)
                w, h = reference.size
                canvas = Image.new("RGB", (3 * w, h + 32), (20, 20, 20))
                draw = ImageDraw.Draw(canvas)
                for col, (panel_label, panel) in enumerate(zip(
                        ("Initial 4DGS", sample_label, "Held-out reference"), panels)):
                    canvas.paste(panel, (col * w, 32))
                    draw.text((col * w + 8, 9), panel_label, fill="white")
                draw.text((8, h + 12), gt.stem, fill="white", stroke_width=1, stroke_fill="black")
                writer.append_data(np.asarray(canvas))
                if index == 0:
                    canvas.save(output / f"{method}-{scene}-cam{camera}-first-frame.png")
        records.append(dict(camera=camera, full_sequence_frames=len(generated),
            held_out_comparison_frames=len(held_out), generated_movie=movie.name,
            generated_sha256=digest(movie), comparison_movie=compare.name,
            comparison_sha256=digest(compare)))
    report = dict(method=method, scene=scene, run=str(run), playback_fps=fps,
        note="MP4s are visual previews only. Metrics use PNGs before lossy encoding. Comparisons show held-out frames in filename/time order, potentially with temporal gaps; playback FPS is a display choice.",
        cameras=records)
    (output / "previews.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps(preview(args.run.resolve(), args.output.resolve(), args.fps), indent=2))
