#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 1x3 comparison video for camera-UI results: "
            "input reference, point-cloud rendering, DiffSynth prediction."
        )
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--inference-root", required=True, type=Path)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--camera-name", default="cam_ui")
    parser.add_argument("--reference-camera", default="cam0")
    parser.add_argument("--pred-subdir", default="pred_rgb_first73_stride2")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--labels",
        nargs=3,
        default=None,
        help="Optional labels to overlay on the three panels.",
    )
    return parser.parse_args()


def load_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        return ImageFont.load_default()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def main() -> None:
    args = parse_args()
    benchmark_root = resolve_path(args.benchmark_root)
    inference_root = resolve_path(args.inference_root)
    scene_root = benchmark_root / args.scene_name
    output_scene_root = inference_root / args.scene_name

    pred_dir = output_scene_root / args.pred_subdir / args.camera_name
    ref_dir = scene_root / "gt" / args.reference_camera
    render_dir = scene_root / "rgb" / args.camera_name
    tmp_dir = (
        resolve_path(args.tmp_dir)
        if args.tmp_dir is not None
        else inference_root / "_comparison_ref_render_pred_frames"
    )
    output = (
        resolve_path(args.output)
        if args.output is not None
        else inference_root
        / f"{args.scene_name}_ref_render_pred_1x3_{args.camera_name}_{args.pred_subdir}.mp4"
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for old in tmp_dir.glob("*.png"):
        old.unlink()

    pred_paths = sorted(pred_dir.glob("*.png"))
    if not pred_paths:
        raise SystemExit(f"No prediction frames found in {pred_dir}")

    font = load_font() if args.labels is not None else None
    for out_idx, pred_path in enumerate(pred_paths):
        name = pred_path.name
        paths = (ref_dir / name, render_dir / name, pred_path)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit("Missing paired frame(s): " + ", ".join(missing))

        images = [Image.open(path).convert("RGB") for path in paths]
        target_size = images[2].size
        images = [
            img
            if img.size == target_size
            else img.resize(target_size, Image.Resampling.BILINEAR)
            for img in images
        ]

        canvas = Image.new("RGB", (target_size[0] * 3, target_size[1]), (0, 0, 0))
        for idx, img in enumerate(images):
            x = idx * target_size[0]
            canvas.paste(img, (x, 0))
        if args.labels is not None:
            draw = ImageDraw.Draw(canvas, "RGBA")
            for idx, label in enumerate(args.labels):
                x = idx * target_size[0]
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.rectangle(
                    (x + 8, 8, x + text_w + 24, text_h + 22),
                    fill=(0, 0, 0, 150),
                )
                draw.text((x + 16, 11), label, fill=(255, 255, 255, 255), font=font)
        canvas.save(tmp_dir / f"{out_idx:05d}.png")

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(tmp_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(args.crf),
        str(output),
    ]
    print(f"Comparison frames: {tmp_dir}")
    print(f"Frame count: {len(pred_paths)}")
    print("FFmpeg command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Comparison video: {output}")


if __name__ == "__main__":
    main()
