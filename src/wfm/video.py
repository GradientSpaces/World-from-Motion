from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float


def _ratio(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / max(float(denominator), 1.0)
    return float(value)


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as exc:
        raise ConfigurationError(f"Cannot inspect input video {path}: {exc}") from exc
    streams = payload.get("streams", [])
    if not streams:
        raise ConfigurationError(f"No video stream found in {path}")
    stream = streams[0]
    fps = _ratio(str(stream.get("avg_frame_rate", "0")))
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    frame_text = stream.get("nb_frames")
    frames = int(frame_text) if frame_text not in (None, "N/A") else int(round(duration * fps))
    if frames <= 0 or fps <= 0:
        raise ConfigurationError(f"Video has invalid frame metadata: frames={frames}, fps={fps}")
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frames=frames,
        duration=duration,
    )


def encode_png_sequence(frame_dir: Path, output: Path, fps: float, max_frames: int) -> None:
    frames = sorted(frame_dir.glob("*.png"))[:max_frames]
    if len(frames) != max_frames:
        raise ConfigurationError(
            f"Expected {max_frames} prediction frames in {frame_dir}, found {len(frames)}"
        )
    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    try:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    finally:
        writer.close()


def make_grid(videos: list[Path], output: Path) -> None:
    if len(videos) != 4:
        raise ValueError("The production preview requires exactly four videos")
    command = ["ffmpeg", "-y"]
    for video in videos:
        command.extend(["-i", str(video)])
    command.extend(
        [
            "-filter_complex",
            "[0:v]drawtext=text='cam_fps_00':x=20:y=20:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.6[v0];"
            "[1:v]drawtext=text='cam_fps_01':x=20:y=20:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.6[v1];"
            "[2:v]drawtext=text='cam_fps_02':x=20:y=20:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.6[v2];"
            "[3:v]drawtext=text='cam_reverse':x=20:y=20:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.6[v3];"
            "[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    subprocess.run(command, check=True)
