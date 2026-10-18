from __future__ import annotations

from pathlib import Path

import pytest

from wfm.cli import build_parser
from wfm.config import load_production_config
from wfm.pipeline import PipelineRunner, RunOptions
from wfm.util import normalize_classes


def test_dynamic_classes_are_deduplicated_and_ordered() -> None:
    assert normalize_classes([" human ", "robot", "Human"]) == ("human", "robot")


def test_dynamic_classes_are_required() -> None:
    with pytest.raises(ValueError):
        normalize_classes([" "])


def test_cli_accepts_repeatable_classes(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--video",
            "input.mp4",
            "--dynamic-class",
            "human",
            "--dynamic-class",
            "robot",
            "--output",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert args.dynamic_classes == ["human", "robot"]


def test_dry_run_needs_no_video_or_weights(tmp_path: Path) -> None:
    profile = load_production_config()
    runner = PipelineRunner(profile)
    result = runner.run(
        RunOptions(
            video=tmp_path / "not-present.mp4",
            dynamic_classes=("human",),
            output=tmp_path / "out",
            seed=0,
            dry_run=True,
        )
    )
    assert result["stages"][-1] == "validate"
    assert result["stages"].index("diffsynth_inference") < result["stages"].index(
        "distill_4dgs"
    )
    assert result["stages"].index("distill_4dgs") < result["stages"].index(
        "export_brush"
    )
    assert result["distillation_steps"] == profile.distillation_steps
