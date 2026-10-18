# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public source includes reproduction tools, without private campaign assets."""
from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_source_without_saved_campaign_assets():
    assert (ROOT / "reproduce/from_raw/run.py").is_file()
    assert (ROOT / "reproduce/sampling/collect.py").is_file()
    assert not (ROOT / "release").exists()
    assert not (ROOT / "skills").exists()
    assert not (ROOT / "runs").exists()
    assert not (ROOT / "vendor/mosca/eval_utils/pck_gt_packs").exists()
    assert not (ROOT / "slurm").exists()
    assert not list((ROOT / "reproduce").rglob("results"))
    for folder in ("src", "config", "docs", "tests", "scripts", "reproduce"):
        assert not list((ROOT / folder).rglob("*.xlsx"))


def test_video_core_has_no_reproduction_imports():
    for path in (ROOT / "src").rglob("*.py"):
        if path.name in {"dycheck_sampling.py", "dycheck_masks.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("reproduce"), path
            if isinstance(node, ast.Import):
                assert all(not item.name.startswith("reproduce") for item in node.names), path


def test_public_document_links_resolve():
    for path in [ROOT / "README.md", ROOT / "REPRODUCTION.md",
                 *(ROOT / "docs").glob("*.md"),
                 *(ROOT / "reproduce").glob("*.md"),
                 *(ROOT / "reproduce").glob("*/README.md"),
                 ROOT / "reproduce/table1/ENVIRONMENT.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path, target)
