#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


TEXT_SUFFIXES = {
    ".cfg", ".cff", ".csv", ".ini", ".json", ".md", ".py", ".sh", ".sbatch", ".toml", ".txt",
    ".yaml", ".yml"
}
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__", "artifacts", "runs"}
DENYLIST_ENV = "WFM_RELEASE_DENYLIST"
# Structural signs of site-specific scheduler, host and storage details. Literal
# names (accounts, hosts, buckets) belong in a private denylist file instead, so
# the public tree never has to contain them. Patterns are written so they do not
# match their own source.
SITE_PATTERNS = (
    ("hardcoded Lustre path", re.compile("/" + r"lustre(?=[/:\s\"'])")),
    ("Slurm job record", re.compile(
        r"\b(?:UserId|GroupId)=[^\s(]+\(\d+\)|\b(?:BatchHost|NodeList)=(?!\(null\))[\w-]")),
    ("site-specific #SBATCH directive", re.compile(
        r"^#SBATCH\s+(?:--account|--partition|--qos|--reservation|-A|-p)\b", re.M)),
    ("recorded Slurm job ID", re.compile(
        r"\"(?:SLURM_(?:ARRAY_)?JOB_ID|job_id|JobID|JobId)\"\s*:\s*\"?\d{4,}")),
    ("cloud host kernel string", re.compile(r"\d+\.\d+\.\d+-\d+-(?:oracle|aws|azure|gcp|gke)\b")),
    ("hardcoded container mount", re.compile(r"--container-mounts=[\"']?(?=/)|_MOUNTS:-(?=/)")),
    ("object-store endpoint", re.compile(r"--endpoint-url(?=[\s=])|\"endpoint\"\s*:\s*\"https?://")),
    ("W&B entity default", re.compile(
        r"wandb_entity[\"'][^\n]*default=[\"'][^\"']+|[\"']wandb_entity[\"'],\s*[\"'][^\"']+[\"']")),
)
# Third-party upstream text reviewed as harmless, keyed by check and path.
REVIEWED_UPSTREAM: set[tuple[str, str]] = set()


def load_denylist(path) -> list[tuple[str, re.Pattern]]:
    """Read one regular expression per line; blank lines and # comments are skipped.

    Each entry is labelled by its denylist line so reports never echo the private term.
    """
    if not path:
        return []
    path = Path(path).expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    return [(f"{path.name}:{number}", re.compile(line.strip()))
            for number, line in enumerate(lines, 1)
            if line.strip() and not line.lstrip().startswith("#")]


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def source_paths(root: Path, tracked_only: bool):
    if tracked_only:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True
        ).strip()
        if Path(top).resolve() != root:
            raise ValueError("--tracked requires the repository root")
        names = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "-z"], text=True
        )
        yield from (root / name for name in names.split("\0") if name)
        return
    # Prune experiment outputs before descent; a checkout can contain terabytes
    # of local runs which are not part of its source package.
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            path = base / name
            if name == ".git" or path.is_symlink():
                yield path
        dirs[:] = [name for name in dirs if name not in IGNORED_PARTS
                   and not (base / name).is_symlink()]
        yield from (base / name for name in files)


def audit(root: Path, *, tracked_only: bool = False,
          denylist: list[tuple[str, re.Pattern]] | None = None) -> dict[str, object]:
    """Audit a source tree. Without an explicit denylist, $WFM_RELEASE_DENYLIST is used if set."""
    root = root.resolve()
    if denylist is None:
        denylist = load_denylist(os.environ.get(DENYLIST_ENV))
    failures: list[str] = []
    checked = 0
    for path in source_paths(root, tracked_only):
        relative = path.relative_to(root)
        if path.is_dir() and path.name == ".git":
            if relative != Path(".git"):
                failures.append(f"nested Git metadata: {relative}")
            continue
        if not tracked_only and any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                failures.append(f"symlink leaves source tree: {relative}")
                continue
            if not path.exists():
                failures.append(f"broken source symlink: {relative}")
                continue
        if not path.is_file():
            if tracked_only:
                failures.append(f"tracked source file missing or not a file: {relative}")
            continue
        checked += 1
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            failures.append(f"model weight is baked into source/image context: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Dockerfile", ".dockerignore"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for reason, pattern in SITE_PATTERNS:
                if (reason, relative.as_posix()) in REVIEWED_UPSTREAM:
                    continue
                match = pattern.search(text)
                if match:
                    failures.append(f"{reason}: {relative}:{line_number(text, match.start())}")
            for label, pattern in denylist:
                match = pattern.search(text)
                if match:
                    failures.append(f"denylisted term ({label}): "
                                    f"{relative}:{line_number(text, match.start())}")
            if ("nv" + "api-") in text or ("NGC_" + "API_KEY=") in text:
                failures.append(f"possible baked NGC credential: {relative}")
    return {"ok": not failures, "files_checked": checked, "denylist_terms": len(denylist),
            "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the WFM source release context")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tracked", action="store_true",
                        help="Audit Git-tracked working files; exclude untracked local assets")
    parser.add_argument("--denylist", type=Path, default=os.environ.get(DENYLIST_ENV),
                        help=f"Private file of regular expressions to reject (default: ${DENYLIST_ENV})")
    args = parser.parse_args()
    result = audit(args.root.expanduser().resolve(), tracked_only=args.tracked,
                   denylist=load_denylist(args.denylist))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
