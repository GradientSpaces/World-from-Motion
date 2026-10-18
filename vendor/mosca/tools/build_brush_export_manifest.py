#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    REPO_ROOT / "data/DAVIS/ws",
    REPO_ROOT / "data/wild_pipeline",
    REPO_ROOT / "data/wild_ws2",
)
CHECKPOINT_NAMES = (
    "photometric_cam.pth",
    "photometric_s_model_{backend}.pth",
    "photometric_d_model_{backend}.pth",
)
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "bundle",
    "camera_ui_exports",
    "epi",
    "export_brush",
    "flow_waft",
    "images",
    "logs",
    "mosca",
    "multiview",
    "src_backup",
    "track",
    "tracks",
    "vggt_omega_depth",
}


@dataclass(frozen=True)
class ExportRow:
    scene_name: str
    workspace: Path
    saved_dir: Path
    output_dir: Path
    ply_name: str
    static_output_dir: Path
    static_ply_name: str


@dataclass(frozen=True)
class SkippedWorkspace:
    workspace: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TSV manifest for all-scene MoSca Brush PLY export."
    )
    parser.add_argument(
        "--root",
        dest="roots",
        type=Path,
        action="append",
        default=None,
        help="Root directory to scan. May be passed multiple times.",
    )
    parser.add_argument("--backend", default="native_add3")
    parser.add_argument("--output-suffix", default="all_frames_rgb_static")
    parser.add_argument("--ply-suffix", default="deform3dgs_all_frames.ply")
    parser.add_argument("--static-output-suffix", default="static_only_rgb")
    parser.add_argument("--static-ply-suffix", default="static_only.ply")
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional TSV file listing skipped workspaces and reasons.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum directory depth below each root to search for workspaces.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "scene"


def checkpoint_paths(saved_dir: Path, backend: str) -> tuple[Path, ...]:
    backend = backend.lower()
    return tuple(saved_dir / name.format(backend=backend) for name in CHECKPOINT_NAMES)


def is_complete_run(saved_dir: Path, backend: str) -> bool:
    return all(path.is_file() for path in checkpoint_paths(saved_dir, backend))


def newest_complete_run(workspace: Path, backend: str) -> Path | None:
    logs_dir = workspace / "logs"
    if not logs_dir.is_dir():
        return None

    candidates = [path for path in logs_dir.iterdir() if path.is_dir()]
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    for candidate in candidates:
        if is_complete_run(candidate, backend):
            return candidate
    return None


def discover_workspace_dirs(root: Path, max_depth: int) -> list[Path]:
    if max_depth < 0:
        raise ValueError("--max-depth must be non-negative")
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found: {root}")

    workspaces: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()

    while queue:
        current, depth = queue.pop(0)
        current = current.resolve()
        if current in seen:
            continue
        seen.add(current)

        if (current / "logs").is_dir():
            workspaces.append(current)
            continue
        if depth >= max_depth:
            continue

        for child in sorted(current.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or child.is_symlink():
                continue
            if child.name in SKIP_DIR_NAMES:
                continue
            queue.append((child, depth + 1))

    return workspaces


def root_label(root: Path) -> str:
    try:
        rel = root.relative_to(REPO_ROOT)
        return safe_name(rel.as_posix())
    except ValueError:
        return safe_name(root.name)


def scene_key(root: Path, workspace: Path) -> str:
    try:
        rel = workspace.relative_to(root)
        return safe_name(f"{root_label(root)}__{rel.as_posix()}")
    except ValueError:
        return safe_name(workspace.name)


def build_rows(
    roots: list[Path],
    backend: str,
    output_suffix: str,
    ply_suffix: str,
    static_output_suffix: str,
    static_ply_suffix: str,
    max_depth: int,
) -> tuple[list[ExportRow], list[SkippedWorkspace]]:
    rows: list[ExportRow] = []
    skipped: list[SkippedWorkspace] = []

    for root in roots:
        for workspace in discover_workspace_dirs(root, max_depth):
            saved_dir = newest_complete_run(workspace, backend)
            if saved_dir is None:
                skipped.append(
                    SkippedWorkspace(
                        workspace=workspace,
                        reason=f"no complete {backend.lower()} photometric checkpoint run",
                    )
                )
                continue

            key = scene_key(root, workspace)
            output_dir = saved_dir / "export_brush" / safe_name(output_suffix)
            ply_name = f"{safe_name(workspace.name)}_{safe_name(ply_suffix)}"
            static_output_dir = saved_dir / "export_brush" / safe_name(static_output_suffix)
            static_ply_name = f"{safe_name(workspace.name)}_{safe_name(static_ply_suffix)}"
            rows.append(
                ExportRow(
                    scene_name=key,
                    workspace=workspace,
                    saved_dir=saved_dir,
                    output_dir=output_dir,
                    ply_name=ply_name,
                    static_output_dir=static_output_dir,
                    static_ply_name=static_ply_name,
                )
            )

    rows.sort(key=lambda row: row.scene_name)
    skipped.sort(key=lambda row: row.workspace.as_posix())
    return rows, skipped


def write_manifest(path: Path, rows: list[ExportRow]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                "\t".join(
                    (
                        row.scene_name,
                        row.workspace.as_posix(),
                        row.saved_dir.as_posix(),
                        row.output_dir.as_posix(),
                        row.ply_name,
                        row.static_output_dir.as_posix(),
                        row.static_ply_name,
                    )
                )
            )
            handle.write("\n")


def write_skipped(path: Path, skipped: list[SkippedWorkspace]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in skipped:
            handle.write(f"{item.workspace.as_posix()}\t{item.reason}\n")


def main() -> None:
    args = parse_args()
    roots = [resolve_path(root) for root in (args.roots or DEFAULT_ROOTS)]
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    rows, skipped = build_rows(
        roots=roots,
        backend=args.backend,
        output_suffix=args.output_suffix,
        ply_suffix=args.ply_suffix,
        static_output_suffix=args.static_output_suffix,
        static_ply_suffix=args.static_ply_suffix,
        max_depth=int(args.max_depth),
    )
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    write_manifest(args.manifest_out, rows)
    if args.summary_out is not None:
        write_skipped(args.summary_out, skipped)

    print(f"Scanned roots: {len(roots)}")
    for root in roots:
        print(f"  {root}")
    print(f"Wrote manifest rows: {len(rows)}")
    print(f"Skipped incomplete workspaces: {len(skipped)}")
    print(f"Manifest: {resolve_path(args.manifest_out)}")
    if args.summary_out is not None:
        print(f"Skipped summary: {resolve_path(args.summary_out)}")


if __name__ == "__main__":
    main()
