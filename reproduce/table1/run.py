#!/usr/bin/env python3
"""Portable entry point for the two Table 1 distillation rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCENES = ("apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel")
FRAMES = dict(zip(SCENES, (532, 667, 419, 681, 722, 688, 219)))
STATIC = ("psnr", "ssim", "lpips", "valid_psnr", "valid_ssim", "valid_lpips",
          "mpsnr", "mssim", "mlpips")
POSE = ("pose_mpsnr", "pose_mssim", "pose_mlpips")
COLUMNS = STATIC + POSE
PATH_FIELDS = ("checkpoint", "workspace", "metric_workspace", "static_metrics",
               "per_frame_metrics")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path, assets_root: Path | None) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported run manifest schema")
    root = (assets_root or path.parent).expanduser().resolve()
    result, seen = [], set()
    for entry in payload["runs"]:
        entry = dict(entry)
        key = (entry["method"], entry["scene"])
        if key[0] not in ("ms", "wt") or key[1] not in SCENES or key in seen:
            raise ValueError(f"Invalid or duplicate run: {key}")
        seen.add(key)
        for field in PATH_FIELDS:
            value = Path(entry[field]).expanduser()
            entry[field] = (root / value).resolve()
        result.append(entry)
    if not result:
        raise ValueError("Empty run manifest")
    return result


def read_workbook(path: Path, columns: tuple[str, ...], expected_frames: int) -> dict:
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        rows = iter(workbook.active.values)
        header = next(rows)
        if len(set(header)) != len(header):
            raise ValueError(f"Duplicate workbook columns: {path}")
        for column in ("fn", *columns):
            if column not in header:
                raise ValueError(f"Missing measured column {column}: {path}")
        records = [dict(zip(header, row)) for row in rows if any(v is not None for v in row)]
    finally:
        workbook.close()
    ave = [row for row in records if row["fn"] == "AVE"]
    frames = [row for row in records if row["fn"] != "AVE"]
    if len(ave) != 1 or len(frames) != expected_frames:
        raise ValueError(f"Expected one AVE and {expected_frames} frames: {path}")
    names = [str(row["fn"]) for row in frames]
    if len(set(names)) != len(names) or any(row["fn"] is None for row in frames):
        raise ValueError(f"Duplicate or empty frame identifiers: {path}")
    values = {}
    for column in columns:
        try:
            samples = [float(row[column]) for row in frames]
            saved = float(ave[0][column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric {column}: {path}") from exc
        if not all(math.isfinite(v) for v in [saved, *samples]):
            raise ValueError(f"Nonfinite {column}: {path}")
        if abs(math.fsum(samples) / len(samples) - saved) > 1e-6:
            raise ValueError(f"AVE differs from frame mean for {column}: {path}")
        if "psnr" in column and all(v == 0 for v in samples):
            raise ValueError(f"Unmeasured zero placeholder for {column}: {path}")
        values[column] = saved
    return values


def average_scenes(records: list[dict]) -> dict:
    result = {}
    for method in ("ms", "wt"):
        selected = [r for r in records if r["method"] == method]
        if not selected:
            continue
        if sorted(r["scene"] for r in selected) != sorted(SCENES):
            raise ValueError(f"{method}: a table row requires exactly all seven scenes")
        result[method] = {}
        for column in COLUMNS:
            values = [float(r["metrics"][column]) for r in selected]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite metric: {method}/{column}")
            result[method][column] = math.fsum(values) / len(SCENES)
    if not result:
        raise ValueError("No complete table rows")
    return result


def markdown(rows: dict) -> str:
    lines = ["| Method | Static Full | Static Valid | Static Covisible | Per-frame Covisible |",
             "|---|---|---|---|---|"]
    for method, values in rows.items():
        cells = []
        for columns in (STATIC[:3], STATIC[3:6], STATIC[6:9], POSE):
            a, b, c = (values[k] for k in columns)
            cells.append(f"{a:.2f} / {b:.3f} / {c:.3f}")
        lines.append(f"| {method.upper()}+Ours (distill) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def fresh_output(output: Path, protected: list[Path]) -> Path:
    output = output.expanduser().resolve()
    for source in protected:
        source = source.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Output overlaps an input asset: {output} and {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    return output


def verify_runs(runs: list[dict], reference: dict, hashes: bool = False) -> dict:
    files, total = 0, 0
    for run in runs:
        key = f"{run['method']}/{run['scene']}"
        root = run["checkpoint"]
        for item in reference["checkpoints"][key]:
            path = root / item["path"]
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing checkpoint asset: {path}")
            if hashes and (path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]):
                raise ValueError(f"Reference checkpoint mismatch: {path}")
            files += 1
            total += path.stat().st_size
        if run["method"] == "wt":
            validate_tree(root)
    return {"runs": len(runs), "checkpoint_and_split_files": files, "bytes": total,
            "reference_hashes_checked": hashes}


def validate_tree(root: Path) -> None:
    import re

    nodes = {}
    for path in root.glob("dst_L*_R*"):
        match = re.fullmatch(r"dst_L(\d+)_R(\d+)", path.name)
        if not match or not path.is_dir():
            raise ValueError(f"Invalid WorldTree node: {path}")
        nodes[path] = tuple(map(int, match.groups()))
    if len(nodes) != 7:
        raise ValueError(f"Expected seven WorldTree nodes: {root}")
    for path, (lo, hi) in nodes.items():
        descendants = [v for p, v in nodes.items() if p != path and lo <= v[0] <= v[1] <= hi]
        if not descendants:
            continue
        split = int((path / "split_ind.txt").read_text())
        left = [v for v in descendants if v[1] < split]
        right = [v for v in descendants if v[0] >= split]
        if (not left or not right or len(left) + len(right) != len(descendants)
                or min(v[0] for v in left) != lo or max(v[1] for v in left) != split - 1
                or min(v[0] for v in right) != split or max(v[1] for v in right) != hi):
            raise ValueError(f"Invalid split or missing child coverage: {path}")


def validate_fit_workspace(run: dict) -> None:
    filename = "multiview_manifest.json" if run["method"] == "ms" else "multiview_8_manifest.json"
    path = run["workspace"] / filename
    manifest = json.loads(path.read_text())
    cameras = manifest["cameras"]
    names = [c["camera_name"] for c in cameras]
    expected = {"cam0", *(f"cam{i}" for i in range(100, 108))}
    if set(names) != expected or len(names) != len(expected):
        raise ValueError(f"Expected cam0 and exactly cam100–cam107: {path}")
    for camera in cameras:
        for field in ("rgb_dir", "camera_json_dir", "reference_c2w_path"):
            asset = run["workspace"] / camera[field]
            if not asset.exists():
                raise FileNotFoundError(asset)


def aggregate(args, runs, reference):
    records = []
    if args.reference:
        for method, scenes in reference["scenes"].items():
            if args.method != "all" and method != args.method:
                continue
            for scene, entry in scenes.items():
                records.append({"method": method, "scene": scene, **entry})
    else:
        for run in runs:
            scene = run["scene"]
            paths = {mode: run[f"{mode}_metrics"] for mode in ("static", "per_frame")}
            if args.results_root:
                paths = {mode: args.results_root / run["method"] / scene / mode /
                         "tto_dycheck_metrics.xlsx" for mode in paths}
            static = read_workbook(paths["static"], STATIC, FRAMES[scene])
            pose = read_workbook(paths["per_frame"], STATIC[6:], FRAMES[scene])
            records.append({"method": run["method"], "scene": scene,
                            "metrics": {**static, **{f"pose_{k}": v for k, v in pose.items()}},
                            "sources": {mode: {"path": str(p), "sha256": digest(p)}
                                        for mode, p in paths.items()}})
    rows = average_scenes(records)
    differences = []
    for method, values in rows.items():
        for column, value in values.items():
            precision = 2 if "psnr" in column else 3
            expected = reference["rows"][method][column]
            if f"{value:.{precision}f}" != f"{expected:.{precision}f}":
                differences.append({"method": method, "metric": column,
                                    "measured": value, "reference": expected})
    print(markdown(rows), end="")
    if args.output:
        output = fresh_output(args.output, [r[f] for r in runs for f in PATH_FIELDS])
        output.mkdir(parents=True)
        (output / "table1_distill.md").write_text(markdown(rows))
        (output / "metrics.json").write_text(json.dumps({
            "source": "archived_reference" if args.reference else "workbook_aggregation",
            "rows": rows, "scenes": records, "differences": differences}, indent=2) + "\n")
        with (output / "table1_distill.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("method", *COLUMNS))
            writer.writeheader()
            for method, row in rows.items():
                writer.writerow({"method": method, **row})
    if differences:
        print(json.dumps({"differences": differences}, indent=2), file=sys.stderr)
    if args.check_reference and differences:
        raise ValueError("Measured row differs from the camera-ready reference")


def jobs(args, runs, reference):
    if args.action == "evaluate":
        verify_runs(runs, reference)
    protected = [r[f] for r in runs for f in PATH_FIELDS]
    output = fresh_output(args.output, protected)
    plans = []
    updated = []
    modes = ("static", "per_frame") if args.mode == "both" else (args.mode,)
    for run in runs:
        for field in ("workspace", "metric_workspace"):
            if not run[field].is_dir():
                raise FileNotFoundError(run[field])
        if args.action == "fit":
            validate_fit_workspace(run)
        if args.action in ("evaluate", "metrics"):
            for folder in ("test_images", "test_covisible", "test_dynamic"):
                path = run["metric_workspace"] / folder
                if not path.is_dir():
                    raise FileNotFoundError(path)
        new_run = {k: str(v) if isinstance(v, Path) else v for k, v in run.items()}
        # A partial evaluation or a new fit must never silently inherit the old
        # workbook for its unexecuted modes and appear to reproduce a whole row.
        for metric_mode in ("static", "per_frame"):
            new_run[f"{metric_mode}_metrics"] = str(
                output / run["method"] / run["scene"] / metric_mode /
                "tto_dycheck_metrics.xlsx")
        for mode in modes if args.action != "fit" else ("fit",):
            destination = output / run["method"] / run["scene"] / mode
            command = [args.python, str(HERE / "worker.py"), "--action", args.action,
                       "--method", run["method"], "--scene", run["scene"],
                       "--workspace", str(run["workspace"]),
                       "--metric-workspace", str(run["metric_workspace"]),
                       "--checkpoint", str(run["checkpoint"]),
                       "--output", str(destination), "--mode", mode]
            if args.action == "fit" and getattr(args, "fit_preset", None) is not None:
                command.extend(["--fit-preset", args.fit_preset])
            if args.action == "metrics":
                prediction = args.predictions_root / run["method"] / run["scene"] / mode / "tto_test"
                if not prediction.is_dir():
                    raise FileNotFoundError(prediction)
                command.extend(["--predictions", str(prediction.resolve())])
            plans.append({"command": command, "method": run["method"], "scene": run["scene"],
                          "mode": mode, "output": str(destination)})
            if mode == "fit":
                new_run["checkpoint"] = str(destination)
            else:
                new_run[f"{mode}_metrics"] = str(destination / "tto_dycheck_metrics.xlsx")
        updated.append(new_run)
    if args.dry_run:
        print(json.dumps({"jobs": plans}, indent=2))
        return
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plans, indent=2) + "\n")
    environment = dict(os.environ, GS_BACKEND="native_add3", PYTHONDONTWRITEBYTECODE="1")
    for plan in plans:
        print(f"{args.action}: {plan['method']}/{plan['scene']}/{plan['mode']}", flush=True)
        logfile = output / f"{plan['method']}_{plan['scene']}_{plan['mode']}.log"
        with logfile.open("w") as log:
            try:
                subprocess.run(plan["command"], cwd=HERE, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as exc:
                raise ValueError(f"Worker exited with status {exc.returncode}; see {logfile}") from exc
    (output / "runs.json").write_text(json.dumps({"schema_version": 1, "runs": updated}, indent=2) + "\n")
    print(f"Completed {len(plans)} jobs. Run manifest: {output / 'runs.json'}")


def main(argv=None):
    if __package__:
        from .efficiency.runtime import FIT_PRESETS
    else:
        from efficiency.runtime import FIT_PRESETS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("aggregate", "verify", "evaluate", "metrics", "fit"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--method", choices=("all", "ms", "wt"), default="all")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--mode", choices=("both", "static", "per_frame"), default="both")
    parser.add_argument("--fit-preset", choices=tuple(FIT_PRESETS),
                        help="Fitting preset (default: efficient with environment overrides); "
                             "an explicit preset overrides all fitting switches")
    parser.add_argument("--reference", action="store_true", help="Aggregate checked-in per-scene reference values")
    parser.add_argument("--check-reference", action="store_true", help="Fail if rounded measured scores differ")
    parser.add_argument("--reference-hashes", action="store_true", help="Verify exact archived checkpoint hashes")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print GPU jobs without executing")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for GPU workers")
    args = parser.parse_args(argv)
    try:
        if args.fit_preset is not None and args.action != "fit":
            raise ValueError("--fit-preset is only valid for fit")
        if args.reference and args.action != "aggregate":
            raise ValueError("--reference is only valid for aggregate")
        if args.scene and args.action == "aggregate":
            raise ValueError("Table aggregation requires all seven scenes; omit --scene")
        if not args.reference and not args.manifest:
            raise ValueError("--manifest is required (or use aggregate --reference)")
        if args.action in ("fit", "evaluate", "metrics") and not args.output:
            raise ValueError("--output is required")
        if args.action == "metrics" and not args.predictions_root:
            raise ValueError("metrics requires --predictions-root")
        reference = json.loads((HERE / "reference.json").read_text())
        runs = read_manifest(args.manifest, args.assets_root) if args.manifest else []
        runs = [r for r in runs if (args.method == "all" or r["method"] == args.method)
                and (not args.scene or r["scene"] == args.scene)]
        if not runs and not args.reference:
            raise ValueError("No runs selected")
        if args.action == "aggregate":
            aggregate(args, runs, reference)
        elif args.action == "verify":
            print(json.dumps(verify_runs(runs, reference, args.reference_hashes), indent=2))
        else:
            jobs(args, runs, reference)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"table1: {exc}\n")


if __name__ == "__main__":
    main()
