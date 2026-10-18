# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Collect completed direct-sample measurements; never fill absent scenes."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("table1_saved_metrics", ROOT / "reproduce/table1/run.py")
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)


def collect(root, scenes, output):
    if output.exists():
        raise FileExistsError(output)
    if len(set(scenes)) != len(scenes):
        raise ValueError("Duplicate scenes")
    results = []
    recipe = None
    for method in ("ms", "wt"):
        for scene in scenes:
            run = root / f"{method}_{scene}"
            stages = ("align", "render", "generate", "score")
            reports = [json.loads((run / f"{stage}-execution.json").read_text()) for stage in stages]
            source_identity = (reports[0]["source_run"], reports[0]["source_execution_sha256"])
            for stage, report in zip(stages, reports):
                if report["stage"] != stage:
                    raise ValueError("Stage identity differs from execution filename")
                if (report["source_run"], report["source_execution_sha256"]) != source_identity:
                    raise ValueError("Initial source identity changed across stages")
                if report["status"] != "complete" or report.get("returncode") != 0:
                    raise ValueError(f"Incomplete stage: {run}/{report['stage']}")
                if report["method"] != method or report["scene"] != scene:
                    raise ValueError("Run identity mismatch")
                if recipe is None:
                    recipe = report["config"]
                if report["config"] != recipe:
                    raise ValueError("Recipe changed across measured stages/scenes")
            if not (run / "source-unchanged.json").is_file():
                raise ValueError("Missing input checkpoint immutability check")
            workbook = run / "scores/tto_dycheck_metrics.xlsx"
            baseline = run / "alignment/tto_dycheck_metrics.xlsx"
            metrics = table.read_workbook(workbook, table.STATIC, table.FRAMES[scene])
            initial = table.read_workbook(baseline, table.STATIC, table.FRAMES[scene])
            from openpyxl import load_workbook
            source_run = Path(reports[0]["source_run"])
            expected = sorted(p.stem for p in (source_run / "workspace/test_images").glob("*.png"))
            for path in (workbook, baseline):
                book = load_workbook(path, data_only=True, read_only=True)
                try:
                    rows = iter(book.active.values); header = next(rows); idx = header.index("fn")
                    actual = sorted(str(row[idx]) for row in rows if row[idx] not in (None, "AVE"))
                finally:
                    book.close()
                if [Path(name).stem for name in actual] != expected:
                    raise ValueError(f"Measured frame identities differ from held-out observations: {path}")
            results.append(dict(method=method, scene=scene, metrics=metrics,
                initial_render_metrics=initial,
                sample_minus_initial={k: metrics[k] - initial[k] for k in table.STATIC},
                workbook=str(workbook), workbook_sha256=table.digest(workbook),
                initial_workbook=str(baseline), initial_workbook_sha256=table.digest(baseline),
                stage_seconds={r["stage"]: r["elapsed_seconds"] for r in reports}))
    full = set(scenes) == set(table.SCENES) and len(scenes) == len(table.SCENES)
    means = {}
    if full:
        for method in ("ms", "wt"):
            selected = [r for r in results if r["method"] == method]
            measured = {k: math.fsum(r["metrics"][k] for r in selected) / 7 for k in table.STATIC}
            paper = recipe["paper_means"][method]
            means[method] = dict(measured=measured, paper_rounded=paper,
                                 signed_delta={k: measured[k] - paper[k] for k in table.STATIC})
    report = dict(scope="seven-scene sample rows" if full else "partial scene pilot; not a table reproduction",
                  recipe=recipe, scenes=results, means=means,
                  mask_uncertainty="Archived dynamic masks remain unresolved for apple, block, paper-windmill, spin and teddy. These masks affect static camera alignment; differences cannot all be attributed to optimization.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scenes", nargs="+", choices=table.SCENES, default=list(table.SCENES))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve(), args.scenes, args.output.resolve()), indent=2))
