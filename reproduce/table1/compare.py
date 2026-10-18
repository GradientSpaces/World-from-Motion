"""Report all reproduction differences without requiring identical rounded scores."""
import argparse
import csv
import json
import math
from pathlib import Path

if __package__:
    from .run import COLUMNS, HERE, average_scenes, digest, fresh_output, markdown
else:
    from run import COLUMNS, HERE, average_scenes, digest, fresh_output, markdown


def compare(results, reference):
    rows = average_scenes(results["scenes"])
    if set(rows) != set(results["rows"]):
        raise ValueError("Result table methods differ from the measured scenes")
    records = []
    for method, values in rows.items():
        for metric, measured in values.items():
            saved = float(results["rows"][method][metric])
            if not math.isfinite(saved) or abs(measured - saved) > 1e-10:
                raise ValueError(f"Result table differs from scene means: {method}/{metric}")
    entries = [(method, "MEAN", values, reference["rows"][method])
               for method, values in rows.items()]
    entries.extend((r["method"], r["scene"], r["metrics"],
                    reference["scenes"][r["method"]][r["scene"]]["metrics"])
                   for r in results["scenes"])
    for method, scene, values, expected in entries:
        for metric in COLUMNS:
            measured, baseline = float(values[metric]), float(expected[metric])
            if not all(math.isfinite(v) for v in (measured, baseline)):
                raise ValueError(f"Nonfinite comparison: {method}/{scene}/{metric}")
            precision = 2 if "psnr" in metric else 3
            records.append({"method": method, "scene": scene, "metric": metric,
                            "measured": measured, "reference": baseline,
                            "delta": measured - baseline,
                            "rounded_equal": f"{measured:.{precision}f}" ==
                                             f"{baseline:.{precision}f}"})
    maxima = {}
    for scope in ("table", "scene"):
        selected = [r for r in records if (r["scene"] == "MEAN") == (scope == "table")]
        maxima[scope] = {metric: max(abs(r["delta"]) for r in selected
                                    if metric in r["metric"])
                         for metric in ("psnr", "ssim", "lpips")}
    return {"delta_definition": "measured minus reference; no tolerance is imposed",
            "max_absolute_delta": maxima, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="metrics.json produced by run.py aggregate")
    parser.add_argument("--reference", type=Path, default=HERE / "reference.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        results = json.loads(args.results.read_text())
        reference = json.loads(args.reference.read_text())
        report = compare(results, reference)
        report["inputs"] = {
            "measurements": {"path": str(args.results.resolve()), "sha256": digest(args.results),
                             "kind": results.get("source", "unspecified")},
            "reference": {"path": str(args.reference.resolve()), "sha256": digest(args.reference)},
        }
        output = fresh_output(args.output, [args.results, args.reference])
        output.mkdir(parents=True)
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        with (output / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["records"][0]))
            writer.writeheader()
            writer.writerows(report["records"])
        selected_reference = {m: reference["rows"][m] for m in results["rows"]}
        text = "# Measured results\n\n" + markdown(results["rows"])
        text += "\n# Archived reference\n\n" + markdown(selected_reference)
        text += "\n# Maximum absolute differences\n\n"
        text += "| Scope | PSNR | SSIM | LPIPS |\n|---|---:|---:|---:|\n"
        for scope, values in report["max_absolute_delta"].items():
            text += f"| {scope} | {values['psnr']:.6f} | {values['ssim']:.6f} | {values['lpips']:.6f} |\n"
        text += "\nAll signed table and per-scene differences are in `comparison.csv`.\n"
        (output / "comparison.md").write_text(text)
        print(text)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"table1-compare: {exc}\n")


if __name__ == "__main__":
    main()
