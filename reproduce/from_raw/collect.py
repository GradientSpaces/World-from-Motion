"""Collect only complete, freshly evaluated independent runs for both table rows."""
import argparse
import json
from pathlib import Path
import sys

from stages import TABLE, save

sys.path.insert(0, str(TABLE))
from run import SCENES, read_manifest


def collect(root):
    rows = []
    for method in ("ms", "wt"):
        for scene in SCENES:
            # A collection root may explicitly select a successful retry by symlink.
            # Compare resolved workbook paths against that selected physical run.
            directory = (root / f"{method}_{scene}").resolve()
            report = json.loads((directory / "execution.json").read_text())
            required = ["prepare", "priors", "masks", "initial", "render", "generate", "materialize", "fit", "evaluate"]
            if (report["status"] != "complete" or
                [s["stage"] for s in report["stages"]] != required or
                any(s["status"] != "complete" or s["returncode"] != 0 for s in report["stages"])):
                raise ValueError(f"Incomplete independent execution: {method}/{scene}")
            entries = read_manifest(directory / "evaluate/runs.json", None)
            if len(entries) != 1 or (entries[0]["method"], entries[0]["scene"]) != (method, scene):
                raise ValueError(f"Wrong evaluated entry: {method}/{scene}")
            for key in ("static_metrics", "per_frame_metrics"):
                path = entries[0][key]
                if not path.is_relative_to(directory) or not path.is_file():
                    raise ValueError(f"Missing fresh workbook: {path}")
            rows.append({k: str(v) if isinstance(v, Path) else v for k, v in entries[0].items()})
    return dict(schema_version=1, runs=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.expanduser().resolve()
    output = root / "runs.json"
    if output.exists():
        raise FileExistsError(output)
    save(output, collect(root))
    print(output)
