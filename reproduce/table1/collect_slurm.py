"""Collect completed per-scene Slurm evaluations into one explicit run manifest."""
import argparse
import json
from pathlib import Path

if __package__:
    from .run import SCENES, read_manifest
else:
    from run import SCENES, read_manifest


def collect(root: Path) -> dict:
    root = root.resolve()
    runs = []
    for method in ("ms", "wt"):
        for scene in SCENES:
            path = root / f"{method}_{scene}" / "runs.json"
            entries = read_manifest(path, None)
            if len(entries) != 1 or (entries[0]["method"], entries[0]["scene"]) != (method, scene):
                raise ValueError(f"Expected exactly {method}/{scene}: {path}")
            entry = entries[0]
            for mode in ("static", "per_frame"):
                metrics = entry[f"{mode}_metrics"]
                if not metrics.is_relative_to(root) or not metrics.is_file():
                    raise ValueError(f"Missing fresh {mode} result within campaign: {metrics}")
            runs.append({k: str(v) if isinstance(v, Path) else v for k, v in entry.items()})
    return {"schema_version": 1, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    try:
        payload = collect(args.campaign)
        output = args.campaign / "runs.json"
        with output.open("x") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        print(output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"table1-slurm: {exc}\n")


if __name__ == "__main__":
    main()
