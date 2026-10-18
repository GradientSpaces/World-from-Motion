"""Rebuild one Table 1 scene from public observations through final evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from stages import HERE, TABLE, REPO, save, sha256

STAGES = ("prepare", "priors", "masks", "initial", "render", "generate", "materialize", "fit", "evaluate")
SCENES = ("apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel")


def commands(args):
    common = ["--method", args.method, "--scene", args.scene, "--output", str(args.output),
              "--data-root", str(args.inputs / "observations"), "--models", str(args.inputs / "models"),
              "--cpu-threads", str(args.cpu_threads)]
    result = [(stage, [sys.executable, str(HERE / "stages.py"), stage, *common]) for stage in STAGES[:-2]]
    result += [("fit", [sys.executable, str(TABLE / "run.py"), "fit", "--manifest", str(args.output / "runs.json"),
                        "--fit-preset", "efficient", "--output", str(args.output / "fit")]),
               ("evaluate", [sys.executable, str(TABLE / "run.py"), "evaluate", "--manifest",
                             str(args.output / "fit/runs.json"), "--mode", "both", "--output", str(args.output / "evaluate")])]
    if getattr(args, "initial_only", False):
        return result[:STAGES.index("initial") + 1]
    return result


def run(args):
    plan = commands(args)
    if args.plan:
        print(json.dumps(dict(method=args.method, scene=args.scene, commands=dict(plan)), indent=2))
        return
    # A failed stage is not silently reused. Keep its artifacts and select a fresh
    # output for another complete campaign; stages.py also permits explicit debugging.
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh experiment directory: {args.output}")
    for path in (args.inputs / "observations" / args.scene, args.inputs / "models"):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.inputs == args.output or args.output in args.inputs.parents or args.inputs in args.output.parents:
        raise ValueError("Keep downloaded inputs and generated outputs in separate directories")
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", GS_BACKEND="native_add3",
               JAX_PLATFORMS="cpu", OMP_NUM_THREADS=str(args.cpu_threads),
               MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    for key, subdir in [("MPLCONFIGDIR", "matplotlib"), ("TRITON_CACHE_DIR", "triton"),
                        ("TORCHINDUCTOR_CACHE_DIR", "inductor")]:
        env[key] = str(args.output / "cache" / subdir)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    report = dict(schema_version=1, method=args.method, scene=args.scene, commit=commit,
                  source_sha256={str(p.relative_to(REPO)): sha256(p) for p in HERE.rglob("*")
                                 if p.is_file() and "__pycache__" not in p.parts},
                  scope="initial_only" if getattr(args, "initial_only", False) else "distillation",
                  status="running", stages=[], started=time.time())
    save(args.output / "execution.json", report)
    try:
        subprocess.run([sys.executable, str(TABLE / "provenance.py"), "--output", str(args.output / "environment")],
                       env=env, cwd=REPO, check=True)
        for stage, command in plan:
            record = dict(stage=stage, command=command, started=time.time(), status="running")
            report["stages"].append(record)
            save(args.output / "execution.json", report)
            print(f"Starting {args.method}/{args.scene}: {stage}", flush=True)
            with (args.output / f"{stage}.log").open("w") as log:
                result = subprocess.run(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, elapsed_seconds=time.time()-record["started"],
                          status="complete" if result.returncode == 0 else "failed")
            save(args.output / "execution.json", report)
            if result.returncode:
                raise RuntimeError(f"{stage} failed; see {args.output / (stage + '.log')}")
        report["status"] = "complete"
        if not getattr(args, "initial_only", False):
            report["metrics_manifest"] = str(args.output / "evaluate/runs.json")
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        report["elapsed_seconds"] = time.time()-report["started"]
        save(args.output / "execution.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("ms", "wt"), required=True)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--initial-only", action="store_true",
                        help="Stop after initial reconstruction, ready for direct video sampling")
    parser.add_argument("--plan", action="store_true", help="Print commands only; no execution or validation claim")
    args = parser.parse_args()
    args.inputs = args.inputs.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
