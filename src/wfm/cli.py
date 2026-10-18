from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .artifacts import ArtifactManager
from .config import load_production_config
from .doctor import run_doctor
from .errors import WFMError
from .pipeline import PipelineRunner, RunOptions
from .util import normalize_classes
from .validation import validate_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wfm", description="World From Motion production CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run reconstruction and novel-view generation")
    run.add_argument("--video", required=True, type=Path)
    run.add_argument("--dynamic-class", dest="dynamic_classes", required=True, action="append")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument(
        "--distillation-steps",
        type=int,
        default=None,
        help="Override the generated-view-to-4DGS optimization steps",
    )
    run.add_argument("--device", default="cuda")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Validate runtime, GPU, disk, and weights")
    doctor.add_argument("--output", required=True, type=Path)
    doctor.add_argument("--skip-artifacts", action="store_true")
    doctor.add_argument("--skip-gpu", action="store_true")

    artifacts = subparsers.add_parser("artifacts", help="Manage private model artifacts")
    artifacts_sub = artifacts.add_subparsers(dest="artifact_command", required=True)
    artifacts_sub.add_parser("pull")
    artifacts_sub.add_parser("verify")

    validate = subparsers.add_parser("validate", help="Validate a completed output directory")
    validate.add_argument("--output", required=True, type=Path)

    cleanup = subparsers.add_parser("cleanup", help="Remove only the resumable work directory")
    cleanup.add_argument("--output", required=True, type=Path)
    cleanup.add_argument("--confirm", action="store_true")
    return parser


def execute(args: argparse.Namespace) -> object:
    profile = load_production_config()
    manager = ArtifactManager()
    if args.command == "run":
        classes = normalize_classes(args.dynamic_classes)
        runner = PipelineRunner(profile, manager)
        return runner.run(
            RunOptions(
                video=args.video,
                dynamic_classes=classes,
                output=args.output,
                seed=int(profile.seed if args.seed is None else args.seed),
                max_frames=args.max_frames,
                distillation_steps=args.distillation_steps,
                device=args.device,
                resume=not args.no_resume,
                dry_run=args.dry_run,
            )
        )
    if args.command == "doctor":
        return run_doctor(
            profile,
            output=args.output,
            artifact_manager=manager,
            require_artifacts=not args.skip_artifacts,
            require_gpu=not args.skip_gpu,
        )
    if args.command == "artifacts":
        if args.artifact_command == "pull":
            manager.pull()
        return manager.validate(full_hash=True)
    if args.command == "validate":
        return validate_output(args.output, profile)
    if args.command == "cleanup":
        if not args.confirm:
            raise WFMError("cleanup requires --confirm")
        work = args.output.expanduser().resolve() / "work"
        if work.is_dir():
            shutil.rmtree(work)
        return {"removed": str(work)}
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except (WFMError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"wfm: error: {exc}\n")
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
