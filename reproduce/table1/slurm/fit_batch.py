"""Fit independent scenes on disjoint GPUs within one Slurm allocation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from ..efficiency.cpu_topology import independent_cpus, topology
    from ..efficiency.runtime import FIT_PRESETS
    from ..provenance import allocation_label
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from efficiency.cpu_topology import independent_cpus, topology
    from efficiency.runtime import FIT_PRESETS
    from provenance import allocation_label

SCENES = ("apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel")


def assignments(method, scenes, gpus, cpus, cpus_per_scene):
    if not scenes or len(set(scenes)) != len(scenes) or set(scenes) - set(SCENES):
        raise ValueError("Choose distinct benchmark scenes")
    if method not in ("ms", "wt") or cpus_per_scene < 1:
        raise ValueError("Invalid method or CPU count")
    # The retained WorldTree GPU discovery code expects numeric device IDs.
    if not gpus or any(not gpu.isdecimal() for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct numeric device IDs")
    width = 1 if method == "ms" else 2
    if len(gpus) < len(scenes) * width or len(cpus) < len(scenes) * cpus_per_scene:
        raise ValueError("Insufficient allocated GPUs or CPUs for concurrent scenes")
    return [dict(method=method, scene=scene,
                 gpus=gpus[i * width:(i + 1) * width],
                 cpus=cpus[i * cpus_per_scene:(i + 1) * cpus_per_scene])
            for i, scene in enumerate(scenes)]


def validate_task(task, phase):
    output = Path(task["output"])
    fresh = [Path(task["cache"])]
    if phase == "evaluate":
        payload = json.loads((output / "fit/runs.json").read_text())
        runs = payload.get("runs", [])
        if len(runs) != 1 or any(runs[0].get(key) != task[key] for key in ("method", "scene")):
            raise ValueError("Evaluation requires this scene's completed fit manifest")
        checkpoint = Path(runs[0]["checkpoint"]).resolve()
        if not checkpoint.is_dir() or not checkpoint.is_relative_to((output / "fit").resolve()):
            raise ValueError("Evaluation must use this task's newly fitted checkpoint")
        fresh.extend([output / "evaluate", output / "runs.json"])
    elif phase in ("fit", "both"):
        fresh.append(output)
    else:
        raise ValueError("Invalid batch phase")
    for path in fresh:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Choose a fresh task output: {path}")


def commands(task, plan):
    repo = Path(plan["repo"])
    output = Path(task["output"])
    steps = [
        [sys.executable, str(repo / "reproduce/table1/provenance.py"),
         "--output", task["cache"]],
        [sys.executable, str(repo / "reproduce/table1/run.py"), "fit",
         "--manifest", plan["manifest"], "--assets-root", plan["assets_root"],
         "--method", task["method"], "--scene", task["scene"],
         "--output", str(output / "fit")],
        [sys.executable, str(repo / "reproduce/table1/run.py"), "evaluate",
         "--manifest", str(output / "fit/runs.json"), "--mode", "both",
         "--output", str(output / "evaluate")],
    ]
    if plan.get("fit_preset") is not None:
        steps[1].extend(["--fit-preset", plan["fit_preset"]])
    phase = plan.get("phase", "both")
    return [steps[0]] + ([steps[1]] if phase != "evaluate" else []) + (
        [steps[2]] if phase != "fit" else [])


def worker(task, plan):
    phase = plan.get("phase", "both")
    validate_task(task, phase)
    os.sched_setaffinity(0, task["cpus"])
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(task["gpus"]),
                      MPLCONFIGDIR=str(Path(task["cache"]) / "matplotlib"),
                      TRITON_CACHE_DIR=str(Path(task["cache"]) / "triton"),
                      TORCHINDUCTOR_CACHE_DIR=str(Path(task["cache"]) / "inductor"),
                      OMP_NUM_THREADS=str(len(task["cpus"])),
                      MKL_NUM_THREADS=str(len(task["cpus"])),
                      OPENBLAS_NUM_THREADS=str(len(task["cpus"])),
                      PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                      JAX_PLATFORMS="cpu", GS_BACKEND="native_add3")
    if phase != "evaluate":
        Path(task["output"]).mkdir(parents=True, exist_ok=False)
    Path(task["cache"]).mkdir(parents=True, exist_ok=False)
    timing_path = Path(task["cache"]) / "timing.json"
    timing = {"task": task, "phase": phase, "fit_preset": plan.get("fit_preset"),
              "allocation": allocation_label(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "stages": []}

    def save_timing():
        temporary = timing_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(timing, indent=2) + "\n")
        temporary.replace(timing_path)

    # Separate processes release provenance's CUDA context before fitting, and
    # release fitting memory before evaluation. No CUDA library enters the parent.
    for index, command in enumerate(commands(task, plan)):
        stage = {"name": "provenance" if index == 0 else command[2],
                 "command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
        timing["stages"].append(stage)
        save_timing()
        started = time.perf_counter()
        try:
            result = subprocess.run(command)
            stage["returncode"] = result.returncode
            result.check_returncode()
        except Exception as exc:
            stage["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stage.update(seconds=time.perf_counter() - started,
                         finished_utc=datetime.now(timezone.utc).isoformat())
            save_timing()
    output = Path(task["output"])
    if phase == "fit":
        if not (output / "fit/runs.json").is_file():
            raise FileNotFoundError("Fitting did not produce its completion manifest")
        return
    if not (output / "evaluate/runs.json").is_file():
        raise FileNotFoundError("Evaluation did not produce its completion manifest")
    (output / "runs.json").symlink_to("evaluate/runs.json")


def run_batch(plan, directory):
    directory.mkdir(parents=True, exist_ok=False)
    plan_file = directory / "plan.json"
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    children = []
    interrupted = None

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, task in enumerate(plan["tasks"]):
            if interrupted:
                break
            with (directory / f"{task['method']}_{task['scene']}.log").open("x") as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker-plan",
                     str(plan_file), "--worker-index", str(index)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((task, process))
        while any(process.poll() is None for _, process in children) and not interrupted:
            time.sleep(2)
    finally:
        # Stop complete process groups, including backend multiprocessing workers.
        # Also handles a partial launch failure without leaving orphaned fits.
        live = [process for _, process in children if process.poll() is None]
        # Keep signals tied to live, unreaped children: a PID from a scene that
        # finished hours earlier could have been reused by an unrelated process.
        for process in live:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        for process in live:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        status = {"finished_utc": datetime.now(timezone.utc).isoformat(),
                  "signal": interrupted,
                  "tasks": [{"method": task["method"], "scene": task["scene"],
                             "returncode": process.returncode}
                            for task, process in children]}
        (directory / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 1 if interrupted or any(p.returncode != 0 for _, p in children) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ms", "wt"))
    parser.add_argument("--scenes", nargs="+", choices=SCENES)
    parser.add_argument("--name")
    parser.add_argument("--phase", choices=("fit", "evaluate", "both"), default="both")
    parser.add_argument("--fit-preset", choices=tuple(FIT_PRESETS),
                        help="Fitting preset; also accepts WFM_TABLE1_FIT_PRESET")
    parser.add_argument("--cpus-per-scene", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        plan = json.loads(args.worker_plan.read_text())
        worker(plan["tasks"][args.worker_index], plan)
        return 0
    preset = args.fit_preset or os.environ.get("WFM_TABLE1_FIT_PRESET")
    if preset is not None and preset not in FIT_PRESETS:
        parser.error("WFM_TABLE1_FIT_PRESET must be efficient or reference")
    if not args.method or not args.scenes or not args.name or Path(args.name).name != args.name:
        parser.error("Provide --method, --scenes, and a single-component --name")
    if args.name in (".", ".."):
        parser.error("Invalid batch name")
    repo = Path(os.environ["WFM_REPO_ROOT"]).resolve()
    root = Path(os.environ["WFM_TABLE1_OUTPUT_ROOT"]).resolve()
    allowed_cpus = sorted(os.sched_getaffinity(0))
    cpu_details = topology(allowed_cpus)
    physical_cpus = independent_cpus(allowed_cpus, cpu_details)
    tasks = assignments(args.method, args.scenes,
                        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","),
                        physical_cpus, args.cpus_per_scene)
    for task in tasks:
        key = f"{task['method']}_{task['scene']}"
        task.update(output=str(root / key), cache=str(root / "cache" / key / args.phase))
        task["cpu_topology"] = {cpu: cpu_details[cpu] for cpu in task["cpus"]}
        validate_task(task, args.phase)
    plan = {"started_utc": datetime.now(timezone.utc).isoformat(), "repo": str(repo),
            "assets_root": str(Path(os.environ["WFM_TABLE1_ASSETS_ROOT"]).resolve()),
            "manifest": str(Path(os.environ.get("WFM_TABLE1_MANIFEST",
                            repo / "reproduce/table1/retained_runs.json")).resolve()),
            "phase": args.phase, "fit_preset": preset, "tasks": tasks}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    return run_batch(plan, root / "batches" / args.name)


if __name__ == "__main__":
    sys.exit(main())
