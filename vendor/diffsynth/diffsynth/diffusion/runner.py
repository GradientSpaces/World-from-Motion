import os, json, torch, shutil, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tqdm import tqdm
from accelerate import Accelerator, skip_first_batches
from PIL import Image
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _parse_slurm_timelimit_to_seconds(value):
    """Parse Slurm timelimit strings like M:S, H:M:S, or D-H:M:S."""
    if not value:
        return None
    text = str(value).strip()
    if text.upper() in {"UNLIMITED", "INFINITE", "NOT_SET", "N/A"}:
        return None

    days = 0
    if "-" in text:
        day_str, text = text.split("-", 1)
        try:
            days = int(day_str)
        except ValueError:
            return None

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        elif len(parts) == 1:
            hours, minutes, seconds = 0, int(parts[0]), 0
        else:
            return None
    except ValueError:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _resolve_autoresume_trigger_elapsed_seconds():
    """
    Derive the self-yield time from WFM_AUTORESUME_BEFORE_TIMELIMIT (minutes) and
    WFM_AUTORESUME_TIMELIMIT (Slurm time format), both set by a scheduler wrapper.
    Returns elapsed seconds from wrapper start when termination should be requested.
    """
    before_timelimit = os.environ.get("WFM_AUTORESUME_BEFORE_TIMELIMIT")
    actual_timelimit = os.environ.get("WFM_AUTORESUME_TIMELIMIT")
    if before_timelimit is None or actual_timelimit is None:
        return None

    try:
        before_seconds = int(float(before_timelimit) * 60)
    except (TypeError, ValueError):
        return None

    timelimit_seconds = _parse_slurm_timelimit_to_seconds(actual_timelimit)
    if timelimit_seconds is None:
        return None

    return max(0, timelimit_seconds - before_seconds)


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _broadcast_object_from_main_process(accelerator: Accelerator, payload):
    """Broadcast a Python object from rank 0 to all ranks."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        payload_list = [payload if accelerator.is_main_process else None]
        torch.distributed.broadcast_object_list(payload_list, src=0)
        return payload_list[0]
    return payload


def _detect_precache_resume_prefix(
    output_path,
    process_index,
    num_processes,
    existing_manifest_paths=None,
    printer=print,
):
    """
    Detect the largest rank-local contiguous cached prefix that is safe to skip.

    Uses on-disk numeric .pth filenames as the source of truth, but only fast-forwards
    through files that also already have manifest entries so split backfill semantics
    remain unchanged for the skipped prefix.
    """
    numeric_rank_dirs = []
    if os.path.isdir(output_path):
        for entry_name in os.listdir(output_path):
            entry_path = os.path.join(output_path, entry_name)
            if os.path.isdir(entry_path) and str(entry_name).isdigit():
                numeric_rank_dirs.append(int(entry_name))
    numeric_rank_dirs = sorted(numeric_rank_dirs)

    if numeric_rank_dirs:
        expected_rank_dirs = set(range(max(0, int(num_processes))))
        observed_rank_dirs = set(numeric_rank_dirs)
        unexpected_rank_dirs = sorted(observed_rank_dirs - expected_rank_dirs)
        missing_rank_dirs = sorted(expected_rank_dirs - observed_rank_dirs)
        if unexpected_rank_dirs or missing_rank_dirs:
            if printer is not None:
                printer(
                    "[Precache autoresume] "
                    f"rank {process_index}: fast-forward disabled because cache rank directories "
                    f"do not match current world size {num_processes} "
                    f"(missing={missing_rank_dirs[:8]}, unexpected={unexpected_rank_dirs[:8]})."
                )
            return 0, False

    rank_dir = os.path.join(output_path, str(process_index))
    if not os.path.isdir(rank_dir):
        if printer is not None:
            printer(
                f"[Precache autoresume] rank {process_index}: no existing cache directory; "
                "starting from local step 0."
            )
        return 0, True

    completed_indices = set()
    ignored_cache_files = []
    for file_name in os.listdir(rank_dir):
        file_path = os.path.join(rank_dir, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith(".pth"):
            continue
        stem = file_name[:-4]
        if stem.isdigit():
            completed_indices.add(int(stem))
        else:
            ignored_cache_files.append(file_name)

    if ignored_cache_files and printer is not None:
        ignored_preview = sorted(ignored_cache_files)[:5]
        ignored_suffix = "..." if len(ignored_cache_files) > len(ignored_preview) else ""
        printer(
            f"[Precache autoresume] rank {process_index}: ignoring non-numeric cache files "
            f"when detecting resume prefix: {ignored_preview}{ignored_suffix}"
        )

    resume_step_offset = 0
    while resume_step_offset in completed_indices:
        rel_path = os.path.join(str(process_index), f"{resume_step_offset}.pth")
        if existing_manifest_paths is not None and rel_path not in existing_manifest_paths:
            if printer is not None:
                printer(
                    "[Precache autoresume] "
                    f"rank {process_index}: fast-forward stops at local step {resume_step_offset} "
                    f"because manifest entry '{rel_path}' is missing."
                )
            break
        resume_step_offset += 1

    if printer is not None:
        if resume_step_offset > 0:
            printer(
                f"[Precache autoresume] rank {process_index}: detected contiguous cached prefix "
                f"of {resume_step_offset} samples."
            )
        else:
            printer(
                f"[Precache autoresume] rank {process_index}: no contiguous cached prefix detected."
            )

    return resume_step_offset, True


def _sync_device_for_timing(device):
    """Synchronize GPU device before/after timed regions."""
    device_type = getattr(device, "type", None)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def _compact_precache_payload(payload):
    """
    Remove PIL image payloads before serializing precached samples.
    Keeps tensors/scalars untouched and preserves tuple/list containers.
    """
    removed_marker = object()
    stats = {
        "removed_images": 0,
        "removed_image_sequences": 0,
        "removed_dict_fields": 0,
        "removed_sequence_items": 0,
    }

    def is_pil_image(value):
        return isinstance(value, Image.Image)

    def compact(value):
        if is_pil_image(value):
            stats["removed_images"] += 1
            return removed_marker

        if isinstance(value, dict):
            compacted = {}
            for key, item in value.items():
                compacted_item = compact(item)
                if compacted_item is removed_marker:
                    stats["removed_dict_fields"] += 1
                    continue
                compacted[key] = compacted_item
            return compacted

        if isinstance(value, list):
            if len(value) > 0 and all(is_pil_image(item) for item in value):
                stats["removed_image_sequences"] += 1
                stats["removed_images"] += len(value)
                return removed_marker

            compacted_items = []
            all_removed = len(value) > 0
            for item in value:
                compacted_item = compact(item)
                if compacted_item is removed_marker:
                    stats["removed_sequence_items"] += 1
                    compacted_items.append(None)
                else:
                    compacted_items.append(compacted_item)
                    all_removed = False
            if all_removed:
                stats["removed_image_sequences"] += 1
                return removed_marker
            return compacted_items

        if isinstance(value, tuple):
            if len(value) > 0 and all(is_pil_image(item) for item in value):
                stats["removed_image_sequences"] += 1
                stats["removed_images"] += len(value)
                return removed_marker

            compacted_items = []
            all_removed = len(value) > 0
            for item in value:
                compacted_item = compact(item)
                if compacted_item is removed_marker:
                    stats["removed_sequence_items"] += 1
                    compacted_items.append(None)
                else:
                    compacted_items.append(compacted_item)
                    all_removed = False
            if all_removed:
                stats["removed_image_sequences"] += 1
                return removed_marker
            return tuple(compacted_items)

        return value

    compacted_payload = compact(payload)
    if compacted_payload is removed_marker:
        if isinstance(payload, tuple):
            compacted_payload = tuple()
        elif isinstance(payload, list):
            compacted_payload = []
        elif isinstance(payload, dict):
            compacted_payload = {}
        else:
            compacted_payload = None
    return compacted_payload, stats


def _values_all_equal(values):
    if len(values) <= 1:
        return True
    first = values[0]
    for value in values[1:]:
        if torch.is_tensor(first) or torch.is_tensor(value):
            return False
        try:
            if value != first:
                return False
        except Exception:
            return False
    return True


def _collate_cached_values(values):
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim == 0:
            return torch.stack(values, dim=0)
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {
            key: _collate_cached_values([value[key] for value in values])
            for key in first.keys()
        }
    if isinstance(first, tuple):
        return tuple(
            _collate_cached_values([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        if _values_all_equal(values):
            return first
        return [value for value in values]
    if _values_all_equal(values):
        return first
    return [value for value in values]


def _collate_cached_samples(batch):
    if len(batch) == 1:
        return batch[0]
    return _collate_cached_values(batch)


def _get_job_id_for_autoresume_file():
    for key in ("SLURM_JOB_ID", "SLURM_JOBID", "JOBID"):
        value = os.environ.get(key)
        if value:
            # JOBID can include rank suffix like "1713236_0".
            return str(value).split("_")[0]
    return None


def _request_autoresume(printer=print):
    """
    Request a follow-up job by writing the file named by WFM_AUTORESUME_FILE.
    A scheduler wrapper that sets this variable can requeue the run.
    """
    resume_file = os.environ.get("WFM_AUTORESUME_FILE")
    if not resume_file:
        return False

    payload = {
        "AUTO_RESUME_DEBUG_MESSAGE": "Termination requested by training worker",
        "AUTO_RESUME_DEBUG_REQUESTED_AT_UTC": datetime.now(timezone.utc).isoformat(),
    }
    try:
        resume_dir = os.path.dirname(resume_file)
        if resume_dir:
            os.makedirs(resume_dir, exist_ok=True)
        _atomic_write_json(resume_file, payload)
        printer(f"Requested auto-resume via {resume_file}")
        return True
    except Exception as e:
        printer(f"Warning: failed to write autoresume request file '{resume_file}': {e}")
        return False


def _save_git_tracked_snapshot(output_path, args=None, printer=print):
    """Save all git-tracked files into <output_path>/code_snapshot* for reproducibility."""
    try:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve repository root and tracked files from git.
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        tracked_result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = [line.strip() for line in tracked_result.stdout.splitlines() if line.strip()]
        if len(tracked_files) == 0:
            printer("Warning: no git-tracked files found; skip code snapshot.")
            return None

        # Keep previous snapshots by creating a unique folder if needed.
        snapshot_dir = output_dir / "code_snapshot"
        if snapshot_dir.exists():
            suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            candidate = output_dir / f"code_snapshot_{suffix}"
            index = 1
            while candidate.exists():
                candidate = output_dir / f"code_snapshot_{suffix}_{index}"
                index += 1
            snapshot_dir = candidate
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for rel_path in tracked_files:
            src = repo_root / rel_path
            if not src.is_file():
                continue
            dst = snapshot_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        commit_hash = commit_result.stdout.strip() if commit_result.returncode == 0 else None

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(repo_root),
            "snapshot_dir": str(snapshot_dir),
            "git_commit": commit_hash,
            "tracked_files_total": len(tracked_files),
            "copied_files": copied,
            "args": vars(args) if args is not None else None,
        }
        with open(snapshot_dir / "snapshot_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        printer(f"Code snapshot saved to {snapshot_dir} ({copied} files)")
        return str(snapshot_dir)
    except Exception as e:
        printer(f"Warning: code snapshot skipped: {e}")
        return None


class _SignalTerminationController:
    def __init__(self):
        self._signal_seen = False
        self._autoresume_term_file = self._resolve_autoresume_term_file()
        self._timer_trigger_elapsed_seconds = _resolve_autoresume_trigger_elapsed_seconds()
        self._run_start_epoch_seconds = self._resolve_run_start_epoch_seconds()

    def install(self):
        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            try:
                signal.signal(sig, self._handler)
            except Exception:
                # Some environments may not allow replacing handlers.
                continue

    def _handler(self, signum, frame):
        del signum, frame
        self._signal_seen = True

    def _resolve_autoresume_term_file(self):
        autoresume_out = os.environ.get("WFM_AUTORESUME_FILE")
        if not autoresume_out:
            return None

        job_id = _get_job_id_for_autoresume_file()
        if not job_id:
            return None

        part_dir = os.path.dirname(autoresume_out)
        if not part_dir:
            return None

        return os.path.join(part_dir, f"job-{job_id}-autoresume.TERMINATION-REQUESTED")

    def _autoresume_termination_requested(self):
        return bool(self._autoresume_term_file and os.path.exists(self._autoresume_term_file))

    def _resolve_run_start_epoch_seconds(self):
        submit_started = os.environ.get("WFM_AUTORESUME_STARTED")
        if submit_started is not None:
            try:
                return float(submit_started)
            except (TypeError, ValueError):
                pass
        return time.time()

    def _timer_termination_requested(self):
        if self._timer_trigger_elapsed_seconds is None:
            return False
        elapsed_seconds = time.time() - self._run_start_epoch_seconds
        return elapsed_seconds >= self._timer_trigger_elapsed_seconds

    def termination_requested_all_ranks(self, device):
        local_flag = int(
            self._signal_seen
            or self._autoresume_termination_requested()
            or self._timer_termination_requested()
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag_tensor = torch.tensor(local_flag, device=device, dtype=torch.int32)
            torch.distributed.all_reduce(flag_tensor, op=torch.distributed.ReduceOp.MAX)
            return bool(int(flag_tensor.item()))
        return bool(local_flag)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    eval_dataset: torch.utils.data.Dataset = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    dit_learning_rate = getattr(args, "dit_learning_rate", None) if args else None
    other_learning_rate = getattr(args, "other_learning_rate", None) if args else None
    uncond_latent_learning_rate = getattr(args, "uncond_latent_learning_rate", None) if args else None
    if (dit_learning_rate is None) != (other_learning_rate is None):
        raise ValueError(
            "Both --dit_learning_rate and --other_learning_rate must be set together."
        )

    # ── Parse validation / checkpointing args (backwards-compatible defaults) ──
    ckpt_save_every = getattr(args, 'ckpt_save_every', None) if args else None
    visualize_every = getattr(args, 'visualize_every', 0) if args else 0
    val_size = getattr(args, 'val_size', 4) if args else 4
    resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', False) if args else False
    warmup_steps = getattr(args, 'warmup_steps', 0) if args else 0
    wandb_enabled = getattr(args, 'wandb', False) if args else False
    initial_global_step_raw = getattr(args, "initial_global_step", 0) if args else 0
    initial_global_step = 0 if initial_global_step_raw is None else max(0, int(initial_global_step_raw))
    max_train_steps_raw = getattr(args, "max_train_steps", None) if args else None
    max_train_steps = None if max_train_steps_raw is None else int(max_train_steps_raw)
    if max_train_steps is not None and max_train_steps <= 0:
        raise ValueError(
            f"--max_train_steps must be > 0, got {max_train_steps}."
        )
    target_global_step = (
        None if max_train_steps is None else initial_global_step + max_train_steps
    )
    autoresume_check_interval = max(1, int(getattr(args, "autoresume_check_interval", 1) if args else 1))
    profile_step_time = bool(getattr(args, "profile_step_time", False) if args else False)
    profile_log_interval = max(1, int(getattr(args, "profile_log_interval", 20) if args else 20))
    profile_warmup_steps = max(0, int(getattr(args, "profile_warmup_steps", 5) if args else 5))
    profile_output_name = str(
        getattr(args, "profile_output_name", "time_profile_summary.json")
        if args else "time_profile_summary.json"
    )

    # ── Save reproducibility code snapshot (main process only) ───────────────
    if accelerator.is_main_process:
        _save_git_tracked_snapshot(
            model_logger.output_path,
            args=args,
            printer=accelerator.print,
        )
    accelerator.wait_for_everyone()

    # ── Initialize trackers (wandb / tensorboard) ────────────────────────────
    if wandb_enabled:
        job_name = getattr(args, 'job_name', None)
        wandb_kwargs = {
            "id": job_name,
            "entity": getattr(args, 'wandb_entity', None),
            "resume": "allow",
        }
        accelerator.init_trackers(
            project_name=getattr(args, 'wandb_project', 'world-from-motion'),
            config=vars(args) if args else {},
            init_kwargs={"wandb": wandb_kwargs},
        )

    trackers_closed = False

    def _safe_end_training():
        nonlocal trackers_closed
        if not wandb_enabled or trackers_closed:
            return
        try:
            accelerator.end_training()
        except Exception as e:
            accelerator.print(f"Warning: failed to finalize trackers cleanly: {e}")
        finally:
            trackers_closed = True

    termination_controller = _SignalTerminationController()
    termination_controller.install()

    def maybe_checkpoint_and_exit(
        epoch_id_for_save,
        step_for_save,
        reason,
        step_in_epoch_for_save=None,
        checkpoint_already_saved=False,
    ):
        should_check = (step_for_save % autoresume_check_interval == 0)
        if not should_check:
            return False
        if not termination_controller.termination_requested_all_ranks(accelerator.device):
            return False

        accelerator.print(
            f"Termination requested ({reason}); checkpointing at step {step_for_save} and exiting cleanly."
        )
        if not checkpoint_already_saved:
            model_logger.save_states(
                accelerator,
                model,
                epoch_id_for_save,
                step_for_save,
                step_in_epoch=step_in_epoch_for_save,
            )
        else:
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            _request_autoresume(printer=accelerator.print)
        accelerator.wait_for_everyone()

        _safe_end_training()
        return True

    # ── Train / val split ────────────────────────────────────────────────────
    val_loader = None
    val_source_dataset = None
    has_train_split = (
        hasattr(dataset, 'train_indices')
        and len(getattr(dataset, 'train_indices', [])) > 0
    )
    has_val_split = (
        hasattr(dataset, 'val_indices')
        and len(getattr(dataset, 'val_indices', [])) > 0
    )
    train_set = torch.utils.data.Subset(dataset, dataset.train_indices) if has_train_split else dataset

    val_set = None
    if eval_dataset is not None:
        eval_has_val_split = (
            hasattr(eval_dataset, 'val_indices')
            and len(getattr(eval_dataset, 'val_indices', [])) > 0
        )
        val_set = torch.utils.data.Subset(eval_dataset, eval_dataset.val_indices) if eval_has_val_split else eval_dataset
        val_source_dataset = eval_dataset
    elif has_train_split and has_val_split:
        val_set = torch.utils.data.Subset(dataset, dataset.val_indices)
        val_source_dataset = dataset

    train_batch_size = 1
    train_collate_fn = lambda x: x[0]
    if getattr(dataset, "load_from_cache", False):
        requested_batch_size = getattr(args, "per_device_train_batch_size", 1) if args else 1
        train_batch_size = max(1, int(requested_batch_size))
        if train_batch_size > 1:
            train_collate_fn = _collate_cached_samples

    train_num_workers = max(0, int(getattr(dataset, "dataloader_num_workers", num_workers)))
    train_loader_kwargs = {
        "dataset": train_set,
        "batch_size": train_batch_size,
        "shuffle": True,
        "collate_fn": train_collate_fn,
        "num_workers": train_num_workers,
        "pin_memory": True,
    }
    if train_num_workers > 0:
        train_prefetch_factor = getattr(dataset, "dataloader_prefetch_factor", None)
        if train_prefetch_factor is not None:
            train_loader_kwargs["prefetch_factor"] = max(1, int(train_prefetch_factor))
        if bool(getattr(dataset, "dataloader_persistent_workers", False)):
            train_loader_kwargs["persistent_workers"] = True
    train_loader = torch.utils.data.DataLoader(**train_loader_kwargs)
    if val_set is not None:
        val_loader = torch.utils.data.DataLoader(
            val_set, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers, pin_memory=True,
        )

    can_run_raw_evaluation = (
        val_loader is not None
        and not getattr(val_source_dataset, 'load_from_cache', False)
    )
    if val_loader is not None and not can_run_raw_evaluation:
        accelerator.print(
            "Validation dataset is cache-backed; skipping baseline/eval metrics. "
            "Provide a separate raw evaluation dataset to enable validation."
        )

    # ── Optimizer & LR scheduler (with optional warmup) ──────────────────────
    if dit_learning_rate is not None:
        if not hasattr(model, "build_optimizer_param_groups"):
            raise ValueError(
                "Split WAN learning rates were requested, but this model does not define "
                "build_optimizer_param_groups()."
            )
        optimizer_param_groups, optimizer_group_summaries = model.build_optimizer_param_groups(
            dit_learning_rate=dit_learning_rate,
            other_learning_rate=other_learning_rate,
            uncond_latent_learning_rate=uncond_latent_learning_rate,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_param_groups)
        accelerator.print("[Optimizer] Split WAN learning rates enabled.")
        for group_summary in optimizer_group_summaries:
            sample_names = ", ".join(group_summary["sample_names"]) if group_summary["sample_names"] else "n/a"
            accelerator.print(
                f"[Optimizer] group={group_summary['name']} lr={group_summary['lr']:.6g} "
                f"tensors={group_summary['num_tensors']} "
                f"parameters={group_summary['num_parameters']} "
                f"samples={sample_names}"
            )
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / max(warmup_steps, 1), total_iters=warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps],
        )

    # ── Prepare with accelerator ─────────────────────────────────────────────
    model.to(device=accelerator.device)
    if val_loader is not None:
        model, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(
            model, optimizer, train_loader, scheduler, val_loader,
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler,
        )

    num_train_steps_per_epoch = None
    try:
        num_train_steps_per_epoch = len(train_loader)
    except TypeError:
        # Some iterable datasets may not define __len__.
        num_train_steps_per_epoch = None

    # ── Auto-resume from checkpoint ──────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    resume_step_in_epoch = 0
    resumed = False
    has_initial_warm_start = bool(
        initial_global_step > 0
        or (getattr(args, "initial_trainable_checkpoint", None) if args else None)
    )
    if resume_from_checkpoint:
        training_state_path = os.path.join(model_logger.output_path, "training_state.json")
        resume_payload = {
            "should_resume": False,
            "checkpoint_path": None,
            "resumed_epoch": -1,
            "global_step": 0,
            "resume_step_in_epoch": 0,
            "start_epoch": 0,
            "error": None,
        }

        if accelerator.is_main_process:
            if os.path.exists(training_state_path):
                try:
                    with open(training_state_path, "r") as f:
                        training_state = json.load(f)
                    resumed_epoch = int(training_state.get("epoch", -1))
                    if resumed_epoch != -1:
                        checkpoint_path = training_state.get(
                            "checkpoint_path",
                            os.path.join(model_logger.output_path, "checkpoint"),
                        )
                        if os.path.isdir(checkpoint_path):
                            restored_global_step = int(training_state.get("global_step", 0))
                            restored_step_in_epoch = training_state.get("step_in_epoch")
                            if restored_step_in_epoch is None and num_train_steps_per_epoch:
                                # Backward-compatible fallback for old checkpoints.
                                restored_step_in_epoch = restored_global_step % num_train_steps_per_epoch
                            restored_step_in_epoch = max(0, int(restored_step_in_epoch or 0))

                            restored_start_epoch = resumed_epoch
                            if (
                                num_train_steps_per_epoch is not None
                                and restored_step_in_epoch >= num_train_steps_per_epoch
                            ):
                                # A step index equal/exceeding epoch length means the next epoch should start.
                                restored_start_epoch = resumed_epoch + 1
                                restored_step_in_epoch = 0

                            resume_payload = {
                                "should_resume": True,
                                "checkpoint_path": checkpoint_path,
                                "resumed_epoch": resumed_epoch,
                                "global_step": restored_global_step,
                                "resume_step_in_epoch": restored_step_in_epoch,
                                "start_epoch": restored_start_epoch,
                                "error": None,
                            }
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    resume_payload["error"] = f"Failed to parse checkpoint metadata: {e}"

        resume_payload = _broadcast_object_from_main_process(accelerator, resume_payload)
        if resume_payload["error"] is not None:
            accelerator.print(f"{resume_payload['error']}. Starting from scratch.")

        if resume_payload["should_resume"]:
            checkpoint_path = str(resume_payload["checkpoint_path"])
            accelerator.print(f"Resuming from checkpoint: {checkpoint_path}")
            accelerator.wait_for_everyone()
            accelerator.load_state(checkpoint_path)

            resumed_epoch = int(resume_payload["resumed_epoch"])
            global_step = int(resume_payload["global_step"])
            resume_step_in_epoch = int(resume_payload["resume_step_in_epoch"])
            start_epoch = int(resume_payload["start_epoch"])

            accelerator.print(
                f"Resumed from epoch {resumed_epoch}, global_step {global_step}. "
                f"Starting from epoch {start_epoch}."
            )
            if resume_step_in_epoch > 0:
                accelerator.print(
                    f"Skipping {resume_step_in_epoch} already-processed steps "
                    f"in epoch {start_epoch}."
                )
            resumed = True

        if not resumed:
            if has_initial_warm_start:
                accelerator.print("No valid checkpoint found. Falling back to configured warm start.")
            else:
                accelerator.print("No valid checkpoint found. Starting training from scratch.")

    if resumed:
        model_logger.num_steps = global_step
    elif initial_global_step > 0:
        global_step = initial_global_step
        model_logger.num_steps = initial_global_step
        accelerator.print(
            f"Initializing global step to {initial_global_step} "
            "for weight-only continuation."
        )

    # ── Determine checkpointing mode ─────────────────────────────────────────
    use_step_ckpt = ckpt_save_every is not None and ckpt_save_every > 0

    def run_post_checkpoint_validation(step_for_eval):
        model_module: Any = accelerator.unwrap_model(model)
        ran_validation = False

        # Run validation if the model supports it (skip when using cached data).
        if can_run_raw_evaluation and hasattr(model_module, 'evaluate'):
            visualize = (
                visualize_every > 0
                and step_for_eval % visualize_every == 0
            )
            model_module.evaluate(
                accelerator, val_loader, step_for_eval, model_logger,
                max_logs=val_size, visualize=visualize,
            )
            ran_validation = True

        # Optional additive DyCheck benchmark hook. Existing jobs are unchanged
        # unless the model explicitly enables benchmark evaluation.
        if (
            hasattr(model_module, 'dycheck_benchmark')
            and bool(getattr(model_module, 'dycheck_benchmark_enabled', False))
        ):
            model_module.dycheck_benchmark(
                accelerator,
                global_step=step_for_eval,
                logger=model_logger,
            )
            ran_validation = True

        # Restore training mode after any validation path.
        if (
            ran_validation
            and hasattr(model_module, 'trainable_models')
            and hasattr(model_module, 'pipe')
        ):
            model_module.pipe.freeze_except(model_module.trainable_models)
            if hasattr(model_module, "_configure_uncond_latent_trainability"):
                model_module._configure_uncond_latent_trainability()
            if hasattr(model_module, "_configure_scene_coord_gate_trainability"):
                model_module._configure_scene_coord_gate_trainability()
            model_module.pipe.scheduler.set_timesteps(1000, training=True)

    # ── Optional step-time profiling ─────────────────────────────────────────
    profile_keys = (
        "data_wait_s",
        "zero_grad_s",
        "forward_s",
        "backward_s",
        "optimizer_s",
        "step_total_s",
        "overhead_s",
    )
    profile_totals = {key: 0.0 for key in profile_keys}
    profiled_steps = 0
    profile_last_step_end_time = time.perf_counter() if profile_step_time else None

    def _reduce_profile_stats():
        values = [profile_totals[key] for key in profile_keys]
        values.append(float(profiled_steps))
        stats_tensor = torch.tensor(values, device=accelerator.device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats_tensor, op=torch.distributed.ReduceOp.SUM)
        reduced_values = stats_tensor.detach().cpu().tolist()
        reduced_totals = {
            key: float(reduced_values[index]) for index, key in enumerate(profile_keys)
        }
        reduced_count = int(round(reduced_values[-1]))
        return reduced_totals, reduced_count

    def emit_profile_snapshot(step, final=False, reason=None):
        nonlocal trackers_closed
        if not profile_step_time:
            return
        if not final and (step % profile_log_interval != 0):
            return

        reduced_totals, reduced_count = _reduce_profile_stats()
        means = {
            key: (reduced_totals[key] / reduced_count) if reduced_count > 0 else None
            for key in profile_keys
        }

        if accelerator.is_main_process:
            if reduced_count > 0:
                iter_total_s = means["step_total_s"] + means["data_wait_s"]
                step_per_second = (1.0 / iter_total_s) if iter_total_s > 0 else 0.0
                step_total_s = means["step_total_s"] if means["step_total_s"] is not None else 0.0

                def pct(key):
                    value = means[key] if means[key] is not None else 0.0
                    return (value / step_total_s * 100.0) if step_total_s > 0 else 0.0

                tag = "[profile][final]" if final else "[profile]"
                suffix = f" ({reason})" if reason else ""
                accelerator.print(
                    f"{tag} step={step}, profiled_steps={reduced_count}{suffix} | "
                    f"data_wait={means['data_wait_s']:.4f}s | "
                    f"zero_grad={means['zero_grad_s']:.4f}s ({pct('zero_grad_s'):.1f}%) | "
                    f"forward={means['forward_s']:.4f}s ({pct('forward_s'):.1f}%) | "
                    f"backward={means['backward_s']:.4f}s ({pct('backward_s'):.1f}%) | "
                    f"optimizer={means['optimizer_s']:.4f}s ({pct('optimizer_s'):.1f}%) | "
                    f"overhead={means['overhead_s']:.4f}s ({pct('overhead_s'):.1f}%) | "
                    f"step_total={step_total_s:.4f}s | iter_sps={step_per_second:.3f}"
                )

                if wandb_enabled and not trackers_closed:
                    try:
                        accelerator.log(
                            {
                                "profile/data_wait_s": means["data_wait_s"],
                                "profile/zero_grad_s": means["zero_grad_s"],
                                "profile/forward_s": means["forward_s"],
                                "profile/backward_s": means["backward_s"],
                                "profile/optimizer_s": means["optimizer_s"],
                                "profile/overhead_s": means["overhead_s"],
                                "profile/step_total_s": means["step_total_s"],
                                "profile/iter_steps_per_second": step_per_second,
                                "profile/profiled_steps": reduced_count,
                            },
                            step=max(step, 0),
                        )
                    except Exception as e:
                        accelerator.print(
                            f"Warning: failed to log profiling metrics to trackers: {e}"
                        )
                        trackers_closed = True
            elif final:
                accelerator.print(
                    "[profile][final] No steps were profiled. "
                    f"Increase num_epochs or reduce --profile_warmup_steps (current={profile_warmup_steps})."
                )

            if final:
                output_file = (
                    profile_output_name
                    if os.path.isabs(profile_output_name)
                    else os.path.join(model_logger.output_path, profile_output_name)
                )
                output_dir = os.path.dirname(output_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                summary_payload = {
                    "global_step": int(step),
                    "profiled_steps": int(reduced_count),
                    "profile_warmup_steps": int(profile_warmup_steps),
                    "profile_log_interval": int(profile_log_interval),
                    "num_processes": int(accelerator.num_processes),
                    "mean_seconds": means,
                    "sum_seconds": reduced_totals,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": reason,
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(summary_payload, f, indent=2)
                accelerator.print(f"[profile][final] Wrote timing summary to {output_file}")

    # ── Baseline evaluation: compare vace_video vs GT before training ────────
    if not resumed and can_run_raw_evaluation:
        model_module = accelerator.unwrap_model(model)
        if hasattr(model_module, "evaluate_vace_baseline"):
            model_module.evaluate_vace_baseline(
                accelerator,
                val_loader,
                global_step=global_step,
            )

    # ── Training loop ────────────────────────────────────────────────────────
    stop_after_current_step = False
    final_epoch_id = start_epoch
    for epoch_id in range(start_epoch, num_epochs):
        final_epoch_id = epoch_id
        epoch_step_offset = resume_step_in_epoch if resumed and epoch_id == start_epoch else 0
        epoch_train_loader = train_loader
        progress_total = None
        if epoch_step_offset > 0:
            epoch_train_loader = skip_first_batches(train_loader, epoch_step_offset)
            progress_total = len(train_loader)
        progress_bar = tqdm(
            epoch_train_loader,
            desc=f"Epoch {epoch_id}",
            disable=not accelerator.is_main_process,
            initial=epoch_step_offset,
            total=progress_total,
        )
        for step_in_epoch_local, data in enumerate(progress_bar):
            step_in_epoch = step_in_epoch_local + epoch_step_offset
            if profile_step_time:
                step_start_time = time.perf_counter()
                data_wait_s = max(
                    0.0,
                    step_start_time - profile_last_step_end_time if profile_last_step_end_time is not None else 0.0,
                )
                zero_grad_s = 0.0
                forward_s = 0.0
                backward_s = 0.0
                optimizer_s = 0.0

            if maybe_checkpoint_and_exit(
                epoch_id,
                global_step,
                "before optimizer step",
                step_in_epoch_for_save=step_in_epoch,
            ):
                emit_profile_snapshot(global_step, final=True, reason="termination before optimizer step")
                return

            with accelerator.accumulate(model):
                if profile_step_time:
                    _sync_device_for_timing(accelerator.device)
                    segment_start_time = time.perf_counter()
                optimizer.zero_grad()
                if profile_step_time:
                    _sync_device_for_timing(accelerator.device)
                    zero_grad_s = max(0.0, time.perf_counter() - segment_start_time)

                    _sync_device_for_timing(accelerator.device)
                    segment_start_time = time.perf_counter()
                if getattr(dataset, 'load_from_cache', False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                if profile_step_time:
                    _sync_device_for_timing(accelerator.device)
                    forward_s = max(0.0, time.perf_counter() - segment_start_time)

                    _sync_device_for_timing(accelerator.device)
                    segment_start_time = time.perf_counter()
                accelerator.backward(loss)
                if profile_step_time:
                    _sync_device_for_timing(accelerator.device)
                    backward_s = max(0.0, time.perf_counter() - segment_start_time)

                    _sync_device_for_timing(accelerator.device)
                    segment_start_time = time.perf_counter()
                optimizer.step()
                scheduler.step()
                if profile_step_time:
                    _sync_device_for_timing(accelerator.device)
                    optimizer_s = max(0.0, time.perf_counter() - segment_start_time)

            # Update progress bar
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            # Log training metrics
            model_logger.on_step_end(
                accelerator, model,
                save_steps=save_steps if not use_step_ckpt else None,
                loss=loss, optimizer=optimizer,
                global_step=global_step, epoch=epoch_id,
            )
            global_step += 1
            just_saved_step_ckpt = False

            # ── Step-based checkpoint + optional validation ──────────────
            if use_step_ckpt and global_step % ckpt_save_every == 0:
                # Save first so a preemption during eval does not lose recent train progress.
                model_logger.save_states(
                    accelerator,
                    model,
                    epoch_id,
                    global_step,
                    step_in_epoch=step_in_epoch + 1,
                )
                just_saved_step_ckpt = True

                if maybe_checkpoint_and_exit(
                    epoch_id,
                    global_step,
                    "after boundary checkpoint before evaluation",
                    step_in_epoch_for_save=step_in_epoch + 1,
                    checkpoint_already_saved=True,
                ):
                    emit_profile_snapshot(global_step, final=True, reason="termination after boundary checkpoint")
                    return

                run_post_checkpoint_validation(global_step)

            if profile_step_time:
                _sync_device_for_timing(accelerator.device)
                step_end_time = time.perf_counter()
                step_total_s = max(0.0, step_end_time - step_start_time)
                overhead_s = max(
                    0.0,
                    step_total_s - (zero_grad_s + forward_s + backward_s + optimizer_s),
                )
                if global_step > profile_warmup_steps:
                    profile_totals["data_wait_s"] += data_wait_s
                    profile_totals["zero_grad_s"] += zero_grad_s
                    profile_totals["forward_s"] += forward_s
                    profile_totals["backward_s"] += backward_s
                    profile_totals["optimizer_s"] += optimizer_s
                    profile_totals["step_total_s"] += step_total_s
                    profile_totals["overhead_s"] += overhead_s
                    profiled_steps += 1
                    emit_profile_snapshot(global_step, final=False)
                profile_last_step_end_time = step_end_time

            if target_global_step is not None and global_step >= target_global_step:
                accelerator.print(
                    f"Reached max_train_steps={max_train_steps}; "
                    f"stopping at global_step {global_step}."
                )
                stop_after_current_step = True
                break

            if maybe_checkpoint_and_exit(
                epoch_id,
                global_step,
                "after optimizer step",
                step_in_epoch_for_save=step_in_epoch + 1,
                checkpoint_already_saved=just_saved_step_ckpt,
            ):
                emit_profile_snapshot(global_step, final=True, reason="termination after optimizer step")
                return

        if stop_after_current_step:
            break

        # Epoch-based saving (fallback when step-based checkpointing is off)
        if not use_step_ckpt and save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    # ── Final save ───────────────────────────────────────────────────────────
    if not use_step_ckpt:
        model_logger.on_training_end(accelerator, model, save_steps)
    elif global_step % ckpt_save_every != 0:
        # Save final checkpoint if we haven't just saved one
        model_logger.save_states(accelerator, model, final_epoch_id, global_step)
        run_post_checkpoint_validation(global_step)

    emit_profile_snapshot(global_step, final=True, reason="training complete")

    # End training (flush trackers)
    if hasattr(model_logger, "flush_video_logs"):
        model_logger.flush_video_logs()
    _safe_end_training()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
    autoresume_check_interval = max(1, int(getattr(args, "autoresume_check_interval", 1) if args else 1))

    termination_controller = _SignalTerminationController()
    termination_controller.install()

    def maybe_autoresume_exit(step_for_check, reason):
        should_check = (step_for_check % autoresume_check_interval == 0)
        if not should_check:
            return False
        if not termination_controller.termination_requested_all_ranks(accelerator.device):
            return False

        accelerator.print(
            f"Termination requested ({reason}); exiting precache cleanly at local step {step_for_check}."
        )
        if accelerator.is_main_process:
            _request_autoresume(printer=accelerator.print)
        accelerator.wait_for_everyone()
        return True

    # ── Save reproducibility code snapshot (main process only) ───────────────
    if accelerator.is_main_process:
        _save_git_tracked_snapshot(
            model_logger.output_path,
            args=args,
            printer=accelerator.print,
        )
    accelerator.wait_for_everyone()
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)

    manifest_path = os.path.join(
        model_logger.output_path,
        f"cache_manifest_rank{accelerator.process_index}.jsonl",
    )
    existing_manifest_paths = set()
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                for line in manifest_file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    path_value = payload.get("path")
                    if path_value:
                        existing_manifest_paths.add(str(path_value))
        except Exception as e:
            accelerator.print(
                f"Warning: failed to parse existing manifest '{manifest_path}': {e}. "
                "Will continue and append new entries."
            )

    def append_manifest_if_needed(rel_path, sample_split):
        if rel_path in existing_manifest_paths:
            return False
        cache_entry = {
            "path": rel_path,
            "split": sample_split,
        }
        with open(manifest_path, "a", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(cache_entry) + "\n")
        existing_manifest_paths.add(rel_path)
        return True

    resume_step_offset, fast_forward_enabled = _detect_precache_resume_prefix(
        model_logger.output_path,
        accelerator.process_index,
        accelerator.num_processes,
        existing_manifest_paths=existing_manifest_paths,
        printer=print,
    )

    dataloader_for_loop = dataloader
    progress_total = None
    if fast_forward_enabled and resume_step_offset > 0:
        dataloader_for_loop = skip_first_batches(dataloader, resume_step_offset)
        progress_total = len(dataloader)
        print(
            f"[Precache autoresume] rank {accelerator.process_index}: fast-forwarding "
            f"prepared dataloader by {resume_step_offset} local steps."
        )

    new_cache_count = 0
    skipped_existing_count = 0
    backfilled_manifest_count = 0
    compacted_sample_count = 0
    removed_images_count = 0
    removed_image_sequence_count = 0
    removed_dict_field_count = 0
    removed_sequence_item_count = 0

    progress_bar = tqdm(
        dataloader_for_loop,
        initial=resume_step_offset,
        total=progress_total,
    )
    for data_id_local, data in enumerate(progress_bar):
        data_id = data_id_local + resume_step_offset
        local_step = data_id + 1
        if maybe_autoresume_exit(local_step, "before processing cached sample"):
            return

        with accelerator.accumulate(model):
            with torch.no_grad():
                sample_split = "train"
                if isinstance(data, dict):
                    split_value = data.get("split", "train")
                    sample_split = str(split_value).strip().lower()

                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                cache_rel_path = os.path.relpath(save_path, model_logger.output_path)

                if os.path.exists(save_path):
                    skipped_existing_count += 1
                    if append_manifest_if_needed(cache_rel_path, sample_split):
                        backfilled_manifest_count += 1
                else:
                    processed = model(data)
                    processed, compaction_stats = _compact_precache_payload(processed)
                    removed_images = int(compaction_stats["removed_images"])
                    removed_image_sequences = int(compaction_stats["removed_image_sequences"])
                    removed_dict_fields = int(compaction_stats["removed_dict_fields"])
                    removed_sequence_items = int(compaction_stats["removed_sequence_items"])
                    removed_images_count += removed_images
                    removed_image_sequence_count += removed_image_sequences
                    removed_dict_field_count += removed_dict_fields
                    removed_sequence_item_count += removed_sequence_items
                    if (
                        removed_images > 0
                        or removed_image_sequences > 0
                        or removed_dict_fields > 0
                        or removed_sequence_items > 0
                    ):
                        compacted_sample_count += 1
                    tmp_save_path = f"{save_path}.tmp"
                    torch.save(processed, tmp_save_path)
                    os.replace(tmp_save_path, save_path)
                    append_manifest_if_needed(cache_rel_path, sample_split)
                    new_cache_count += 1

        if maybe_autoresume_exit(local_step, "after processing cached sample"):
            return

    counter_tensor = torch.tensor(
        [
            new_cache_count,
            skipped_existing_count,
            backfilled_manifest_count,
            compacted_sample_count,
            removed_images_count,
            removed_image_sequence_count,
            removed_dict_field_count,
            removed_sequence_item_count,
        ],
        device=accelerator.device,
        dtype=torch.int64,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(counter_tensor, op=torch.distributed.ReduceOp.SUM)

    (
        total_new_cache_count,
        total_skipped_existing_count,
        total_backfilled_manifest_count,
        total_compacted_sample_count,
        total_removed_images_count,
        total_removed_image_sequence_count,
        total_removed_dict_field_count,
        total_removed_sequence_item_count,
    ) = [int(value) for value in counter_tensor.detach().cpu().tolist()]

    accelerator.print(
        "Precache complete on main process. "
        f"new={total_new_cache_count}, skipped_existing={total_skipped_existing_count}, "
        f"manifest_backfilled={total_backfilled_manifest_count}, "
        f"compacted_samples={total_compacted_sample_count}, "
        f"removed_images={total_removed_images_count}, "
        f"removed_image_sequences={total_removed_image_sequence_count}, "
        f"removed_dict_fields={total_removed_dict_field_count}, "
        f"removed_sequence_items={total_removed_sequence_item_count}"
    )
