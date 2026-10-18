import os, json, math, shutil, torch, imageio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from accelerate import Accelerator


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _atomic_torch_save(path, payload):
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter: Callable[[dict], dict] = lambda x: x,
        save_eval_videos_locally: bool = False,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter: Callable[[dict], dict] = state_dict_converter
        self.save_eval_videos_locally = bool(save_eval_videos_locally)
        self.num_steps = 0
        self._video_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_video_futures = []
        self._step_weight_retention_state = None


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None,
                    loss=None, optimizer=None, global_step=None, epoch=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

        # Log training metrics via accelerator trackers (wandb / tensorboard)
        if global_step is not None:
            metrics = {}
            if loss is not None:
                metrics["train/loss"] = loss.item() if hasattr(loss, 'item') else loss
            if optimizer is not None:
                metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                if len(optimizer.param_groups) > 1:
                    for group_idx, group in enumerate(optimizer.param_groups):
                        group_name = group.get("name", f"group{group_idx}")
                        metrics[f"train/lr_{group_name}"] = group["lr"]
            if metrics:
                accelerator.log(metrics, step=global_step)


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


    def save_states(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        epoch,
        global_step,
        step_in_epoch=None,
    ):
        """Save accelerator state (optimizer, scheduler, model), training_state.json, and model weights.

        The accelerator state is saved to ``<output_path>/checkpoint/`` so that training can be
        resumed later.  A lightweight ``training_state.json`` is written alongside it with the
        current epoch and global step.  Finally the trainable model weights are exported as a
        standalone ``.safetensors`` file.
        """
        state_path = os.path.join(self.output_path, "checkpoint")
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(state_path, exist_ok=True)

        # Save full accelerator state (optimizer, scheduler, model, RNG states)
        accelerator.save_state(output_dir=state_path)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Save training-state metadata for easy resume
            training_state = {
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_path": state_path,
            }
            if step_in_epoch is not None:
                training_state["step_in_epoch"] = step_in_epoch
            _atomic_write_json(os.path.join(self.output_path, "training_state.json"), training_state)

            # Save model weights as safetensors
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            weight_path = os.path.join(self.output_path, f"step-{global_step}.safetensors")
            accelerator.save(state_dict, weight_path, safe_serialization=True)
            self._register_latest_step_weight(global_step)

    def _step_weight_retention_path(self):
        return os.path.join(self.output_path, "step_weight_retention.json")

    @staticmethod
    def _step_weight_file_name(global_step):
        return f"step-{int(global_step)}.safetensors"

    def _step_weight_file_path(self, file_name):
        if not file_name:
            return None
        return os.path.join(self.output_path, file_name)

    @staticmethod
    def _parse_step_weight_step(file_name):
        if not isinstance(file_name, str):
            return None
        if not file_name.startswith("step-") or not file_name.endswith(".safetensors"):
            return None
        step_text = file_name[len("step-"):-len(".safetensors")]
        if not step_text.isdigit():
            return None
        return int(step_text)

    def _list_step_weight_files(self):
        if not os.path.isdir(self.output_path):
            return []
        step_files = [
            file_name
            for file_name in os.listdir(self.output_path)
            if self._parse_step_weight_step(file_name) is not None
        ]
        step_files.sort(key=lambda file_name: int(self._parse_step_weight_step(file_name) or -1))
        return step_files

    def _normalize_step_weight_retention_state(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        state: dict[str, int | float | str | None] = {
            "latest_step": None,
            "latest_file_name": None,
            "best_masked_mpsnr": None,
            "best_step": None,
            "best_file_name": None,
        }

        latest_file_name = payload.get("latest_file_name")
        latest_step = self._parse_step_weight_step(latest_file_name)
        if latest_step is None or latest_file_name is None:
            latest_file_name = None
        else:
            latest_file_path = os.path.join(self.output_path, latest_file_name)
            if not os.path.isfile(latest_file_path):
                latest_step = None
                latest_file_name = None
        if latest_file_name is None:
            latest_step = None
        state["latest_step"] = latest_step
        state["latest_file_name"] = latest_file_name

        best_file_name = payload.get("best_file_name")
        best_step = self._parse_step_weight_step(best_file_name)
        if best_step is None or best_file_name is None:
            best_file_name = None
        else:
            best_file_path = os.path.join(self.output_path, best_file_name)
            if not os.path.isfile(best_file_path):
                best_step = None
                best_file_name = None
        if best_file_name is None:
            best_step = None
        state["best_step"] = best_step
        state["best_file_name"] = best_file_name

        best_masked_mpsnr = payload.get("best_masked_mpsnr")
        try:
            best_masked_mpsnr = (
                None if best_masked_mpsnr is None else float(best_masked_mpsnr)
            )
        except (TypeError, ValueError):
            best_masked_mpsnr = None
        if best_masked_mpsnr is not None and not math.isfinite(best_masked_mpsnr):
            best_masked_mpsnr = None
        if state["best_file_name"] is None:
            best_masked_mpsnr = None
        state["best_masked_mpsnr"] = best_masked_mpsnr

        if state["latest_file_name"] is None and state["best_file_name"] is None:
            existing_step_files = self._list_step_weight_files()
            if existing_step_files:
                latest_file_name = existing_step_files[-1]
                latest_step = self._parse_step_weight_step(latest_file_name)
                state["latest_step"] = latest_step
                state["latest_file_name"] = latest_file_name
                # Preserve the most recent pre-existing weight until DyCheck
                # establishes a real best metric for the run.
                state["best_step"] = latest_step
                state["best_file_name"] = latest_file_name
                state["best_masked_mpsnr"] = None

        return state

    def _load_step_weight_retention_state(self):
        if self._step_weight_retention_state is not None:
            return dict(self._step_weight_retention_state)

        payload: dict[str, Any] = {}
        state_path = self._step_weight_retention_path()
        if os.path.isfile(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                payload = {}

        self._step_weight_retention_state = self._normalize_step_weight_retention_state(
            payload
        )
        return dict(self._step_weight_retention_state)

    def _save_step_weight_retention_state(self, state):
        normalized_state = self._normalize_step_weight_retention_state(state)
        os.makedirs(self.output_path, exist_ok=True)
        _atomic_write_json(self._step_weight_retention_path(), normalized_state)
        self._step_weight_retention_state = normalized_state
        return dict(normalized_state)

    def _prune_step_weight_files(self, keep_file_names):
        keep_file_names = {file_name for file_name in keep_file_names if file_name}
        for file_name in self._list_step_weight_files():
            if file_name in keep_file_names:
                continue
            file_path = self._step_weight_file_path(file_name)
            if file_path is None:
                continue
            try:
                os.remove(file_path)
            except FileNotFoundError:
                continue

    def _register_latest_step_weight(self, global_step):
        global_step = int(global_step)
        file_name = self._step_weight_file_name(global_step)
        state = self._load_step_weight_retention_state()
        state["latest_step"] = global_step
        state["latest_file_name"] = file_name
        state = self._save_step_weight_retention_state(state)
        self._prune_step_weight_files(
            {state.get("latest_file_name"), state.get("best_file_name")}
        )
        return dict(state)

    def maybe_promote_best_step_weight(self, global_step, masked_mpsnr):
        file_name = self._step_weight_file_name(global_step)
        file_path = self._step_weight_file_path(file_name)
        if file_path is None or not os.path.isfile(file_path):
            return {
                "promoted": False,
                "best_step": None,
                "best_masked_mpsnr": None,
                "best_file_name": None,
                "reason": "missing_weight_file",
            }

        try:
            masked_mpsnr = float(masked_mpsnr)
        except (TypeError, ValueError):
            return {
                "promoted": False,
                "best_step": None,
                "best_masked_mpsnr": None,
                "best_file_name": None,
                "reason": "invalid_metric",
            }
        if not math.isfinite(masked_mpsnr):
            return {
                "promoted": False,
                "best_step": None,
                "best_masked_mpsnr": None,
                "best_file_name": None,
                "reason": "invalid_metric",
            }

        state = self._load_step_weight_retention_state()
        current_best = state.get("best_masked_mpsnr")
        if current_best is not None and masked_mpsnr <= current_best:
            return {
                "promoted": False,
                "best_step": state.get("best_step"),
                "best_masked_mpsnr": current_best,
                "best_file_name": state.get("best_file_name"),
                "reason": "not_improved",
            }

        state["best_step"] = int(global_step)
        state["best_file_name"] = file_name
        state["best_masked_mpsnr"] = masked_mpsnr
        state = self._save_step_weight_retention_state(state)
        self._prune_step_weight_files(
            {state.get("latest_file_name"), state.get("best_file_name")}
        )
        return {
            "promoted": True,
            "best_step": state.get("best_step"),
            "best_masked_mpsnr": state.get("best_masked_mpsnr"),
            "best_file_name": state.get("best_file_name"),
            "reason": "improved",
        }

    @staticmethod
    def _video_tensor_to_uint8_permuted(video_tensor):
        video_f32 = (video_tensor.detach().cpu().to(torch.float32).clamp(-1.0, 1.0) + 1.0) / 2.0
        video_uint8 = (video_f32.clamp(0.0, 1.0).permute(0, 2, 1, 3, 4) * 255.0).round().to(torch.uint8)
        return video_uint8

    def _get_wandb_run(self, accelerator: Accelerator):
        wandb_run = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            if wandb_tracker:
                wandb_run = getattr(wandb_tracker, "tracker", None)
        except Exception:
            wandb_run = None
        return wandb_run

    def _staged_eval_video_dir(self, global_step, namespace=None):
        if namespace is None:
            return os.path.join(self.output_path, ".staged_eval_videos", f"step-{int(global_step)}")
        namespace = self._sanitize_path_token(namespace)
        return os.path.join(self.output_path, ".staged_eval_videos", namespace, f"step-{int(global_step)}")

    def _local_eval_video_dir(self, global_step, namespace=None):
        if namespace is None:
            return os.path.join(self.output_path, "eval_videos", f"step-{int(global_step)}")
        namespace = self._sanitize_path_token(namespace)
        return os.path.join(self.output_path, "eval_videos", namespace, f"step-{int(global_step)}")

    @staticmethod
    def _sanitize_path_token(text, default="unknown"):
        text = str(text or "").strip()
        if not text:
            return default
        sanitized = []
        for ch in text:
            if ch.isalnum() or ch in ("-", "_", "."):
                sanitized.append(ch)
            else:
                sanitized.append("_")
        result = "".join(sanitized).strip("._")
        return result or default

    @staticmethod
    def _iter_video_batches(video_uint8_permuted):
        video_uint8_permuted = video_uint8_permuted.detach().cpu()
        if video_uint8_permuted.ndim == 4:
            yield 0, video_uint8_permuted
            return
        if video_uint8_permuted.ndim != 5:
            raise ValueError(
                f"Expected video tensor with 4 or 5 dims, got shape {tuple(video_uint8_permuted.shape)}"
            )
        for batch_idx in range(video_uint8_permuted.shape[0]):
            yield batch_idx, video_uint8_permuted[batch_idx]

    def _build_local_eval_video_paths(self, payload):
        video_uint8 = payload["video_uint8"]
        global_step = int(payload.get("global_step", 0))
        sample_idx = int(payload.get("sample_idx", 0))
        rank = int(payload.get("rank", 0))
        dataset_name = self._sanitize_path_token(payload.get("dataset_name", "unknown"))
        namespace = payload.get("namespace")
        if namespace is not None:
            namespace = self._sanitize_path_token(namespace)
        relpath = self._sanitize_relpath(payload.get("local_video_relpath"))

        if relpath:
            rel_dir, rel_file = os.path.split(relpath)
            file_stem, file_ext = os.path.splitext(rel_file)
            output_dir = (
                os.path.join(self.output_path, rel_dir)
                if rel_dir
                else self.output_path
            )
            base_name = file_stem or f"sample-{sample_idx:06d}-rank-{rank:06d}"
            ext = file_ext or ".mp4"
        else:
            output_dir = os.path.join(self._local_eval_video_dir(global_step, namespace=namespace), dataset_name)
            base_name = f"sample-{sample_idx:06d}-rank-{rank:06d}"
            ext = ".mp4"

        if isinstance(video_uint8, torch.Tensor) and video_uint8.ndim == 5:
            batch_size = int(video_uint8.shape[0])
        else:
            batch_size = 1

        paths = []
        for batch_idx in range(batch_size):
            batch_suffix = "" if batch_size == 1 else f"-batch-{batch_idx:02d}"
            file_name = f"{base_name}{batch_suffix}{ext}"
            paths.append(os.path.join(output_dir, file_name))
        return paths

    @classmethod
    def _sanitize_relpath(cls, relpath):
        relpath = str(relpath or "").strip().replace("\\", "/")
        if not relpath:
            return None
        parts = []
        for part in relpath.split("/"):
            part = part.strip()
            if not part or part in (".", ".."):
                continue
            parts.append(cls._sanitize_path_token(part))
        if not parts:
            return None
        return os.path.join(*parts)

    @staticmethod
    def _write_local_video_mp4(video_uint8_permuted, save_path, fps=18):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = imageio.get_writer(save_path, fps=fps, quality=9)
        try:
            for frame in video_uint8_permuted:
                writer.append_data(frame.permute(1, 2, 0).cpu().numpy())
        finally:
            writer.close()

    @staticmethod
    def _write_video_bg(video_uint8_permuted, name, global_step, wandb_run):
        if wandb_run is None:
            return

        try:
            import wandb
            wandb_video = wandb.Video(video_uint8_permuted.cpu().numpy(), fps=18, format="mp4")
            wandb_run.log({name: wandb_video}, step=global_step)
        except Exception:
            pass

    @staticmethod
    def _write_video_file_bg(video_path, name, global_step, wandb_run):
        if wandb_run is None:
            return
        try:
            import wandb
            wandb_video = wandb.Video(str(video_path), format="mp4")
            wandb_run.log({name: wandb_video}, step=global_step)
        except Exception:
            pass

    @classmethod
    def _write_eval_video_artifacts_bg(
        cls,
        video_uint8_permuted,
        local_video_uint8_permuted,
        name,
        global_step,
        wandb_run,
        local_paths=None,
    ):
        try:
            if local_paths:
                local_video_source = (
                    local_video_uint8_permuted
                    if local_video_uint8_permuted is not None
                    else video_uint8_permuted
                )
                batches = list(cls._iter_video_batches(local_video_source))
                for (_, batch_video), local_path in zip(batches, local_paths):
                    cls._write_local_video_mp4(batch_video, local_path, fps=18)
        except Exception:
            pass
        cls._write_video_bg(video_uint8_permuted, name, global_step, wandb_run)


    def log_video(self, accelerator: Accelerator, video_tensor, name, global_step):
        """Queue video logging to the WandB tracker.

        Args:
            accelerator: The HuggingFace Accelerator instance.
            video_tensor: Tensor of shape ``[B, C, T, H, W]`` in range ``[-1, 1]``.
            name: Name / tag for the video log entry.
            global_step: Current training step.
        """
        video_uint8_permuted = self._video_tensor_to_uint8_permuted(video_tensor)
        wandb_run = self._get_wandb_run(accelerator)

        future = self._video_executor.submit(
            self._write_video_bg,
            video_uint8_permuted,
            name,
            global_step,
            wandb_run,
        )
        self._pending_video_futures.append(future)

    def log_video_file(self, accelerator: Accelerator, video_path, name, global_step):
        wandb_run = self._get_wandb_run(accelerator)
        future = self._video_executor.submit(
            self._write_video_file_bg,
            str(video_path),
            name,
            global_step,
            wandb_run,
        )
        self._pending_video_futures.append(future)

    def begin_eval_video_staging(self, accelerator: Accelerator, global_step, namespace=None):
        stage_dir = self._staged_eval_video_dir(global_step, namespace=namespace)
        if accelerator.is_main_process and os.path.isdir(stage_dir):
            shutil.rmtree(stage_dir, ignore_errors=True)
        accelerator.wait_for_everyone()
        os.makedirs(stage_dir, exist_ok=True)
        accelerator.wait_for_everyone()

    def stage_eval_video(
        self,
        video_tensor,
        global_step,
        sample_idx,
        rank,
        dataset_name,
        local_video_relpath=None,
        local_video_tensor=None,
        namespace=None,
        wandb_name=None,
    ):
        stage_dir = self._staged_eval_video_dir(global_step, namespace=namespace)
        os.makedirs(stage_dir, exist_ok=True)
        payload = {
            "dataset_name": str(dataset_name or "unknown"),
            "global_step": int(global_step),
            "name": str(wandb_name or f"video/{int(sample_idx)}_r{int(rank)}"),
            "rank": int(rank),
            "sample_idx": int(sample_idx),
            "video_uint8": self._video_tensor_to_uint8_permuted(video_tensor),
        }
        if namespace is not None:
            payload["namespace"] = str(namespace)
        if local_video_relpath:
            payload["local_video_relpath"] = str(local_video_relpath)
        if local_video_tensor is not None:
            payload["local_video_uint8"] = self._video_tensor_to_uint8_permuted(
                local_video_tensor
            )
        file_name = f"sample-{int(sample_idx):06d}-rank-{int(rank):06d}.pt"
        _atomic_torch_save(os.path.join(stage_dir, file_name), payload)

    def flush_staged_eval_videos(self, accelerator: Accelerator, global_step, max_logs=None, namespace=None):
        accelerator.wait_for_everyone()
        logged = 0
        stage_dir = self._staged_eval_video_dir(global_step, namespace=namespace)
        if accelerator.is_main_process and os.path.isdir(stage_dir):
            staged_files = sorted(
                os.path.join(stage_dir, name)
                for name in os.listdir(stage_dir)
                if name.endswith(".pt")
            )
            wandb_run = self._get_wandb_run(accelerator)
            for staged_path in staged_files:
                if max_logs is not None and max_logs >= 0 and logged >= max_logs:
                    break
                payload = torch.load(staged_path, map_location="cpu")
                local_paths = (
                    self._build_local_eval_video_paths(payload)
                    if self.save_eval_videos_locally
                    else None
                )
                future = self._video_executor.submit(
                    self._write_eval_video_artifacts_bg,
                    payload["video_uint8"],
                    payload.get("local_video_uint8"),
                    payload["name"],
                    int(payload.get("global_step", global_step)),
                    wandb_run,
                    local_paths,
                )
                self._pending_video_futures.append(future)
                logged += 1
            # W&B rejects logs that arrive for an older step after training has
            # already advanced. Drain staged eval video jobs here so the next
            # optimizer step cannot overtake visualization uploads.
            self.flush_video_logs()
            shutil.rmtree(stage_dir, ignore_errors=True)
        accelerator.wait_for_everyone()
        return logged


    def flush_video_logs(self):
        """Wait for all queued video logging jobs to finish."""
        for future in self._pending_video_futures:
            future.result()
        self._pending_video_futures.clear()
