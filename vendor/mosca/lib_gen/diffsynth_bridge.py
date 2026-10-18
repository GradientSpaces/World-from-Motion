from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import torch

DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS = (
    "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,"
    "Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,"
    "Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth"
)


@dataclass(frozen=True)
class DiffSynthCheckpointConfig:
    diffsynth_root: Path
    checkpoint: Path
    model_paths: str | None = None
    model_id_with_origin_paths: str = DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS
    tokenizer_path: str | None = None
    audio_processor_path: str | None = None
    trainable_models: str = "vace"
    remove_cached_keys: str = ""
    fp8_models: str | None = None
    offload_models: str | None = None
    split_cam_embed: bool = False
    scene_coord_pe: bool = False
    scene_coord_xyzt_pe: bool = False
    scene_coord_gated_residual: bool = False
    num_inference_steps: int = 20
    use_scene_flow: bool = False
    device: str | None = None


@dataclass
class LoadedDiffSynthModel:
    model: Any
    pipe: Any
    device: torch.device


def resolve_torch_device(device_name: str | None) -> torch.device:
    if device_name is None or str(device_name).strip() == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(str(device_name))


@contextmanager
def prepend_sys_path(path: Path) -> Iterator[None]:
    text = str(path.expanduser().resolve())
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:  # pragma: no cover - defensive
            pass


def load_python_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_wan_train_module(diffsynth_root: Path) -> ModuleType:
    train_py = diffsynth_root / "examples" / "wanvideo" / "model_training" / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"Missing WAN training module: {train_py}")
    return load_python_module("mosca_lib_gen_diffsynth_train", train_py)


def build_extra_inputs(use_scene_flow: bool) -> str:
    keys = [
        "source_video",
        "cameras_npz",
        "target_cam",
        "source_cam",
        "vace_video",
        "vace_depth",
        "vace_normal",
        "vace_alpha",
        "source_vace_video",
        "source_vace_depth",
        "source_vace_normal",
        "source_vace_alpha",
    ]
    if use_scene_flow:
        keys.extend(["scene_flow_target", "scene_flow_source"])
    return ",".join(keys)


def load_trained_checkpoint(pipe: Any, checkpoint: Path, *, verbose: bool = True) -> tuple[int, int]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    from diffsynth.core.loader.file import load_state_dict  # type: ignore

    ckpt = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16)
    pipe_state: dict[str, Any] = {}
    for key, value in ckpt.items():
        if key.startswith("pipe."):
            pipe_state[key[len("pipe.") :]] = value
        else:
            pipe_state["vace." + key] = value
    model_state = pipe.state_dict()
    unexpected = []
    loaded = 0
    with torch.no_grad():
        for key, value in pipe_state.items():
            if key not in model_state:
                unexpected.append(key)
                continue
            target = model_state[key]
            if value.device != target.device or value.dtype != target.dtype:
                value = value.to(device=target.device, dtype=target.dtype)
            target.copy_(value)
            loaded += 1
    if verbose:
        print(f"[Checkpoint] Loaded {loaded} params from {checkpoint}")
        if unexpected:
            print(
                f"[Checkpoint] {len(unexpected)} unexpected keys "
                f"(first 5): {unexpected[:5]}"
            )
    return loaded, len(unexpected)


def load_diffsynth_model(
    config: DiffSynthCheckpointConfig,
    *,
    verbose: bool = True,
) -> LoadedDiffSynthModel:
    diffsynth_root = config.diffsynth_root.expanduser().resolve()
    device = resolve_torch_device(config.device)
    if not diffsynth_root.is_dir():
        raise FileNotFoundError(f"Missing diffsynth root: {diffsynth_root}")

    with prepend_sys_path(diffsynth_root):
        wan_train = load_wan_train_module(diffsynth_root)
        model = wan_train.WanTrainingModule(
            model_paths=config.model_paths,
            model_id_with_origin_paths=config.model_id_with_origin_paths,
            tokenizer_path=config.tokenizer_path,
            audio_processor_path=config.audio_processor_path,
            trainable_models=config.trainable_models,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            extra_inputs=build_extra_inputs(config.use_scene_flow),
            remove_cached_keys=config.remove_cached_keys,
            fp8_models=config.fp8_models,
            offload_models=config.offload_models,
            device=device,
            task="sft",
            split_cam_embed=config.split_cam_embed,
            scene_coord_pe=config.scene_coord_pe,
            scene_coord_xyzt_pe=config.scene_coord_xyzt_pe,
            scene_coord_gated_residual=config.scene_coord_gated_residual,
            eval_num_inference_steps=config.num_inference_steps,
        )
        load_trained_checkpoint(model.pipe, config.checkpoint, verbose=verbose)
        model.eval()
        pipe = model.pipe
        if hasattr(pipe, "_all_units"):
            pipe.units = pipe._all_units
        return LoadedDiffSynthModel(model=model, pipe=pipe, device=device)
