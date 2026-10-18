import torch, os, argparse, accelerate, re, imageio, shutil
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from diffsynth.core import UnifiedDataset
from diffsynth.core.loader.file import load_state_dict
from diffsynth.core.data.operators import LoadTorchPickle, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath, ToStr
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    ModelConfig,
    WanVideoUnit_SceneCoordLoader,
    _normalize_target_gbuffer_dropout_scope,
    _normalize_target_gbuffer_uncond_fill_mode,
    _apply_target_gbuffer_uncond_fill,
)
from diffsynth.diffusion import *
from diffsynth.metrics import video_msssim, video_psnr, video_psnr_masked
from diffsynth.models.wan_video_dit import SceneCoordEncoder
from diffsynth.models.wan_video_vace import _OPTIONAL_STATE_DICT_KEYS as _VACE_OPTIONAL_STATE_DICT_KEYS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from examples.wanvideo.model_training import dycheck_benchmark as dycheck_benchmark_lib
except ModuleNotFoundError:
    import dycheck_benchmark as dycheck_benchmark_lib


_SCENE_COORD_WORKER_CACHE_MAXSIZE = 2
_SCENE_COORD_WORKER_NUM_WORKERS = 1
_SCENE_COORD_WORKER_PREFETCH_FACTOR = 1
_COMPACT_CACHED_VACE_LATENT_CHANNELS_PER_SIGNAL = 16
_COMPACT_CACHED_VACE_GROUPS = ("video", "depth", "normal", "alpha", "scene_flow")
# Older joint caches predate cached scene flow and therefore only store the
# first four compact groups (64 channels). Newer caches may include the
# appended scene_flow group for 80 channels.
_COMPACT_CACHED_VACE_GROUP_LAYOUTS = (
    _COMPACT_CACHED_VACE_GROUPS[:-1],
    _COMPACT_CACHED_VACE_GROUPS,
)
_COMPACT_CACHED_VACE_GROUP_LAYOUTS_BY_CHANNELS = {
    len(group_layout) * _COMPACT_CACHED_VACE_LATENT_CHANNELS_PER_SIGNAL: group_layout
    for group_layout in _COMPACT_CACHED_VACE_GROUP_LAYOUTS
}
_COMPACT_CACHED_VACE_GROUP_TO_INPUT_KEYS = {
    "video": ("vace_video", "source_vace_video"),
    "depth": ("vace_depth", "source_vace_depth"),
    "normal": ("vace_normal", "source_vace_normal"),
    "alpha": ("vace_alpha", "source_vace_alpha"),
    "scene_flow": ("scene_flow_target", "scene_flow_source"),
}


def _cached_to_python_scalar(value):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def _cached_to_int(value):
    value = _cached_to_python_scalar(value)
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = _cached_to_python_scalar(value[0])
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for integer conversion, got shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError(f"Cannot convert value to int: {type(value).__name__}")


def _cached_to_str(value):
    value = _cached_to_python_scalar(value)
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = _cached_to_python_scalar(value[0])
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return None
    return str(value)


def _parse_compact_cached_vace_groups(groups, arg_name: str = "drop_cached_vace_groups"):
    if groups is None:
        return tuple()
    if isinstance(groups, str):
        raw_groups = [group.strip().lower() for group in groups.split(",") if group.strip()]
    else:
        raw_groups = [
            str(group).strip().lower()
            for group in groups
            if str(group).strip()
        ]

    deduped_groups = []
    seen = set()
    invalid_groups = []
    for group in raw_groups:
        if group not in _COMPACT_CACHED_VACE_GROUPS:
            invalid_groups.append(group)
            continue
        if group in seen:
            continue
        deduped_groups.append(group)
        seen.add(group)
    if invalid_groups:
        raise ValueError(
            f"Unsupported {arg_name} values: {invalid_groups}. "
            f"Supported groups: {', '.join(_COMPACT_CACHED_VACE_GROUPS)}."
        )
    return tuple(deduped_groups)


@torch.no_grad()
def video_ssim(video1: torch.Tensor, video2: torch.Tensor) -> torch.Tensor:
    """Compute full-frame SSIM across all frames in a video batch."""
    assert video1.shape == video2.shape, "Input videos must have the same shape"
    b, c, t, h, w = video1.shape
    video1_reshaped = video1.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    video2_reshaped = video2.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(video1.device)
    return ssim_metric(video1_reshaped, video2_reshaped)


def build_eval_lpips_model(device, lpips_net="alex"):
    try:
        import lpips  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "The 'lpips' package is required to compute evaluation LPIPS metrics."
        ) from exc

    model = lpips.LPIPS(net=lpips_net).to(device)
    model.eval()
    return model


@torch.no_grad()
def video_lpips(video1: torch.Tensor, video2: torch.Tensor, lpips_model) -> torch.Tensor:
    """Compute full-frame LPIPS averaged across all frames in a video batch."""
    assert video1.shape == video2.shape, "Input videos must have the same shape"
    b, c, t, h, w = video1.shape
    video1_reshaped = video1.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    video2_reshaped = video2.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    scores = lpips_model(video1_reshaped * 2.0 - 1.0, video2_reshaped * 2.0 - 1.0)
    return scores.mean()


def _infer_cached_video_shape(inputs_shared, input_latents):
    height = inputs_shared.get("height", input_latents.shape[3] * 8)
    width = inputs_shared.get("width", input_latents.shape[4] * 8)
    num_frames = inputs_shared.get("num_frames", (input_latents.shape[2] - 1) * 4 + 1)
    return _cached_to_int(height), _cached_to_int(width), _cached_to_int(num_frames)


def _resolve_scene_coord_cached_job(cameras_npz, target_cam, source_cam):
    cameras_npz = _cached_to_str(cameras_npz)
    target_cam = _cached_to_str(target_cam)
    source_cam = _cached_to_str(source_cam)
    if cameras_npz is None or target_cam is None or source_cam is None:
        raise ValueError(
            "Scene coordinate loading requires cameras_npz, target_cam, and source_cam in cached inputs."
        )
    vid_match = re.search(r"(vid\d+)", os.path.basename(cameras_npz))
    vid_tag = vid_match.group(1) if vid_match else None
    return os.path.dirname(cameras_npz), target_cam, source_cam, vid_tag


def _load_scene_coord_pair_from_job(
    loader: WanVideoUnit_SceneCoordLoader,
    scene_root_dir: str,
    target_cam: str,
    source_cam: str,
    height: int,
    width: int,
    num_frames: int,
    vid_tag: str | None = None,
):
    coords_target = loader._load_single(
        scene_root_dir, target_cam, height, width, num_frames, vid_tag=vid_tag
    )
    coords_source = loader._load_single(
        scene_root_dir, source_cam, height, width, num_frames, vid_tag=vid_tag
    )
    coords_target, coords_source = loader._joint_normalize_pair(coords_target, coords_source)
    if loader.xyzt:
        coords_target = loader._append_temporal_channel(coords_target)
        coords_source = loader._append_temporal_channel(coords_source)
    return coords_target, coords_source


class LoadCachedWanTuple(LoadTorchPickle):
    """Rewrite cached WAN tuples inside DataLoader workers before training."""

    def __init__(
        self,
        load_scene_coords: bool = False,
        scene_coord_xyzt: bool = False,
        remove_cached_keys=None,
        drop_cached_vace_groups=None,
        map_location: str = "cpu",
        scene_coord_cache_maxsize: int = _SCENE_COORD_WORKER_CACHE_MAXSIZE,
    ):
        super().__init__(map_location=map_location)
        self.load_scene_coords = bool(load_scene_coords)
        self.scene_coord_xyzt = bool(scene_coord_xyzt)
        if isinstance(remove_cached_keys, str):
            remove_cached_keys = [k.strip() for k in remove_cached_keys.split(",") if k.strip()]
        self.remove_cached_keys = tuple(remove_cached_keys or ())
        self.drop_cached_vace_groups = _parse_compact_cached_vace_groups(drop_cached_vace_groups)
        self.scene_coord_cache_maxsize = max(0, int(scene_coord_cache_maxsize))
        self._scene_coord_loader = None
        self._cached_vace_keep_slices_by_channels = self._build_cached_vace_keep_slices()

    def _get_scene_coord_loader(self):
        if self._scene_coord_loader is None:
            self._scene_coord_loader = WanVideoUnit_SceneCoordLoader(
                xyzt=self.scene_coord_xyzt,
                load_cache_maxsize=self.scene_coord_cache_maxsize,
            )
        return self._scene_coord_loader

    def _build_cached_vace_keep_slices(self):
        if not self.drop_cached_vace_groups:
            return {}
        drop_groups = set(self.drop_cached_vace_groups)
        keep_slices_by_channels = {}
        for channel_count, group_layout in _COMPACT_CACHED_VACE_GROUP_LAYOUTS_BY_CHANNELS.items():
            keep_slices = []
            for index, group in enumerate(group_layout):
                if group in drop_groups:
                    continue
                start = index * _COMPACT_CACHED_VACE_LATENT_CHANNELS_PER_SIGNAL
                stop = start + _COMPACT_CACHED_VACE_LATENT_CHANNELS_PER_SIGNAL
                keep_slices.append(slice(start, stop))
            keep_slices_by_channels[channel_count] = tuple(keep_slices)
        return keep_slices_by_channels

    def _maybe_trim_cached_vace_context(self, inputs_shared):
        if not self.drop_cached_vace_groups:
            return inputs_shared
        vace_context = inputs_shared.get("vace_context")
        if not isinstance(vace_context, torch.Tensor) or vace_context.ndim < 2:
            raise ValueError(
                "Dropping cached VACE groups requires tensor input 'vace_context' "
                f"with at least 2 dimensions, got {type(vace_context).__name__}."
            )
        channel_count = int(vace_context.shape[1])
        keep_slices = self._cached_vace_keep_slices_by_channels.get(channel_count)
        if keep_slices is None:
            supported_channels = ", ".join(
                str(count) for count in sorted(_COMPACT_CACHED_VACE_GROUP_LAYOUTS_BY_CHANNELS)
            )
            raise ValueError(
                "Cached VACE group drop expects compact grouped channels "
                f"({supported_channels}), got vace_context shape {tuple(vace_context.shape)}."
            )
        if not keep_slices:
            group_layout = _COMPACT_CACHED_VACE_GROUP_LAYOUTS_BY_CHANNELS[channel_count]
            raise ValueError(
                "Dropping cached VACE groups would remove every signal from the "
                f"{channel_count}-channel compact layout ({', '.join(group_layout)})."
            )
        trimmed_context = torch.cat(
            [vace_context[:, channel_slice, ...] for channel_slice in keep_slices],
            dim=1,
        )
        inputs_shared["vace_context"] = trimmed_context.contiguous()
        return inputs_shared

    def __call__(self, data):
        loaded = super().__call__(data)
        if not (
            isinstance(loaded, tuple)
            and len(loaded) == 3
            and isinstance(loaded[0], dict)
            and isinstance(loaded[1], dict)
            and isinstance(loaded[2], dict)
        ):
            return loaded

        inputs_shared, inputs_posi, inputs_nega = loaded
        needs_trim = bool(self.drop_cached_vace_groups)
        needs_scene_coords = self.load_scene_coords and (
            inputs_shared.get("scene_coord_target") is None
            or inputs_shared.get("scene_coord_source") is None
        )
        if not self.remove_cached_keys and not needs_trim and not needs_scene_coords:
            return loaded

        inputs_shared = dict(inputs_shared)
        if needs_trim:
            inputs_shared = self._maybe_trim_cached_vace_context(inputs_shared)
        for key in self.remove_cached_keys:
            inputs_shared.pop(key, None)

        if needs_scene_coords:
            input_latents = inputs_shared.get("input_latents")
            has_camera_keys = all(
                key in inputs_shared for key in ("cameras_npz", "target_cam", "source_cam")
            )
            if isinstance(input_latents, torch.Tensor) and has_camera_keys:
                height, width, num_frames = _infer_cached_video_shape(inputs_shared, input_latents)
                scene_root_dir, target_cam, source_cam, vid_tag = _resolve_scene_coord_cached_job(
                    inputs_shared.get("cameras_npz"),
                    inputs_shared.get("target_cam"),
                    inputs_shared.get("source_cam"),
                )
                scene_target, scene_source = _load_scene_coord_pair_from_job(
                    self._get_scene_coord_loader(),
                    scene_root_dir,
                    target_cam,
                    source_cam,
                    height,
                    width,
                    num_frames,
                    vid_tag,
                )
                inputs_shared["scene_coord_target"] = scene_target.unsqueeze(0)
                inputs_shared["scene_coord_source"] = scene_source.unsqueeze(0)

        return (inputs_shared, inputs_posi, inputs_nega)


class LoadCachedWanSceneCoords(LoadCachedWanTuple):
    """Backward-compatible alias for cached WAN tuple rewriting with scene coords."""


class WanTrainingModule(DiffusionTrainingModule):
    _TURBO_R_COEFFS = (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943)
    _TURBO_G_COEFFS = (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604)
    _TURBO_B_COEFFS = (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973)
    _PIL_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else 2
    _PIL_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else 0
    _DIT_PARAM_PREFIXES = ("pipe.dit.", "pipe.dit2.")
    _OTHER_LR_DIT_PARAM_MARKERS = (
        ".cam_encoder.",
        ".projector.",
        ".scene_coord_encoder.",
        ".coord_gate",
    )
    _UNCOND_LATENT_PARAM_SUFFIXES = ("_uncond_vace_context",)
    _VACE_LATENT_CHANNELS_PER_SIGNAL = 16
    _OPTIONAL_RESUME_STATE_DICT_KEYS = frozenset(
        f"pipe.{vace_name}.{key}"
        for vace_name in ("vace", "vace2")
        for key in _VACE_OPTIONAL_STATE_DICT_KEYS
    )
    _SUPPORTED_SHARED_GBUFFER_DROPOUT_GROUPS = ("depth", "normal", "alpha", "scene_flow")
    _VACE_KEY_TO_SHARED_GROUP = {
        "vace_video": "video",
        "source_vace_video": "video",
        "vace_depth": "depth",
        "source_vace_depth": "depth",
        "vace_normal": "normal",
        "source_vace_normal": "normal",
        "vace_alpha": "alpha",
        "source_vace_alpha": "alpha",
        "scene_flow_target": "scene_flow",
        "scene_flow_source": "scene_flow",
    }

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        remove_cached_keys=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        split_cam_embed=False,
        zero_source_cam_emb_when_missing=False,
        strip_source_gbuffer=False,
        target_gbuffer_dropout_prob=0.0,
        target_gbuffer_dropout_scope="target_only",
        target_gbuffer_uncond_fill_mode="zero",
        shared_gbuffer_dropout_prob=0.0,
        shared_gbuffer_rgb_only_prob=0.0,
        shared_gbuffer_dropout_groups="depth,normal,alpha,scene_flow",
        scene_coord_pe=False,
        scene_coord_xyzt_pe=False,
        scene_coord_gated_residual=False,
        scene_coord_gate_init=0.0,
        eval_height=None,
        eval_width=None,
        eval_dataset_resolutions=None,
        eval_num_inference_steps=20,
        eval_target_gbuffer_cfg_scale=1.0,
        eval_metrics_mode="legacy",
        eval_metrics_device="auto",
        eval_metrics_lpips_net="alex",
        dycheck_benchmark_root=None,
        dycheck_benchmark_data_root=None,
        dycheck_benchmark_seq_names=None,
        dycheck_benchmark_chunk_size=73,
    ):
        super().__init__()
        self.scene_coord_xyzt = bool(scene_coord_xyzt_pe)
        self.scene_coord_pe = bool(scene_coord_pe or self.scene_coord_xyzt)
        self.scene_coord_gated_residual = bool(scene_coord_gated_residual)
        self.scene_coord_gate_init = float(scene_coord_gate_init)
        self.scene_coord_loader = WanVideoUnit_SceneCoordLoader(xyzt=self.scene_coord_xyzt) if self.scene_coord_pe else None
        # Warning
        # if not use_gradient_checkpointing:
        #     warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
        #     use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = [key.strip() for key in extra_inputs.split(",") if key.strip()] if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.trainable_models = trainable_models.split(",") if trainable_models is not None else []
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLossWithTargetSlice(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLossWithTargetSlice(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.split_cam_embed = split_cam_embed
        self.zero_source_cam_emb_when_missing = bool(zero_source_cam_emb_when_missing)
        self.strip_source_gbuffer = strip_source_gbuffer
        self.target_gbuffer_dropout_prob = float(target_gbuffer_dropout_prob)
        self.target_gbuffer_dropout_scope = _normalize_target_gbuffer_dropout_scope(
            target_gbuffer_dropout_scope
        )
        self.target_gbuffer_uncond_fill_mode = _normalize_target_gbuffer_uncond_fill_mode(
            target_gbuffer_uncond_fill_mode
        )
        self.shared_gbuffer_dropout_prob = float(shared_gbuffer_dropout_prob)
        self.shared_gbuffer_rgb_only_prob = float(shared_gbuffer_rgb_only_prob)
        if not 0.0 <= self.target_gbuffer_dropout_prob <= 1.0:
            raise ValueError(
                f"target_gbuffer_dropout_prob must be in [0, 1], got {self.target_gbuffer_dropout_prob}."
            )
        if not 0.0 <= self.shared_gbuffer_dropout_prob <= 1.0:
            raise ValueError(
                f"shared_gbuffer_dropout_prob must be in [0, 1], got {self.shared_gbuffer_dropout_prob}."
            )
        if not 0.0 <= self.shared_gbuffer_rgb_only_prob <= 1.0:
            raise ValueError(
                f"shared_gbuffer_rgb_only_prob must be in [0, 1], got {self.shared_gbuffer_rgb_only_prob}."
            )
        if shared_gbuffer_dropout_groups is None:
            parsed_dropout_groups = list(self._SUPPORTED_SHARED_GBUFFER_DROPOUT_GROUPS)
        elif isinstance(shared_gbuffer_dropout_groups, str):
            parsed_dropout_groups = [
                group.strip().lower()
                for group in shared_gbuffer_dropout_groups.split(",")
                if group.strip()
            ]
        else:
            parsed_dropout_groups = [
                str(group).strip().lower()
                for group in shared_gbuffer_dropout_groups
                if str(group).strip()
            ]
        if not parsed_dropout_groups:
            parsed_dropout_groups = list(self._SUPPORTED_SHARED_GBUFFER_DROPOUT_GROUPS)
        deduped_dropout_groups = []
        seen_dropout_groups = set()
        for group in parsed_dropout_groups:
            if group in seen_dropout_groups:
                continue
            deduped_dropout_groups.append(group)
            seen_dropout_groups.add(group)
        invalid_dropout_groups = [
            group for group in deduped_dropout_groups
            if group not in self._SUPPORTED_SHARED_GBUFFER_DROPOUT_GROUPS
        ]
        if invalid_dropout_groups:
            raise ValueError(
                "Unsupported shared_gbuffer_dropout_groups: "
                f"{invalid_dropout_groups}. Supported groups: "
                f"{self._SUPPORTED_SHARED_GBUFFER_DROPOUT_GROUPS}."
            )
        self.shared_gbuffer_dropout_groups = tuple(deduped_dropout_groups)
        self._target_gbuffer_dropout_logged = False
        self._shared_gbuffer_dropout_logged = False
        if (eval_height is None) != (eval_width is None):
            raise ValueError("eval_height and eval_width must be both set or both unset.")
        self.eval_height = eval_height
        self.eval_width = eval_width
        self.eval_num_inference_steps = max(1, int(eval_num_inference_steps))
        self.eval_target_gbuffer_cfg_scale = float(eval_target_gbuffer_cfg_scale)
        if self.eval_target_gbuffer_cfg_scale < 0.0:
            raise ValueError(
                f"eval_target_gbuffer_cfg_scale must be >= 0, got {self.eval_target_gbuffer_cfg_scale}."
            )
        self.eval_metrics_mode = str(eval_metrics_mode or "legacy").strip().lower()
        if self.eval_metrics_mode not in {"legacy", "multicam"}:
            raise ValueError(
                f"Unsupported eval_metrics_mode '{eval_metrics_mode}'. "
                "Expected one of: legacy, multicam."
            )
        self.eval_metrics_device = str(eval_metrics_device or "auto").strip().lower()
        if self.eval_metrics_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"Unsupported eval_metrics_device '{eval_metrics_device}'. "
                "Expected one of: auto, cpu, cuda."
            )
        self.eval_metrics_lpips_net = str(eval_metrics_lpips_net or "alex").strip() or "alex"
        self.eval_dataset_resolutions = {}
        if eval_dataset_resolutions:
            for key, value in eval_dataset_resolutions.items():
                key_norm = str(key).strip().lower()
                if not key_norm:
                    continue
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"Invalid eval resolution override for '{key}': {value}")
                h, w = int(value[0]), int(value[1])
                if h <= 0 or w <= 0:
                    raise ValueError(f"Invalid eval resolution override for '{key}': {(h, w)}")
                self.eval_dataset_resolutions[key_norm] = (h, w)
        self.dycheck_benchmark_root = _normalize_optional_cli_path(dycheck_benchmark_root)
        self.dycheck_benchmark_data_root = _normalize_optional_cli_path(
            dycheck_benchmark_data_root
        )
        self.dycheck_benchmark_seq_names = tuple(
            str(name).strip()
            for name in (dycheck_benchmark_seq_names or [])
            if str(name).strip()
        )
        self.dycheck_benchmark_chunk_size = max(1, int(dycheck_benchmark_chunk_size))
        self.dycheck_benchmark_enabled = self.dycheck_benchmark_root is not None
        self.remove_cached_keys = [k.strip() for k in remove_cached_keys.split(",") if k.strip()] if remove_cached_keys else []
        self.remove_cached_keys_set = set(self.remove_cached_keys)
        # File-level debiasing now creates explicit reversed samples.
        # Keep runtime reversal disabled to avoid pose/video temporal mismatch.
        self.reverse_video_probability = 0.0
        compact_target_vace_key_order = ("vace_video", "vace_depth", "vace_normal", "vace_alpha")
        compact_source_vace_key_order = (
            "source_vace_video",
            "source_vace_depth",
            "source_vace_normal",
            "source_vace_alpha",
        )
        self.base_vace_joint_video_keys = tuple(
            key for key in compact_target_vace_key_order if key in self.extra_inputs
        )
        self.base_source_vace_joint_video_keys = tuple(
            key for key in compact_source_vace_key_order if key in self.extra_inputs
        )
        self.scene_flow_target_key = "scene_flow_target"
        self.scene_flow_source_key = "scene_flow_source"
        self.scene_flow_target_enabled = self.scene_flow_target_key in self.extra_inputs
        self.scene_flow_source_enabled = self.scene_flow_source_key in self.extra_inputs
        self.scene_flow_enabled = (
            self.scene_flow_target_enabled or self.scene_flow_source_enabled
        )
        self.vace_joint_video_keys = self.base_vace_joint_video_keys + (
            (self.scene_flow_target_key,) if self.scene_flow_target_enabled else ()
        )
        self.source_vace_joint_video_keys = self.base_source_vace_joint_video_keys + (
            (self.scene_flow_source_key,) if self.scene_flow_source_enabled else ()
        )
        # Pose-only mode should not require source_video.
        self.camera_pose_keys = ("cameras_npz", "target_cam", "source_cam")
        self.cached_vace_group_channel_slices = self._build_cached_vace_group_channel_slices()
        self.shared_gbuffer_available_dropout_groups = tuple(
            group for group in self.shared_gbuffer_dropout_groups
            if group in self.cached_vace_group_channel_slices
        )

        # Reinitialize VACE patch embedding whenever a compact gbuffer layout is
        # requested so the channel count matches the active conditioning subset
        # (for example 16, 64, or 80 channels instead of the legacy 96-channel
        # path). This is safe when training VACE from scratch.
        compact_vace_reinit_keys = (
            self.vace_joint_video_keys
            if self.vace_joint_video_keys
            else self.source_vace_joint_video_keys
        )
        if (
            self.pipe.vace is not None
            and compact_vace_reinit_keys
            and not task.endswith(":data_process")
        ):
            vace_in_dim = 16 * len(compact_vace_reinit_keys)
            for vace_name in ("vace", "vace2"):
                vace_model = getattr(self.pipe, vace_name, None)
                if vace_model is None or vace_in_dim == vace_model.vace_in_dim:
                    continue
                old_pe = vace_model.vace_patch_embedding
                old_vace_in_dim = int(vace_model.vace_in_dim)
                vace_model.vace_in_dim = vace_in_dim
                vace_model.vace_patch_embedding = torch.nn.Conv3d(
                    vace_in_dim, old_pe.out_channels,
                    kernel_size=old_pe.kernel_size, stride=old_pe.stride,
                    padding=old_pe.padding, dilation=old_pe.dilation, groups=old_pe.groups,
                    bias=old_pe.bias is not None,
                    device=old_pe.weight.device,
                    dtype=old_pe.weight.dtype,
                )
                vace_model.reset_uncond_vace_contexts(vace_in_dim)
                print(
                    "[GBuffer] Reinitialized "
                    f"{vace_name} patch embedding and unconditional latents: "
                    f"{old_vace_in_dim} -> {vace_in_dim} channels"
                )
        if self._camera_conditioning_requested() and "cam_emb" not in self.remove_cached_keys_set:
            self._enable_camera_conditioning_modules()
        if self.scene_coord_pe:
            self._enable_scene_coord_conditioning_modules()
        self._configure_uncond_latent_trainability()

    def _configure_uncond_latent_trainability(self):
        # The learned uncond VACE contexts are registered as BUFFERS by default
        # on VaceWanModel. This keeps them out of:
        #   - self.pipe.vace.parameters() (so DeepSpeed Zero never puts them in
        #     the optimizer flat buffer, and its frozen-param restore path
        #     never tries to copy them from saved_frozen_params)
        #   - trainable_modules() (so they do not appear in any optimizer group)
        # Only when this run is actually learning them do we promote them to
        # nn.Parameter so autograd will update them.
        should_learn = (
            self.target_gbuffer_uncond_fill_mode == "learned"
            and any(name in self.trainable_models for name in ("vace", "vace2"))
        )
        if not should_learn:
            return
        for vace_name in ("vace", "vace2"):
            vace_model = getattr(self.pipe, vace_name, None)
            if vace_model is None:
                continue
            if not hasattr(vace_model, "promote_uncond_vace_contexts_to_parameters"):
                continue
            vace_model.promote_uncond_vace_contexts_to_parameters()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        missing_keys = [
            key for key in incompatible.missing_keys
            if key not in self._OPTIONAL_RESUME_STATE_DICT_KEYS
        ]
        unexpected_keys = list(incompatible.unexpected_keys)
        if strict and (missing_keys or unexpected_keys):
            error_msgs = []
            if missing_keys:
                quoted_missing = ", ".join(f'"{key}"' for key in missing_keys)
                error_msgs.append(
                    f"Missing key(s) in state_dict: {quoted_missing}. "
                )
            if unexpected_keys:
                quoted_unexpected = ", ".join(f'"{key}"' for key in unexpected_keys)
                error_msgs.append(
                    f"Unexpected key(s) in state_dict: {quoted_unexpected}. "
                )
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                + "\n\t".join(error_msgs)
            )
        return type(incompatible)(missing_keys, unexpected_keys)

    def _camera_conditioning_requested(self):
        return all(key in self.extra_inputs for key in self.camera_pose_keys)

    @classmethod
    def _canonicalize_shared_vace_group(cls, key):
        return cls._VACE_KEY_TO_SHARED_GROUP.get(key)

    def _build_cached_vace_group_channel_slices(self):
        target_groups = []
        for key in self.vace_joint_video_keys:
            group = self._canonicalize_shared_vace_group(key)
            if group is None:
                raise ValueError(f"Unsupported target VACE key in compact layout: {key}")
            target_groups.append(group)
        source_groups = []
        for key in self.source_vace_joint_video_keys:
            group = self._canonicalize_shared_vace_group(key)
            if group is None:
                raise ValueError(f"Unsupported source VACE key in compact layout: {key}")
            source_groups.append(group)
        if target_groups and source_groups and target_groups != source_groups:
            raise ValueError(
                "Target/source VACE channel ordering mismatch for cached shared dropout: "
                f"target={target_groups}, source={source_groups}."
            )
        channel_groups = target_groups or source_groups
        return {
            group: slice(
                index * self._VACE_LATENT_CHANNELS_PER_SIGNAL,
                (index + 1) * self._VACE_LATENT_CHANNELS_PER_SIGNAL,
            )
            for index, group in enumerate(channel_groups)
        }

    @staticmethod
    def _infer_target_vace_frame_count(vace_context, source_latents=None):
        if not isinstance(vace_context, torch.Tensor) or vace_context.ndim < 3:
            return None
        total_frames = int(vace_context.shape[2])
        if total_frames <= 0:
            return None
        if isinstance(source_latents, torch.Tensor) and source_latents.ndim >= 3:
            source_frames = int(source_latents.shape[2])
            if 0 < source_frames < total_frames:
                target_frames = total_frames - source_frames
                if target_frames > 0:
                    return target_frames
        return total_frames

    def _maybe_apply_target_gbuffer_dropout(self, inputs_shared):
        if not self.training or self.target_gbuffer_dropout_prob <= 0.0:
            return
        vace_context = inputs_shared.get("vace_context", None)
        if not isinstance(vace_context, torch.Tensor) or vace_context.ndim != 5:
            return

        batch_size = int(vace_context.shape[0]) if vace_context.shape[0] > 0 else 1
        sample_device = vace_context.device
        drop_mask = torch.rand(batch_size, device=sample_device) < self.target_gbuffer_dropout_prob
        drop_count = int(drop_mask.sum().item())
        if drop_count == 0:
            return

        masked_context = vace_context.clone()
        fill_info = _apply_target_gbuffer_uncond_fill(
            masked_context,
            source_latents=inputs_shared.get("source_latents", None),
            dropout_scope=self.target_gbuffer_dropout_scope,
            fill_mode=self.target_gbuffer_uncond_fill_mode,
            vace_model=self.pipe.vace,
            batch_mask=drop_mask,
        )
        if fill_info is None:
            return
        inputs_shared["vace_context"] = masked_context

        if not self._target_gbuffer_dropout_logged:
            print(
                "[GBuffer] Target cached VACE dropout active: "
                f"prob={self.target_gbuffer_dropout_prob:.3f} "
                f"scope={self.target_gbuffer_dropout_scope} "
                f"fill={self.target_gbuffer_uncond_fill_mode} "
                f"first_batch_drops={drop_count}/{batch_size} "
                f"fill_frames={fill_info['fill_frames']} "
                f"target_frames={fill_info['target_frames']}"
            )
            self._target_gbuffer_dropout_logged = True

    def _maybe_apply_shared_cached_vace_dropout(self, inputs_shared):
        if not self.training:
            return
        if (
            self.shared_gbuffer_dropout_prob <= 0.0
            and self.shared_gbuffer_rgb_only_prob <= 0.0
        ):
            return
        if not self.shared_gbuffer_available_dropout_groups:
            return
        vace_context = inputs_shared.get("vace_context", None)
        if not isinstance(vace_context, torch.Tensor) or vace_context.ndim != 5:
            return
        expected_channels = (
            len(self.cached_vace_group_channel_slices)
            * self._VACE_LATENT_CHANNELS_PER_SIGNAL
        )
        if vace_context.shape[1] != expected_channels:
            raise ValueError(
                "Shared cached VACE dropout expects compact grouped channels "
                f"({expected_channels}), got vace_context shape {tuple(vace_context.shape)}."
            )

        batch_size = int(vace_context.shape[0]) if vace_context.shape[0] > 0 else 1
        sample_device = vace_context.device
        rgb_only_mask = torch.zeros(batch_size, dtype=torch.bool, device=sample_device)
        if self.shared_gbuffer_rgb_only_prob > 0.0:
            rgb_only_mask = torch.rand(batch_size, device=sample_device) < self.shared_gbuffer_rgb_only_prob

        masked_context = None
        applied_drop_counts = {}
        for group in self.shared_gbuffer_available_dropout_groups:
            drop_mask = rgb_only_mask.clone()
            if self.shared_gbuffer_dropout_prob > 0.0:
                drop_mask |= (
                    torch.rand(batch_size, device=sample_device)
                    < self.shared_gbuffer_dropout_prob
                )
            drop_count = int(drop_mask.sum().item())
            if drop_count == 0:
                continue
            if masked_context is None:
                masked_context = vace_context.clone()
            channel_slice = self.cached_vace_group_channel_slices[group]
            keep_mask = (~drop_mask).to(dtype=masked_context.dtype).view(batch_size, 1, 1, 1, 1)
            masked_context[:, channel_slice, :, :, :].mul_(keep_mask)
            applied_drop_counts[group] = drop_count

        if masked_context is None:
            return

        inputs_shared["vace_context"] = masked_context
        if not self._shared_gbuffer_dropout_logged:
            eligible_groups = ",".join(self.shared_gbuffer_available_dropout_groups)
            drop_summary = ", ".join(
                f"{group}:{applied_drop_counts[group]}/{batch_size}"
                for group in self.shared_gbuffer_available_dropout_groups
                if group in applied_drop_counts
            ) or "none"
            print(
                "[GBuffer] Shared cached VACE dropout active: "
                f"groups={eligible_groups} "
                f"per_group_prob={self.shared_gbuffer_dropout_prob:.3f} "
                f"rgb_only_prob={self.shared_gbuffer_rgb_only_prob:.3f} "
                f"first_batch_drops={drop_summary} "
                f"rgb_only={int(rgb_only_mask.sum().item())}/{batch_size}"
            )
            self._shared_gbuffer_dropout_logged = True

    def _get_optional_extra_input(self, data, key):
        return data.get(key, None) if key in self.extra_inputs else None

    def _resize_vace_inputs(self, named_inputs, width, height):
        resized = {}
        for key, value in named_inputs.items():
            resample = self._PIL_NEAREST if key.endswith("alpha") else self._PIL_BILINEAR
            resized[key] = self._resize_eval_video(value, width, height, resample)
        return resized

    def _iter_dit_blocks(self):
        for dit_name in ("dit", "dit2"):
            dit_model = getattr(self.pipe, dit_name, None)
            if dit_model is None:
                continue
            for block_id, block in enumerate(dit_model.blocks):
                yield dit_name, block_id, block

    def _iter_dit_models(self):
        for dit_name in ("dit", "dit2"):
            dit_model = getattr(self.pipe, dit_name, None)
            if dit_model is not None:
                yield dit_name, dit_model

    def _configure_scene_coord_gate_trainability(self):
        if not self.scene_coord_gated_residual:
            return 0
        enabled_count = 0
        for _, _, block in self._iter_dit_blocks():
            coord_gate = getattr(block, "coord_gate", None)
            if coord_gate is None:
                continue
            coord_gate.requires_grad_(True)
            enabled_count += 1
        return enabled_count

    @classmethod
    def _is_dit_param_name(cls, param_name):
        return str(param_name).startswith(cls._DIT_PARAM_PREFIXES)

    @classmethod
    def _uses_other_learning_rate(cls, param_name):
        if not cls._is_dit_param_name(param_name):
            return True
        return any(marker in param_name for marker in cls._OTHER_LR_DIT_PARAM_MARKERS)

    @classmethod
    def _uses_uncond_latent_learning_rate(cls, param_name):
        return str(param_name).endswith(cls._UNCOND_LATENT_PARAM_SUFFIXES)

    def build_optimizer_param_groups(
        self,
        dit_learning_rate,
        other_learning_rate,
        weight_decay,
        uncond_latent_learning_rate=None,
    ):
        group_specs = {
            "dit": {
                "lr": float(dit_learning_rate),
                "params": [],
                "param_names": [],
            },
            "other": {
                "lr": float(other_learning_rate),
                "params": [],
                "param_names": [],
            },
        }
        if uncond_latent_learning_rate is not None:
            group_specs["uncond_latent"] = {
                "lr": float(uncond_latent_learning_rate),
                "params": [],
                "param_names": [],
            }

        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (
                uncond_latent_learning_rate is not None
                and self._uses_uncond_latent_learning_rate(param_name)
            ):
                group_name = "uncond_latent"
            elif self._uses_other_learning_rate(param_name):
                group_name = "other"
            else:
                group_name = "dit"
            group_specs[group_name]["params"].append(param)
            group_specs[group_name]["param_names"].append(param_name)

        optimizer_param_groups = []
        optimizer_group_summaries = []
        for group_name in ("dit", "other", "uncond_latent"):
            spec = group_specs.get(group_name)
            if spec is None:
                continue
            if len(spec["params"]) == 0:
                continue
            optimizer_param_groups.append(
                {
                    "params": spec["params"],
                    "lr": spec["lr"],
                    "weight_decay": weight_decay,
                    "name": group_name,
                }
            )
            optimizer_group_summaries.append(
                {
                    "name": group_name,
                    "lr": spec["lr"],
                    "num_tensors": len(spec["params"]),
                    "num_parameters": sum(param.numel() for param in spec["params"]),
                    "sample_names": spec["param_names"][:5],
                }
            )

        if len(optimizer_param_groups) == 0:
            raise ValueError("Split WAN learning rates requested, but no trainable parameters were found.")

        return optimizer_param_groups, optimizer_group_summaries

    def _resize_eval_video(self, video, width, height, resample):
        if video is None or not isinstance(video, list) or len(video) == 0:
            return video
        if not hasattr(video[0], "resize"):
            return video
        target_size = (width, height)
        if video[0].size == target_size:
            return video
        return [frame.resize(target_size, resample=resample) for frame in video]

    def _resolve_eval_resolution(self, dataset_name, default_height, default_width):
        name = (dataset_name or "").strip().lower()
        if name and self.eval_dataset_resolutions:
            if name in self.eval_dataset_resolutions:
                return self.eval_dataset_resolutions[name]
            for key, value in self.eval_dataset_resolutions.items():
                if key in name:
                    return value
        if self.eval_height is not None and self.eval_width is not None:
            return self.eval_height, self.eval_width
        return default_height, default_width

    def _resolve_eval_metric_device(self, accelerator_device):
        if self.eval_metrics_device == "auto":
            return torch.device(accelerator_device)
        if self.eval_metrics_device == "cpu":
            return torch.device("cpu")
        if self.eval_metrics_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA metrics requested but CUDA is not available.")
            return torch.device("cuda")
        raise ValueError(f"Unsupported eval_metrics_device: {self.eval_metrics_device}")

    def _enable_scene_coord_conditioning_modules(self):
        if isinstance(self.task, str) and self.task.endswith(":data_process"):
            return

        scene_module_names = []
        attn_module_names = []
        projector_module_names = []
        coord_gate_params = []
        scene_coord_in_channels = 4 if self.scene_coord_xyzt else 3
        for dit_name, dit_model in self._iter_dit_models():
            dim = getattr(dit_model, "dim", dit_model.patch_embedding.out_channels)
            ref_param = dit_model.patch_embedding.weight
            device = ref_param.device
            dtype = ref_param.dtype

            if getattr(dit_model, "scene_coord_encoder", None) is None:
                dit_model.scene_coord_encoder = SceneCoordEncoder(
                    dim=dim,
                    in_channels=scene_coord_in_channels,
                    temporal_group=4,
                ).to(device=device, dtype=dtype)
            else:
                dit_model.scene_coord_encoder = dit_model.scene_coord_encoder.to(device=device, dtype=dtype)
            dit_model.scene_coord_cond = True
            scene_module_names.append(f"{dit_name}.scene_coord_encoder")

            for block_id, block in enumerate(dit_model.blocks):
                block_dim = block.self_attn.q.weight.shape[0]
                block_device = block.self_attn.q.weight.device
                block_dtype = block.self_attn.q.weight.dtype
                if getattr(block, "projector", None) is None:
                    block.projector = torch.nn.Linear(
                        block_dim, block_dim, device=block_device, dtype=block_dtype
                    )
                    torch.nn.init.eye_(block.projector.weight)
                    torch.nn.init.zeros_(block.projector.bias)
                else:
                    block.projector = block.projector.to(
                        device=block_device, dtype=block_dtype
                    )
                if self.scene_coord_gated_residual:
                    coord_gate = getattr(block, "coord_gate", None)
                    if coord_gate is None:
                        block.coord_gate = torch.nn.Parameter(
                            torch.full(
                                (1, 1, block_dim),
                                self.scene_coord_gate_init,
                                device=block_device,
                                dtype=block_dtype,
                            )
                        )
                    else:
                        block.coord_gate = torch.nn.Parameter(
                            coord_gate.to(device=block_device, dtype=block_dtype)
                        )
                    coord_gate_params.append(block.coord_gate)

                attn_module_names.append(f"{dit_name}.blocks.{block_id}.self_attn")
                projector_module_names.append(f"{dit_name}.blocks.{block_id}.projector")

        if len(scene_module_names) == 0:
            return

        module_names = scene_module_names + attn_module_names + projector_module_names
        for module_name in module_names:
            if module_name not in self.trainable_models:
                self.trainable_models.append(module_name)

        self.pipe.freeze_except(self.trainable_models)
        # coord_gate is a Parameter (not a Module), so it cannot be passed via freeze_except.
        # Enable gradients explicitly after module-level freezing.
        self._configure_scene_coord_gate_trainability()
        self._configure_uncond_latent_trainability()
        self.pipe.scheduler.set_timesteps(1000, training=True)
        print(
            "[SceneCoord] Enabled scene coordinate conditioning modules on "
            f"{len(scene_module_names)} DiT model(s): "
            f"encoders={len(scene_module_names)}, "
            f"self_attn={len(attn_module_names)}, "
            f"projector={len(projector_module_names)}, "
            f"coord_gate={len(coord_gate_params)}."
        )

    @staticmethod
    def _normalize_cached_batch_field(value, batch_size):
        if isinstance(value, (list, tuple)):
            if len(value) == batch_size:
                return list(value)
            if len(value) == 1:
                return list(value) * batch_size
            raise ValueError(f"Cached batch field has length {len(value)} but batch size is {batch_size}")
        return [value] * batch_size

    @staticmethod
    def _to_python_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
        return value

    def _to_int(self, value):
        return _cached_to_int(value)

    def _to_str(self, value):
        return _cached_to_str(value)

    def _build_local_eval_video_relpath(self, cameras_npz, target_cam):
        cameras_npz = self._to_str(cameras_npz)
        target_cam = self._to_str(target_cam)
        if cameras_npz is None or target_cam is None:
            return None

        vid_match = re.search(r"(vid\d+)", os.path.basename(cameras_npz))
        if vid_match is None:
            return None

        normalized = cameras_npz.replace("\\", "/")
        marker = "/posed_renderings/"
        if marker not in normalized:
            return None

        scene_name = normalized.split(marker, 1)[0].rsplit("/", 1)[-1]
        if not scene_name:
            return None

        return os.path.join(scene_name, f"{target_cam}_{vid_match.group(1)}.mp4")

    def _infer_cached_video_shape(self, inputs_shared, input_latents):
        return _infer_cached_video_shape(inputs_shared, input_latents)

    def _maybe_lazy_load_scene_coords_from_cache(self, inputs_shared):
        if not self.scene_coord_pe or self.scene_coord_loader is None:
            return
        if inputs_shared.get("scene_coord_target") is not None and inputs_shared.get("scene_coord_source") is not None:
            return
        if not all(key in inputs_shared for key in self.camera_pose_keys):
            return

        input_latents = inputs_shared.get("input_latents")
        if not isinstance(input_latents, torch.Tensor):
            return

        batch_size = input_latents.shape[0]
        height, width, num_frames = self._infer_cached_video_shape(inputs_shared, input_latents)
        cameras_npz_list = self._normalize_cached_batch_field(inputs_shared.get("cameras_npz"), batch_size)
        target_cam_list = self._normalize_cached_batch_field(inputs_shared.get("target_cam"), batch_size)
        source_cam_list = self._normalize_cached_batch_field(inputs_shared.get("source_cam"), batch_size)

        loader = self.scene_coord_loader
        jobs = [
            _resolve_scene_coord_cached_job(
                cameras_npz_list[idx], target_cam_list[idx], source_cam_list[idx]
            )
            for idx in range(batch_size)
        ]

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [
                pool.submit(
                    _load_scene_coord_pair_from_job,
                    loader,
                    root,
                    tgt,
                    src,
                    height,
                    width,
                    num_frames,
                    vid_tag,
                )
                for root, tgt, src, vid_tag in jobs
            ]
            raw_pairs = [future.result() for future in futures]

        scene_targets, scene_sources = [], []
        for scene_target, scene_source in raw_pairs:
            scene_targets.append(scene_target.unsqueeze(0))
            scene_sources.append(scene_source.unsqueeze(0))

        inputs_shared["scene_coord_target"] = torch.cat(scene_targets, dim=0)
        inputs_shared["scene_coord_source"] = torch.cat(scene_sources, dim=0)

    def _enable_camera_conditioning_modules(self):
        if isinstance(self.task, str) and self.task.endswith(":data_process"):
            return

        camera_module_names = []
        for dit_name, block_id, block in self._iter_dit_blocks():
            dim = block.self_attn.q.weight.shape[0]
            device = block.self_attn.q.weight.device
            dtype = block.self_attn.q.weight.dtype

            if getattr(block, "cam_encoder", None) is None:
                block.cam_encoder = torch.nn.Linear(12, dim, device=device, dtype=dtype)
            if getattr(block, "projector", None) is None:
                block.projector = torch.nn.Linear(dim, dim, device=device, dtype=dtype)

            torch.nn.init.zeros_(block.cam_encoder.weight)
            torch.nn.init.zeros_(block.cam_encoder.bias)
            torch.nn.init.eye_(block.projector.weight)
            torch.nn.init.zeros_(block.projector.bias)
            block.add_camera_conditioning = True

            camera_module_names.extend([
                f"{dit_name}.blocks.{block_id}.cam_encoder",
                f"{dit_name}.blocks.{block_id}.projector",
                f"{dit_name}.blocks.{block_id}.self_attn",
            ])

        if len(camera_module_names) == 0:
            return

        for module_name in camera_module_names:
            if module_name not in self.trainable_models:
                self.trainable_models.append(module_name)

        self.pipe.freeze_except(self.trainable_models)
        self._configure_uncond_latent_trainability()
        self.pipe.scheduler.set_timesteps(1000, training=True)
        print(f"[CameraPose] Enabled camera conditioning modules (cam_encoder, projector, self_attn) on {len(camera_module_names) // 3} DiT blocks.")

    @staticmethod
    def _eval_poly5(x, coeffs):
        out = torch.full_like(x, coeffs[-1])
        for coeff in reversed(coeffs[:-1]):
            out = out * x + coeff
        return out

    @classmethod
    def _turbo_colormap(cls, values_01):
        """Approximate Turbo colormap for input values in [0, 1]."""
        r = cls._eval_poly5(values_01, cls._TURBO_R_COEFFS)
        g = cls._eval_poly5(values_01, cls._TURBO_G_COEFFS)
        b = cls._eval_poly5(values_01, cls._TURBO_B_COEFFS)
        return torch.cat([r, g, b], dim=1).clamp(0, 1)

    def _build_turbo_error_map(self, pred_01, gt_01, gain=5.0):
        """Return Turbo-colored per-pixel MAE in [-1, 1] for visualization."""
        err_mae = (pred_01 - gt_01).abs().mean(dim=1, keepdim=True)
        err_norm = (err_mae * gain).clamp(0, 1)
        return self._turbo_colormap(err_norm) * 2 - 1

    @staticmethod
    def _eval_poly5_np(values, coeffs):
        values = np.asarray(values, dtype=np.float32)
        out = np.full_like(values, float(coeffs[-1]), dtype=np.float32)
        for coeff in reversed(coeffs[:-1]):
            out = out * values + float(coeff)
        return out

    @classmethod
    def _turbo_colormap_np(cls, values_01):
        values_01 = np.clip(np.asarray(values_01, dtype=np.float32), 0.0, 1.0)
        r = cls._eval_poly5_np(values_01, cls._TURBO_R_COEFFS)
        g = cls._eval_poly5_np(values_01, cls._TURBO_G_COEFFS)
        b = cls._eval_poly5_np(values_01, cls._TURBO_B_COEFFS)
        return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)

    @classmethod
    def _build_turbo_error_map_frame_np(cls, pred_rgb, gt_rgb, gain=5.0):
        pred_01 = np.asarray(pred_rgb, dtype=np.float32) / 255.0
        gt_01 = np.asarray(gt_rgb, dtype=np.float32) / 255.0
        err_mae = np.abs(pred_01 - gt_01).mean(axis=2)
        err_norm = np.clip(err_mae * float(gain), 0.0, 1.0)
        return (cls._turbo_colormap_np(err_norm) * 255.0).round().astype(np.uint8)

    def _build_dycheck_benchmark_output_root(self, logger, global_step):
        if logger is None:
            raise ValueError("DyCheck benchmark requires a logger to determine output paths.")
        return (
            Path(logger.output_path).expanduser().resolve()
            / "dycheck_benchmark"
            / "latest"
        )

    def _build_dycheck_benchmark_config(self, logger, global_step):
        if self.dycheck_benchmark_root is None:
            raise ValueError("DyCheck benchmark is not configured.")
        benchmark_root = Path(self.dycheck_benchmark_root).expanduser().resolve()
        data_root = Path(
            self.dycheck_benchmark_data_root or self.dycheck_benchmark_root
        ).expanduser().resolve()
        disable_gbuffer = not any(
            key in self.extra_inputs
            for key in (
                *self.base_vace_joint_video_keys,
                *self.base_source_vace_joint_video_keys,
                self.scene_flow_target_key,
                self.scene_flow_source_key,
            )
        )
        use_source_video = (
            "source_video" in self.extra_inputs
            and "source_latents" not in self.remove_cached_keys_set
        )
        return dycheck_benchmark_lib.DyCheckBenchmarkConfig(
            benchmark_root=benchmark_root,
            data_root=data_root,
            output_root=self._build_dycheck_benchmark_output_root(logger, global_step),
            seq_names=self.dycheck_benchmark_seq_names or None,
            chunk_size=self.dycheck_benchmark_chunk_size,
            pred_subdir="pred_rgb",
            prompt=" ",
            negative_prompt=None,
            inference_res=(512, 384),
            eval_res=(480, 360),
            num_inference_steps=self.eval_num_inference_steps,
            cfg_scale=1.0,
            target_gbuffer_cfg_scale=self.eval_target_gbuffer_cfg_scale,
            target_gbuffer_dropout_scope=self.target_gbuffer_dropout_scope,
            target_gbuffer_uncond_fill_mode=self.target_gbuffer_uncond_fill_mode,
            vace_scale=1.0,
            seed=0,
            tiled=False,
            use_scene_flow=self.scene_flow_enabled,
            split_cam_embed=self.split_cam_embed,
            zero_source_cam_emb_when_missing=self.zero_source_cam_emb_when_missing,
            disable_gbuffer=disable_gbuffer,
            use_source_video=use_source_video,
            # Pin the benchmark's VACE signals to exactly the compact target /
            # source subsets the VACE patch embedding was (re)initialized for.
            # Without this, build_chunk_inputs would load all 4 target + 4
            # source gbuffer signals and feed a 64-channel VACE context into a
            # vace_patch_embedding that expects the compact channel count
            # (e.g. 16 for vace_video-only runs). These tuples already include
            # scene_flow_* when --extra_inputs selected it, so we do not rely
            # on use_scene_flow for gating here.
            active_target_vace_keys=tuple(self.vace_joint_video_keys),
            active_source_vace_keys=tuple(self.source_vace_joint_video_keys),
            scene_coord_pe=self.scene_coord_pe,
            scene_coord_xyzt_pe=self.scene_coord_xyzt,
            scene_coord_gated_residual=self.scene_coord_gated_residual,
            metrics_device=self.eval_metrics_device,
            lpips_net=self.eval_metrics_lpips_net,
            mask_frame_name_mode="target",
            binarize_masks=True,
        )

    @staticmethod
    def _resize_rgb_frame_for_video(frame_rgb, size):
        if frame_rgb.shape[:2] != (size[1], size[0]):
            return dycheck_benchmark_lib.resize_rgb_array(frame_rgb, size)
        return frame_rgb

    @staticmethod
    def _build_dycheck_benchmark_video_name(sequence):
        return (
            "WT_Dycheck_benchmark/video/"
            f"{sequence.scene_name}/{sequence.pair_name}/{sequence.target_cam}"
        )

    @staticmethod
    def _build_dycheck_benchmark_video_path(config, sequence):
        return (
            config.output_root
            / "videos"
            / sequence.scene_name
            / sequence.pair_name
            / f"{sequence.target_cam}.mp4"
        )

    def _write_dycheck_benchmark_video(self, sequence, config):
        output_path = self._build_dycheck_benchmark_video_path(config, sequence)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        eval_height, eval_width = (
            config.eval_res if isinstance(config.eval_res, tuple) else (480, 360)
        )
        panel_size = (int(eval_width), int(eval_height))
        black_frame = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
        pred_dir = (
            config.output_root
            / sequence.scene_name
            / sequence.pair_name
            / config.pred_subdir
            / sequence.target_cam
        )
        writer = imageio.get_writer(
            str(output_path),
            fps=18,
            quality=9,
            macro_block_size=1,
        )
        try:
            for record in sequence.records:
                pred_frame = self._resize_rgb_frame_for_video(
                    dycheck_benchmark_lib.read_rgb_image(
                        pred_dir / f"{record.frame_name}.png"
                    ),
                    panel_size,
                )
                target_input = self._resize_rgb_frame_for_video(
                    dycheck_benchmark_lib.read_rgb_image(
                        sequence.pair_root
                        / "rgb"
                        / sequence.target_cam
                        / f"{record.frame_name}.png"
                    ),
                    panel_size,
                )
                source_frame = self._resize_rgb_frame_for_video(
                    dycheck_benchmark_lib.read_rgb_image(
                        sequence.pair_root
                        / "gt"
                        / sequence.source_cam
                        / f"{record.source_input_frame_name}.png"
                    ),
                    panel_size,
                )
                if record.has_gt:
                    gt_frame = self._resize_rgb_frame_for_video(
                        dycheck_benchmark_lib.read_rgb_image(
                            sequence.pair_root
                            / "gt"
                            / sequence.target_cam
                            / f"{record.frame_name}.png"
                        ),
                        panel_size,
                    )
                    input_error = self._build_turbo_error_map_frame_np(
                        target_input,
                        gt_frame,
                        gain=5.0,
                    )
                    pred_error = self._build_turbo_error_map_frame_np(
                        pred_frame,
                        gt_frame,
                        gain=5.0,
                    )
                else:
                    gt_frame = black_frame
                    input_error = black_frame
                    pred_error = black_frame
                top_row = np.concatenate([target_input, source_frame, input_error], axis=1)
                bottom_row = np.concatenate([pred_frame, gt_frame, pred_error], axis=1)
                writer.append_data(np.concatenate([top_row, bottom_row], axis=0))
        finally:
            writer.close()
        return output_path

    @torch.no_grad()
    def dycheck_benchmark(self, accelerator, global_step=0, logger=None):
        if not self.dycheck_benchmark_enabled:
            return None
        accelerator.print("Running DyCheck benchmark...")
        saved_units = self.pipe.units
        self.pipe.freeze_except([])
        if hasattr(self.pipe, "_all_units"):
            self.pipe.units = self.pipe._all_units
        try:
            config = self._build_dycheck_benchmark_config(logger, global_step)
            if accelerator.is_main_process and config.output_root.exists():
                shutil.rmtree(config.output_root, ignore_errors=True)
            accelerator.wait_for_everyone()
            result = dycheck_benchmark_lib.run_dycheck_benchmark(
                pipe=self.pipe,
                accelerator=accelerator,
                config=config,
            )
            postprocess_done_path = dycheck_benchmark_lib.dycheck_postprocess_done_path(
                result.metrics_dir
            )
            if accelerator.is_main_process and result.summary is not None:
                mean_metrics = result.summary.get("mean_metrics", {})
                if mean_metrics:
                    accelerator.log(
                        {
                            "WT_Dycheck_benchmark/mspnr": mean_metrics["masked_mPSNR"],
                            "WT_Dycheck_benchmark/mssim": mean_metrics["masked_mSSIM"],
                            "WT_Dycheck_benchmark/mlpips": mean_metrics["masked_mLPIPS"],
                        },
                        step=global_step,
                    )
                    accelerator.print(
                        f"[Step {global_step}] WT DyCheck benchmark: "
                        f"mPSNR={mean_metrics['masked_mPSNR']:.4f}, "
                        f"mSSIM={mean_metrics['masked_mSSIM']:.4f}, "
                        f"mLPIPS={mean_metrics['masked_mLPIPS']:.4f}"
                    )
                    if logger is not None:
                        best_update = logger.maybe_promote_best_step_weight(
                            global_step=global_step,
                            masked_mpsnr=mean_metrics["masked_mPSNR"],
                        )
                        if best_update["promoted"]:
                            accelerator.print(
                                f"[Step {global_step}] Promoted "
                                f"{best_update['best_file_name']} as best DyCheck "
                                f"checkpoint (mPSNR={best_update['best_masked_mpsnr']:.4f})."
                            )
                if logger is not None:
                    total_sequences = len(result.dataset.sequences)
                    for seq_idx, sequence in enumerate(result.dataset.sequences):
                        accelerator.print(
                            f"[DyCheck] Writing benchmark video "
                            f"{seq_idx + 1}/{total_sequences}: "
                            f"{sequence.scene_name}/{sequence.pair_name}/{sequence.target_cam}"
                        )
                        video_path = self._write_dycheck_benchmark_video(sequence, config)
                        logger.log_video_file(
                            accelerator,
                            video_path,
                            self._build_dycheck_benchmark_video_name(sequence),
                            global_step,
                        )
                    logger.flush_video_logs()
                dycheck_benchmark_lib.write_json(
                    postprocess_done_path,
                    {
                        "global_step": int(global_step),
                        "summary_path": (
                            None if result.summary_path is None else str(result.summary_path)
                        ),
                        "num_sequences": len(result.dataset.sequences),
                    },
                )
                accelerator.print(
                    "[DyCheck] Main-process DyCheck postprocessing complete."
                )
            else:
                dycheck_benchmark_lib.wait_for_files(
                    [postprocess_done_path],
                    progress_prefix=(
                        f"[DyCheck][Rank {accelerator.process_index}] Waiting for "
                        "main-process DyCheck logging"
                    ),
                )
            return result
        finally:
            self.pipe.units = saved_units


    def maybe_reverse_video_pair(self, data):
        if not self.training:
            return data
        if isinstance(self.task, str) and self.task.endswith(":data_process"):
            return data
        if torch.rand(1).item() >= self.reverse_video_probability:
            return data

        gt_video = data.get("video")
        paired_keys = []
        for key in (*self.vace_joint_video_keys, *self.source_vace_joint_video_keys, "input_video", "source_video"):
            if key in paired_keys:
                continue
            if isinstance(data.get(key), list):
                paired_keys.append(key)
        if not isinstance(gt_video, list) or len(paired_keys) == 0:
            return data

        data = data.copy()
        data["video"] = gt_video[::-1]
        for key in paired_keys:
            data[key] = data[key][::-1]
        return data
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                if extra_input in data:
                    inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input in data:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        data = self.maybe_reverse_video_pair(data)
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "split_cam_embed": self.split_cam_embed,
            "zero_source_cam_emb_when_missing": self.zero_source_cam_emb_when_missing,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        if self.remove_cached_keys and isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            for key in self.remove_cached_keys:
                inputs[0].pop(key, None)
        if self.strip_source_gbuffer and isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            vace_context = inputs[0].get("vace_context", None)
            if isinstance(vace_context, torch.Tensor) and vace_context.ndim >= 3 and vace_context.shape[2] > 1:
                target_context_frames = vace_context.shape[2] // 2
                inputs[0]["vace_context"] = vace_context[:, :, :target_context_frames]
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            self._maybe_apply_target_gbuffer_dropout(inputs[0])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            self._maybe_apply_shared_cached_vace_dropout(inputs[0])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            self._maybe_lazy_load_scene_coords_from_cache(inputs[0])
        inputs[0]["use_gradient_checkpointing"] = self.use_gradient_checkpointing
        inputs[0]["use_gradient_checkpointing_offload"] = self.use_gradient_checkpointing_offload
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def _get_eval_sample_count(self, val_loader):
        dataset = getattr(val_loader, "dataset", None)
        if dataset is None:
            return None
        try:
            return int(len(dataset))
        except TypeError:
            return None

    def _is_valid_eval_sample(self, accelerator, sample_idx, total_eval_samples):
        if total_eval_samples is None:
            return True
        # Validation uses batch_size=1, so distributed shards interleave samples
        # by rank and pad only the uneven tail.
        global_sample_idx = sample_idx * accelerator.num_processes + accelerator.process_index
        return global_sample_idx < total_eval_samples

    @staticmethod
    def _tensor_to_list(tensor):
        return tensor.detach().cpu().reshape(-1).tolist()

    @torch.no_grad()
    def evaluate_vace_baseline(self, accelerator, val_loader, global_step=0):
        """Compute baseline metrics (PSNR, SSIM, LPIPS, MSE) for vace_video vs GT.

        This establishes a reference for how close the conditioning input already
        is to the ground truth, so later we can quantify whether training improves
        upon the vace_video input.
        """
        if "vace_video" not in self.extra_inputs:
            accelerator.print("Skipping VACE baseline eval (vace_video not in extra_inputs).")
            return

        accelerator.print("Computing baseline metrics (vace_video vs GT)...")
        progress_bar = tqdm(val_loader, desc="Baseline eval", disable=not accelerator.is_main_process)
        total_eval_samples = self._get_eval_sample_count(val_loader)
        metric_device = self._resolve_eval_metric_device(accelerator.device)
        lpips_model = build_eval_lpips_model(metric_device, self.eval_metrics_lpips_net)

        all_msssim, all_psnr, all_lpips, all_mse = [], [], [], []
        all_psnr_masked, all_mse_masked, all_mask_coverage = [], [], []

        # Build dataset_name <-> index mapping for per-dataset metrics.
        # Unwrap nested Subset wrappers to reach the underlying dataset with .data.
        underlying_ds = val_loader.dataset
        val_indices = list(range(len(underlying_ds)))
        while hasattr(underlying_ds, "dataset"):
            if hasattr(underlying_ds, "indices"):
                val_indices = [underlying_ds.indices[i] for i in val_indices]
            underlying_ds = underlying_ds.dataset
        unique_names = sorted(set(
            underlying_ds.data[i].get("dataset_name", "unknown")
            if hasattr(underlying_ds, "data") and isinstance(underlying_ds.data[i], dict)
            else "unknown"
            for i in val_indices
        ))
        name_to_idx = {name: i for i, name in enumerate(unique_names)}
        idx_to_name = {i: name for i, name in enumerate(unique_names)}
        per_dataset = {
            name: {
                "msssim": [],
                "psnr": [],
                "lpips": [],
                "mse": [],
                "psnr_masked": [],
                "mse_masked": [],
                "mask_coverage": [],
            }
            for name in unique_names
        }

        for idx, data in enumerate(progress_bar):
            local_is_valid = self._is_valid_eval_sample(accelerator, idx, total_eval_samples)
            vace_video = data.get("vace_video", None)
            vace_alpha = data.get("vace_alpha", None)
            gt_video = data.get("video", None)
            dataset_name = self._to_str(data.get("dataset_name", "unknown")) or "unknown"
            name_idx = torch.tensor(name_to_idx.get(dataset_name, 0), device=accelerator.device)
            zero_scalar = torch.zeros((), device=accelerator.device)
            msssim = zero_scalar
            psnr = zero_scalar
            lpips_score = zero_scalar
            mse = zero_scalar
            psnr_m = zero_scalar
            mse_m = zero_scalar
            mask_cov = zero_scalar
            valid_sample = torch.tensor(0, device=accelerator.device, dtype=torch.int32)
            if local_is_valid and (gt_video is None or vace_video is None):
                accelerator.print(f"Baseline eval skipping sample {idx}: missing vace_video or ground truth.")
                local_is_valid = False

            if local_is_valid:
                try:
                    default_height = gt_video[0].size[1]
                    default_width = gt_video[0].size[0]
                    height, width = self._resolve_eval_resolution(dataset_name, default_height, default_width)
                    if (height, width) != (default_height, default_width):
                        gt_video = self._resize_eval_video(gt_video, width, height, self._PIL_BILINEAR)
                        vace_video = self._resize_eval_video(vace_video, width, height, self._PIL_BILINEAR)
                        vace_alpha = self._resize_eval_video(vace_alpha, width, height, self._PIL_NEAREST)

                    # Convert PIL image lists to tensors [1, C, T, H, W] in [-1, 1]
                    gt_tensor = self.pipe.preprocess_video(gt_video)
                    input_tensor = self.pipe.preprocess_video(vace_video)

                    # Ensure same number of frames
                    min_t = min(gt_tensor.shape[2], input_tensor.shape[2])
                    gt_tensor = gt_tensor[:, :, :min_t]
                    input_tensor = input_tensor[:, :, :min_t]

                    # Normalize from [-1, 1] to [0, 1] for metrics
                    gt_01 = (gt_tensor.float() + 1) / 2
                    vace_01 = (input_tensor.float() + 1) / 2

                    # Build covisibility mask from opacity map
                    opacity_mask = None
                    if vace_alpha is not None:
                        alpha_tensor = self.pipe.preprocess_video(vace_alpha)
                        alpha_01 = (alpha_tensor.float() + 1) / 2
                        opacity_mask = alpha_01.mean(dim=1, keepdim=True)
                        opacity_mask = (opacity_mask > 0.01).float()
                        opacity_mask = opacity_mask[:, :, :min_t]

                    msssim = video_msssim(vace_01, gt_01).detach()
                    psnr = video_psnr(vace_01, gt_01).detach()
                    lpips_score = (
                        video_lpips(
                            vace_01.to(metric_device),
                            gt_01.to(metric_device),
                            lpips_model,
                        )
                        .to(accelerator.device)
                        .detach()
                    )
                    mse = F.mse_loss(vace_01, gt_01).detach()

                    if opacity_mask is not None and opacity_mask.sum() > 0:
                        psnr_m = video_psnr_masked(vace_01, gt_01, opacity_mask).detach()
                        diff_sq = (vace_01 - gt_01) ** 2
                        mse_m = (diff_sq * opacity_mask).sum() / (opacity_mask.sum() * vace_01.shape[1])
                        mse_m = mse_m.detach()
                        mask_cov = (opacity_mask.sum() / opacity_mask.numel()).detach()
                    else:
                        psnr_m = psnr
                        mse_m = mse
                        mask_cov = torch.tensor(1.0, device=accelerator.device)
                except Exception as e:
                    accelerator.print(f"Baseline metrics failed for sample {idx}: {e}")
                    local_is_valid = False

            if local_is_valid:
                valid_sample = torch.tensor(1, device=accelerator.device, dtype=torch.int32)
            else:
                name_idx = torch.zeros((), device=accelerator.device, dtype=torch.long)

            # Gather from all processes
            msssim_g = accelerator.gather(msssim)
            psnr_g = accelerator.gather(psnr)
            lpips_g = accelerator.gather(lpips_score)
            mse_g = accelerator.gather(mse)
            psnr_m_g = accelerator.gather(psnr_m)
            mse_m_g = accelerator.gather(mse_m)
            mask_cov_g = accelerator.gather(mask_cov)
            name_idx_g = accelerator.gather(name_idx)
            valid_g = accelerator.gather(valid_sample)

            if accelerator.is_main_process:
                msssim_list = self._tensor_to_list(msssim_g)
                psnr_list = self._tensor_to_list(psnr_g)
                lpips_list = self._tensor_to_list(lpips_g)
                mse_list = self._tensor_to_list(mse_g)
                psnr_m_list = self._tensor_to_list(psnr_m_g)
                mse_m_list = self._tensor_to_list(mse_m_g)
                mask_cov_list = self._tensor_to_list(mask_cov_g)
                name_idx_list = self._tensor_to_list(name_idx_g)
                valid_list = self._tensor_to_list(valid_g)

                for is_valid, m_ss, m_p, m_lp, m_m, m_pm, m_mm, m_cov, n_i in zip(
                    valid_list,
                    msssim_list,
                    psnr_list,
                    lpips_list,
                    mse_list,
                    psnr_m_list,
                    mse_m_list,
                    mask_cov_list,
                    name_idx_list,
                ):
                    if not bool(is_valid):
                        continue
                    all_msssim.append(m_ss)
                    all_psnr.append(m_p)
                    all_lpips.append(m_lp)
                    all_mse.append(m_m)
                    all_psnr_masked.append(m_pm)
                    all_mse_masked.append(m_mm)
                    all_mask_coverage.append(m_cov)
                    ds_name = idx_to_name.get(int(n_i), "unknown")
                    if ds_name not in per_dataset:
                        per_dataset[ds_name] = {
                            "msssim": [],
                            "psnr": [],
                            "lpips": [],
                            "mse": [],
                            "psnr_masked": [],
                            "mse_masked": [],
                            "mask_coverage": [],
                        }
                    per_dataset[ds_name]["msssim"].append(m_ss)
                    per_dataset[ds_name]["psnr"].append(m_p)
                    per_dataset[ds_name]["lpips"].append(m_lp)
                    per_dataset[ds_name]["mse"].append(m_m)
                    per_dataset[ds_name]["psnr_masked"].append(m_pm)
                    per_dataset[ds_name]["mse_masked"].append(m_mm)
                    per_dataset[ds_name]["mask_coverage"].append(m_cov)

        # Log aggregate metrics
        if accelerator.is_main_process:
            if len(all_msssim) > 0:
                avg_msssim = sum(all_msssim) / len(all_msssim)
                avg_psnr = sum(all_psnr) / len(all_psnr)
                avg_lpips = sum(all_lpips) / len(all_lpips)
                avg_mse = sum(all_mse) / len(all_mse)
                avg_psnr_m = sum(all_psnr_masked) / len(all_psnr_masked)
                avg_mse_m = sum(all_mse_masked) / len(all_mse_masked)
                avg_cov = sum(all_mask_coverage) / len(all_mask_coverage)
                accelerator.log({
                    "baseline/ssim_rgb": avg_msssim,
                    "baseline/psnr_rgb": avg_psnr,
                    "baseline/lpips_rgb": avg_lpips,
                    "baseline/mse_rgb": avg_mse,
                    "baseline/psnr_rgb_masked": avg_psnr_m,
                    "baseline/mse_rgb_masked": avg_mse_m,
                    "baseline/mask_coverage": avg_cov,
                }, step=global_step)
                accelerator.print(
                    f"[Baseline] MS-SSIM: {avg_msssim:.4f}, "
                    f"PSNR: {avg_psnr:.2f}, LPIPS: {avg_lpips:.4f}, MSE: {avg_mse:.6f} | "
                    f"Masked PSNR: {avg_psnr_m:.2f}, Masked MSE: {avg_mse_m:.6f}, Coverage: {avg_cov:.2%} "
                    f"({len(all_msssim)} samples)"
                )

                # Log per-dataset metrics
                for ds_name, metrics in sorted(per_dataset.items()):
                    if len(metrics["msssim"]) == 0:
                        continue
                    n = len(metrics["msssim"])
                    avg_ss = sum(metrics["msssim"]) / n
                    avg_p = sum(metrics["psnr"]) / n
                    avg_lp = sum(metrics["lpips"]) / n
                    avg_m = sum(metrics["mse"]) / n
                    avg_pm = sum(metrics["psnr_masked"]) / n
                    avg_mm = sum(metrics["mse_masked"]) / n
                    avg_mc = sum(metrics["mask_coverage"]) / n
                    accelerator.log({
                        f"baseline/ssim_rgb/{ds_name}": avg_ss,
                        f"baseline/psnr_rgb/{ds_name}": avg_p,
                        f"baseline/lpips_rgb/{ds_name}": avg_lp,
                        f"baseline/mse_rgb/{ds_name}": avg_m,
                        f"baseline/psnr_rgb_masked/{ds_name}": avg_pm,
                        f"baseline/mse_rgb_masked/{ds_name}": avg_mm,
                        f"baseline/mask_coverage/{ds_name}": avg_mc,
                    }, step=global_step)
                    accelerator.print(
                        f"  [{ds_name}] MS-SSIM: {avg_ss:.4f}, PSNR: {avg_p:.2f}, "
                        f"LPIPS: {avg_lp:.4f}, MSE: {avg_m:.6f} | "
                        f"Masked PSNR: {avg_pm:.2f}, Masked MSE: {avg_mm:.6f}, Cov: {avg_mc:.2%} ({n} samples)"
                    )
            else:
                accelerator.print("Baseline: no valid vace_video/video pairs found in val set.")

    @torch.no_grad()
    def evaluate(self, accelerator, val_loader, global_step=0, logger=None, max_logs=4, visualize=False):
        """Run validation: inference on val set, compute metrics, and optionally log videos."""
        accelerator.print("Evaluating...")
        progress_bar = tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process)
        total_eval_samples = self._get_eval_sample_count(val_loader)
        use_multicam_metrics = self.eval_metrics_mode == "multicam"
        metric_device = self._resolve_eval_metric_device(accelerator.device)
        lpips_model = build_eval_lpips_model(metric_device, self.eval_metrics_lpips_net)

        if use_multicam_metrics:
            all_ssim, all_psnr, all_lpips = [], [], []
            all_input_ssim, all_input_psnr, all_input_lpips = [], [], []
        else:
            all_msssim, all_psnr, all_lpips, all_mse = [], [], [], []
            all_psnr_masked, all_mse_masked, all_mask_coverage = [], [], []
        should_stage_videos = (
            visualize
            and logger is not None
            and (max_logs is None or max_logs != 0)
        )

        # Build dataset_name <-> index mapping for per-dataset metrics.
        # Unwrap nested Subset wrappers to reach the underlying dataset with .data.
        underlying_ds = val_loader.dataset
        val_indices = list(range(len(underlying_ds)))
        while hasattr(underlying_ds, "dataset"):
            if hasattr(underlying_ds, "indices"):
                val_indices = [underlying_ds.indices[i] for i in val_indices]
            underlying_ds = underlying_ds.dataset
        unique_names = sorted(set(
            underlying_ds.data[i].get("dataset_name", "unknown")
            if hasattr(underlying_ds, "data") and isinstance(underlying_ds.data[i], dict)
            else "unknown"
            for i in val_indices
        ))
        name_to_idx = {name: i for i, name in enumerate(unique_names)}
        idx_to_name = {i: name for i, name in enumerate(unique_names)}
        if use_multicam_metrics:
            per_dataset = {
                name: {
                    "ssim": [],
                    "psnr": [],
                    "lpips": [],
                    "input_ssim": [],
                    "input_psnr": [],
                    "input_lpips": [],
                }
                for name in unique_names
            }
        else:
            per_dataset = {
                name: {
                    "msssim": [],
                    "psnr": [],
                    "lpips": [],
                    "mse": [],
                    "psnr_masked": [],
                    "mse_masked": [],
                    "mask_coverage": [],
                }
                for name in unique_names
            }

        # Freeze all models for inference
        self.pipe.freeze_except([])

        # Restore full pipeline units for inference (they may have been
        # stripped to training-only units when task is "sft:train").
        saved_units = self.pipe.units
        if hasattr(self.pipe, '_all_units'):
            self.pipe.units = self.pipe._all_units
        if should_stage_videos and logger is not None:
            logger.begin_eval_video_staging(accelerator, global_step)

        for idx, data in enumerate(progress_bar):
            local_is_valid = self._is_valid_eval_sample(accelerator, idx, total_eval_samples)
            # Prepare inputs for inference
            prompt = data.get("prompt", " ")
            target_vace_inputs = {
                key: self._get_optional_extra_input(data, key)
                for key in self.vace_joint_video_keys
            }
            source_vace_inputs = {
                key: self._get_optional_extra_input(data, key)
                for key in self.source_vace_joint_video_keys
            }
            vace_video = target_vace_inputs.get("vace_video")
            vace_alpha = target_vace_inputs.get("vace_alpha")
            gt_video = data.get("video", None)
            dataset_name = self._to_str(data.get("dataset_name", "unknown")) or "unknown"
            source_enabled = "source_video" in self.extra_inputs and "source_latents" not in self.remove_cached_keys_set
            camera_enabled = self._camera_conditioning_requested() and (
                "cam_emb" not in self.remove_cached_keys_set or self.scene_coord_pe
            )
            source_video = data.get("source_video", None) if source_enabled else None
            cameras_npz = data.get("cameras_npz", None) if camera_enabled else None
            target_cam = data.get("target_cam", None) if camera_enabled else None
            source_cam = data.get("source_cam", None) if camera_enabled else None
            name_idx = torch.tensor(name_to_idx.get(dataset_name, 0), device=accelerator.device)
            can_stage_video = False
            zero_scalar = torch.zeros((), device=accelerator.device)
            psnr = zero_scalar
            lpips_score = zero_scalar
            if use_multicam_metrics:
                ssim = zero_scalar
                input_ssim = zero_scalar
                input_psnr = zero_scalar
                input_lpips_score = zero_scalar
                input_valid_sample = torch.tensor(0, device=accelerator.device, dtype=torch.int32)
            else:
                msssim = zero_scalar
                mse = zero_scalar
                psnr_m = zero_scalar
                mse_m = zero_scalar
                mask_cov = zero_scalar
            valid_sample = torch.tensor(0, device=accelerator.device, dtype=torch.int32)
            pred_tensor = None
            gt_tensor = None
            pred_01 = None
            gt_01 = None
            min_t = 0

            if local_is_valid and gt_video is None:
                accelerator.print(f"Validation eval skipping sample {idx}: missing ground truth video.")
                local_is_valid = False

            if local_is_valid:
                try:
                    default_height = gt_video[0].size[1]
                    default_width = gt_video[0].size[0]
                    height, width = self._resolve_eval_resolution(dataset_name, default_height, default_width)
                    num_frames = len(gt_video)
                    if (height, width) != (default_height, default_width):
                        gt_video = self._resize_eval_video(gt_video, width, height, self._PIL_BILINEAR)
                        target_vace_inputs = self._resize_vace_inputs(target_vace_inputs, width, height)
                        source_vace_inputs = self._resize_vace_inputs(source_vace_inputs, width, height)
                        vace_video = target_vace_inputs.get("vace_video")
                        vace_alpha = target_vace_inputs.get("vace_alpha")
                        source_video = self._resize_eval_video(source_video, width, height, self._PIL_BILINEAR)

                    # Run inference through the pipeline
                    pred_video = self.pipe(
                        prompt=prompt,
                        source_video=source_video,
                        cameras_npz=cameras_npz,
                        target_cam=target_cam,
                        source_cam=source_cam,
                        split_cam_embed=self.split_cam_embed,
                        zero_source_cam_emb_when_missing=self.zero_source_cam_emb_when_missing,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        cfg_scale=1.0,
                        target_gbuffer_cfg_scale=self.eval_target_gbuffer_cfg_scale,
                        target_gbuffer_dropout_scope=self.target_gbuffer_dropout_scope,
                        target_gbuffer_uncond_fill_mode=self.target_gbuffer_uncond_fill_mode,
                        tiled=False,
                        num_inference_steps=self.eval_num_inference_steps,
                        vace_scale=1.0,
                        output_type="floatpoint",
                        **target_vace_inputs,
                        **source_vace_inputs,
                    )

                    # Convert GT video (list of PIL Images) to tensor [1, C, T, H, W] in [-1, 1]
                    gt_tensor = self.pipe.preprocess_video(gt_video)

                    # pred_video is already [1, C, T, H, W] in [-1, 1] (floatpoint output)
                    pred_tensor = pred_video
                    if not isinstance(gt_tensor, torch.Tensor) or not isinstance(pred_tensor, torch.Tensor):
                        raise TypeError("Evaluation expected tensor videos after preprocessing.")

                    # Ensure same number of frames (pipeline may trim for VACE reference image)
                    min_t = min(pred_tensor.shape[2], gt_tensor.shape[2])
                    pred_tensor = pred_tensor[:, :, :min_t]
                    gt_tensor = gt_tensor[:, :, :min_t]

                    # Normalize to [0, 1] for metrics
                    pred_01 = (pred_tensor.float() + 1) / 2
                    gt_01 = (gt_tensor.float() + 1) / 2
                    if use_multicam_metrics:
                        pred_metric = pred_01.to(metric_device)
                        gt_metric = gt_01.to(metric_device)
                        ssim = video_ssim(pred_metric, gt_metric).to(accelerator.device).detach()
                        psnr = video_psnr(pred_metric, gt_metric).to(accelerator.device).detach()
                        lpips_score = (
                            video_lpips(pred_metric, gt_metric, lpips_model)
                            .to(accelerator.device)
                            .detach()
                        )
                        if vace_video is not None:
                            input_tensor = self.pipe.preprocess_video(vace_video)
                            input_min_t = min(input_tensor.shape[2], gt_metric.shape[2])
                            if input_min_t > 0:
                                input_metric = ((input_tensor[:, :, :input_min_t].float() + 1) / 2).to(metric_device)
                                input_gt_metric = gt_metric[:, :, :input_min_t]
                                input_ssim = (
                                    video_ssim(input_metric, input_gt_metric)
                                    .to(accelerator.device)
                                    .detach()
                                )
                                input_psnr = (
                                    video_psnr(input_metric, input_gt_metric)
                                    .to(accelerator.device)
                                    .detach()
                                )
                                input_lpips_score = (
                                    video_lpips(input_metric, input_gt_metric, lpips_model)
                                    .to(accelerator.device)
                                    .detach()
                                )
                                input_valid_sample = torch.tensor(
                                    1, device=accelerator.device, dtype=torch.int32
                                )
                    else:
                        # Build covisibility mask from opacity map
                        opacity_mask = None
                        if vace_alpha is not None:
                            alpha_tensor = self.pipe.preprocess_video(vace_alpha)        # [1, C, T, H, W] in [-1, 1]
                            alpha_01 = (alpha_tensor.float() + 1) / 2                   # [0, 1]
                            opacity_mask = alpha_01.mean(dim=1, keepdim=True)            # [1, 1, T, H, W]
                            opacity_mask = (opacity_mask > 0.01).float()
                            opacity_mask = opacity_mask[:, :, :min_t]

                        msssim = video_msssim(pred_01, gt_01).detach()
                        psnr = video_psnr(pred_01, gt_01).detach()
                        lpips_score = (
                            video_lpips(
                                pred_01.to(metric_device),
                                gt_01.to(metric_device),
                                lpips_model,
                            )
                            .to(accelerator.device)
                            .detach()
                        )
                        mse = F.mse_loss(pred_01, gt_01).detach()

                        if opacity_mask is not None and opacity_mask.sum() > 0:
                            psnr_m = video_psnr_masked(pred_01, gt_01, opacity_mask).detach()
                            diff_sq = (pred_01 - gt_01) ** 2
                            mse_m = (diff_sq * opacity_mask).sum() / (opacity_mask.sum() * pred_01.shape[1])
                            mse_m = mse_m.detach()
                            mask_cov = (opacity_mask.sum() / opacity_mask.numel()).detach()
                        else:
                            psnr_m = psnr
                            mse_m = mse
                            mask_cov = torch.tensor(1.0, device=accelerator.device)
                    can_stage_video = True
                    valid_sample = torch.tensor(1, device=accelerator.device, dtype=torch.int32)
                except Exception as e:
                    accelerator.print(f"Metrics computation failed for sample {idx}: {e}")
                    local_is_valid = False

            if not local_is_valid:
                name_idx = torch.zeros((), device=accelerator.device, dtype=torch.long)

            # Gather from all processes
            psnr_g = accelerator.gather(psnr)
            if use_multicam_metrics:
                ssim_g = accelerator.gather(ssim)
                lpips_g = accelerator.gather(lpips_score)
                input_ssim_g = accelerator.gather(input_ssim)
                input_psnr_g = accelerator.gather(input_psnr)
                input_lpips_g = accelerator.gather(input_lpips_score)
                input_valid_g = accelerator.gather(input_valid_sample)
            else:
                msssim_g = accelerator.gather(msssim)
                lpips_g = accelerator.gather(lpips_score)
                mse_g = accelerator.gather(mse)
                psnr_m_g = accelerator.gather(psnr_m)
                mse_m_g = accelerator.gather(mse_m)
                mask_cov_g = accelerator.gather(mask_cov)
            name_idx_g = accelerator.gather(name_idx)
            valid_g = accelerator.gather(valid_sample)

            # Stage merged video tensors rank-locally for a main-process flush.
            if should_stage_videos and can_stage_video and logger is not None:
                assert isinstance(pred_tensor, torch.Tensor) and isinstance(gt_tensor, torch.Tensor)
                assert isinstance(pred_01, torch.Tensor) and isinstance(gt_01, torch.Tensor)
                local_video_relpath = self._build_local_eval_video_relpath(
                    cameras_npz, target_cam
                )
                # Visualization grid layout (all tensors [1, C, T, H, W] in [-1, 1]):
                #
                #   3 panels  [pred, gt, error]                  -> 1x3 horizontal
                #   4 panels  [vace/source, pred, gt, error]     -> 1x4 horizontal
                #   5 panels  [vace, source, pred, gt, error]    -> 2x3:
                #       row 0: vace      | source | error(vace-gt)
                #       row 1: pred      | gt     | error(pred-gt)
                #
                # Error map = Turbo(|pred - gt|_mean_rgb), amplified 5x for visibility.
                error_map = self._build_turbo_error_map(pred_01, gt_01, gain=5.0)
                error_map = error_map.to(pred_tensor.dtype)
                panels = []
                if vace_video is not None:
                    panels.append(self.pipe.preprocess_video(vace_video)[:, :, :min_t])
                if source_video is not None:
                    panels.append(self.pipe.preprocess_video(source_video)[:, :, :min_t])
                panels.extend([pred_tensor, gt_tensor, error_map])
                if len(panels) == 5:
                    vace_01 = (panels[0].float() + 1) / 2
                    input_error_map = self._build_turbo_error_map(vace_01, gt_01, gain=5.0)
                    input_error_map = input_error_map.to(pred_tensor.dtype)
                    top_row = torch.cat([panels[0], panels[1], input_error_map], dim=-1)
                    bottom_row = torch.cat([panels[2], panels[3], panels[4]], dim=-1)
                    merged = torch.cat([top_row, bottom_row], dim=-2)
                else:
                    merged = torch.cat(panels, dim=-1)
                logger.stage_eval_video(
                    merged.contiguous(),
                    global_step=global_step,
                    sample_idx=idx,
                    rank=accelerator.process_index,
                    dataset_name=dataset_name,
                    local_video_relpath=local_video_relpath,
                    local_video_tensor=pred_tensor.contiguous(),
                )
                del merged, panels

            if accelerator.is_main_process:
                psnr_list = self._tensor_to_list(psnr_g)
                if use_multicam_metrics:
                    ssim_list = self._tensor_to_list(ssim_g)
                    lpips_list = self._tensor_to_list(lpips_g)
                    input_ssim_list = self._tensor_to_list(input_ssim_g)
                    input_psnr_list = self._tensor_to_list(input_psnr_g)
                    input_lpips_list = self._tensor_to_list(input_lpips_g)
                    input_valid_list = self._tensor_to_list(input_valid_g)
                else:
                    msssim_list = self._tensor_to_list(msssim_g)
                    lpips_list = self._tensor_to_list(lpips_g)
                    mse_list = self._tensor_to_list(mse_g)
                    psnr_m_list = self._tensor_to_list(psnr_m_g)
                    mse_m_list = self._tensor_to_list(mse_m_g)
                    mask_cov_list = self._tensor_to_list(mask_cov_g)
                name_idx_list = self._tensor_to_list(name_idx_g)
                valid_list = self._tensor_to_list(valid_g)

                # Accumulate only real, non-padded samples.
                if use_multicam_metrics:
                    for is_valid, m_ss, m_p, m_lp, n_i in zip(
                        valid_list, ssim_list, psnr_list, lpips_list, name_idx_list
                    ):
                        if not bool(is_valid):
                            continue
                        all_ssim.append(m_ss)
                        all_psnr.append(m_p)
                        all_lpips.append(m_lp)
                        ds_name = idx_to_name.get(int(n_i), "unknown")
                        if ds_name not in per_dataset:
                            per_dataset[ds_name] = {
                                "ssim": [],
                                "psnr": [],
                                "lpips": [],
                                "input_ssim": [],
                                "input_psnr": [],
                                "input_lpips": [],
                            }
                        per_dataset[ds_name]["ssim"].append(m_ss)
                        per_dataset[ds_name]["psnr"].append(m_p)
                        per_dataset[ds_name]["lpips"].append(m_lp)
                    for is_valid, m_ss, m_p, m_lp, n_i in zip(
                        input_valid_list,
                        input_ssim_list,
                        input_psnr_list,
                        input_lpips_list,
                        name_idx_list,
                    ):
                        if not bool(is_valid):
                            continue
                        all_input_ssim.append(m_ss)
                        all_input_psnr.append(m_p)
                        all_input_lpips.append(m_lp)
                        ds_name = idx_to_name.get(int(n_i), "unknown")
                        if ds_name not in per_dataset:
                            per_dataset[ds_name] = {
                                "ssim": [],
                                "psnr": [],
                                "lpips": [],
                                "input_ssim": [],
                                "input_psnr": [],
                                "input_lpips": [],
                            }
                        per_dataset[ds_name]["input_ssim"].append(m_ss)
                        per_dataset[ds_name]["input_psnr"].append(m_p)
                        per_dataset[ds_name]["input_lpips"].append(m_lp)
                else:
                    for is_valid, m_ss, m_p, m_lp, m_m, m_pm, m_mm, m_cov, n_i in zip(
                        valid_list,
                        msssim_list,
                        psnr_list,
                        lpips_list,
                        mse_list,
                        psnr_m_list,
                        mse_m_list,
                        mask_cov_list,
                        name_idx_list,
                    ):
                        if not bool(is_valid):
                            continue
                        all_msssim.append(m_ss)
                        all_psnr.append(m_p)
                        all_lpips.append(m_lp)
                        all_mse.append(m_m)
                        all_psnr_masked.append(m_pm)
                        all_mse_masked.append(m_mm)
                        all_mask_coverage.append(m_cov)
                        ds_name = idx_to_name.get(int(n_i), "unknown")
                        if ds_name not in per_dataset:
                            per_dataset[ds_name] = {
                                "msssim": [],
                                "psnr": [],
                                "lpips": [],
                                "mse": [],
                                "psnr_masked": [],
                                "mse_masked": [],
                                "mask_coverage": [],
                            }
                        per_dataset[ds_name]["msssim"].append(m_ss)
                        per_dataset[ds_name]["psnr"].append(m_p)
                        per_dataset[ds_name]["lpips"].append(m_lp)
                        per_dataset[ds_name]["mse"].append(m_m)
                        per_dataset[ds_name]["psnr_masked"].append(m_pm)
                        per_dataset[ds_name]["mse_masked"].append(m_mm)
                        per_dataset[ds_name]["mask_coverage"].append(m_cov)

        if should_stage_videos and logger is not None:
            logger.flush_staged_eval_videos(
                accelerator,
                global_step,
                max_logs=max_logs,
            )

        # Log aggregate metrics
        if accelerator.is_main_process:
            if use_multicam_metrics:
                if len(all_ssim) > 0:
                    avg_ssim = sum(all_ssim) / len(all_ssim)
                    avg_psnr = sum(all_psnr) / len(all_psnr)
                    avg_lpips = sum(all_lpips) / len(all_lpips)
                    log_payload = {
                        "eval/ssim_rgb": avg_ssim,
                        "eval/psnr_rgb": avg_psnr,
                        "eval/lpips_rgb": avg_lpips,
                    }
                    if len(all_input_ssim) > 0:
                        log_payload.update(
                            {
                                "eval/input_ssim_rgb": sum(all_input_ssim) / len(all_input_ssim),
                                "eval/input_psnr_rgb": sum(all_input_psnr) / len(all_input_psnr),
                                "eval/input_lpips_rgb": sum(all_input_lpips) / len(all_input_lpips),
                            }
                        )
                    accelerator.log(log_payload, step=global_step)
                    accelerator.print(
                        f"[Step {global_step}] Pred Val SSIM: {avg_ssim:.4f}, "
                        f"PSNR: {avg_psnr:.2f}, LPIPS: {avg_lpips:.4f} "
                        f"({len(all_ssim)} samples)"
                    )
                    if len(all_input_ssim) > 0:
                        accelerator.print(
                            f"[Step {global_step}] Input Val SSIM: "
                            f"{sum(all_input_ssim)/len(all_input_ssim):.4f}, "
                            f"PSNR: {sum(all_input_psnr)/len(all_input_psnr):.2f}, "
                            f"LPIPS: {sum(all_input_lpips)/len(all_input_lpips):.4f} "
                            f"({len(all_input_ssim)} samples)"
                        )

                    for ds_name, metrics in sorted(per_dataset.items()):
                        if len(metrics["ssim"]) == 0:
                            continue
                        n = len(metrics["ssim"])
                        avg_ss = sum(metrics["ssim"]) / n
                        avg_p = sum(metrics["psnr"]) / n
                        avg_lp = sum(metrics["lpips"]) / n
                        ds_payload = {
                            f"eval/ssim_rgb/{ds_name}": avg_ss,
                            f"eval/psnr_rgb/{ds_name}": avg_p,
                            f"eval/lpips_rgb/{ds_name}": avg_lp,
                        }
                        if len(metrics["input_ssim"]) > 0:
                            ds_payload.update(
                                {
                                    f"eval/input_ssim_rgb/{ds_name}": (
                                        sum(metrics["input_ssim"]) / len(metrics["input_ssim"])
                                    ),
                                    f"eval/input_psnr_rgb/{ds_name}": (
                                        sum(metrics["input_psnr"]) / len(metrics["input_psnr"])
                                    ),
                                    f"eval/input_lpips_rgb/{ds_name}": (
                                        sum(metrics["input_lpips"]) / len(metrics["input_lpips"])
                                    ),
                                }
                            )
                        accelerator.log(ds_payload, step=global_step)
                        accelerator.print(
                            f"  [pred:{ds_name}] SSIM: {avg_ss:.4f}, PSNR: {avg_p:.2f}, "
                            f"LPIPS: {avg_lp:.4f} ({n} samples)"
                        )
                        if len(metrics["input_ssim"]) > 0:
                            input_n = len(metrics["input_ssim"])
                            accelerator.print(
                                f"  [input:{ds_name}] SSIM: "
                                f"{sum(metrics['input_ssim'])/input_n:.4f}, "
                                f"PSNR: {sum(metrics['input_psnr'])/input_n:.2f}, "
                                f"LPIPS: {sum(metrics['input_lpips'])/input_n:.4f} "
                                f"({input_n} samples)"
                            )
                else:
                    accelerator.print("Validation data is empty, skipping metrics logging.")
            else:
                if len(all_msssim) > 0:
                    avg_lpips = sum(all_lpips) / len(all_lpips)
                    avg_psnr_m = sum(all_psnr_masked) / len(all_psnr_masked)
                    avg_mse_m = sum(all_mse_masked) / len(all_mse_masked)
                    avg_cov = sum(all_mask_coverage) / len(all_mask_coverage)
                    accelerator.log({
                        "eval/ssim_rgb": sum(all_msssim) / len(all_msssim),
                        "eval/psnr_rgb": sum(all_psnr) / len(all_psnr),
                        "eval/lpips_rgb": avg_lpips,
                        "eval/mse_rgb": sum(all_mse) / len(all_mse),
                        "eval/psnr_rgb_masked": avg_psnr_m,
                        "eval/mse_rgb_masked": avg_mse_m,
                        "eval/mask_coverage": avg_cov,
                    }, step=global_step)
                    accelerator.print(
                        f"[Step {global_step}] Val MS-SSIM: {sum(all_msssim)/len(all_msssim):.4f}, "
                        f"PSNR: {sum(all_psnr)/len(all_psnr):.2f}, "
                        f"LPIPS: {avg_lpips:.4f}, MSE: {sum(all_mse)/len(all_mse):.6f} | "
                        f"Masked PSNR: {avg_psnr_m:.2f}, Masked MSE: {avg_mse_m:.6f}, Coverage: {avg_cov:.2%} "
                        f"({len(all_msssim)} samples)"
                    )

                    # Log per-dataset metrics
                    for ds_name, metrics in sorted(per_dataset.items()):
                        if len(metrics["msssim"]) == 0:
                            continue
                        n = len(metrics["msssim"])
                        avg_ss = sum(metrics["msssim"]) / n
                        avg_p = sum(metrics["psnr"]) / n
                        avg_lp = sum(metrics["lpips"]) / n
                        avg_m = sum(metrics["mse"]) / n
                        avg_pm = sum(metrics["psnr_masked"]) / n
                        avg_mm = sum(metrics["mse_masked"]) / n
                        avg_mc = sum(metrics["mask_coverage"]) / n
                        accelerator.log({
                            f"eval/ssim_rgb/{ds_name}": avg_ss,
                            f"eval/psnr_rgb/{ds_name}": avg_p,
                            f"eval/lpips_rgb/{ds_name}": avg_lp,
                            f"eval/mse_rgb/{ds_name}": avg_m,
                            f"eval/psnr_rgb_masked/{ds_name}": avg_pm,
                            f"eval/mse_rgb_masked/{ds_name}": avg_mm,
                            f"eval/mask_coverage/{ds_name}": avg_mc,
                        }, step=global_step)
                        accelerator.print(
                            f"  [{ds_name}] MS-SSIM: {avg_ss:.4f}, PSNR: {avg_p:.2f}, "
                            f"LPIPS: {avg_lp:.4f}, MSE: {avg_m:.6f} | "
                            f"Masked PSNR: {avg_pm:.2f}, Masked MSE: {avg_mm:.6f}, Cov: {avg_mc:.2%} ({n} samples)"
                        )
                else:
                    accelerator.print("Validation data is empty, skipping metrics logging.")

        # Restore training-only pipeline units.
        self.pipe.units = saved_units


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--split_cam_embed", default=False, action="store_true", help="Use separate source/target pose embeddings with shared cam_encoder.")
    parser.add_argument("--zero_source_cam_emb_when_missing", default=False, action="store_true", help="When source_video is enabled without split source pose embeddings, zero the source-half camera embedding instead of duplicating the target pose embedding.")
    parser.add_argument("--scene_coord_pe", default=False, action="store_true", help="Enable scene-coordinate positional encoding conditioning.")
    parser.add_argument("--scene_coord_xyzt_pe", default=False, action="store_true", help="Enable XYZT scene-coordinate positional encoding conditioning (adds a temporal channel).")
    parser.add_argument("--scene_coord_gated_residual", default=False, action="store_true", help="Apply scene-coordinate PE as a gated residual before self-attention projections.")
    parser.add_argument("--scene_coord_gate_init", type=float, default=0.0, help="Initial value for gated scene-coordinate residual parameters when new coord_gate tensors are created.")
    parser.add_argument(
        "--drop_cached_vace_groups",
        type=str,
        default=None,
        help=(
            "Comma-separated compact cached VACE groups to drop at load time when "
            "training from cache. Supported: video,depth,normal,alpha,scene_flow."
        ),
    )
    parser.add_argument(
        "--scene_coord_worker_cache_maxsize",
        type=int,
        default=_SCENE_COORD_WORKER_CACHE_MAXSIZE,
        help="Per-worker scene-coordinate tensor cache size used when augmenting cached training samples.",
    )
    parser.add_argument(
        "--scene_coord_worker_num_workers",
        type=int,
        default=_SCENE_COORD_WORKER_NUM_WORKERS,
        help="Max DataLoader workers to use for cached scene-coordinate loading when scene-coordinate PE is enabled.",
    )
    parser.add_argument(
        "--scene_coord_worker_prefetch_factor",
        type=int,
        default=_SCENE_COORD_WORKER_PREFETCH_FACTOR,
        help="Prefetch factor for cached scene-coordinate DataLoader workers when worker loading is enabled.",
    )
    parser.add_argument("--strip_source_gbuffer", default=False, action="store_true", help="Truncate cached vace_context to target-only frames (drop source gbuffer half).")
    parser.add_argument(
        "--target_gbuffer_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of zeroing the full target half of cached vace_context, including target RGB, to train target-gbuffer CFG.",
    )
    parser.add_argument(
        "--target_gbuffer_dropout_scope",
        type=str,
        choices=("target_only", "full_context"),
        default="target_only",
        help="Scope of target-gbuffer unconditional conditioning. 'target_only' affects only the target half; 'full_context' affects both target and source cached VACE context.",
    )
    parser.add_argument(
        "--target_gbuffer_uncond_fill_mode",
        type=str,
        choices=("zero", "learned"),
        default="zero",
        help="How to fill dropped cached VACE context. 'zero' keeps the existing behavior; 'learned' uses a learned non-zero unconditional VACE latent.",
    )
    parser.add_argument(
        "--shared_gbuffer_dropout_prob",
        type=float,
        default=0.0,
        help="Per-group dropout probability for cached VACE auxiliary groups, shared across target/source views.",
    )
    parser.add_argument(
        "--shared_gbuffer_rgb_only_prob",
        type=float,
        default=0.0,
        help="Probability of zeroing all configured non-RGB cached VACE auxiliary groups together, leaving RGB latents active.",
    )
    parser.add_argument(
        "--shared_gbuffer_dropout_groups",
        type=str,
        default="depth,normal,alpha,scene_flow",
        help="Comma-separated cached VACE auxiliary groups eligible for shared dropout. Supported: depth,normal,alpha,scene_flow.",
    )
    parser.add_argument("--eval_height", type=int, default=None, help="Optional evaluation-only height override. Must be used with --eval_width.")
    parser.add_argument("--eval_width", type=int, default=None, help="Optional evaluation-only width override. Must be used with --eval_height.")
    parser.add_argument(
        "--eval_dataset_resolutions",
        type=str,
        default=None,
        help="Optional per-dataset eval resolution overrides. Format: dataset_name:HxW,dataset_name2:HxW",
    )
    parser.add_argument(
        "--eval_target_gbuffer_cfg_scale",
        type=float,
        default=1.0,
        help="Target-gbuffer guidance scale used by evaluate() and DyCheck benchmark renders. Default: 1.0",
    )
    parser.add_argument(
        "--dycheck_benchmark_root",
        type=str,
        default=None,
        help=(
            "Optional DyCheck/WT benchmark root. When unset, the additive "
            "DyCheck benchmark path is disabled and existing jobs are unchanged."
        ),
    )
    parser.add_argument(
        "--dycheck_benchmark_data_root",
        type=str,
        default=None,
        help=(
            "Optional DyCheck benchmark data root for covisible masks. "
            "Defaults to --dycheck_benchmark_root when unset."
        ),
    )
    parser.add_argument(
        "--dycheck_benchmark_seq_names",
        type=str,
        default=None,
        help="Optional comma-separated DyCheck benchmark scene names to evaluate.",
    )
    parser.add_argument(
        "--dycheck_benchmark_chunk_size",
        type=int,
        default=73,
        help="Chunk size for additive DyCheck benchmark inference.",
    )
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument(
        "--initial_trainable_checkpoint",
        type=str,
        default=None,
        help="Optional exported trainable checkpoint (.safetensors) used to warm-start weights before optimizer creation.",
    )
    parser.add_argument(
        "--initial_global_step",
        type=int,
        default=None,
        help="Optional global-step seed for weight-only continuation when no local resume checkpoint exists.",
    )
    return parser


def _normalize_optional_cli_path(path_value):
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "":
        return None
    if text.upper() in {"NONE", "NULL", "__NONE__"}:
        return None
    return text


def _load_initial_trainable_checkpoint(pipe, ckpt_path: str, verbose: bool = True):
    """Warm-start trainable weights from an exported training checkpoint."""
    ckpt = load_state_dict(ckpt_path, torch_dtype=torch.bfloat16)

    # Training exports use remove_prefix="pipe.vace.":
    # - plain keys belong under "vace."
    # - keys still starting with "pipe." should drop that prefix.
    pipe_state = {}
    for key, value in ckpt.items():
        if key.startswith("pipe."):
            pipe_state[key[len("pipe."):]] = value
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
        print(f"[Checkpoint] Loaded {loaded} params from {ckpt_path}")
        if unexpected:
            print(
                f"[Checkpoint] {len(unexpected)} unexpected keys "
                f"(first 5): {unexpected[:5]}"
            )

    return loaded, len(unexpected)


def _parse_eval_dataset_resolutions(raw_value, parser):
    if raw_value is None:
        return {}
    text = str(raw_value).strip()
    if text == "":
        return {}

    overrides = {}
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        name_and_dims = item.split(":", 1)
        if len(name_and_dims) != 2:
            parser.error(
                "Invalid --eval_dataset_resolutions entry: "
                f"'{item}'. Expected format dataset_name:HxW"
            )
        name = name_and_dims[0].strip().lower()
        dims_text = name_and_dims[1].strip().lower().replace(" ", "")
        dim_parts = dims_text.split("x", 1)
        if len(dim_parts) != 2 or not dim_parts[0].isdigit() or not dim_parts[1].isdigit():
            parser.error(
                "Invalid --eval_dataset_resolutions entry: "
                f"'{item}'. Expected format dataset_name:HxW"
            )
        height = int(dim_parts[0])
        width = int(dim_parts[1])
        if name == "":
            parser.error(
                f"Invalid --eval_dataset_resolutions entry: '{item}'. Dataset name cannot be empty."
            )
        if height <= 0 or width <= 0:
            parser.error(
                f"Invalid --eval_dataset_resolutions entry: '{item}'. Height/width must be > 0."
            )
        overrides[name] = (height, width)
    return overrides


def _parse_optional_name_list(raw_value):
    if raw_value is None:
        return ()
    text = str(raw_value).strip()
    if text == "":
        return ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _build_wan_dataset(args, base_path, metadata_path, repeat, dynamic_resolution=False):
    data_file_keys = [k.strip() for k in args.data_file_keys.split(",") if k.strip()]
    extra_inputs = {k.strip() for k in (args.extra_inputs or "").split(",") if k.strip()}
    configured_drop_cached_vace_groups = _parse_compact_cached_vace_groups(
        args.drop_cached_vace_groups
    )
    # Loader-time VACE trimming only applies to cached tuples. Raw/metadata-backed
    # datasets already control conditioning through data_file_keys/extra_inputs.
    drop_cached_vace_groups = (
        configured_drop_cached_vace_groups if metadata_path is None else tuple()
    )
    optional_video_camera_keys = {
        "source_video", "cameras_npz", "target_cam", "source_cam",
        "source_vace_video", "source_vace_depth", "source_vace_normal", "source_vace_alpha",
        "scene_flow_target", "scene_flow_source",
    }
    for key in optional_video_camera_keys:
        if key in extra_inputs and key not in data_file_keys:
            data_file_keys.append(key)

    camera_base_path = "" if base_path is None else base_path
    scene_coord_requested = bool(args.scene_coord_pe or args.scene_coord_xyzt_pe)
    cached_data_operator = None
    for group in drop_cached_vace_groups:
        if any(key in extra_inputs for key in _COMPACT_CACHED_VACE_GROUP_TO_INPUT_KEYS[group]):
            raise ValueError(
                f"Cannot drop cached VACE group '{group}' while --extra_inputs still request "
                f"{_COMPACT_CACHED_VACE_GROUP_TO_INPUT_KEYS[group]}."
            )
    if metadata_path is None and (scene_coord_requested or drop_cached_vace_groups):
        cached_data_operator = LoadCachedWanTuple(
            load_scene_coords=scene_coord_requested,
            scene_coord_xyzt=args.scene_coord_xyzt_pe,
            remove_cached_keys=args.remove_cached_keys,
            drop_cached_vace_groups=drop_cached_vace_groups,
            scene_coord_cache_maxsize=args.scene_coord_worker_cache_maxsize,
        )
        if drop_cached_vace_groups:
            print(
                "[CacheFilter] Dropping compact cached VACE groups at load time: "
                f"{', '.join(drop_cached_vace_groups)}."
            )

    dataset = UnifiedDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        repeat=repeat,
        data_file_keys=data_file_keys,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=base_path,
            max_pixels=(1 << 62) if dynamic_resolution else args.max_pixels,
            height=None if dynamic_resolution else args.height,
            width=None if dynamic_resolution else args.width,
            height_division_factor=1 if dynamic_resolution else 16,
            width_division_factor=1 if dynamic_resolution else 16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(base_path) >> LoadAudio(sr=16000),
            "cameras_npz": ToAbsolutePath(camera_base_path),
            "target_cam": ToStr(),
            "source_cam": ToStr(),
        },
        cached_data_operator=cached_data_operator,
    )
    if metadata_path is None and scene_coord_requested:
        requested_dataset_workers = max(0, int(args.dataset_num_workers))
        scene_coord_worker_limit = max(0, int(args.scene_coord_worker_num_workers))
        scene_coord_prefetch_factor = max(1, int(args.scene_coord_worker_prefetch_factor))
        effective_scene_coord_workers = min(requested_dataset_workers, scene_coord_worker_limit)
        if requested_dataset_workers > 0:
            setattr(dataset, "dataloader_num_workers", effective_scene_coord_workers)
            if effective_scene_coord_workers > 0:
                setattr(dataset, "dataloader_prefetch_factor", scene_coord_prefetch_factor)
                setattr(dataset, "dataloader_persistent_workers", True)
        print(
            "[SceneCoord] Cached loader tuning: "
            f"workers={effective_scene_coord_workers} "
            f"(requested_dataset_workers={requested_dataset_workers}, limit={scene_coord_worker_limit}), "
            f"prefetch={scene_coord_prefetch_factor}, "
            f"worker_cache={max(0, int(args.scene_coord_worker_cache_maxsize))}."
        )
    return dataset


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    if (args.eval_height is None) != (args.eval_width is None):
        parser.error("--eval_height and --eval_width must be set together.")
    args.eval_dataset_resolutions = _parse_eval_dataset_resolutions(
        args.eval_dataset_resolutions, parser
    )
    eval_resolution_overrides_enabled = (
        bool(args.eval_dataset_resolutions)
        or (args.eval_height is not None and args.eval_width is not None)
    )
    args.dataset_metadata_path = _normalize_optional_cli_path(args.dataset_metadata_path)
    args.eval_dataset_base_path = _normalize_optional_cli_path(args.eval_dataset_base_path)
    args.eval_dataset_metadata_path = _normalize_optional_cli_path(args.eval_dataset_metadata_path)
    args.dycheck_benchmark_root = _normalize_optional_cli_path(args.dycheck_benchmark_root)
    args.dycheck_benchmark_data_root = _normalize_optional_cli_path(args.dycheck_benchmark_data_root)
    args.dycheck_benchmark_seq_names = _parse_optional_name_list(args.dycheck_benchmark_seq_names)
    args.initial_trainable_checkpoint = _normalize_optional_cli_path(args.initial_trainable_checkpoint)
    if args.dycheck_benchmark_root is None:
        if args.dycheck_benchmark_data_root is not None:
            parser.error(
                "--dycheck_benchmark_data_root requires --dycheck_benchmark_root."
            )
        if len(args.dycheck_benchmark_seq_names) > 0:
            parser.error(
                "--dycheck_benchmark_seq_names requires --dycheck_benchmark_root."
            )
    if args.dycheck_benchmark_root is not None and not os.path.exists(args.dycheck_benchmark_root):
        parser.error(
            "--dycheck_benchmark_root does not exist: "
            f"{args.dycheck_benchmark_root}"
        )
    if (
        args.dycheck_benchmark_data_root is not None
        and not os.path.exists(args.dycheck_benchmark_data_root)
    ):
        parser.error(
            "--dycheck_benchmark_data_root does not exist: "
            f"{args.dycheck_benchmark_data_root}"
        )
    if args.dycheck_benchmark_chunk_size < 1:
        parser.error("--dycheck_benchmark_chunk_size must be >= 1.")
    if not 0.0 <= args.target_gbuffer_dropout_prob <= 1.0:
        parser.error("--target_gbuffer_dropout_prob must be in [0, 1].")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        parser.error("--max_train_steps must be > 0.")
    if args.eval_target_gbuffer_cfg_scale < 0.0:
        parser.error("--eval_target_gbuffer_cfg_scale must be >= 0.")
    if (
        args.initial_trainable_checkpoint is not None
        and not os.path.exists(args.initial_trainable_checkpoint)
    ):
        parser.error(
            "--initial_trainable_checkpoint does not exist: "
            f"{args.initial_trainable_checkpoint}"
        )
    if args.initial_global_step is not None and args.initial_global_step < 0:
        parser.error("--initial_global_step must be >= 0.")
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        log_with="wandb" if args.wandb else None,
        project_dir=args.output_path if args.wandb else None,
    )
    dataset = _build_wan_dataset(
        args,
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
    )

    eval_dataset = None
    if args.eval_dataset_metadata_path is not None:
        eval_dataset_base_path = args.eval_dataset_base_path or args.dataset_base_path
        eval_dataset = _build_wan_dataset(
            args,
            base_path=eval_dataset_base_path,
            metadata_path=args.eval_dataset_metadata_path,
            repeat=1,
            dynamic_resolution=eval_resolution_overrides_enabled,
        )
        accelerator.print(
            "Using separate raw evaluation dataset: "
            f"base_path={eval_dataset_base_path}, metadata_path={args.eval_dataset_metadata_path}"
        )
        if eval_resolution_overrides_enabled:
            accelerator.print(
                "Evaluation resolution overrides enabled: using dynamic/raw frame loading "
                "for eval dataset before applying per-dataset resize."
            )
    elif eval_resolution_overrides_enabled:
        accelerator.print(
            "Warning: evaluation resolution overrides are enabled, but no separate eval dataset "
            "is configured. Validation samples from train/val split may already be center-cropped "
            "by training data loader settings."
        )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        remove_cached_keys=args.remove_cached_keys,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        split_cam_embed=args.split_cam_embed,
        zero_source_cam_emb_when_missing=args.zero_source_cam_emb_when_missing,
        strip_source_gbuffer=args.strip_source_gbuffer,
        target_gbuffer_dropout_prob=args.target_gbuffer_dropout_prob,
        target_gbuffer_dropout_scope=args.target_gbuffer_dropout_scope,
        target_gbuffer_uncond_fill_mode=args.target_gbuffer_uncond_fill_mode,
        shared_gbuffer_dropout_prob=args.shared_gbuffer_dropout_prob,
        shared_gbuffer_rgb_only_prob=args.shared_gbuffer_rgb_only_prob,
        shared_gbuffer_dropout_groups=args.shared_gbuffer_dropout_groups,
        scene_coord_pe=args.scene_coord_pe,
        scene_coord_xyzt_pe=args.scene_coord_xyzt_pe,
        scene_coord_gated_residual=args.scene_coord_gated_residual,
        scene_coord_gate_init=args.scene_coord_gate_init,
        eval_height=args.eval_height,
        eval_width=args.eval_width,
        eval_dataset_resolutions=args.eval_dataset_resolutions,
        eval_num_inference_steps=args.eval_num_inference_steps,
        eval_target_gbuffer_cfg_scale=args.eval_target_gbuffer_cfg_scale,
        dycheck_benchmark_root=args.dycheck_benchmark_root,
        dycheck_benchmark_data_root=args.dycheck_benchmark_data_root,
        dycheck_benchmark_seq_names=args.dycheck_benchmark_seq_names,
        dycheck_benchmark_chunk_size=args.dycheck_benchmark_chunk_size,
    )
    if args.initial_trainable_checkpoint is not None:
        accelerator.print(
            "Warm-starting trainable weights from "
            f"{args.initial_trainable_checkpoint}"
        )
        _load_initial_trainable_checkpoint(
            model.pipe,
            args.initial_trainable_checkpoint,
            verbose=accelerator.is_main_process,
        )
        accelerator.wait_for_everyone()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher = launcher_map[args.task]
    if args.task.endswith(":data_process"):
        launcher(accelerator, dataset, model, model_logger, args=args)
    else:
        launcher(accelerator, dataset, model, model_logger, args=args, eval_dataset=eval_dataset)
