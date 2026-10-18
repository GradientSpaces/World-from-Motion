import math
import torch, types, os, re, threading
import numpy as np
from collections import OrderedDict
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from transformers import Wav2Vec2Processor

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.wav2vec import WanS2VAudioEncoder
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel


_TARGET_GBUFFER_DROPOUT_SCOPES = ("target_only", "full_context")
_TARGET_GBUFFER_UNCOND_FILL_MODES = ("zero", "learned")


def _guidance_scale_enabled(scale, eps=1e-6):
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return False
    return not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=eps)


def _resolve_wan_guidance_scale(cfg_scale, target_gbuffer_cfg_scale):
    cfg_enabled = _guidance_scale_enabled(cfg_scale)
    target_enabled = _guidance_scale_enabled(target_gbuffer_cfg_scale)
    if cfg_enabled and target_enabled:
        raise ValueError(
            "Simultaneous prompt cfg_scale and target_gbuffer_cfg_scale are not supported yet."
        )
    if target_enabled:
        return float(target_gbuffer_cfg_scale), True
    return float(cfg_scale), False


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


def _normalize_target_gbuffer_dropout_scope(scope):
    scope = "target_only" if scope is None else str(scope).strip().lower()
    if scope not in _TARGET_GBUFFER_DROPOUT_SCOPES:
        raise ValueError(
            "Unsupported target_gbuffer_dropout_scope "
            f"'{scope}'. Expected one of: {', '.join(_TARGET_GBUFFER_DROPOUT_SCOPES)}."
        )
    return scope


def _normalize_target_gbuffer_uncond_fill_mode(mode):
    mode = "zero" if mode is None else str(mode).strip().lower()
    if mode not in _TARGET_GBUFFER_UNCOND_FILL_MODES:
        raise ValueError(
            "Unsupported target_gbuffer_uncond_fill_mode "
            f"'{mode}'. Expected one of: {', '.join(_TARGET_GBUFFER_UNCOND_FILL_MODES)}."
        )
    return mode


def _resolve_target_gbuffer_fill_info(
    vace_context,
    source_latents=None,
    dropout_scope="target_only",
):
    if not isinstance(vace_context, torch.Tensor) or vace_context.ndim != 5:
        return None
    total_frames = int(vace_context.shape[2])
    if total_frames <= 0:
        return None
    dropout_scope = _normalize_target_gbuffer_dropout_scope(dropout_scope)
    target_frames = _infer_target_vace_frame_count(
        vace_context, source_latents=source_latents
    )
    if target_frames is None:
        return None
    fill_frames = (
        total_frames if dropout_scope == "full_context" else int(target_frames)
    )
    if fill_frames <= 0:
        return None
    return {
        "dropout_scope": dropout_scope,
        "frame_start": 0,
        "frame_end": int(fill_frames),
        "fill_frames": int(fill_frames),
        "target_frames": int(target_frames),
        "total_frames": int(total_frames),
    }


def _apply_target_gbuffer_uncond_fill(
    vace_context,
    *,
    source_latents=None,
    dropout_scope="target_only",
    fill_mode="zero",
    vace_model: VaceWanModel | None = None,
    batch_mask=None,
):
    fill_info = _resolve_target_gbuffer_fill_info(
        vace_context,
        source_latents=source_latents,
        dropout_scope=dropout_scope,
    )
    if fill_info is None:
        return None
    fill_mode = _normalize_target_gbuffer_uncond_fill_mode(fill_mode)
    frame_start = int(fill_info["frame_start"])
    frame_end = int(fill_info["frame_end"])
    if batch_mask is None:
        batch_indices = None
        batch_size = int(vace_context.shape[0])
    else:
        batch_mask = torch.as_tensor(
            batch_mask,
            device=vace_context.device,
            dtype=torch.bool,
        ).view(-1)
        if batch_mask.numel() != int(vace_context.shape[0]):
            raise ValueError(
                "target-gbuffer batch mask size mismatch: "
                f"expected {int(vace_context.shape[0])}, got {batch_mask.numel()}."
            )
        batch_indices = torch.nonzero(batch_mask, as_tuple=False).flatten()
        batch_size = int(batch_indices.numel())
        if batch_size == 0:
            return fill_info

    if fill_mode == "zero":
        if batch_indices is None:
            vace_context[:, :, frame_start:frame_end, :, :] = 0
        else:
            vace_context[batch_indices, :, frame_start:frame_end, :, :] = 0
        return fill_info

    if vace_model is None:
        raise ValueError(
            "target_gbuffer_uncond_fill_mode='learned' requires a VACE model."
        )
    fill_value = vace_model.get_uncond_vace_context(
        fill_info["dropout_scope"],
        dtype=vace_context.dtype,
        device=vace_context.device,
    )
    expanded_fill = fill_value.expand(
        batch_size,
        vace_context.shape[1],
        frame_end - frame_start,
        vace_context.shape[3],
        vace_context.shape[4],
    )
    if batch_indices is None:
        vace_context[:, :, frame_start:frame_end, :, :] = expanded_fill
    else:
        vace_context[batch_indices, :, frame_start:frame_end, :, :] = expanded_fill
    return fill_info


def _build_target_gbuffer_dropped_vace_context(
    vace_context,
    source_latents=None,
    dropout_scope="target_only",
    fill_mode="zero",
    vace_model: VaceWanModel | None = None,
):
    if not isinstance(vace_context, torch.Tensor) or vace_context.ndim != 5:
        return None
    dropped_vace_context = vace_context.clone()
    fill_info = _apply_target_gbuffer_uncond_fill(
        dropped_vace_context,
        source_latents=source_latents,
        dropout_scope=dropout_scope,
        fill_mode=fill_mode,
        vace_model=vace_model,
    )
    if fill_info is None:
        return None
    return dropped_vace_context


def _normalize_target_gbuffer_guidance_mode(mode):
    mode = "cfg" if mode is None else str(mode).strip().lower()
    if mode not in {"cfg", "apg"}:
        raise ValueError(
            "Unsupported target_gbuffer_guidance_mode "
            f"'{mode}'. Expected one of: cfg, apg."
        )
    return mode


def _normalize_target_gbuffer_apg_space(space):
    space = "velocity" if space is None else str(space).strip().lower()
    aliases = {
        "v": "velocity",
        "velocity": "velocity",
        "native": "velocity",
        "model_output": "velocity",
        "x0": "x0",
        "x_0": "x0",
        "denoised": "x0",
    }
    normalized = aliases.get(space)
    if normalized is None:
        raise ValueError(
            "Unsupported target_gbuffer_apg_space "
            f"'{space}'. Expected one of: velocity, x0."
        )
    return normalized


DEFAULT_TARGET_GBUFFER_APG_CFG_SCALE = 3.4
DEFAULT_TARGET_GBUFFER_APG_ETA = 0.0
DEFAULT_TARGET_GBUFFER_APG_NORM_THRESHOLD = 8.0
DEFAULT_TARGET_GBUFFER_APG_MOMENTUM = -0.5
DEFAULT_TARGET_GBUFFER_APG_SPACE = "x0"


def _guidance_reduce_dims(x: torch.Tensor):
    return tuple(range(1, x.ndim))


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = float(momentum)
        self.running_average: Optional[torch.Tensor] = None

    def update(self, update_value: torch.Tensor):
        if self.running_average is None:
            self.running_average = update_value.clone()
        else:
            self.running_average = update_value + self.momentum * self.running_average


def _project_guidance_update(
    v0: torch.Tensor,
    v1: torch.Tensor,
    eps: float = 1e-12,
):
    dims = _guidance_reduce_dims(v0)
    orig_dtype = v0.dtype
    compute_dtype = (
        torch.float32 if v0.dtype in (torch.float16, torch.bfloat16) else v0.dtype
    )

    v0_f = v0.to(compute_dtype)
    v1_f = v1.to(compute_dtype)

    v1_norm = torch.linalg.vector_norm(
        v1_f, ord=2, dim=dims, keepdim=True
    ).clamp_min(eps)
    v1_unit = v1_f / v1_norm

    v0_parallel = (v0_f * v1_unit).sum(dim=dims, keepdim=True) * v1_unit
    v0_orthogonal = v0_f - v0_parallel
    return v0_parallel.to(orig_dtype), v0_orthogonal.to(orig_dtype)


def _apply_adaptive_projected_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
):
    dims = _guidance_reduce_dims(pred_cond)
    diff = pred_cond - pred_uncond

    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0.0:
        diff_f = diff.float()
        diff_norm = torch.linalg.vector_norm(
            diff_f, ord=2, dim=dims, keepdim=True
        ).clamp_min(1e-12)
        scale = torch.minimum(
            torch.ones_like(diff_norm),
            diff_norm.new_full(diff_norm.shape, float(norm_threshold)) / diff_norm,
        )
        diff = (diff_f * scale).to(diff.dtype)

    diff_parallel, diff_orthogonal = _project_guidance_update(diff, pred_cond)
    normalized_update = diff_orthogonal + float(eta) * diff_parallel
    return pred_cond + (float(guidance_scale) - 1.0) * normalized_update


def _combine_wan_guidance_predictions(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    *,
    use_target_gbuffer_apg: bool = False,
    target_gbuffer_apg_buffer: MomentumBuffer = None,
    target_gbuffer_apg_eta: float = 0.0,
    target_gbuffer_apg_norm_threshold: float = 2.5,
    target_gbuffer_apg_space: str = "velocity",
    scheduler: FlowMatchScheduler | None = None,
    timestep: torch.Tensor | None = None,
    sample: torch.Tensor | None = None,
):
    if use_target_gbuffer_apg:
        if target_gbuffer_apg_space == "x0":
            if scheduler is None or timestep is None or sample is None:
                raise ValueError(
                    "scheduler, timestep, and sample are required for target-gbuffer "
                    "APG in denoised x0 space."
                )
            pred_cond = scheduler.model_output_to_denoised(
                pred_cond,
                timestep,
                sample,
            )
            pred_uncond = scheduler.model_output_to_denoised(
                pred_uncond,
                timestep,
                sample,
            )
            guided_x0 = _apply_adaptive_projected_guidance(
                pred_cond=pred_cond,
                pred_uncond=pred_uncond,
                guidance_scale=guidance_scale,
                momentum_buffer=target_gbuffer_apg_buffer,
                eta=target_gbuffer_apg_eta,
                norm_threshold=target_gbuffer_apg_norm_threshold,
            )
            return scheduler.denoised_to_model_output(
                guided_x0,
                timestep,
                sample,
            )
        return _apply_adaptive_projected_guidance(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guidance_scale=guidance_scale,
            momentum_buffer=target_gbuffer_apg_buffer,
            eta=target_gbuffer_apg_eta,
            norm_threshold=target_gbuffer_apg_norm_threshold,
        )
    return pred_uncond + float(guidance_scale) * (pred_cond - pred_uncond)


class WanVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_CameraPose(),
            WanVideoUnit_SceneCoordLoader(),
            WanVideoUnit_SourceVideo(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_TargetGBufferCFG(),
            WanVideoUnit_AnimateVideoSplit(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_VAP(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path (only for ModelScope; the redirect repo doesn't exist on Hugging Face)
        if redirect_common_files:
            # Check if any model config uses huggingface as download source
            _sample_config = model_configs[0] if model_configs else ModelConfig()
            _download_source = _sample_config.parse_download_source()
            if _download_source.lower() != "huggingface":
                redirect_dict = {
                    "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                    "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                    "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
                }
                for model_config in model_configs:
                    if model_config.origin_file_pattern is None or model_config.model_id is None:
                        continue
                    if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                        print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                        model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                        model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
            import torch.distributed as dist
            from ..core.device.npu_compatible_device import get_device_name
            if dist.is_available() and dist.is_initialized():
                device = get_device_name()
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # Multi-camera cross-view generation
        # Generates a video from a target camera viewpoint conditioned on a source camera video.
        # Requires all four parameters below to be set together; if any is None the feature is skipped.
        cameras_npz: Optional[str] = None,      # Path to .npz file with per-camera extrinsic matrices.
                                                 # Expected keys: "<cam_name>_c2w" → float array of shape (N, 4, 4).
        target_cam: Optional[str] = None,        # Camera name for the desired output viewpoint (looked up as "<target_cam>_c2w").
        source_cam: Optional[str] = None,        # Camera name for the conditioning viewpoint (looked up as "<source_cam>_c2w").
                                                 # Relative pose is computed as: inv(source_c2w[0]) @ target_c2w[t].
        split_cam_embed: bool = False,           # Enable separate source/target pose embeddings with shared cam_encoder.
        zero_source_cam_emb_when_missing: bool = False,  # Keep target pose conditioning on the target half only when no separate source pose embedding is provided.
        source_video: Optional[list[Image.Image]] = None,  # Frames from the source camera, VAE-encoded into latents
                                                           # and concatenated channel-wise with the denoising target.
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_depth: Optional[list[Image.Image]] = None,
        vace_normal: Optional[list[Image.Image]] = None,
        vace_alpha: Optional[list[Image.Image]] = None,
        scene_flow_target: Optional[list[Image.Image]] = None,
        source_vace_video: Optional[list[Image.Image]] = None,
        source_vace_depth: Optional[list[Image.Image]] = None,
        source_vace_normal: Optional[list[Image.Image]] = None,
        source_vace_alpha: Optional[list[Image.Image]] = None,
        scene_flow_source: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # VAP
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        target_gbuffer_cfg_scale: Optional[float] = None,
        target_gbuffer_dropout_scope: Optional[
            Literal["target_only", "full_context"]
        ] = "target_only",
        target_gbuffer_uncond_fill_mode: Optional[
            Literal["zero", "learned"]
        ] = "zero",
        target_gbuffer_guidance_mode: Optional[Literal["cfg", "apg"]] = "cfg",
        target_gbuffer_apg_eta: Optional[float] = DEFAULT_TARGET_GBUFFER_APG_ETA,
        target_gbuffer_apg_norm_threshold: Optional[float] = DEFAULT_TARGET_GBUFFER_APG_NORM_THRESHOLD,
        target_gbuffer_apg_momentum: Optional[float] = DEFAULT_TARGET_GBUFFER_APG_MOMENTUM,
        target_gbuffer_apg_space: Optional[Literal["velocity", "x0"]] = DEFAULT_TARGET_GBUFFER_APG_SPACE,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        target_gbuffer_guidance_mode = _normalize_target_gbuffer_guidance_mode(
            target_gbuffer_guidance_mode
        )
        if target_gbuffer_cfg_scale is None:
            target_gbuffer_cfg_scale = (
                DEFAULT_TARGET_GBUFFER_APG_CFG_SCALE
                if target_gbuffer_guidance_mode == "apg"
                else 1.0
            )
        guidance_scale, target_gbuffer_guidance_enabled = _resolve_wan_guidance_scale(
            cfg_scale,
            target_gbuffer_cfg_scale,
        )
        target_gbuffer_dropout_scope = _normalize_target_gbuffer_dropout_scope(
            target_gbuffer_dropout_scope
        )
        target_gbuffer_uncond_fill_mode = _normalize_target_gbuffer_uncond_fill_mode(
            target_gbuffer_uncond_fill_mode
        )
        target_gbuffer_apg_space = _normalize_target_gbuffer_apg_space(
            target_gbuffer_apg_space
        )
        target_gbuffer_apg_eta = float(
            0.0 if target_gbuffer_apg_eta is None else target_gbuffer_apg_eta
        )
        target_gbuffer_apg_norm_threshold = float(
            0.0
            if target_gbuffer_apg_norm_threshold is None
            else target_gbuffer_apg_norm_threshold
        )
        target_gbuffer_apg_momentum = (
            None
            if target_gbuffer_apg_momentum is None
            else float(target_gbuffer_apg_momentum)
        )
        if target_gbuffer_apg_eta < 0.0:
            raise ValueError(
                "target_gbuffer_apg_eta must be >= 0, got "
                f"{target_gbuffer_apg_eta}."
            )
        if target_gbuffer_apg_norm_threshold < 0.0:
            raise ValueError(
                "target_gbuffer_apg_norm_threshold must be >= 0, got "
                f"{target_gbuffer_apg_norm_threshold}."
            )
        use_target_gbuffer_apg = (
            target_gbuffer_guidance_enabled
            and _guidance_scale_enabled(guidance_scale)
            and target_gbuffer_guidance_mode == "apg"
        )
        target_gbuffer_apg_buffer = (
            MomentumBuffer(target_gbuffer_apg_momentum)
            if use_target_gbuffer_apg and target_gbuffer_apg_momentum is not None
            else None
        )
        target_gbuffer_apg_logged = False
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "cameras_npz": cameras_npz, "target_cam": target_cam, "source_cam": source_cam,
            "split_cam_embed": split_cam_embed,
            "zero_source_cam_emb_when_missing": zero_source_cam_emb_when_missing,
            "source_video": source_video,
            "vace_video": vace_video, "vace_depth": vace_depth, "vace_normal": vace_normal, "vace_alpha": vace_alpha, "scene_flow_target": scene_flow_target,
            "source_vace_video": source_vace_video, "source_vace_depth": source_vace_depth, "source_vace_normal": source_vace_normal, "source_vace_alpha": source_vace_alpha, "scene_flow_source": scene_flow_source,
            "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "target_gbuffer_cfg_scale": target_gbuffer_cfg_scale,
            "target_gbuffer_dropout_scope": target_gbuffer_dropout_scope,
            "target_gbuffer_uncond_fill_mode": target_gbuffer_uncond_fill_mode,
            "target_gbuffer_guidance_mode": target_gbuffer_guidance_mode,
            "target_gbuffer_apg_eta": target_gbuffer_apg_eta,
            "target_gbuffer_apg_norm_threshold": target_gbuffer_apg_norm_threshold,
            "target_gbuffer_apg_momentum": target_gbuffer_apg_momentum,
            "target_gbuffer_apg_space": target_gbuffer_apg_space,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, 
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if _guidance_scale_enabled(guidance_scale):
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                if use_target_gbuffer_apg and not target_gbuffer_apg_logged:
                    print(
                        "[Guidance] Target-gbuffer APG active: "
                        f"space={target_gbuffer_apg_space} "
                        f"eta={target_gbuffer_apg_eta:.3f} "
                        f"norm_threshold={target_gbuffer_apg_norm_threshold:.3f} "
                        f"momentum={target_gbuffer_apg_momentum}"
                    )
                    target_gbuffer_apg_logged = True
                noise_pred = _combine_wan_guidance_predictions(
                    pred_cond=noise_pred_posi,
                    pred_uncond=noise_pred_nega,
                    guidance_scale=guidance_scale,
                    use_target_gbuffer_apg=use_target_gbuffer_apg,
                    target_gbuffer_apg_buffer=target_gbuffer_apg_buffer,
                    target_gbuffer_apg_eta=target_gbuffer_apg_eta,
                    target_gbuffer_apg_norm_threshold=target_gbuffer_apg_norm_threshold,
                    target_gbuffer_apg_space=target_gbuffer_apg_space,
                    scheduler=self.scheduler,
                    timestep=timestep,
                    sample=inputs_shared["latents"],
                )
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            output_params=("clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            output_params=("reference_latents", "clip_feature"),
            onload_model_names=("vae", "image_encoder")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_CameraPose(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("cameras_npz", "target_cam", "source_cam", "num_frames", "split_cam_embed"),
            output_params=("cam_emb", "source_cam_emb")
        )

    def process(self, pipe: WanVideoPipeline, cameras_npz, target_cam, source_cam, num_frames, split_cam_embed=False):
        if cameras_npz is None or target_cam is None or source_cam is None:
            return {}
        latent_frames = (num_frames - 1) // 4 + 1
        target_key = f"{target_cam}_c2w"
        source_key = f"{source_cam}_c2w"
        with np.load(cameras_npz) as data:
            if target_key not in data or source_key not in data:
                available = sorted([k for k in data.keys() if k.endswith("_c2w")])
                raise KeyError(
                    f"Missing camera key(s) in {cameras_npz}: target={target_key}, source={source_key}. "
                    f"Available keys: {available}"
                )
            target_c2w = data[target_key]
            source_c2w = data[source_key]

        max_frames = min(target_c2w.shape[0], source_c2w.shape[0])
        frame_indices = list(range(0, max_frames, 4))[:latent_frames]
        if len(frame_indices) == 0:
            raise ValueError(f"No valid camera frames found in {cameras_npz}")
        target_c2w = target_c2w[frame_indices]
        source_c2w = source_c2w[frame_indices]

        # Relative pose: trajectories w.r.t. source camera frame-0.
        source_w2c_0 = np.linalg.inv(source_c2w[0])
        target_relative_poses = []
        source_relative_poses = [] if split_cam_embed else None
        for target_pose, source_pose in zip(target_c2w, source_c2w):
            target_rel = source_w2c_0 @ target_pose
            target_relative_poses.append(target_rel[:3, :].flatten())
            if split_cam_embed:
                source_rel = source_w2c_0 @ source_pose
                source_relative_poses.append(source_rel[:3, :].flatten())

        cam_emb = np.stack(target_relative_poses, axis=0)
        cam_emb = torch.from_numpy(cam_emb).unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        outputs = {"cam_emb": cam_emb}
        if split_cam_embed:
            source_cam_emb = np.stack(source_relative_poses, axis=0)
            source_cam_emb = torch.from_numpy(source_cam_emb).unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            outputs["source_cam_emb"] = source_cam_emb
        return outputs



class WanVideoUnit_SceneCoordLoader(PipelineUnit):
    _LOAD_CACHE_MAXSIZE = 32

    def __init__(self, xyzt: bool = False, load_cache_maxsize: int | None = None):
        super().__init__(
            input_params=("cameras_npz", "target_cam", "source_cam", "height", "width", "num_frames"),
            output_params=("scene_coord_target", "scene_coord_source"),
        )
        self.xyzt = bool(xyzt)
        if load_cache_maxsize is None:
            load_cache_maxsize = self._LOAD_CACHE_MAXSIZE
        self.load_cache_maxsize = max(0, int(load_cache_maxsize))
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()

    def _ensure_thwc(self, array: np.ndarray, source_path: str):
        """Return numpy array in (T, H, W, 3) layout for preprocessing."""
        if array.ndim == 3:
            if array.shape[-1] == 3:
                return array[None]
            elif array.shape[0] == 3:
                return array.transpose(1, 2, 0)[None]
            else:
                raise ValueError(f"Unsupported scene coord shape {array.shape} in {source_path}")
        elif array.ndim == 4:
            if array.shape[-1] == 3:
                return array
            elif array.shape[1] == 3:
                return array.transpose(0, 2, 3, 1)
            elif array.shape[0] == 3:
                return array.transpose(1, 2, 3, 0)
            else:
                raise ValueError(f"Unsupported scene coord shape {array.shape} in {source_path}")
        else:
            raise ValueError(f"Unsupported scene coord ndim {array.ndim} in {source_path}")

    def _np_temporal_slice_and_crop(
        self, array: np.ndarray, num_frames: int, height: int, width: int, apply_spatial_crop: bool = True
    ):
        """Slice temporally and nearest-neighbor resize + center crop in numpy.

        Mirrors ImageCropAndResize used for RGB frames:
          1. scale = max(target_w / src_w, target_h / src_h)
          2. nearest-neighbor resize to (round(h*scale), round(w*scale))
          3. center crop to (height, width)

        Operates on (T, H, W, 3) numpy arrays. Returns (T', 3, H', W') float32 torch tensor.
        """
        t, h_src, w_src = array.shape[0], array.shape[1], array.shape[2]
        latent_frames = (num_frames - 1) // 4 + 1
        target_frames = latent_frames * 4

        if t > target_frames:
            array = array[:target_frames]
        elif t < target_frames:
            pad_count = target_frames - t
            array = np.concatenate([np.repeat(array[:1], pad_count, axis=0), array], axis=0)

        if h_src != height or w_src != width:
            if not apply_spatial_crop:
                raise ValueError(
                    f"Preprocessed scene-coord resolution mismatch: got {(h_src, w_src)}, "
                    f"expected {(height, width)}"
                )
            scale = max(width / w_src, height / h_src)
            h_scaled = round(h_src * scale)
            w_scaled = round(w_src * scale)

            row_idx = np.linspace(0, h_src - 1, h_scaled).astype(np.intp)
            col_idx = np.linspace(0, w_src - 1, w_scaled).astype(np.intp)
            array = array[:, row_idx][:, :, col_idx]

            top = (h_scaled - height) // 2
            left = (w_scaled - width) // 2
            array = array[:, top:top + height, left:left + width]

        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))
        if not tensor.is_floating_point() or tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor

    def _load_single(
        self,
        scene_root_dir: str,
        cam_name: str,
        height: int,
        width: int,
        num_frames: int,
        vid_tag: str | None = None,
    ):
        cache_key = (scene_root_dir, cam_name, height, width, num_frames, vid_tag)
        if self.load_cache_maxsize > 0:
            with self._cache_lock:
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    return self._cache[cache_key]

        name_variants = [f"{cam_name}_{vid_tag}", cam_name] if vid_tag else [cam_name]
        tried: list[str] = []

        result = None
        for name in name_variants:
            preprocessed_npy = os.path.join(
                scene_root_dir, f"scene_coords_{height}x{width}", f"{name}.npy"
            )
            if os.path.isfile(preprocessed_npy):
                array = np.load(preprocessed_npy, mmap_mode="r")
                array = self._ensure_thwc(array, preprocessed_npy)
                result = self._np_temporal_slice_and_crop(
                    array, num_frames, height, width, apply_spatial_crop=False
                )
                break
            tried.append(preprocessed_npy)

            npz_path = os.path.join(scene_root_dir, "scene_coords", f"{name}.npz")
            if os.path.isfile(npz_path):
                with np.load(npz_path) as data:
                    if len(data.files) == 0:
                        raise ValueError(f"Scene coordinate file has no arrays: {npz_path}")
                    for key in ("scene_coords", "coords", "xyz", "arr_0"):
                        if key in data:
                            array = data[key]
                            break
                    else:
                        array = data[data.files[0]]
                array = self._ensure_thwc(array, npz_path)
                result = self._np_temporal_slice_and_crop(
                    array, num_frames, height, width, apply_spatial_crop=True
                )
                break
            tried.append(npz_path)

        if result is None:
            raise FileNotFoundError(
                f"Missing scene-coordinate file for camera '{cam_name}'"
                + (f" vid_tag='{vid_tag}'" if vid_tag else "")
                + f". Tried: {tried}"
            )

        if self.load_cache_maxsize > 0:
            with self._cache_lock:
                if cache_key not in self._cache:
                    if len(self._cache) >= self.load_cache_maxsize:
                        self._cache.popitem(last=False)
                    self._cache[cache_key] = result
                self._cache.move_to_end(cache_key)
        return result

    def _valid_mask(self, coords: torch.Tensor):
        return (coords.abs().sum(dim=1, keepdim=True) > 1e-6)

    def _joint_normalize_pair(self, coords_target: torch.Tensor, coords_source: torch.Tensor):
        target_dtype = coords_target.dtype
        source_dtype = coords_source.dtype
        if coords_target.dtype != torch.float32:
            coords_target = coords_target.float()
        if coords_source.dtype != torch.float32:
            coords_source = coords_source.float()

        target_valid = self._valid_mask(coords_target)
        source_valid = self._valid_mask(coords_source)

        target_flat = coords_target.permute(0, 2, 3, 1)[target_valid.squeeze(1)]
        source_flat = coords_source.permute(0, 2, 3, 1)[source_valid.squeeze(1)]

        parts = [p for p in (target_flat, source_flat) if p.numel() > 0]
        if parts:
            all_vals = torch.cat(parts, dim=0)
            min_v = all_vals.min()
            max_v = all_vals.max()
            if max_v > min_v:
                coords_target = (coords_target - min_v) / (max_v - min_v) * 2 - 1
                coords_source = (coords_source - min_v) / (max_v - min_v) * 2 - 1

        coords_target = coords_target * target_valid.to(dtype=coords_target.dtype)
        coords_source = coords_source * source_valid.to(dtype=coords_source.dtype)
        if coords_target.dtype != target_dtype:
            coords_target = coords_target.to(dtype=target_dtype)
        if coords_source.dtype != source_dtype:
            coords_source = coords_source.to(dtype=source_dtype)
        return coords_target, coords_source

    def _append_temporal_channel(self, coords: torch.Tensor):
        if coords.ndim != 4:
            raise ValueError(f"Expected [T, C, H, W] scene coords, got {coords.shape}")
        if coords.shape[1] != 3:
            raise ValueError(
                f"Temporal channel augmentation expects 3-channel XYZ input, got {coords.shape[1]} channels."
            )

        t, _, h, w = coords.shape
        if t <= 1:
            t_values = torch.zeros((t, 1, 1, 1), dtype=coords.dtype, device=coords.device)
        else:
            t_values = torch.linspace(0.0, 1.0, steps=t, dtype=coords.dtype, device=coords.device).view(t, 1, 1, 1)
        t_values = t_values.expand(-1, 1, h, w)

        # Preserve invalid-pixel masking semantics from XYZ-only mode.
        valid = self._valid_mask(coords).to(dtype=coords.dtype)
        t_channel = t_values * valid
        return torch.cat([coords, t_channel], dim=1)

    def _load_pair(
        self,
        scene_root_dir: str,
        target_cam: str,
        source_cam: str,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        vid_tag: str | None = None,
    ):
        coords_target = self._load_single(scene_root_dir, target_cam, height, width, num_frames, vid_tag=vid_tag)
        coords_source = self._load_single(scene_root_dir, source_cam, height, width, num_frames, vid_tag=vid_tag)
        coords_target, coords_source = self._joint_normalize_pair(coords_target, coords_source)
        if self.xyzt:
            coords_target = self._append_temporal_channel(coords_target)
            coords_source = self._append_temporal_channel(coords_source)
        coords_target = coords_target.unsqueeze(0).to(dtype=dtype, device=device)
        coords_source = coords_source.unsqueeze(0).to(dtype=dtype, device=device)
        return coords_target, coords_source

    def process(self, pipe: WanVideoPipeline, cameras_npz, target_cam, source_cam, height, width, num_frames):
        scene_coord_encoders = []
        for dit_name in ("dit", "dit2"):
            dit_model = getattr(pipe, dit_name, None)
            scene_encoder = None if dit_model is None else getattr(dit_model, "scene_coord_encoder", None)
            if scene_encoder is not None:
                scene_coord_encoders.append(scene_encoder)
        if len(scene_coord_encoders) == 0:
            return {}
        self.xyzt = any(getattr(scene_encoder, "in_channels", 3) == 4 for scene_encoder in scene_coord_encoders)
        if cameras_npz is None or target_cam is None or source_cam is None:
            return {}
        scene_root_dir = os.path.dirname(cameras_npz)
        vid_match = re.search(r"(vid\d+)", os.path.basename(cameras_npz))
        vid_tag = vid_match.group(1) if vid_match else None
        try:
            scene_coord_target, scene_coord_source = self._load_pair(
                scene_root_dir,
                target_cam,
                source_cam,
                height,
                width,
                num_frames,
                pipe.torch_dtype,
                pipe.device,
                vid_tag=vid_tag,
            )
        except FileNotFoundError:
            return {}
        return {"scene_coord_target": scene_coord_target, "scene_coord_source": scene_coord_source}


class WanVideoUnit_SourceVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("source_video", "tiled", "tile_size", "tile_stride"),
            output_params=("source_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, source_video, tiled, tile_size, tile_stride):
        if source_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        source_video = pipe.preprocess_video(source_video)
        source_latents = pipe.vae.encode(source_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"source_latents": source_latents}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("control_camera_latents_input", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",),
            output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video", "vace_depth", "vace_normal", "vace_alpha",
                "scene_flow_target",
                "source_vace_video", "source_vace_depth", "source_vace_normal", "source_vace_alpha",
                "scene_flow_source",
                "vace_video_mask", "vace_reference_image", "vace_scale",
                "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"
            ),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",)
        )

    def _encode_video_latents(self, pipe: WanVideoPipeline, video, tiled, tile_size, tile_stride):
        if video is None:
            return None
        if not isinstance(video, list):
            video = [video]
        video = pipe.preprocess_video(video)
        latents = pipe.vae.encode(
            video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return latents

    def _write_latent_slice(self, context, latent, channel_start):
        if latent is None:
            return
        channel_size = min(16, latent.shape[1], context.shape[1] - channel_start)
        if channel_size <= 0:
            return
        t = min(context.shape[2], latent.shape[2])
        h = min(context.shape[3], latent.shape[3])
        w = min(context.shape[4], latent.shape[4])
        context[:, channel_start:channel_start+channel_size, :t, :h, :w] = latent[:, :channel_size, :t, :h, :w]

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_depth, vace_normal, vace_alpha,
        scene_flow_target,
        source_vace_video, source_vace_depth, source_vace_normal, source_vace_alpha,
        scene_flow_source,
        vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        # GBuffer-only path: encode each requested signal independently and
        # concatenate the active groups. Full target/source conditioning still
        # yields 4 * 16 = 64 channels by default, or 5 * 16 = 80 channels with
        # scene flow, while partial ablations such as target-RGB-only can now
        # shrink the compact layout down to 16 channels. The old 96-channel
        # standard VACE path (inactive/reactive + mask) is intentionally removed.
        target_signal_pairs = [
            ("vace_video", vace_video),
            ("vace_depth", vace_depth),
            ("vace_normal", vace_normal),
            ("vace_alpha", vace_alpha),
        ]
        source_signal_pairs = [
            ("source_vace_video", source_vace_video),
            ("source_vace_depth", source_vace_depth),
            ("source_vace_normal", source_vace_normal),
            ("source_vace_alpha", source_vace_alpha),
        ]
        if scene_flow_target is not None:
            target_signal_pairs.append(("scene_flow_target", scene_flow_target))
        if scene_flow_source is not None:
            source_signal_pairs.append(("scene_flow_source", scene_flow_source))
        if not any(signal is not None for _, signal in target_signal_pairs) and not any(
            signal is not None for _, signal in source_signal_pairs
        ):
            return {"vace_context": None, "vace_scale": vace_scale}

        pipe.load_models_to_device(["vae"])

        def _encode_signal_group(signal_pairs):
            encoded = []
            active_names = []
            for name, signal in signal_pairs:
                latent = self._encode_video_latents(pipe, signal, tiled, tile_size, tile_stride)
                if latent is None:
                    continue
                encoded.append(latent)
                active_names.append(name)
            if not encoded:
                return None, tuple()
            return torch.cat(encoded, dim=1), tuple(active_names)

        target_context, target_signal_names = _encode_signal_group(target_signal_pairs)
        source_context, source_signal_names = _encode_signal_group(source_signal_pairs)
        if target_context is None:
            vace_context = source_context
        elif source_context is None:
            vace_context = target_context
        else:
            if target_context.shape[1] != source_context.shape[1]:
                raise ValueError(
                    "Target/source compact VACE channel mismatch: "
                    f"target={target_signal_names}, source={source_signal_names}."
                )
            vace_context = torch.cat([target_context, source_context], dim=2)
        return {"vace_context": vace_context, "vace_scale": vace_scale}


class WanVideoUnit_TargetGBufferCFG(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        _, target_guidance_enabled = _resolve_wan_guidance_scale(
            inputs_shared.get("cfg_scale", 1.0),
            inputs_shared.get("target_gbuffer_cfg_scale", 1.0),
        )
        if not target_guidance_enabled:
            return inputs_shared, inputs_posi, inputs_nega

        has_target_signal = any(
            inputs_shared.get(name) is not None
            for name in ("vace_video", "vace_depth", "vace_normal", "vace_alpha", "scene_flow_target")
        )
        if not has_target_signal:
            return inputs_shared, inputs_posi, inputs_nega

        vace_context = inputs_shared.get("vace_context")
        if not isinstance(vace_context, torch.Tensor) or vace_context.ndim < 3:
            return inputs_shared, inputs_posi, inputs_nega

        dropped_vace_context = _build_target_gbuffer_dropped_vace_context(
            vace_context,
            source_latents=inputs_shared.get("source_latents"),
            dropout_scope=inputs_shared.get("target_gbuffer_dropout_scope", "target_only"),
            fill_mode=inputs_shared.get("target_gbuffer_uncond_fill_mode", "zero"),
            vace_model=pipe.vace,
        )
        if dropped_vace_context is None:
            return inputs_shared, inputs_posi, inputs_nega

        # Move vace_context out of the shared kwargs before assigning
        # branch-specific copies; otherwise model_fn receives it twice.
        inputs_shared.pop("vace_context", None)
        inputs_posi["vace_context"] = vace_context
        inputs_nega["vace_context"] = dropped_vace_context
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_VAP(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder", "vae", "image_encoder"),
            input_params=("vap_video", "vap_prompt", "negative_vap_prompt", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("vap_clip_feature", "vap_hidden_state", "context_vap")
        )
        
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("vap_video") is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            # 1. encode vap prompt
            pipe.load_models_to_device(["text_encoder"])
            vap_prompt, negative_vap_prompt = inputs_posi.get("vap_prompt", ""), inputs_nega.get("negative_vap_prompt", "")
            vap_prompt_emb = self.encode_prompt(pipe, vap_prompt)
            negative_vap_prompt_emb = self.encode_prompt(pipe, negative_vap_prompt)
            inputs_posi.update({"context_vap":vap_prompt_emb})
            inputs_nega.update({"context_vap":negative_vap_prompt_emb})
            # 2. prepare vap image clip embedding
            pipe.load_models_to_device(["vae", "image_encoder"])
            vap_video, end_image = inputs_shared.get("vap_video"), inputs_shared.get("end_image")

            num_frames, height, width = inputs_shared.get("num_frames"),inputs_shared.get("height"), inputs_shared.get("width")
            
            image_vap = pipe.preprocess_image(vap_video[0].resize((width, height))).to(pipe.device)

            vap_clip_context = pipe.image_encoder.encode_image([image_vap])
            if end_image is not None:
                vap_end_image = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                if pipe.dit.has_image_pos_emb:
                    vap_clip_context = torch.concat([vap_clip_context, pipe.image_encoder.encode_image([vap_end_image])], dim=1)
            vap_clip_context = vap_clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_clip_feature":vap_clip_context})

            # 3. prepare vap latents            
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            if end_image is not None:
                msk[:, -1:] = 1
                last_image_vap = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                vae_input = torch.concat([image_vap.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image_vap.device), last_image_vap.transpose(0,1)],dim=1)
            else:
                vae_input = torch.concat([image_vap.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image_vap.device)], dim=1)
            
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]

            tiled,tile_size,tile_stride = inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")

            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_video = pipe.preprocess_video(vap_video)
            vap_latent = pipe.vae.encode(vap_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_latent = torch.concat([vap_latent,y], dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_hidden_state":vap_latent})

            return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    @staticmethod
    def _build_cache(num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return None
        return TeaCache(
            num_inference_steps,
            rel_l1_thresh=tea_cache_l1_thresh,
            model_id=tea_cache_model_id,
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        tea_cache_posi = self._build_cache(
            inputs_posi.get("num_inference_steps"),
            inputs_posi.get("tea_cache_l1_thresh"),
            inputs_posi.get("tea_cache_model_id"),
        )
        if tea_cache_posi is not None:
            inputs_posi["tea_cache"] = tea_cache_posi

        tea_cache_nega = self._build_cache(
            inputs_nega.get("num_inference_steps"),
            inputs_nega.get("tea_cache_l1_thresh"),
            inputs_nega.get("tea_cache_model_id"),
        )
        if tea_cache_nega is not None:
            inputs_nega["tea_cache"] = tea_cache_nega
        return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = [
            "context", "clip_feature", "y", "reference_latents",
            "source_latents", "cam_emb", "source_cam_emb", "vace_context",
            "scene_coord_target", "scene_coord_source",
        ]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        guidance_scale, _ = _resolve_wan_guidance_scale(
            inputs_shared.get("cfg_scale", 1.0),
            inputs_shared.get("target_gbuffer_cfg_scale", 1.0),
        )
        if not _guidance_scale_enabled(guidance_scale):
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",),
            input_params=("input_audio", "audio_embeds", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "audio_sample_rate", "s2v_pose_video", "s2v_pose_latents", "motion_video"),
            output_params=("audio_embeds", "motion_latents", "drop_motion_frames", "s2v_pose_latents"),
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None:
            assert motion_video.shape[2] == motion_frames, f"motion video must have {motion_frames} frames, but got {motion_video.shape[2]}"
            motion_latents = motion_video
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"),
            output_params=("animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video")
        )

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            output_params=("pose_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("animate_face_video",),
            output_params=("face_pixel_values"),
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device=get_device_type()):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class WanVideoUnit_LongCatVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("longcat_video",),
            output_params=("longcat_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, longcat_video):
        if longcat_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        longcat_video = pipe.preprocess_video(longcat_video)
        longcat_latents = pipe.vae.encode(longcat_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"longcat_latents": longcat_latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state = None,
    vap_clip_feature = None,
    context_vap = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    cam_emb: Optional[torch.Tensor] = None,
    source_cam_emb: Optional[torch.Tensor] = None,
    source_latents: Optional[torch.Tensor] = None,
    zero_source_cam_emb_when_missing: bool = False,
    scene_coord_target: Optional[torch.Tensor] = None,
    scene_coord_source: Optional[torch.Tensor] = None,
    **kwargs,
):
    if (
        sliding_window_size is not None and sliding_window_stride is not None
        and (cam_emb is not None or source_latents is not None or scene_coord_target is not None or scene_coord_source is not None)
    ):
        raise ValueError("Sliding-window mode is not supported with source-video camera/scene-coordinate conditioning.")
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            cam_emb=cam_emb,
            source_cam_emb=source_cam_emb,
            source_latents=source_latents,
            zero_source_cam_emb_when_missing=zero_source_cam_emb_when_missing,
            scene_coord_target=scene_coord_target,
            scene_coord_source=scene_coord_source,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # LongCat-Video
    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if source_latents is not None:
        if source_latents.shape[0] != context.shape[0]:
            source_latents = torch.concat([source_latents] * context.shape[0], dim=0)
        x = torch.cat([x, source_latents], dim=2)
        if cam_emb is not None:
            if cam_emb.shape[0] != context.shape[0]:
                cam_emb = torch.concat([cam_emb] * context.shape[0], dim=0)
            if source_cam_emb is not None:
                if source_cam_emb.shape[0] != context.shape[0]:
                    source_cam_emb = torch.concat([source_cam_emb] * context.shape[0], dim=0)
                cam_emb = torch.cat([cam_emb, source_cam_emb], dim=1)
            else:
                source_cam_emb_fallback = torch.zeros_like(cam_emb) if zero_source_cam_emb_when_missing else cam_emb
                cam_emb = torch.cat([cam_emb, source_cam_emb_fallback], dim=1)

    scene_coord_pe = None
    if scene_coord_target is not None and getattr(dit, "scene_coord_encoder", None) is not None:
        if scene_coord_target.shape[0] != context.shape[0]:
            scene_coord_target = torch.concat([scene_coord_target] * context.shape[0], dim=0)
        scene_coord_target = scene_coord_target.to(dtype=x.dtype, device=x.device)
        target_scene_coord_pe = dit.scene_coord_encoder(scene_coord_target)
        scene_coord_pe = target_scene_coord_pe
        if source_latents is not None:
            if scene_coord_source is not None:
                if scene_coord_source.shape[0] != context.shape[0]:
                    scene_coord_source = torch.concat([scene_coord_source] * context.shape[0], dim=0)
                scene_coord_source = scene_coord_source.to(dtype=x.dtype, device=x.device)
                source_scene_coord_pe = dit.scene_coord_encoder(scene_coord_source)
            else:
                source_scene_coord_pe = target_scene_coord_pe
            scene_coord_pe = torch.cat([target_scene_coord_pe, source_scene_coord_pe], dim=1)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    if scene_coord_pe is not None:
        if reference_latents is not None and scene_coord_pe.shape[1] + (h * w) == x.shape[1]:
            ref_pad = scene_coord_pe.new_zeros((scene_coord_pe.shape[0], h * w, scene_coord_pe.shape[2]))
            scene_coord_pe = torch.cat([ref_pad, scene_coord_pe], dim=1)
        if scene_coord_pe.shape[1] != x.shape[1]:
            raise ValueError(
                f"Scene coordinate PE token mismatch: pe_seq={scene_coord_pe.shape[1]}, x_seq={x.shape[1]}"
            )
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # VAP 
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f,h,w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
            if scene_coord_pe is not None:
                scene_chunks = torch.chunk(scene_coord_pe, get_sequence_parallel_world_size(), dim=1)
                scene_chunks = [
                    torch.nn.functional.pad(chunk, (0, 0, 0, scene_chunks[0].shape[1] - chunk.shape[1]), value=0)
                    for chunk in scene_chunks
                ]
                scene_coord_pe = scene_chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                            use_reentrant=False
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                        use_reentrant=False
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs, cam_emb, scene_coord_pe
                )
              
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    if source_latents is not None:
        x = x[:, :, :x.shape[2]//2]
    return x


def model_fn_longcat_video(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    context = context.unsqueeze(0)
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        num_cond_latents=num_cond_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    output = -output
    output = output.to(latents.dtype)
    return output


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, seq_len_x, pre_compute_freqs[0]
            )
        x = gradient_checkpoint_forward(
            lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x),
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x
        )

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
