from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_production_config, vendor_root


def _prepend_runtime_paths() -> tuple[Path, Path]:
    vendor = vendor_root()
    mosca = vendor / "mosca"
    diffsynth = vendor / "diffsynth"
    for path in (str(diffsynth), str(mosca)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return mosca, diffsynth


def _model_paths(wan_root: Path) -> str:
    diffusion_shards = sorted(wan_root.glob("diffusion_pytorch_model*.safetensors"))
    if not diffusion_shards:
        raise FileNotFoundError(f"No local Wan diffusion shards found under {wan_root}")
    auxiliary_paths = [
        wan_root / "models_t5_umt5-xxl-enc-bf16.pth",
        wan_root / "Wan2.1_VAE.pth",
    ]
    missing = [
        path for path in [*diffusion_shards, *auxiliary_paths] if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing local Wan model files: {missing}")
    # DiffSynth identifies a sharded model from the union of keys across all
    # shards. Keep that collection as one ModelConfig.path list; passing a flat
    # list makes every shard look like a separate, unrecognizable model.
    return json.dumps(
        [
            [str(path.resolve()) for path in diffusion_shards],
            *[str(path.resolve()) for path in auxiliary_paths],
        ]
    )


def _load_checkpoint(pipe, checkpoint: Path) -> tuple[int, int]:
    import torch
    from diffsynth.core.loader.file import load_state_dict

    state = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16)
    remapped = {
        (key[len("pipe.") :] if key.startswith("pipe.") else f"vace.{key}"): value
        for key, value in state.items()
    }
    model_state = pipe.state_dict()
    loaded = 0
    unexpected = 0
    with torch.no_grad():
        for key, value in remapped.items():
            target = model_state.get(key)
            if target is None:
                unexpected += 1
                continue
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded += 1
    return loaded, unexpected


def load_production_pipe(
    wan_root: Path, tokenizer_root: Path, checkpoint: Path, device: str, steps: int
):
    import torch
    from examples.wanvideo.model_training.train import WanTrainingModule

    extra_inputs = ",".join(
        [
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
            "scene_flow_target",
            "scene_flow_source",
        ]
    )
    model = WanTrainingModule(
        model_paths=_model_paths(wan_root),
        model_id_with_origin_paths=None,
        tokenizer_path=str(tokenizer_root.resolve()),
        audio_processor_path=None,
        trainable_models="vace",
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=extra_inputs,
        remove_cached_keys="",
        fp8_models=None,
        offload_models=None,
        device=torch.device(device),
        task="sft",
        split_cam_embed=True,
        scene_coord_pe=False,
        scene_coord_xyzt_pe=False,
        scene_coord_gated_residual=False,
        eval_num_inference_steps=steps,
    )
    loaded, unexpected = _load_checkpoint(model.pipe, checkpoint)
    if loaded == 0:
        raise RuntimeError(f"Checkpoint did not match any production model parameters: {checkpoint}")
    print(f"[wfm] loaded checkpoint parameters={loaded} unexpected={unexpected}", flush=True)
    model.eval()
    pipe = model.pipe
    if hasattr(pipe, "_all_units"):
        pipe.units = pipe._all_units
    return model, pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen World From Motion DiffSynth inference")
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--wan-root", required=True, type=Path)
    parser.add_argument("--tokenizer-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    _prepend_runtime_paths()
    args = parse_args()
    profile = load_production_config()
    from lib_gen.worldtree_runner import WorldTreeInferenceConfig, run_worldtree_inference

    model, pipe = load_production_pipe(
        args.wan_root.expanduser().resolve(),
        args.tokenizer_root.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        args.device,
        int(profile.diffsynth_steps),
    )
    try:
        result = run_worldtree_inference(
            WorldTreeInferenceConfig(
                benchmark_root=args.benchmark_root.expanduser().resolve(),
                output_root=args.output_root.expanduser().resolve(),
                diffsynth_root=vendor_root() / "diffsynth",
                checkpoint=None,
                scene_names=(args.scene_name,),
                reference_camera=str(profile.reference_camera),
                target_cameras=tuple(profile.target_cameras),
                chunk_size=int(profile.diffsynth_chunk_size),
                pred_subdir="pred_rgb",
                prompt=" ",
                inference_res=(int(profile.diffsynth_height), int(profile.diffsynth_width)),
                num_inference_steps=int(profile.diffsynth_steps),
                cfg_scale=float(profile.diffsynth_cfg_scale),
                target_gbuffer_cfg_scale=float(profile.target_gbuffer_cfg_scale),
                target_gbuffer_dropout_scope="target_only",
                target_gbuffer_uncond_fill_mode="zero",
                vace_scale=float(profile.vace_scale),
                seed=int(args.seed),
                use_scene_flow=True,
                split_cam_embed=True,
                device=args.device,
                strict_layout=True,
            ),
            pipe=pipe,
        )
        print(f"[wfm] predictions={result.num_saved_frames} chunks={result.num_chunks}")
    finally:
        del pipe
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
