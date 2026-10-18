#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_gen import (
    DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS,
    WorldTreeInferenceConfig,
    parse_resolution,
    run_worldtree_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DiffSynth inference on a WorldTree-style benchmark exported from MoSca."
    )
    parser.add_argument("--diffsynth-root", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ckpt", required=True, help="Trained DiffSynth checkpoint.")
    parser.add_argument("--scene-names", nargs="+", default=None)
    parser.add_argument("--reference-camera", default=None)
    parser.add_argument("--target-cameras", nargs="+", default=None)
    parser.add_argument("--evaluation-unit-ids", nargs="+", default=None)
    parser.add_argument("--chunk-size", type=int, default=73)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only run inference on the first N frames from each target sequence.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Subsample target frames at this stride after the dense prefix. "
            "Use with --dense-prefix to build a single strided inference window."
        ),
    )
    parser.add_argument(
        "--dense-prefix",
        type=int,
        default=0,
        help=(
            "Number of leading frames kept at stride 1 before --frame-stride "
            "applies (e.g. 9 dense + stride 3 over 200 frames yields 73 frames)."
        ),
    )
    parser.add_argument("--pred-subdir", default="pred_rgb")
    parser.add_argument("--prompt", default=" ")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument(
        "--inference-res",
        default=None,
        help="Override inference resolution with HxW.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--target-gbuffer-cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--target-gbuffer-dropout-scope",
        choices=("target_only", "full_context"),
        default="target_only",
        help=(
            "Scope for gbuffer unconditional guidance. full_context drops both "
            "source and target gbuffer context."
        ),
    )
    parser.add_argument(
        "--target-gbuffer-uncond-fill-mode",
        choices=("zero", "learned"),
        default="zero",
        help="How to fill dropped gbuffer context for the unconditional branch.",
    )
    parser.add_argument("--vace-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-image",
        default=None,
        help=(
            "Optional first-frame (image-to-video) condition: path to an image used "
            "to anchor the generated chunk's first frame (e.g. the previous segment's "
            "last predicted frame)."
        ),
    )
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-scene-flow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--split-cam-embed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scene-coord-pe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--scene-coord-xyzt-pe", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--scene-coord-gated-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--model-paths", default=None)
    parser.add_argument("--model-id-with-origin-paths", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--audio-processor-path", default=None)
    parser.add_argument("--trainable-models", default="vace")
    parser.add_argument("--remove-cached-keys", default="")
    parser.add_argument("--fp8-models", default=None)
    parser.add_argument("--offload-models", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--strict-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate every expected frame before inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_worldtree_inference(
        WorldTreeInferenceConfig(
            benchmark_root=Path(args.benchmark_root).expanduser().resolve(),
            output_root=Path(args.output_root).expanduser().resolve(),
            diffsynth_root=Path(args.diffsynth_root).expanduser().resolve(),
            checkpoint=Path(args.ckpt).expanduser().resolve(),
            scene_names=None if args.scene_names is None else tuple(args.scene_names),
            reference_camera=args.reference_camera,
            target_cameras=None if args.target_cameras is None else tuple(args.target_cameras),
            evaluation_unit_ids=(
                None if args.evaluation_unit_ids is None else tuple(args.evaluation_unit_ids)
            ),
            chunk_size=int(args.chunk_size),
            max_frames=None if args.max_frames is None else int(args.max_frames),
            frame_stride=int(args.frame_stride),
            dense_prefix=int(args.dense_prefix),
            pred_subdir=str(args.pred_subdir),
            prompt=str(args.prompt),
            negative_prompt=args.negative_prompt,
            inference_res=parse_resolution(args.inference_res),
            num_inference_steps=int(args.num_inference_steps),
            cfg_scale=float(args.cfg_scale),
            target_gbuffer_cfg_scale=float(args.target_gbuffer_cfg_scale),
            target_gbuffer_dropout_scope=str(args.target_gbuffer_dropout_scope),
            target_gbuffer_uncond_fill_mode=str(args.target_gbuffer_uncond_fill_mode),
            vace_scale=float(args.vace_scale),
            seed=args.seed,
            input_image=args.input_image,
            tiled=bool(args.tiled),
            use_scene_flow=bool(args.use_scene_flow),
            split_cam_embed=bool(args.split_cam_embed),
            scene_coord_pe=bool(args.scene_coord_pe),
            scene_coord_xyzt_pe=bool(args.scene_coord_xyzt_pe),
            scene_coord_gated_residual=bool(args.scene_coord_gated_residual),
            model_paths=args.model_paths,
            model_id_with_origin_paths=(
                args.model_id_with_origin_paths
                if args.model_id_with_origin_paths is not None
                else DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS
            ),
            tokenizer_path=args.tokenizer_path,
            audio_processor_path=args.audio_processor_path,
            trainable_models=args.trainable_models,
            remove_cached_keys=args.remove_cached_keys,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            device=args.device,
            strict_layout=bool(args.strict_layout),
        )
    )
    print(f"Saved {result.num_saved_frames} frames across {result.num_chunks} chunk(s).")
    print(f"Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
