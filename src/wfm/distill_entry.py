from __future__ import annotations

import argparse
import json
from pathlib import Path

from .distillation import DistillationConfig, distill_generated_views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill DiffSynth pseudo views into a MoSca 4D Gaussian reconstruction"
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--source-native", required=True, type=Path)
    parser.add_argument("--mosca-root", required=True, type=Path)
    parser.add_argument("--reference-camera", default="cam0")
    parser.add_argument("--target-camera", dest="target_cameras", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--virtual-weight", type=float, default=1.0)
    parser.add_argument("--ssim-lambda", type=float, default=0.1)
    parser.add_argument("--lr-scale", type=float, default=0.25)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--debug-every", type=int, default=25)
    parser.add_argument(
        "--use-alpha-mask", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = distill_generated_views(
        DistillationConfig(
            benchmark_root=args.benchmark_root,
            prediction_root=args.prediction_root,
            output_root=args.output_root,
            scene_name=args.scene_name,
            source_native=args.source_native,
            mosca_root=args.mosca_root,
            target_cameras=tuple(args.target_cameras),
            reference_camera=args.reference_camera,
            device=args.device,
            seed=args.seed,
            steps=args.steps,
            source_weight=args.source_weight,
            virtual_weight=args.virtual_weight,
            ssim_lambda=args.ssim_lambda,
            lr_scale=args.lr_scale,
            grad_clip_norm=args.grad_clip_norm,
            use_alpha_mask=args.use_alpha_mask,
            debug_every=args.debug_every,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
