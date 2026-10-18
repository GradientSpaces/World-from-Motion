"""Run one backend per process to isolate the research modules' import names."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def configure_backend(method):
    backend = HERE / "backends" / method
    os.environ["GS_BACKEND"] = "native_add3"
    # Original research modules also import siblings by their unqualified names.
    sys.path[:0] = [str(backend), str(backend / "lib_mosca"), str(backend / "lib_moca")]
    return backend


def score(workspace, predictions, output, scene, *, wt_per_frame=False):
    from run import FRAMES
    from eval_utils.eval_dyncheck import eval_dycheck

    expected = None
    for directory in [predictions, *(workspace / name for name in
                                    ("test_images", "test_covisible", "test_dynamic"))]:
        names = sorted(p.name for p in directory.iterdir() if p.is_file())
        if len(names) != FRAMES[scene] or any(not n.endswith(".png") for n in names):
            raise ValueError(f"Expected {FRAMES[scene]} PNG frames: {directory}")
        if expected is not None and names != expected:
            raise ValueError(f"Prediction/GT/mask filename mismatch: {directory}")
        expected = names
    options = {} if wt_per_frame else {"dynamic_mask_dir": str(workspace / "test_dynamic")}
    eval_dycheck(save_dir=str(output), gt_rgb_dir=str(workspace / "test_images"),
                 gt_mask_dir=str(workspace / "test_covisible"), pred_dir=str(predictions),
                 strict_eval_all_gt_flag=True, eval_non_masked=not wt_per_frame,
                 save_prefix="tto_", **options)


def fit(args, backend):
    from omegaconf import OmegaConf
    import recon_utils
    from efficiency.runtime import configure_fit

    settings = configure_fit()
    (args.output / "efficiency_settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    config = HERE / "configs" / f"{args.method}_fit.yaml"

    def setup(*positional, fit_cfg, **kwargs):
        recon_utils.seed_everything(recon_utils.SEED)
        OmegaConf.save(fit_cfg, args.output / "fit_config.yaml")
        (args.output / "fit_commandline_args.txt").write_text(
            json.dumps(sys.argv, indent=2) + "\n")
        return str(args.output)

    # Use the explicitly requested fresh output, without latest-run discovery,
    # shell-based source copying, or writing logs into the input workspace.
    recon_utils.setup_recon_ws = setup
    recon_utils.setup_recon_ours = setup
    sys.argv = [str(backend / "mosca_reconstruct.py"), "--ws", str(args.workspace),
                "--cfg", str(config), "--no_viz"]
    if args.method == "wt":
        sys.argv.extend(["--model_path", str(args.output)])
    runpy.run_path(str(backend / "mosca_reconstruct.py"), run_name="__main__")


def evaluate(args, backend):
    from omegaconf import OmegaConf

    if args.method == "ms":
        from mosca_evaluate_ours import main

        config = HERE / "configs/ms_fit.yaml"
        sys.argv = [str(backend / "mosca_evaluate_ours.py"),
                    "--saved_dir", str(args.checkpoint), "--ws", str(args.workspace),
                    "--cfg", str(config), "--output_dir", str(args.output),
                    "--device", "cuda:0", "--eval_tto_mode", "per_frame",
                    "--eval_tto_static_camera_mode",
                    "global_first_frame" if args.mode == "static" else "none",
                    "--skip_metrics"]
        # Both audited MoSca evaluations inherit static-background TTO masking
        # from the same fit profile; only camera constancy changes.
        main()
    else:
        from mosca_onlyeval import process_layer
        from recon_utils import seed_everything, SEED

        config = HERE / "configs" / f"wt_{args.mode}.yaml"
        cfg = OmegaConf.load(config)
        # process_layer creates this child; the wrapper has reserved its parent.
        render_dir = args.output / "render"
        cfg.eval_rendering_dir_name = str(render_dir)
        seed_everything(SEED)
        process_layer(2, str(args.checkpoint), SimpleNamespace(ws=str(args.workspace)),
                      cfg, "iphone_ours_v3", 0)
        (args.output / "tto_test").symlink_to(render_dir / "tto_test", target_is_directory=True)
        if args.mode == "per_frame":
            # The published WT per-frame row comes from WorldTree's own scorer.
            source = render_dir / "tto_dycheck_metrics.xlsx"
            (args.output / "tto_dycheck_metrics.xlsx").write_bytes(source.read_bytes())
            return
    # Use the MoSca Full/Valid/Covisible implementation for both static rows,
    # matching the retained WT metrics wrapper. Isolate imports in a new process.
    subprocess.run([sys.executable, str(HERE / "worker.py"), "--action", "metrics",
                    "--method", "ms", "--scene", args.scene,
                    "--workspace", str(args.workspace),
                    "--metric-workspace", str(args.metric_workspace),
                    "--checkpoint", str(args.checkpoint), "--mode", args.mode,
                    "--predictions", str(args.output / "tto_test"),
                    "--output", str(args.output / "scores")], check=True)
    source = args.output / "scores/tto_dycheck_metrics.xlsx"
    (args.output / "tto_dycheck_metrics.xlsx").write_bytes(source.read_bytes())


def main():
    from efficiency.runtime import FIT_PRESETS, configure_fit_environment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("fit", "evaluate", "metrics"), required=True)
    parser.add_argument("--method", choices=("ms", "wt"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--metric-workspace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--mode", choices=("static", "per_frame", "fit"), required=True)
    parser.add_argument("--fit-preset", choices=tuple(FIT_PRESETS))
    args = parser.parse_args()
    if args.fit_preset is not None and args.action != "fit":
        parser.error("--fit-preset is only valid for fitting")
    if args.action == "fit":
        configure_fit_environment(args.fit_preset, method=args.method)
    from run import fresh_output

    inputs = [args.workspace, args.metric_workspace, args.checkpoint]
    # Scoring may write a sibling of its predictions in a fresh evaluation run.
    if args.predictions:
        inputs.append(args.predictions)
    args.output = fresh_output(args.output, inputs)
    if args.action in ("fit", "evaluate"):
        import torch

        if not torch.cuda.is_available():
            parser.error("Fitting and checkpoint rendering require a visible CUDA GPU")
    args.output.mkdir(parents=True)
    if args.action != "fit":
        from efficiency.runtime import configure_evaluation

        settings = configure_evaluation()
        (args.output / "efficiency_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    wt_per_frame = args.method == "wt" and args.mode == "per_frame"
    metric_backend = "wt" if wt_per_frame else "ms"
    backend = configure_backend(metric_backend if args.action == "metrics" else args.method)
    (args.output / "reproduction_job.json").write_text(json.dumps({
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }, indent=2) + "\n")
    if args.action == "fit":
        fit(args, backend)
    elif args.action == "evaluate":
        evaluate(args, backend)
    else:
        if not args.predictions:
            parser.error("metrics requires --predictions")
        workspace = args.workspace if wt_per_frame else args.metric_workspace
        score(workspace, args.predictions, args.output, args.scene, wt_per_frame=wt_per_frame)


if __name__ == "__main__":
    main()
