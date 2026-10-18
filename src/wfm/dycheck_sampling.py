# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Direct DyCheck video samples from independently reconstructed initial 4DGS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "reproduce/from_raw/runtime"
TABLE = ROOT / "reproduce/table1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_fixed_pose(path, expected_count):
    import numpy as np
    poses = np.load(path)["poses"]
    if poses.shape != (expected_count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"Invalid test pose archive: {path}")
    if not np.allclose(poses, poses[0], rtol=0, atol=1e-6):
        raise ValueError(f"Per-frame camera motion is forbidden for sample rows: {path}")
    if not np.allclose(poses[0, 3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("Invalid homogeneous pose")
    return poses[0]


def align(args):
    model_files = sorted(args.initial.rglob("*.pth"))
    if not model_files:
        raise ValueError("Initial reconstruction checkpoint is missing")
    save(args.output / "initial-checkpoints.json", {
        str(p.relative_to(args.initial)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in model_files})
    command = [sys.executable, str(TABLE / "worker.py"), "--action", "evaluate",
               "--method", args.method, "--scene", args.scene, "--mode", "static",
               "--checkpoint", str(args.initial), "--workspace", str(args.workspace),
               "--metric-workspace", str(args.workspace), "--output", str(args.output / "alignment")]
    subprocess.run(command, check=True)


def render(args):
    # Research modules use bare imports such as `from camera import ...`.
    # Match the existing backend worker's import isolation before loading them.
    backend = TABLE / "backends/wt"
    sys.path[:0] = [str(backend), str(backend / "lib_mosca"), str(backend / "lib_moca")]
    import numpy as np
    import torch
    from gen4dgs.render.worldtree_render_bridge import WorldTreeRenderConfig, WorldTreeStationaryRenderer
    from gen4dgs.render.export_worldtree_layout import (
        ExportFrameRecord, RenderedFrame, EvaluationUnitPlan, SceneExportPlan,
        WorldTreeExportConfig, WorldTreeLayoutExporter)

    renderer = WorldTreeStationaryRenderer(WorldTreeRenderConfig(
        worldtree_root=TABLE / "backends/wt", saved_dir=args.initial,
        scene_root=args.workspace, device="cuda:0",
        process_height=2 if args.method == "wt" else None,
        fit_cfg_path=args.initial / "fit_config.yaml",
        use_fachain=True if args.method == "wt" else None))
    module = renderer.module
    tests = module.discover_camera_frames(str(args.workspace / "test_cameras"),
                                          str(args.workspace / "test_images"))
    pose_root = args.output / "alignment"
    if args.method == "wt":
        pose_root /= "render"
    poses = {cam: load_fixed_pose(pose_root / f"tto_test_pose_{i}.npz", len(specs))
             for i, (cam, specs) in enumerate(tests.items())}
    # Static test cameras have fixed intrinsics. Never substitute a source-camera K.
    for cam, specs in tests.items():
        if any(s.img_wh != specs[0].img_wh or not np.allclose(s.K_raw, specs[0].K_raw)
               for s in specs):
            raise ValueError(f"Varying test intrinsics require an explicit recipe: {cam}")
    source = renderer.build_source_records(reference_rgb_source="render")

    class Adapter:
        def build_scene_plan(self, reference_camera, *, target_cameras=None, max_frames=None):
            cameras = {"cam0": source}
            for cam, specs in tests.items():
                observed = {s.time_id for s in specs}
                cameras[f"cam{cam}"] = tuple(ExportFrameRecord(
                    frame_idx=i, time_id=r.time_id, frame_name=f"{cam}_{r.time_id:05d}",
                    has_gt=False, has_dataset_frame_at_time=r.time_id in observed,
                    render_pose_source_kind="initial_4dgs_static_test_alignment",
                    render_pose_archive_path=str(pose_root), rgb_source_kind="render",
                    reference_frame_name=r.frame_name, source_frame_name=r.frame_name)
                    for i, r in enumerate(source))
            return SceneExportPlan(scene_name=args.scene, reference_camera="cam0",
                camera_records=cameras, evaluation_units=tuple(EvaluationUnitPlan(
                    unit_id=f"cam0_to_cam{cam}", source_camera="cam0", target_camera=f"cam{cam}",
                    reference_camera="cam0") for cam in tests),
                scene_summary_extra={"adapter": "independent_test_camera_samples",
                    "target_gt_exported": False, "source_rgb_conditioning": "render",
                    "source_video": "observed RGB", "initial_checkpoint": str(args.initial)})

        def render_camera_frames(self, camera_name, records, outputs):
            if camera_name == "cam0":
                yield from renderer.render_source_frames(records=records, outputs=outputs,
                                                         reference_rgb_source="render")
                return
            cam = camera_name[3:]
            first = tests[cam][0]
            render_outputs = renderer._normalize_render_outputs(outputs)
            with torch.inference_mode():
                for record in records:
                    assignment = module.PoseAssignment(T_cw=poses[cam],
                        source_kind="static_test_alignment", source_time_id=first.time_id,
                        source_frame_name=first.frame_name, source_camera_json_path=first.camera_json_path,
                        pose_archive_path=str(pose_root), has_exact_tto_pose=True)
                    spec = module.CameraRenderSpec(cam_id=cam, time_id=record.time_id,
                        frame_name=record.frame_name, camera_json_path=first.camera_json_path,
                        image_path=None, K_raw=first.K_raw, img_wh=first.img_wh, T_cw=poses[cam],
                        has_dataset_frame_at_time=record.has_dataset_frame_at_time,
                        pose_assignment=assignment)
                    gs, _, _ = renderer._build_gs_for_time(record.time_id)
                    next_gs = None
                    if record.time_id < renderer.max_time_id:
                        next_gs, _, _ = renderer._build_gs_for_time(record.time_id + 1)
                    images, tensors, meta = module.render_single_camera_frame(spec=spec,
                        gs_list=gs, next_gs_list=next_gs, outputs=render_outputs, scene_space="world",
                        bg_color=[1., 1., 1.], alpha_threshold=1e-3, device=renderer.device)
                    yield RenderedFrame(frame_name=record.frame_name,
                        c2w=np.asarray(meta["c2w"], dtype=np.float32), images=images, tensors=tensors)

    output = args.output / "conditioning"
    if output.exists():
        raise FileExistsError(output)
    WorldTreeLayoutExporter(WorldTreeExportConfig(output_root=output, scene_name=args.scene,
        reference_camera="cam0", target_cameras=tuple(f"cam{c}" for c in tests),
        outputs=("rgb", "depth", "normal", "alpha", "gt", "scene_flow", "scene_flow_rgb"))).export(Adapter())
    for cam in tests:
        if any((output / args.scene / "gt" / f"cam{cam}").iterdir()):
            raise ValueError("Held-out RGB must not be exported to generator inputs")
    save(args.output / "camera-audit.json", {"targets": list(tests),
        "full_sequence_frames": len(source), "evaluated_frames": sum(map(len, tests.values())),
        "poses": {cam: pose.tolist() for cam, pose in poses.items()}, "target_gt_exported": False,
        "pose_archives": {p.name: digest(p) for p in pose_root.glob("tto_test_pose_*.npz")}})


def generate(args):
    from gen4dgs.inference.worldtree_runner import WorldTreeInferenceConfig, run_worldtree_inference
    cfg = args.recipe
    checkpoint = args.models / cfg["checkpoint"]
    if digest(checkpoint) != cfg["checkpoint_sha256"]:
        raise ValueError("Wrong WfM checkpoint")
    wan = args.models / "Wan2.1-VACE-14B"
    shards = sorted(wan.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if len(shards) != 7:
        raise ValueError("Expected seven Wan shards")
    output = args.output / "samples"
    if output.exists():
        raise FileExistsError(output)
    config = WorldTreeInferenceConfig(benchmark_root=args.output / "conditioning",
        output_root=output, diffsynth_root=ROOT / "vendor/diffsynth", checkpoint=checkpoint,
        scene_names=(args.scene,), reference_camera="cam0", chunk_size=cfg["chunk_size"],
        prompt=" ", inference_res=tuple(cfg["inference_res_hw"]),
        num_inference_steps=cfg["num_inference_steps"], cfg_scale=1.0,
        target_gbuffer_cfg_scale=cfg["guidance_scale"], target_gbuffer_dropout_scope="full_context",
        target_gbuffer_uncond_fill_mode="learned", target_gbuffer_guidance_mode=cfg["guidance_mode"],
        target_gbuffer_apg_eta=cfg["apg_eta"], target_gbuffer_apg_norm_threshold=cfg["apg_norm_threshold"],
        target_gbuffer_apg_momentum=cfg["apg_momentum"], target_gbuffer_apg_space=cfg["apg_space"],
        vace_scale=1.0, seed=cfg["seed"], tiled=False, use_scene_flow=True, split_cam_embed=True,
        model_paths=json.dumps([[str(p) for p in shards], str(wan / "models_t5_umt5-xxl-enc-bf16.pth"),
                                str(wan / "Wan2.1_VAE.pth")]), model_id_with_origin_paths=None,
        tokenizer_path=str(args.models / "tokenizer"), device="cuda:0", strict_layout=True)
    run_worldtree_inference(config)


def score(args):
    from PIL import Image
    predictions = args.output / "predictions"
    predictions.mkdir(exist_ok=False)
    records = []
    for gt in sorted((args.workspace / "test_images").glob("*.png")):
        source = args.output / "samples" / args.scene / "pred_rgb" / f"cam{gt.stem.split('_')[0]}" / gt.name
        with Image.open(gt) as image:
            size = image.size
        with Image.open(source) as image:
            image.convert("RGB").resize(size, Image.Resampling.BILINEAR).save(predictions / gt.name)
        records.append({"frame": gt.name, "generated_sha256": digest(source),
                        "scored_sha256": digest(predictions / gt.name), "scored_wh": list(size)})
    save(args.output / "prediction-audit.json", {"resampling": "PIL bilinear", "frames": records})
    subprocess.run([sys.executable, str(TABLE / "worker.py"), "--action", "metrics",
        "--method", args.method, "--scene", args.scene, "--mode", "static",
        "--checkpoint", str(args.initial), "--workspace", str(args.workspace),
        "--metric-workspace", str(args.workspace), "--predictions", str(predictions),
        "--output", str(args.output / "scores")], check=True)
    initial = json.loads((args.output / "initial-checkpoints.json").read_text())
    for name, identity in initial.items():
        path = args.initial / name
        if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            raise ValueError("Initial reconstruction changed during sampling")
    save(args.output / "source-unchanged.json", {"verified_initial_checkpoint_files": len(initial)})


def main():
    # Direct execution must not expose wfm/camera.py as a top-level `camera`.
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("align", "render", "generate", "score"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--method", choices=("ms", "wt"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/dycheck_sampling.json")
    args = parser.parse_args()
    for key in ("source_run", "output", "models", "config"):
        setattr(args, key, getattr(args, key).resolve())
    args.workspace, args.initial = args.source_run / "workspace", args.source_run / "initial"
    args.recipe = json.loads(args.config.read_text())
    if args.output == args.source_run or args.output.is_relative_to(args.source_run):
        raise ValueError("Use output outside the source campaign")
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / f"{args.stage}-execution.json"
    if record.exists():
        raise FileExistsError(record)
    source = json.loads((args.source_run / "execution.json").read_text())
    if source["method"] != args.method or source["scene"] != args.scene:
        raise ValueError("Source run identity mismatch")
    initial_stage = [s for s in source["stages"] if s["stage"] == "initial"]
    if len(initial_stage) != 1 or initial_stage[0]["status"] != "complete":
        raise ValueError("Source initial reconstruction has not completed")
    report = dict(stage=args.stage, method=args.method, scene=args.scene,
        command=sys.argv, source_run=str(args.source_run), config=args.recipe,
        script_sha256=digest(__file__), started=time.time(), status="running",
        source_execution_sha256=digest(args.source_run / "execution.json"))
    save(record, report)
    sys.path[:0] = [str(RUNTIME), str(ROOT / "vendor/diffsynth")]
    os.environ.update(GS_BACKEND="native_add3", HF_HUB_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="true", WANDB_MODE="disabled")
    try:
        globals()[args.stage](args)
        report.update(status="complete", returncode=0)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        save(record, report)


if __name__ == "__main__":
    main()
