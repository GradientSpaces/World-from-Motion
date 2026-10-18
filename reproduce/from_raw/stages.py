"""One isolated process per input-generation stage; no sibling repository imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TABLE = HERE.parent / "table1"
CAMERAS = ["cam0", *[f"cam{i}" for i in range(100, 108)]]
OBSERVATIONS = ("images", "cameras", "sensor_depth", "test_images", "test_cameras",
                "test_covisible")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def backend(method):
    path = TABLE / "backends" / method
    sys.path[:0] = [str(TABLE), str(path), str(path / "lib_moca"), str(path / "lib_mosca")]
    os.environ.update(GS_BACKEND="native_add3", JAX_PLATFORMS="cpu")
    return path


def prepare(args):
    """Copy only public observation exports; never import supplied learned priors."""
    from PIL import Image
    sys.path.insert(0, str(TABLE))
    from run import FRAMES

    source = args.data_root / args.scene
    workspace = args.output / "workspace"
    if workspace.exists():
        raise FileExistsError(workspace)
    for name in OBSERVATIONS:
        if not (source / name).is_dir():
            raise FileNotFoundError(f"Missing public observation export: {source / name}")
    expected = sorted(p.stem for p in (source / "images").glob("*.png"))
    if not expected or expected != [f"0_{i:05d}" for i in range(len(expected))]:
        raise ValueError("Expected contiguous training frames 0_00000.png onwards")
    for name, suffix in [("cameras", ".json"), ("sensor_depth", ".npz")]:
        if sorted(p.stem for p in (source / name).glob("*" + suffix)) != expected:
            raise ValueError(f"Training observation alignment mismatch: {name}")
    test = sorted(p.name for p in (source / "test_images").glob("*.png"))
    if len(test) != FRAMES[args.scene]:
        raise ValueError(f"Expected {FRAMES[args.scene]} held-out frames, got {len(test)}")
    for name in ("test_covisible",):
        if sorted(p.name for p in (source / name).glob("*.png")) != test:
            raise ValueError(f"Evaluation mask alignment mismatch: {name}")
    if sorted(p.stem for p in (source / "test_cameras").glob("*.json")) != [Path(x).stem for x in test]:
        raise ValueError("Test camera alignment mismatch")
    records = []
    for name in OBSERVATIONS:
        for src in sorted((source / name).iterdir()):
            if not src.is_file():
                raise ValueError(f"Unexpected nested observation entry: {src}")
            dst = workspace / name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            records.append({"path": str(dst.relative_to(workspace)), "bytes": dst.stat().st_size,
                            "sha256": sha256(dst)})
    with Image.open(workspace / "images" / (expected[0] + ".png")) as im:
        size = list(im.size)
    save(args.output / "observations.json", {"source": str(source), "frames": len(expected),
         "image_size": size, "test_frames": len(test), "files": records,
         "scope": "Only RGB, camera, sensor-depth and evaluation-mask observations; no tracks, flow, epipolar priors, teacher views or reconstruction checkpoints copied."})


def priors(args):
    os.environ.update(GS_BACKEND="native_add3", JAX_PLATFORMS="cpu")
    vendor = REPO / "vendor/mosca"
    sys.path[:0] = [str(REPO / "src"), str(vendor), str(vendor / "lib_prior"),
                    str(HERE / "runtime/cotracker3")]
    import torch
    from omegaconf import OmegaConf
    import optical_flow.raft_wrapper as raft
    original = raft.get_raft_model
    raft.get_raft_model = lambda ckpt_path, device, **kw: original(
        str(args.models / "raft-things.pth"), device, **kw)
    if args.method == "ms":
        import tracking.bootstapir_wrapper as tap
        original_tap = tap.get_bootstapir_model
        tap.get_bootstapir_model = lambda device, **kw: original_tap(
            model_path=str(args.models / "bootstapir_checkpoint_v2.pt"), device=device)
    else:
        import tracking.cotracker_wrapper as cot
        from cotracker.predictor import CoTrackerOnlinePredictor
        def load_tracker(device, online_flag=True, cotrakcer_version="3"):
            if not online_flag or str(cotrakcer_version) != "3":
                raise ValueError("The recorded WorldTree recipe uses CoTracker3 online")
            model = CoTrackerOnlinePredictor(checkpoint=str(args.models / "scaled_online.pth"),
                                            window_len=16, v2=False).to(device)
            return model.eval()
        cot.get_cotracker = load_tracker
    import mosca_precompute as prep
    cfg = OmegaConf.create(dict(dep_mode="sensor", tap_mode="bootstapir" if args.method == "ms" else "cotracker",
        flow_mode="raft", dynamic_mask_mode="raft_epi", epi_th=0.001, depth_boundary_th=1.0,
        tap_chunk_size=5000, n_track_uniform=8192, n_track_dynamic=16384,
        flow_steps=[1, 3], epi_num_threads=args.cpu_threads, align_metric_flag=True,
        boundary_enhance_th=-1, seed=12345))
    torch.set_num_threads(args.cpu_threads)
    OmegaConf.save(cfg, args.output / "prior_config.yaml")
    workspace = args.output / "workspace"
    images, names = prep.load_imgs_from_dir(str(workspace))
    prep.preprocess(images, names, str(workspace), prep.get_moca_processor(cfg), cfg)


def masks(args):
    """Install pinned released masks, or regenerate evaluation masks from RGB."""
    bundle = getattr(args, "evaluation_mask_bundle", None)
    manifest_hash = getattr(args, "evaluation_mask_manifest_sha256", None)
    if bool(bundle) != bool(manifest_hash):
        raise ValueError("Released masks require both a bundle and its pinned manifest SHA-256")
    if bundle:
        sys.path.insert(0, str(REPO / "src"))
        from wfm.released_masks import install_bundle
        report = install_bundle(bundle, manifest_hash, args.method, args.scene,
                                args.output / "workspace")
        save(args.output / "evaluation_masks.json", report)
        return
    sys.path[:0] = [str(REPO / "src"), str(HERE / "runtime"), str(REPO / "vendor/mosca")]
    import extract_test_masks as implementation
    import numpy as np
    from PIL import Image
    workspace = args.output / "workspace"
    destination = workspace / "test_dynamic"
    if destination.exists():
        raise FileExistsError(destination)
    prompts = json.loads((HERE / "configs/mask_prompts.json").read_text())[args.scene]
    transport = json.loads((HERE / "configs/mask_transport.json").read_text()).get(args.scene, "legacy")
    options = SimpleNamespace(prompt_override=None, prompt_override_list=prompts,
        output_dirname="test_sam3_dynamic_masks_regenerated", force=False, track_prefix_sep="_",
        gpus="0", sam3_checkpoint=str(args.models / "sam3/sam3.pt"),
        sam3_bpe=str(REPO / "vendor/mosca/lib_prior/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        sam3_lib_prior_root=str(REPO / "vendor/mosca/lib_prior"), fps=10.0,
        rgb_transport=transport)
    implementation.process_scene(str(workspace), args.scene, options)
    source = workspace / options.output_dirname / "masks"
    expected = sorted(p.name for p in (workspace / "test_images").glob("*.png"))
    if sorted(p.name for p in source.glob("*.png")) != expected:
        raise ValueError("Regenerated test masks do not cover all held-out images")
    nonzero = sum(int(np.count_nonzero(np.asarray(Image.open(source / name)))) for name in expected)
    if nonzero == 0:
        raise ValueError("SAM3 returned empty masks for the entire scene")
    track_pixels = {}
    for name in expected:
        track = name.split("_", 1)[0]
        track_pixels[track] = track_pixels.get(track, 0) + int(
            np.count_nonzero(np.asarray(Image.open(source / name))))
    empty_tracks = [track for track, count in track_pixels.items() if count == 0]
    if empty_tracks:
        raise ValueError(f"SAM3 returned empty masks for test tracks: {empty_tracks}")
    shutil.copytree(source, destination)
    save(args.output / "evaluation_masks.json", dict(prompts=prompts, frames=len(expected),
         rgb_transport=transport, track_nonzero_pixels=track_pixels,
         nonzero_pixels=nonzero, checkpoint_sha256=sha256(args.models / "sam3/sam3.pt"),
         note="Fresh SAM3 prompt unions; exact agreement with archived masks is not assumed."))


def initial(args):
    path = backend(args.method)
    from omegaconf import OmegaConf
    from efficiency.runtime import configure_fit_environment
    configure_fit_environment("reference", method=args.method)
    import recon_utils
    output = args.output / "initial"
    output.mkdir(exist_ok=False)
    config = HERE / "configs" / f"{args.method}_initial.yaml"
    def setup(*pos, fit_cfg, **kwargs):
        recon_utils.seed_everything(recon_utils.SEED)
        OmegaConf.save(fit_cfg, output / "fit_config.yaml")
        (output / "fit_commandline_args.txt").write_text(
            shlex.join(["--ws", str(args.output / "workspace"), "--cfg", str(output / "fit_config.yaml")]) + "\n")
        return str(output)
    recon_utils.setup_recon_ws = setup
    recon_utils.setup_recon_ours = setup
    sys.argv = [str(path / "mosca_reconstruct.py"), "--ws", str(args.output / "workspace"),
                "--cfg", str(config), "--no_viz"]
    if args.method == "wt":
        sys.argv += ["--model_path", str(output)]
    runpy.run_path(str(path / "mosca_reconstruct.py"), run_name="__main__")


def render(args):
    # The recorded Gen4DGS export reads both native MoSca and WorldTree checkpoints
    # through the WorldTree-compatible renderer, in its own process.
    path = backend("wt")
    sys.path.insert(0, str(HERE / "runtime"))
    from gen4dgs.render.trajectory_fullseq_stationary_adapter import TrajectoryFullseqStationaryAdapter
    from gen4dgs.render.export_worldtree_layout import WorldTreeLayoutExporter, WorldTreeExportConfig
    adapter = TrajectoryFullseqStationaryAdapter(scene_root=args.output / "workspace", scene_name=args.scene,
        num_anchors=8, worldtree_root=path, worldtree_saved_dir=args.output / "initial",
        worldtree_device="cuda:0", worldtree_process_height=2 if args.method == "wt" else None,
        worldtree_fit_cfg_path=args.output / "initial/fit_config.yaml",
        worldtree_use_fachain=True if args.method == "wt" else None,
        reference_rgb_source="gt")
    config = WorldTreeExportConfig(output_root=args.output / "conditioning", scene_name=args.scene,
        reference_camera="cam0", target_cameras=tuple(CAMERAS[1:]),
        outputs=("rgb", "depth", "normal", "alpha", "gt", "scene_flow", "scene_flow_rgb"))
    result = WorldTreeLayoutExporter(config).export(adapter)
    print(result)


def wfm_checkpoint(models, spec=None):
    """Return the pinned WfM checkpoint only after its size and SHA-256 match the lock."""
    spec = spec or json.loads((HERE / "assets.lock.json").read_text())["wfm"]
    checkpoint = Path(models) / Path(spec["path"]).name
    if checkpoint.stat().st_size != spec["bytes"] or sha256(checkpoint) != spec["sha256"]:
        raise ValueError(f"Wrong WfM checkpoint: {checkpoint}")
    return checkpoint


def generate(args):
    checkpoint = wfm_checkpoint(args.models)
    sys.path[:0] = [str(HERE / "runtime"), str(REPO / "vendor/diffsynth")]
    os.environ.update(DIFFSYNTH_SKIP_DOWNLOAD="true", HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    from gen4dgs.inference.worldtree_runner import WorldTreeInferenceConfig, run_worldtree_inference
    wan = args.models / "Wan2.1-VACE-14B"
    shards = sorted(wan.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if len(shards) != 7:
        raise ValueError(f"Expected seven Wan diffusion shards under {wan}")
    paths = json.dumps([[str(p) for p in shards], str(wan / "models_t5_umt5-xxl-enc-bf16.pth"),
                        str(wan / "Wan2.1_VAE.pth")])
    config = WorldTreeInferenceConfig(benchmark_root=args.output / "conditioning",
        output_root=args.output / "teacher", diffsynth_root=REPO / "vendor/diffsynth",
        checkpoint=checkpoint, scene_names=(args.scene,),
        reference_camera="cam0", target_cameras=tuple(CAMERAS[1:]), chunk_size=73,
        prompt=" ", inference_res=(512, 384), num_inference_steps=30, cfg_scale=1.0,
        target_gbuffer_cfg_scale=3.0 if args.method == "ms" else 3.5,
        target_gbuffer_dropout_scope="full_context", target_gbuffer_uncond_fill_mode="learned",
        target_gbuffer_guidance_mode="apg", target_gbuffer_apg_eta=0.0,
        target_gbuffer_apg_norm_threshold=8.0, target_gbuffer_apg_momentum=-0.5,
        target_gbuffer_apg_space="x0", vace_scale=1.0, seed=0, tiled=False,
        use_scene_flow=True, split_cam_embed=True, model_paths=paths,
        model_id_with_origin_paths=None, tokenizer_path=str(args.models / "tokenizer"),
        device="cuda:0", strict_layout=True)
    run_worldtree_inference(config)


def materialize(args):
    import numpy as np
    workspace = args.output / "workspace"
    benchmark = args.output / "conditioning" / args.scene
    teacher = args.output / "teacher"
    prefix = "multiview" if args.method == "ms" else "multiview_8"
    mv = workspace / prefix
    mv.mkdir(exist_ok=False)
    names = sorted(p.stem for p in (workspace / "images").glob("*.png"))
    manifest = dict(scene_name=args.scene, num_frames=len(names), anchor_camera="cam0",
        anchor_rgb_camera="cam0", override_anchor_rgb=True, anchor_saved2d_rgb_dir="images", cameras=[])
    for camera in CAMERAS:
        cid = int(camera[3:])
        image_dir = mv / "rgb" / camera
        camera_dir = mv / "camera_jsons" / camera
        image_dir.mkdir(parents=True); camera_dir.mkdir(parents=True)
        src_pose = benchmark / "cameras" / f"{camera}_c2w.npz"
        pose_path = mv / "cameras" / src_pose.name
        pose_path.parent.mkdir(exist_ok=True)
        shutil.copyfile(src_pose, pose_path)
        poses = np.load(pose_path)["c2w"]
        if len(poses) != len(names):
            raise ValueError(f"Camera/frame count mismatch: {camera}")
        frame_names = []
        for t, anchor in enumerate(names):
            name = f"{cid}_{t:05d}"
            src = (benchmark / "gt" / camera / f"{name}.png" if cid == 0 else
                   teacher / args.scene / "pred_rgb" / camera / f"{name}.png")
            if not src.is_file():
                raise FileNotFoundError(f"Missing generated supervision: {src}")
            shutil.copyfile(src, image_dir / (name + ".png"))
            # Intrinsics follow the source observation at the same timestep.
            payload = json.loads((workspace / "cameras" / (anchor + ".json")).read_text())
            payload.update(orientation=poses[t, :3, :3].T.tolist(), position=poses[t, :3, 3].tolist())
            save(camera_dir / (name + ".json"), payload)
            frame_names.append(name)
        manifest["cameras"].append(dict(camera_name=camera, legacy_camera_id=cid,
            rgb_dir=str(image_dir.relative_to(workspace)), camera_json_dir=str(camera_dir.relative_to(workspace)),
            reference_c2w_path=str(pose_path.relative_to(workspace)), frame_names=frame_names,
            weight=1.0 if cid == 0 else (0.2 if args.method == "ms" else 0.1)))
    save(workspace / ("multiview_manifest.json" if args.method == "ms" else "multiview_8_manifest.json"), manifest)
    row = dict(method=args.method, scene=args.scene, workspace="workspace", metric_workspace="workspace",
        checkpoint="initial", static_metrics="not-yet-evaluated/static.xlsx",
        per_frame_metrics="not-yet-evaluated/per_frame.xlsx")
    save(args.output / "runs.json", {"schema_version": 1, "runs": [row]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "masks", "priors", "initial", "render", "generate", "materialize"))
    parser.add_argument("--method", choices=("ms", "wt"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--evaluation-mask-bundle", type=Path)
    parser.add_argument("--evaluation-mask-manifest-sha256")
    args = parser.parse_args()
    for name in ("output", "data_root", "models"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
