# MoSca Camera UI

`tools/mosca_camera_ui.py` is a lightweight Viser UI for designing camera
trajectories directly from a MoSca photometric log directory. It loads:

- `photometric_cam.pth`
- `photometric_s_model_native_add3.pth`
- `photometric_d_model_native_add3.pth`

The Gaussians are visualized as point clouds for fast interaction.

## Launch

Run this on an interactive/GPU node:

```bash
conda activate mosca
python tools/mosca_camera_ui.py \
  --saved-dir data/wild_workspaces/<scene>/logs/<run> \
  --device cuda \
  --port 9997
```

If the node is remote, forward the Viser port to your laptop, for example:

```bash
ssh -N -L 9997:<compute-node>:9997 <user>@<login-host>
```

Then open:

```text
http://localhost:9997
```

## Capture And Export

Use the **Frame** slider to choose a timestep, move the Viser viewport to the
desired camera pose, and click **Capture current view**. You only need sparse
keyframes; the tool interpolates a full per-frame trajectory.

Interpolation behavior:

- With zero keyframes, export uses the original solved source trajectory.
- With one keyframe, the same camera is held for all frames.
- With multiple keyframes, translation is smoothstep-interpolated and rotation
  uses quaternion slerp.
- Before the first keyframe, the first captured pose is held.
- After the last keyframe, the last captured pose is held.

Click **Export cameras** to write:

```text
<saved-dir>/camera_ui_exports/cam_ui_c2w.npz
```

The exported file contains a full trajectory array:

```text
cam_ui_c2w: [T, 4, 4]
```

You can also set a custom export path in the UI, e.g. `cam_ui_c2w_2.npz`.

## Render Stride 2

Example command using a second exported trajectory:

```bash
python scripts/wild/render_wild_fixed_camera_gbuffers.py \
  --ws data/wild_workspaces/astronaut_on_dunes_00 \
  --saved-dir data/wild_workspaces/<scene>/logs/<run> \
  --output-root data/wild_workspaces/astronaut_on_dunes_00/diffsynth_camera_ui_benchmark_stride2_v2 \
  --external-camera-npz data/wild_workspaces/<scene>/logs/<run>/camera_ui_exports/cam_ui_c2w_2.npz \
  --external-camera-align-mode none \
  --camera-names cam_ui \
  --frame-stride 2 \
  --overwrite
```

The important flags are:

- `--external-camera-npz`: points to the exported UI trajectory.
- `--external-camera-align-mode none`: the UI trajectory is already in MoSca
  world coordinates.
- `--frame-stride 2`: renders every other timestep.

## Automated G-Buffer And DiffSynth Pipeline

Use `tools/run_camera_ui_pipeline.sh` to render a camera-UI trajectory and then
run DiffSynth inference on the rendered `cam_ui` target camera.

For `animal_motion_00`, the defaults are:

- workspace: `data/wild_workspaces/animal_motion_00`
- saved dir: `data/wild_workspaces/<scene>/logs/<run>`
- camera trajectory: newest `.npz` under `<saved-dir>/camera_ui_exports/`
- g-buffer output: `data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_benchmark_first73_stride2`
- inference output: `data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_inference_first73_stride2_sceneflow`

Run on an interactive/GPU node with the `mosca` environment active:

```bash
conda activate mosca
bash tools/run_camera_ui_pipeline.sh \
  --camera-npz data/wild_workspaces/<scene>/logs/<run>/camera_ui_exports/cam_ui_c2w.npz
```

By default the script uses:

- DiffSynth root: `/path/to/diffsynth`
- checkpoint: `/path/to/step-16000.safetensors`
- g-buffer CFG: no dropout, `--target-gbuffer-dropout-scope target_only --target-gbuffer-uncond-fill-mode zero`
- camera conditioning: `--use-scene-flow --split-cam-embed`

The script does:

1. Render stride-2 g-buffers for the first 73 sampled frames:

```bash
python scripts/wild/render_wild_fixed_camera_gbuffers.py \
  --ws data/wild_workspaces/animal_motion_00 \
  --saved-dir data/wild_workspaces/<scene>/logs/<run> \
  --output-root data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_benchmark_first73_stride2 \
  --external-camera-npz data/wild_workspaces/<scene>/logs/<run>/camera_ui_exports/cam_ui_c2w.npz \
  --external-camera-align-mode none \
  --camera-names cam_ui \
  --frame-stride 2 \
  --max-frames 73 \
  --overwrite
```

2. Run DiffSynth inference on `cam_ui` for 73 frames:

```bash
python run_worldtree_inference.py \
  --diffsynth-root /path/to/diffsynth \
  --benchmark-root data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_benchmark_first73_stride2 \
  --output-root data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_inference_first73_stride2_sceneflow \
  --ckpt /path/to/step-16000.safetensors \
  --scene-names animal_motion_00 \
  --reference-camera cam0 \
  --target-cameras cam_ui \
  --chunk-size 73 \
  --max-frames 73 \
  --pred-subdir pred_rgb_first73_stride2 \
  --prompt " " \
  --inference-res 384x512 \
  --num-inference-steps 20 \
  --cfg-scale 1.0 \
  --target-gbuffer-cfg-scale 1.0 \
  --target-gbuffer-dropout-scope target_only \
  --target-gbuffer-uncond-fill-mode zero \
  --vace-scale 1.0 \
  --seed 0 \
  --device cuda \
  --use-scene-flow \
  --split-cam-embed
```

3. Create a 1x3 comparison video with input reference, original rendering, and
   model prediction:

```bash
python tools/make_camera_ui_video_grid.py \
  --benchmark-root data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_benchmark_first73_stride2 \
  --inference-root data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_inference_first73_stride2_sceneflow \
  --scene-name animal_motion_00 \
  --camera-name cam_ui \
  --reference-camera cam0 \
  --pred-subdir pred_rgb_first73_stride2 \
  --fps 15
```

By default this writes:

```text
data/wild_workspaces/animal_motion_00/diffsynth_camera_ui_inference_first73_stride2_sceneflow/animal_motion_00_ref_render_pred_1x3_cam_ui_pred_rgb_first73_stride2.mp4
```

To point at a specific export instead of the newest one:

```bash
bash tools/run_camera_ui_pipeline.sh \
  --camera-npz data/wild_workspaces/<scene>/logs/<run>/camera_ui_exports/cam_ui_c2w_2.npz
```

Use `--dry-run` to print the render and inference commands without executing.

## Slurm submission

Slurm wrappers are not included. To run g-buffer rendering and DiffSynth
inference in one GPU job, call `tools/run_camera_ui_pipeline.sh` from your own job
script with the same options shown above. To play a UI camera path backward,
reverse the trajectory NPZ first and pass the reversed file with `--camera-npz`.

