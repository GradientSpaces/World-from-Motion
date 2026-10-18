# Reconstruct a monocular video

Build the [environment](INSTALL.md) and prepare the [model bundle](ASSETS.md).
Select the semantic classes that should move, such as `human` or `robot`.

```bash
docker run --rm --gpus all \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  -v "$WFM_MODEL_CACHE:/cache" \
  world-from-motion:local \
  wfm run --video /input/clip.mp4 \
    --dynamic-class human --output /output/scene
```

Set `WFM_MODEL_CACHE` to your model cache's absolute path before running the
container. To inspect the stages without a video, weights or GPU:

```bash
PYTHONPATH=src python -m wfm.cli run --video input/clip.mp4 \
  --dynamic-class human --output output/scene --dry-run
```

A real run computes priors, fits the initial MoSca scene, renders camera paths,
generates novel views, distills them into the scene and exports the result.
`--max-frames` bounds the input length and `--distillation-steps` overrides the
optimization count. The current profile is described in [model setup](ASSETS.md).

The output includes native reconstruction checkpoints, a dynamic Gaussian PLY,
novel-view videos and a run manifest. Working files and logs stay under the
selected output directory. Runs can resume when their input and configuration
fingerprints match. Inspect the completed output with:

```bash
wfm validate --output /path/to/output/scene
```
