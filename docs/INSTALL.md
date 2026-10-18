# Installation

For Table 1 reproduction without Docker, follow the
[native environment instructions](../reproduce/table1/ENVIRONMENT.md), then
[REPRODUCTION.md](../REPRODUCTION.md). They specify a separate Python 3.10 /
PyTorch 2.3.1 / CUDA 12.1 environment, system prerequisites and compiled
dependencies. Both sampling and distillation recipes are included.

The container recipe below belongs to the separate video-demo pipeline.

The core video pipeline uses Python 3.10 and CUDA. The Dockerfile supplies the
pinned runtime and compiled reconstruction dependencies. Model weights are kept
in a separate cache and are not included in the image.

```bash
docker build -t world-from-motion:local .
```

The configured hardware target is an NVIDIA A100 or H100 with 80 GB of memory.
Keep at least 250 GB of writable storage for the model cache and working files.
See [model setup](ASSETS.md), then [run your video](USAGE.md).

For source inspection and CPU development checks:

```bash
python -m pip install -e . --no-deps
python -m pip install pytest numpy Pillow
PYTHONPATH=src:. python -m pytest tests/test_config_and_cli.py tests/test_public_source.py
```

The editable install alone does not install the complete GPU stack. A container
build and GPU run have not been performed as part of this repository cleanup.
