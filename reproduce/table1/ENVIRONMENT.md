# Table 1 environment

Use a separate Python 3.10 environment with a compatible CUDA 12.1 toolkit and
FFmpeg on PATH. The historical GPU campaigns used Python 3.10.20,
PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121 and PyTorch3D 0.7.6.
[requirements-tested.txt](requirements-tested.txt) records the core package pins.
On October 15, 2026, the commands below completed in an empty virtual environment
and passed dependency checks, all five import probes, and 105 CPU tests plus 107
subtests. A one-scene MS wheel sampling run also completed on an H100 from a saved
initial reconstruction; see the [measured validation report](NATIVE_VALIDATION.md)
for scores and scope. The root Dockerfile uses a newer stack and is not the Table 1 environment.

## System prerequisites

This recipe targets Linux x86_64 with Python 3.10 (Ubuntu 22.04 is suitable),
an NVIDIA GPU and a working NVIDIA driver. Use an 80 GB A100 or H100 for the
sampling pilot. Sampling uses one GPU; WT reconstruction uses two. Allow at
least 250 GB of writable space for packages, model weights and scene outputs.

Install the system tools before creating the Python environment. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev \
  build-essential git ca-certificates ffmpeg libgl1 libglib2.0-0
```

Also install or load the **CUDA 12.1 development toolkit**, including `nvcc`,
headers and libraries. The CUDA libraries installed by pip do not supply the
complete extension-build toolchain. Follow NVIDIA's
[CUDA 12.1 Linux installation guide](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/index.html#ubuntu)
to configure its package repository. With that repository configured, Ubuntu
users can install the toolkit with `sudo apt-get install cuda-toolkit-12-1`.
This toolkit package does not replace the installed driver.
Set `CUDA_HOME` to your toolkit root (for
example `/usr/local/cuda-12.1`, or your cluster's CUDA module directory).
The CUDA version displayed by `nvidia-smi` is the driver's supported version;
check the toolkit separately:

```bash
export CUDA_HOME=/absolute/path/to/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
g++ --version
ffmpeg -version
nvidia-smi
```

Use GCC/G++ 11 with this recipe. A machine without a GPU can build the extensions
with the explicit architecture list below; GPU execution still needs an
allocation. No container is required.

## Create a clean environment

Run from the repository root. Choose a new environment directory; do not use
`--system-site-packages` or append another environment's packages to `PYTHONPATH`.

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
python3.10 -m venv .venv-table1
source .venv-table1/bin/activate
export PIP_CONSTRAINT="$PWD/reproduce/table1/constraints-native-py310-cu121.txt"
python -m pip install pip==24.0 setuptools==79.0.1 wheel==0.45.1
```

Upgrade pip **before** installing PyTorch: the older pip bundled with some Python
3.10 environments rejects normalized dependency names on the PyTorch index and
can fail while resolving `typing-extensions`. Use the pinned pip 24.0 here:
pip 25.3's `pip check` reports Decord 0.6.0 as unsupported because of its incorrect
internal `cp36` wheel tag, although its distributed wheel is named `py3-none`
and uses ctypes. Decord decoding is checked separately in the validation run; the pip
version pin does not replace the import and GPU checks below.

Keep `PIP_CONSTRAINT` set for all installation commands below. The
[complete dependency snapshot](constraints-native-py310-cu121.txt) also pins
transitive dependencies; it is a constraints file, not a standalone install list.
For a cluster, prefer a local SSD for the environment and checkout if shared
storage is slow. Use the same Python environment for the checks and GPU run.

## Install Python packages and CUDA extensions

Keep this order, and stop if a command fails. Use a Bash session with
`set -euo pipefail` when running the commands as a script.

```bash
python -m pip install -c reproduce/table1/requirements-tested.txt \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r reproduce/table1/requirements-tested.txt
python -m pip install -r reproduce/table1/requirements-pyg.txt
python -m pip install \
  'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl#sha256=e2845e8317768ed590b5999457879397e4876330b79bf01c2966c9e9dc116b87'
export TORCH_CUDA_ARCH_LIST='8.0;9.0'  # A100 and H100
export MAX_JOBS=8
python -m pip install --no-build-isolation \
  ./vendor/mosca/lib_render/diff-gaussian-rasterization-alphadep
python -m pip install --no-build-isolation \
  ./vendor/mosca/lib_render/diff-gaussian-rasterization-alphadep-add3
python -m pip install --no-build-isolation ./vendor/mosca/lib_render/simple-knn
python -m pip install -c reproduce/table1/requirements-tested.txt \
  -r reproduce/from_raw/requirements.txt
python -m pip install pytest pytest-subtests
python -m pip check
```

The wheel above is for Linux x86_64, CPython 3.10 and CUDA 12.1. Use the
recorded PyTorch3D version when matching the observed stack; the vendored 0.7.8
snapshot belongs to the newer video-demo image. `torch-cluster` supplies voxel-grid
operations; its compiled PyG dependencies must match PyTorch/CUDA.
The tested pins include setuptools 79.0.1 because SAM3 imports `pkg_resources`.
Keep that constraint when adding generation dependencies.
Both rasterizers are required: visualization imports the original alpha/depth
renderer even when the selected reconstruction backend is `native_add3`.

The initial setup downloads packages and model weights. Obtain the necessary
upstream access and comply with their licenses. Preserve build logs and
investigate dependency conflicts before accepting a new installation as validated.

## Cache model weights and check the installation

LPIPS uses its packaged calibration weights and the torchvision AlexNet backbone.
Populate the Torch cache while network access is available:

```bash
export TORCH_HOME=/absolute/path/to/writable/torch-cache
python -c "import lpips; lpips.LPIPS(net='alex', spatial=True)"
```

Use the same `TORCH_HOME` in your GPU job. Download the remaining model weights
and observations using the asset instructions in [REPRODUCTION.md](../../REPRODUCTION.md).
The sampling runner operates offline after those inputs are prepared.

Use `JAX_PLATFORMS=cpu`; rendering, generation, optimization and LPIPS use CUDA.
The `efficient` fitting preset needs a compiler toolchain and writable Triton and
TorchInductor caches. The raw runner assigns caches beneath each fresh run.
It retains MoSca's original Gaussian preparation and compiles WorldTree's
preparation. Evaluation uses the reference implementation in a separate process.
The research versions of Triton, SymPy and Ninja are recorded in the tested pins.

Run `reproduce/from_raw/check_environment.py` to probe the priors, both
reconstruction backends, SAM3 and generation imports in isolated processes.
Inside your GPU allocation, verify that the installed PyTorch can use the GPU:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Then run the CPU tests and wheel pilots in [REPRODUCTION.md](../../REPRODUCTION.md).
Successful imports do not establish GPU execution or numerical agreement.
