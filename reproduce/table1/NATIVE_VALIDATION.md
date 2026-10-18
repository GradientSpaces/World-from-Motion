# Native installation validation — October 15, 2026

The [native installation guide](ENVIRONMENT.md) was validated with a fresh
Python environment and one **MS / wheel / Ours (sample)** GPU run. No Docker
image was used. This is a one-scene sampling check, not a seven-scene Table 1
reproduction or a fresh distillation campaign.

## Installation checks

The final installation command blocks were executed from beginning to end in
an empty virtual environment, with only the interpreter, environment directory
and CUDA toolkit paths adapted to the host. All 191 installed package versions
matched the independently constructed fresh environment used for the GPU run.
Neither environment inherited or copied installed packages from the old research
environment. The existing operating system, Python interpreter/standard library,
CUDA compiler, driver and FFmpeg were reused; this was not a fresh OS installation.

- Python 3.10.20; PyTorch 2.3.1+cu121; torchvision 0.18.1+cu121; PyTorch3D 0.7.6.
- CUDA toolkit 12.1.66; GCC/G++ 11.4; NVIDIA driver 535.216.03.
- `pip check`: passed with the documented pip 24.0.
- Priors, MS reconstruction, WT reconstruction, SAM3 and generation import probes: all passed in both environments.
- Full CPU suite on a clean source export: **105 tests and 107 subtests passed**, with 16 existing Pillow deprecation warnings.
- Decord 0.6.0 decoded a new five-frame H.264 video successfully in both environments; its incorrect internal wheel tag is documented in the installation guide.
- Package-isolation checks: all 191 distributions were inside each new environment, with user site-packages disabled. This also passed on the GPU node.

The installation fixes include the original alpha/depth rasterizer (in addition
to add3 and simple-knn), BootsTAPIR's `einshape` dependency, installer/build-tool
pins, explicit CUDA architecture targets, and the full
[dependency constraints](constraints-native-py310-cu121.txt).

## GPU sampling result

One H100 80 GB completed alignment, conditioning rendering, video generation
and scoring. The job completed successfully in **34 minutes 15 seconds**, including
runtime staging. The sampling recipe was step-12500, 20 denoising steps, APG x0
3.0, seed 0, height 512 / width 384 and 73-frame chunks, as specified in
[the sampling config](../../config/dycheck_sampling.json).

The run regenerated **437 frames** and scored all **219 held-out frames** for
wheel camera 1. Frame identities and prediction hashes passed validation.
Target ground-truth RGB was absent from generation inputs; static camera
alignment used the documented held-out masking protocol. All **seven initial
checkpoint files** retained their original hashes after scoring.

The input was the saved initial MS wheel reconstruction and workspace from the
October 5–7 independent campaign, before generated-view fitting. Existing
observation data, evaluation masks, pretrained weights and the LPIPS backbone
cache were reused. Initial reconstruction, priors, mask generation, WT GPU
execution and distillation were **not** rerun here. The new `--initial-only`
runner option was unit-tested; this pilot used the already available initial model.

All cells below are **PSNR / SSIM / LPIPS**. The comparison is with the October 7
wheel measurement using the same initial reconstruction, not with the paper's
seven-scene averages. These are close measurements, not bit-identical results;
the small regressions are retained explicitly.

| Evaluation | Fresh environment | October 7 wheel | Signed difference |
| --- | --- | --- | --- |
| Static Full | 16.165331 / 0.564135 / 0.212638 | 16.168860 / 0.564316 / 0.212440 | -0.003529 / -0.000180 / +0.000199 |
| Static Valid | 18.032517 / 0.651238 / 0.184618 | 18.036011 / 0.651335 / 0.184512 | -0.003494 / -0.000097 / +0.000105 |
| Static Covisible | 18.571810 / 0.689875 / 0.168944 | 18.573106 / 0.689935 / 0.168871 | -0.001296 / -0.000060 / +0.000073 |

| Stage | Seconds |
| --- | ---: |
| align | 256.4 |
| render | 297.8 |
| generate | 1268.9 |
| score | 178.3 |

The resulting score workbook has SHA-256
`f9af26b65358a7da019c0727507488825a409be39f2ae473005a418f46c023bd`.
The WfM checkpoint hash is
`32aa1215af2dd36a6dc07900e6196486c914003a5f5934b7fa51498c05301c01`.
Generated outputs and build logs are validation artifacts, not bundled source.

The full sampling and distillation instructions remain in
[REPRODUCTION.md](../../REPRODUCTION.md). This check does not establish full-table
numerical parity.
