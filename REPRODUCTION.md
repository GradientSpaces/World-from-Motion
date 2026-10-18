# Reproduce Ours (sample) and Ours (distill)

This guide covers the DyCheck Table 1 rows for **MoSca (MS)** and
**WorldTree (WT)**. It uses pretrained WfM to generate views and reconstruct
scenes; training the WfM video model is a separate experiment.

**Release status:** the reproduction code and pinned input identities are included.
WfM step-12500 is published on Hugging Face at
[`liyzzz/World-from-Motion`](https://huggingface.co/liyzzz/World-from-Motion), and
the downloader fetches it at a pinned commit.

**Validation:** the native environment (Python 3.10, PyTorch 2.3.1/CUDA 12.1) and
one MS wheel sampling run from a saved initial reconstruction have been tested.
Initial fitting and the full distillation pipeline have not been revalidated in
this checkout. See the optional [validation report](reproduce/table1/NATIVE_VALIDATION.md)
for measurements and test details.

## What each row measures

| Setting | Ours (sample) | Ours (distill) |
| --- | --- | --- |
| Measured output | Generated test-camera video conditioned on the initial monocular 4DGS | Final 4DGS fitted using generated auxiliary views |
| WfM checkpoint | Step-12500 | Step-12500 |
| Denoising steps | 20 | 30 |
| Target-buffer APG x0 guidance | 3.0 for MS and WT | 3.0 for MS; 3.5 for WT |
| Generated cameras | Held-out test-camera trajectories after static alignment | Eight stationary auxiliary cameras |
| Inference resolution | Height 512, width 384 | Height 512, width 384 |
| Temporal chunk size / seed | 73 frames / 0 | 73 frames / 0 |
| Evaluation | Static Full, Valid and Covisible | Static Full, Valid and Covisible; per-frame Covisible |

Both generation recipes use APG eta 0, norm threshold 8 and momentum -0.5,
learned full-context unconditional inputs, scene flow, split camera embeddings,
text CFG 1.0 and VACE scale 1.0. Sampling settings are in
[`config/dycheck_sampling.json`](config/dycheck_sampling.json); teacher generation
is specified in [`stages.py`](reproduce/from_raw/stages.py). Distillation uses
the method-specific [fit configs](reproduce/table1/configs) with the `efficient`
preset, followed by evaluation with the reference implementation.

The four-view, step-16000 `wfm run` demo has a different recipe. Follow this guide
for Table 1 rather than changing the demo's distillation-step flag.

## Reference numbers and evaluation

All cells below are **PSNR / SSIM / LPIPS**. Higher PSNR and SSIM, and lower
LPIPS, are better. These are the rounded paper values retained with the research
recipe, not newly measured results from this checkout.

| Paper row | Static Full | Static Valid | Static Covisible | Per-frame Covisible |
| --- | --- | --- | --- | --- |
| MS + Ours (sample) | 18.43 / 0.548 / 0.269 | 18.68 / 0.638 / 0.242 | 19.18 / 0.688 / 0.227 | — |
| WT + Ours (sample) | 18.74 / 0.560 / 0.265 | 19.02 / 0.649 / 0.239 | 19.50 / 0.701 / 0.224 | — |
| MS + Ours (distill) | 17.36 / 0.570 / 0.264 | 18.88 / 0.661 / 0.202 | 19.78 / 0.710 / 0.180 | 19.98 / 0.716 / 0.178 |
| WT + Ours (distill) | 17.55 / 0.588 / 0.304 | 19.08 / 0.675 / 0.244 | 19.96 / 0.724 / 0.218 | 20.26 / 0.732 / 0.215 |

Use all seven scenes: `apple`, `block`, `paper-windmill`, `space-out`, `spin`,
`teddy`, and `wheel`. Average frames within each scene, then weight the seven
scenes equally. Static evaluation fixes one optimized pose per test camera;
per-frame evaluation optimizes poses individually. Camera calibration uses
held-out RGB under the backend's masking protocol. **Target ground-truth RGB
is excluded from the video generator inputs.** Preserve original held-out frame
identities, masks and metric implementations; MP4 previews are not scoring inputs.

The frozen sampling recipe uses guidance 3.0 for both methods; exact parity with
all archived sampling inputs/settings is unverified. Distillation comparisons use
[`public-pool-reference.json`](reproduce/table1/public-pool-reference.json),
which differs from both the paper values and the legacy reference in
[`reference.json`](reproduce/table1/reference.json). Label the reference used
and report signed differences, including regressions; command completion alone
does not establish numerical agreement.

## 1. Prepare the environment and inputs

Run commands from this repository root in a Bash session. Follow the separate
[Table 1 environment guide](reproduce/table1/ENVIRONMENT.md). Its observed stack
is Python 3.10, PyTorch 2.3.1/CUDA 12.1 and PyTorch3D 0.7.6; the root Dockerfile
uses a different stack. Use a GPU allocation for inference and reconstruction.
WT reconstruction needs two visible GPUs; sampling uses one. Check memory
requirements and runtime on your hardware with the wheel pilots.

On a scheduler, keep its assigned GPU visibility. The commands below use the
visible devices; do not replace a scheduler-provided `CUDA_VISIBLE_DEVICES` with
physical device numbers. On a workstation, select the desired GPUs before
running the commands.

Choose absolute paths. Keep inputs under a data directory, outside all generated
output roots. Put the three new output roots below outside the source checkout.
The source-packaging test expects a clean source tree without a local `runs/`
directory; use a clean checkout or source export for that test if you already
keep experiments under `runs/`.

```bash
set -euo pipefail
export WFM_INPUTS=/absolute/path/to/data/wfm-inputs
export WFM_DISTILL_ROOT=/absolute/path/to/new-distill-campaign
export WFM_SAMPLE_ROOT=/absolute/path/to/new-sample-campaign
export WFM_CHECK_ROOT=/absolute/path/to/new-checks
mkdir -p "$(dirname "$WFM_CHECK_ROOT")"
mkdir "$WFM_CHECK_ROOT"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

git rev-parse HEAD > "$WFM_CHECK_ROOT/commit.txt"
git status --short > "$WFM_CHECK_ROOT/working-tree.txt"
python -m pip freeze > "$WFM_CHECK_ROOT/environment.txt"
python reproduce/from_raw/check_environment.py 2>&1 | tee "$WFM_CHECK_ROOT/imports.log"
python -m pytest tests/test_from_raw.py tests/test_dycheck_mask_transport.py \
  tests/test_dycheck_sampling.py tests/test_sampling_preview.py \
  tests/test_table1_reproduction.py tests/test_table1_efficiency_runtime.py \
  tests/test_public_source.py 2>&1 | tee "$WFM_CHECK_ROOT/cpu-tests.log"
```

Stop on failed checks. Import and CPU tests do not run GPU reproduction.

The [asset lock](reproduce/from_raw/assets.lock.json) pins sizes and SHA-256
hashes for the observation export and pretrained models. The downloader extracts
only RGB, cameras, sensor depth, test RGB/cameras and covisibility masks from the
WorldTree DyCheck export. It regenerates flow, tracks, initial reconstructions
and teacher views locally. Arbitrary canonical DyCheck layouts are not accepted
without conversion to this export's layout.

The default `all` group includes the gated SAM3 weights, which need your own
approved SAM3 Hugging Face access. If you use the released evaluation masks
(below), you can skip SAM3. The asset lock pins the WfM checkpoint to the
[`liyzzz/World-from-Motion`](https://huggingface.co/liyzzz/World-from-Motion)
model repository at an exact commit:

```bash
python reproduce/from_raw/download.py --root "$WFM_INPUTS"
```

A checkpoint already at `$WFM_INPUTS/models/wfm-step-12500.safetensors` is
verified before any other download and reused. `--wfm-url` instead fetches the
same checkpoint from another HTTPS or S3 location. For S3, `--profile` and
`--endpoint` are optional explicit overrides; AWS CLI is required. No internal
endpoint or profile is selected automatically.

`--mirror DIR` (or `WFM_ASSET_MIRROR=DIR`) reads assets from a local copy laid
out like `--root`, such as a shared read-only input root, before using the
network. `--offline` (or `HF_HUB_OFFLINE=1`, which the root Dockerfile sets)
never uses the network: every selected asset must already be under `--root` or
the mirror, or the downloader stops before copying anything. Whatever the
source, each file is checked against its pinned size and SHA-256. Hugging Face
files download into a staging directory next to their destination, not through
the shared Hugging Face cache. `downloads-<group>.json` under `--root` records
each asset's source and, for Hugging Face assets, the pinned `repo@revision`.

The checkpoint is 16,591,588,312 bytes with SHA-256:

```text
32aa1215af2dd36a6dc07900e6196486c914003a5f5934b7fa51498c05301c01
```

Download all seven scenes initially, then limit GPU work to wheel. A wheel-only
observation extraction cannot be expanded in place; use a fresh input root if
you initially supplied `--scenes wheel`. Verified downloaded files are reusable,
but observation extraction requires a fresh `observations/` directory. Large
HTTPS downloads require range support. Weights and datasets are not included
in Git and remain subject to their original access and license terms.

### Released evaluation masks (optional)

By default the runner regenerates evaluation masks with SAM3. Instead, you can
use the released `dycheck-evaluation-masks-sam3-20261007` set from the
[`liyzzz/World-from-Motion-DyCheck-Masks`](https://huggingface.co/datasets/liyzzz/World-from-Motion-DyCheck-Masks)
dataset, pinned in the asset lock. Those masks were regenerated with
SAM3 on 2026-10-07 during an independent reproduction; they are not the masks
behind the paper's numbers. Download them separately (the default `all` group
omits them):

```bash
python reproduce/from_raw/download.py --root "$WFM_INPUTS" --group masks
```

The downloader fetches a single archive into `$WFM_INPUTS/downloads/`, checks
its SHA-256, the manifest SHA-256 and every mask, then installs the bundle under
`$WFM_INPUTS/evaluation-masks/`. Then add
`--evaluation-masks released` to each `run.py` command. The runner checks each
mask against the held-out RGB before any GPU stage starts. Its mask stage
installs the exact bytes instead of running SAM3, and `evaluation_masks.json`
records the bundle label and hashes. With released masks the SAM3 weights are
not needed: download the `data`, `models` and `wfm` groups instead of `all`,
which skips the separate `sam3` group.

## 2. Reproduce Ours (distill): wheel pilots

The raw runner performs nine stages: prepare observations, compute priors,
create evaluation masks, fit the initial reconstruction, render conditioning,
generate eight teachers, materialize fitting inputs, distill, and evaluate.
MoSca uses BootsTAPIR tracks; WorldTree uses CoTracker3 online tracks. Both use
RAFT flow, supplied sensor depth and cameras.

```bash
wfm_distill_scene() {
  python reproduce/from_raw/run.py \
    --inputs "$WFM_INPUTS" --output "$WFM_DISTILL_ROOT/${1}_${2}" \
    --method "$1" --scene "$2"
}
wfm_distill_scene ms wheel
wfm_distill_scene wt wheel
```

Require `execution.json` to report `complete`, with all nine stages successful,
and both fresh metric workbooks listed in each run's `evaluate/runs.json`.
`--plan` only prints the commands and cannot count as a completed pilot.
Failed attempts retain their outputs; use a fresh directory for a retry.

## 3. Reproduce Ours (sample): wheel pilots

Use the completed runs' **initial** reconstructions, before generated-view
fitting. The sampler chooses `initial/` and `workspace/` automatically. Although
the shared workflow above completes distillation first, the final `fit/` model
is not used for sampling. Wait for each source run to finish so its recorded
identity remains stable across sampling stages.

For **sampling only**, skip Section 2 and prepare initial reconstructions with
the following commands. These execute only `prepare`, `priors`, `masks` and
`initial`; they do not generate teachers or distill. Both methods below are
needed by the two-method wheel collector. For a one-scene, one-method installation
check, run just `ms wheel` and inspect its score workbook directly.

```bash
export WFM_SOURCE_ROOT=/absolute/path/to/new-initial-campaign
for method in ms wt; do
  python reproduce/from_raw/run.py \
    --inputs "$WFM_INPUTS" --output "$WFM_SOURCE_ROOT/${method}_wheel" \
    --method "$method" --scene wheel --initial-only
done
```

If you completed Section 2 instead, set `WFM_SOURCE_ROOT="$WFM_DISTILL_ROOT"`.
An initial-only run records `scope: initial_only` and four successful stages in
`execution.json`; it is not a completed distillation run. Either source works
with the same sampling commands:

```bash
wfm_sample_scene() {
  local sample_output="$WFM_SAMPLE_ROOT/${1}_${2}"
  mkdir -p "$WFM_SAMPLE_ROOT"
  mkdir "$sample_output"
  for stage in align render generate score; do
    python src/wfm/dycheck_sampling.py "$stage" \
      --method "$1" --scene "$2" \
      --source-run "$WFM_SOURCE_ROOT/${1}_${2}" \
      --models "$WFM_INPUTS/models" --output "$sample_output" \
      > "$sample_output/$stage.log" 2>&1
  done
}
wfm_sample_scene ms wheel
wfm_sample_scene wt wheel
python reproduce/sampling/collect.py "$WFM_SAMPLE_ROOT" --scenes wheel \
  --output "$WFM_CHECK_ROOT/sample-wheel.json"
```

The stages align static test cameras, render initial-4DGS conditioning, generate
full videos and score the original held-out frames. Generated RGB is resized
with PIL bilinear interpolation for scoring. Initial checkpoint hashes are
checked again after scoring. The wheel report is explicitly partial and has no
seven-scene means.

For a one-method installation check, invoke only `wfm_sample_scene ms wheel`
and omit the two-method collector. Read the `AVE` row in
`$WFM_SAMPLE_ROOT/ms_wheel/scores/tto_dycheck_metrics.xlsx`; the wheel workbook
must also contain all 219 held-out frame rows. Require all four
`<stage>-execution.json` files to report `complete` and the
`source-unchanged.json` checkpoint check to exist.

## 4. Run all seven scenes and collect numbers

After both wheel pilots pass for both rows, run the remaining scenes:

```bash
export WFM_SOURCE_ROOT="$WFM_DISTILL_ROOT"
for method in ms wt; do
  for scene in apple block paper-windmill space-out spin teddy; do
    wfm_distill_scene "$method" "$scene"
    wfm_sample_scene "$method" "$scene"
  done
done

python reproduce/from_raw/collect.py "$WFM_DISTILL_ROOT"
python reproduce/table1/run.py aggregate \
  --manifest "$WFM_DISTILL_ROOT/runs.json" \
  --output "$WFM_CHECK_ROOT/distill-table"
python reproduce/table1/compare.py "$WFM_CHECK_ROOT/distill-table/metrics.json" \
  --reference reproduce/table1/public-pool-reference.json \
  --output "$WFM_CHECK_ROOT/distill-vs-public-pool"
python reproduce/sampling/collect.py "$WFM_SAMPLE_ROOT" \
  --output "$WFM_CHECK_ROOT/sample-table.json"
```

Distillation requires 14 completed runs and 28 workbooks, reporting 24 means.
Sampling requires 14 completed runs and reports 18 means with differences from
the rounded paper values. The distillation aggregator also writes a default
comparison to the legacy reference; `distill-vs-public-pool` uses the public-pool
reference selected above. Keep both references clearly labeled.

Keep per-scene differences as well as means. Collectors expect one successful
`ms_<scene>` and `wt_<scene>` directory per scene. If retries were needed, use a
separate collection directory with symlinks to explicitly selected successful
attempts; preserve the failed attempts. Never fill missing metrics from a saved
table. Optional visual previews, separate from scoring:

```bash
python reproduce/sampling/preview.py "$WFM_SAMPLE_ROOT/ms_wheel" \
  --output "$WFM_CHECK_ROOT/previews/ms_wheel"
```

## Numerical limits

Evaluation dynamic masks are regenerated with SAM3 prompt unions unless the
released set is selected; either way they are not the archived masks. Archived mask
provenance remains unresolved for apple, block, paper-windmill, spin and teddy;
wheel/space-out prompt provenance alone does not establish pixel parity. The
paper-windmill recipe uses `pinwheel` and lossless FFV1 RGB transport; see the
[mask prompts](reproduce/from_raw/configs/mask_prompts.json) and
[transport settings](reproduce/from_raw/configs/mask_transport.json). These
choices affect alignment and must be reported. Training uses RAFT epipolar masks.

Differences can come from priors, masks/alignment, initial fits, teacher generation
and optimization. Record the commit and working-tree changes, environment,
asset hashes, commands, failures, stage times, and all measurements. The imported
source revision and per-file hashes are in [`reproduce/SOURCE.json`](reproduce/SOURCE.json).
The original WfM code and step-12500 checkpoint use [Apache-2.0](LICENSE), with
attribution in [NOTICE](NOTICE). Bundled dependencies retain their own terms and
the documented [third-party license gap](THIRD_PARTY_NOTICES.md#dependency-license-gap).
