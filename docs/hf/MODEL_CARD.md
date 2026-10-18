---
license: apache-2.0
base_model: Wan-AI/Wan2.1-VACE-14B
base_model_relation: finetune
tags:
- novel-view-synthesis
- dynamic-reconstruction
- gaussian-splatting
- video-generation
- wan2.1
- vace
---

# World from Motion: WfM step-12500

WfM step-12500 is the camera-conditioned video model used for the DyCheck Table 1
results of **World from Motion: Generative Dynamic Gaussian Reconstruction from
Monocular Video** (NeurIPS 2026). Given a monocular video and an initial dynamic
reconstruction, it generates the scene as seen from new cameras. World from Motion
distills those generated views back into a dynamic 3D Gaussian scene.

[Project page](https://research.nvidia.com/labs/amri/projects/world-from-motion/) ·
[Paper](https://arxiv.org/abs/2607.01202) ·
[Code](https://github.com/GradientSpaces/World-from-Motion)

> **This is not a standalone model.** The checkpoint is a partial state dict that is
> loaded on top of Wan2.1-VACE-14B at a pinned revision, using the code in the
> GitHub repository. Stock Diffusers or DiffSynth-Studio pipelines cannot load it.

## Model details

| | |
| --- | --- |
| Developed by | Liyuan Zhu, Shengyu Huang, Amrita Mazumdar, Tianye Li, Zan Gojcic, Gordon Wetzstein, Iro Armeni, Shalini De Mello, Alex Trevithick (Stanford University, NVIDIA) |
| Model type | Camera-conditioned video diffusion transformer, fine-tuned from Wan2.1-VACE-14B |
| Base model | [`Wan-AI/Wan2.1-VACE-14B`](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) at revision `539c162b1387eac9dc4c20bd3f74671309e76a4c` (Apache-2.0) |
| Checkpoint | `wfm-step-12500.safetensors`, training step 12,500 |
| Format | Partial state dict: 798 tensors, all BF16, 8.30 B parameters |
| Size | 16,591,588,312 bytes |
| SHA-256 | `32aa1215af2dd36a6dc07900e6196486c914003a5f5934b7fa51498c05301c01` |
| License | Apache-2.0; see `LICENSE` and `NOTICE` |

## What the checkpoint contains

Only the modules that WfM adds or fine-tunes are stored:

| Key prefix | Tensors | Parameters | Role |
| --- | ---: | ---: | --- |
| `pipe.dit.blocks.{0-39}.self_attn` | 400 | 4.20 B | Fine-tuned DiT self-attention |
| `pipe.dit.blocks.{0-39}.cam_encoder` | 80 | 2.7 M | Added per-block camera encoder |
| `pipe.dit.blocks.{0-39}.projector` | 80 | 1.05 B | Added projection after camera-conditioned self-attention |
| `vace_blocks.{0-7}.*` | 234 | 3.05 B | Fine-tuned VACE context blocks |
| `vace_patch_embedding` | 2 | 1.6 M | VACE context patch embedding (80 input channels) |
| `full_context_uncond_vace_context`, `target_only_uncond_vace_context` | 2 | 160 | Learned unconditional context for guidance |

Keys beginning with `pipe.` load into the pipeline. All other keys load into its
VACE module. Every other weight is taken unchanged from the base revision: the
umt5-xxl text encoder, the VAE, and the remaining DiT layers (cross-attention,
feed-forward, norms, embeddings and head).

Files needed from upstream:

| Source | Revision | Files |
| --- | --- | --- |
| [`Wan-AI/Wan2.1-VACE-14B`](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) | `539c162b1387eac9dc4c20bd3f74671309e76a4c` | `diffusion_pytorch_model-0000{1-7}-of-00007.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth` |
| [`Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | `37ec512624d61f7aa208f7ea8140a131f93afc9a` | `google/umt5-xxl/` tokenizer files |

The [asset lock](https://github.com/GradientSpaces/World-from-Motion/blob/main/reproduce/from_raw/assets.lock.json)
in the code repository pins each file's size and SHA-256.

## Inputs and outputs

The model works on 73-frame chunks at 512 × 384 (height × width). Its inputs are:

- the observed source video and its cameras;
- the target camera trajectory, as split camera embeddings;
- RGB, depth, normal and alpha buffers rendered from the initial dynamic
  reconstruction at the target and source cameras (VACE context);
- scene flow for the target and source views;
- a blank text prompt (`" "`), with text CFG 1.0.

Its output is an RGB video seen from the target cameras. Ground-truth target
RGB is never an input.

## Recipes

These are the Table 1 settings. The code repository holds them in
[`config/dycheck_sampling.json`](https://github.com/GradientSpaces/World-from-Motion/blob/main/config/dycheck_sampling.json)
(sample) and
[`reproduce/from_raw/stages.py`](https://github.com/GradientSpaces/World-from-Motion/blob/main/reproduce/from_raw/stages.py)
(distill).

| Setting | Ours (sample) | Ours (distill) |
| --- | --- | --- |
| Generated cameras | Held-out test-camera trajectories after static alignment | Eight stationary auxiliary cameras |
| Denoising steps | 20 | 30 |
| Target-buffer APG guidance (x0 space) | 3.0 for MoSca and WorldTree | 3.0 for MoSca; 3.5 for WorldTree |
| APG eta / norm threshold / momentum | 0 / 8 / -0.5 | 0 / 8 / -0.5 |
| Unconditional input | Learned, full context | Learned, full context |
| Text CFG / VACE scale | 1.0 / 1.0 | 1.0 / 1.0 |
| Resolution (H × W) / chunk / seed | 512 × 384 / 73 frames / 0 | 512 × 384 / 73 frames / 0 |

## How to use

Follow [REPRODUCTION.md](https://github.com/GradientSpaces/World-from-Motion/blob/main/REPRODUCTION.md)
in the code repository. Its downloader fetches this checkpoint at a pinned commit
and checks its SHA-256 before use:

```bash
git clone https://github.com/GradientSpaces/World-from-Motion
cd World-from-Motion
python reproduce/from_raw/download.py --root /absolute/path/to/wfm-inputs --group wfm
```

To fetch only the checkpoint:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("liyzzz/World-from-Motion", "wfm-step-12500.safetensors")
```

Then check that the file's SHA-256 matches the value above. The code
repository's asset lock pins the exact commit of this repository.

## Results

DyCheck Table 1, from the paper. Each cell is PSNR / SSIM / LPIPS, averaged
equally over `apple`, `block`, `paper-windmill`, `space-out`, `spin`, `teddy` and
`wheel`.

| Paper row | Static Full | Static Valid | Static Covisible | Per-frame Covisible |
| --- | --- | --- | --- | --- |
| MoSca + Ours (sample) | 18.43 / 0.548 / 0.269 | 18.68 / 0.638 / 0.242 | 19.18 / 0.688 / 0.227 | — |
| WorldTree + Ours (sample) | 18.74 / 0.560 / 0.265 | 19.02 / 0.649 / 0.239 | 19.50 / 0.701 / 0.224 | — |
| MoSca + Ours (distill) | 17.36 / 0.570 / 0.264 | 18.88 / 0.661 / 0.202 | 19.78 / 0.710 / 0.180 | 19.98 / 0.716 / 0.178 |
| WorldTree + Ours (distill) | 17.55 / 0.588 / 0.304 | 19.08 / 0.675 / 0.244 | 19.96 / 0.724 / 0.218 | 20.26 / 0.732 / 0.215 |

These are the paper's values. The reproduction guide explains which reference
each re-run is compared against, and why exact parity with archived inputs is
not guaranteed.

## Intended use

- Research on novel-view synthesis and dynamic scene reconstruction from
  monocular video.
- Reproducing and extending the World from Motion results.

## Out-of-scope use

- Identifying, tracking or surveilling people.
- Producing deceptive or non-consensual media of real people.
- Safety-critical decisions, or metric measurement taken from generated views.

## Limitations

- Generated views are plausible completions, not observations. Regions the
  source video never saw are hallucinated and may be wrong or temporally
  inconsistent.
- Quality depends on the initial reconstruction and its camera estimates.
- Evaluation covered the seven DyCheck iPhone scenes and the paper's
  qualitative videos; behaviour on other domains has not been measured.
- Inference loads the full 14B base model plus this checkpoint. One 80 GB H100
  ran a complete MoSca `wheel` sampling job (alignment, rendering, generation
  and scoring of 219 held-out frames) in about 34 minutes.

## Files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `wfm-step-12500.safetensors` | 16,591,588,312 | `32aa1215af2dd36a6dc07900e6196486c914003a5f5934b7fa51498c05301c01` |
| `LICENSE` | | Apache License 2.0, copied from the code repository |
| `NOTICE` | | Attribution and base-model identity, copied from the code repository |
| `README.md` | | This card |

## License

The checkpoint is licensed under the Apache License, Version 2.0. It is a
modified, fine-tuned version of Wan2.1-VACE-14B, which Wan-AI publishes under
Apache-2.0. Redistribute `LICENSE` and `NOTICE` with the checkpoint. The
separately downloaded Wan components and other pretrained models keep their own
licenses.

## Citation

```bibtex
@inproceedings{zhu2026worldfrommotion,
  title = {World from Motion: Generative Dynamic Gaussian Reconstruction from Monocular Video},
  author = {Liyuan Zhu and Shengyu Huang and Amrita Mazumdar and Tianye Li and Zan Gojcic and Gordon Wetzstein and Iro Armeni and Shalini De Mello and Alex Trevithick},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2026}
}
```
