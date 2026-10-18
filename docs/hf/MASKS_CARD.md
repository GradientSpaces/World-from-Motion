---
license: apache-2.0
pretty_name: World from Motion DyCheck evaluation masks (SAM 3, 2026-10-07)
viewer: false
tags:
- dycheck
- novel-view-synthesis
- evaluation
- segmentation-masks
size_categories:
- 1K<n<10K
---

# World from Motion: DyCheck evaluation masks

Dynamic-region masks for the held-out test frames of the seven DyCheck iPhone
scenes. They are used only to **evaluate** the Table 1 reproduction of
**World from Motion: Generative Dynamic Gaussian Reconstruction from Monocular
Video** (NeurIPS 2026). Using these masks lets you skip SAM 3 (a gated model)
when reproducing the evaluation.

[Project page](https://research.nvidia.com/labs/amri/projects/world-from-motion/) ·
[Paper](https://arxiv.org/abs/2607.01202) ·
[Code](https://github.com/GradientSpaces/World-from-Motion) ·
[Model](https://huggingface.co/liyzzz/World-from-Motion)

## Which masks these are

`dycheck-evaluation-masks-sam3-20261007` was regenerated with SAM 3 on 2026-10-07,
during an independent reproduction that used only public inputs. These are
**not the masks behind the paper's Table 1 numbers.** Those historical masks are
not part of this release. Scores computed with this set count as a new
reproduction, not a re-scoring of the paper's runs.

How they were made:

- Model: [`facebook/sam3`](https://huggingface.co/facebook/sam3) at revision
  `3c879f39826c281e95690f02c7821c4de09afae7` (`sam3.pt`, SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`).
- Input: held-out test RGB from the DyCheck iPhone export. Each mask is the
  union of the prompts below.
- `paper-windmill` frames were passed to SAM 3 as a lossless FFV1 video; all
  other scenes used the default transport.

| Scene | Prompts | Frames |
| --- | --- | ---: |
| `apple` | apple, arm | 532 |
| `block` | gray block, block | 667 |
| `paper-windmill` | pinwheel | 419 |
| `space-out` | person, human | 681 |
| `spin` | person, human | 722 |
| `teddy` | teddy, hand | 688 |
| `wheel` | steering wheel, arm | 219 |

## How the masks are used

During evaluation the masks exclude moving content when the held-out test
cameras are aligned, and they split the Valid metrics into dynamic and static
regions. They are never used for training, reconstruction or video generation.

## Layout

The masks ship as a single uncompressed ZIP, so they download as one file
rather than thousands:

```text
dycheck-evaluation-masks-sam3-20261007.zip
  dycheck-evaluation-masks-sam3-20261007/
    manifest.json
    ms/<scene>/test_dynamic/<camera>_<frame>.png    # MoSca runs
    wt/<scene>/test_dynamic/<camera>_<frame>.png    # WorldTree runs
```

The archive is 17,644,275 bytes with SHA-256
`a106a67d856acc241109333b2839706b6099fd7466287c4c89ec8488fc8e1e6e`. Its entries
have fixed timestamps, so the code repository's packager rebuilds it
byte for byte. There are 14 sets (two methods × seven scenes): 7,856 PNGs totalling 11,264,264
bytes. File names match the DyCheck held-out frames. Each mask is an 8-bit
single-channel PNG at the frame resolution (360 × 480, width × height): 255
marks dynamic content and 0 marks static background. The MoSca and WorldTree sets were produced
separately and are byte-identical for every scene. Both are kept so that each
method's run reads its own set.

`manifest.json` has SHA-256 `daa2e01082bb660718347daeafd893c5ff780bb04dd0dd90d1577883c12eab53`.
For every mask it records the file size, the SHA-256 of its bytes and of its
decoded pixels, its image size and mode, and the decoded-pixel identity of the
matching held-out RGB frame. The loader therefore rejects a mask paired with
the wrong frame. The manifest contains no paths and no RGB data.

## Usage

With the code repository, download the masks and run a scene with them in place
of SAM 3:

```bash
python reproduce/from_raw/download.py --root "$WFM_INPUTS" --group masks
python reproduce/from_raw/run.py --inputs "$WFM_INPUTS" --output "$RUN_DIR" \
  --method ms --scene wheel --evaluation-masks released
```

The downloader pins this repository to an exact commit and checks the archive,
the manifest SHA-256 and every mask before use. The runner checks each mask against
your downloaded held-out RGB before any GPU stage starts. See
[REPRODUCTION.md](https://github.com/GradientSpaces/World-from-Motion/blob/main/REPRODUCTION.md).

To fetch the archive on its own:

```bash
hf download liyzzz/World-from-Motion-DyCheck-Masks \
  dycheck-evaluation-masks-sam3-20261007.zip --repo-type dataset --local-dir .
```

## License and attribution

The masks are released under Apache-2.0. They are derived from the DyCheck
iPhone dataset (Gao et al., *Monocular Dynamic View Synthesis: A Reality Check*,
NeurIPS 2022) and were produced with SAM 3 from Meta. Follow the DyCheck terms
for the underlying frames, and acknowledge SAM 3 when publishing results that
use these masks.

## Citation

```bibtex
@inproceedings{zhu2026worldfrommotion,
  title = {World from Motion: Generative Dynamic Gaussian Reconstruction from Monocular Video},
  author = {Liyuan Zhu and Shengyu Huang and Amrita Mazumdar and Tianye Li and Zan Gojcic and Gordon Wetzstein and Iro Armeni and Shalini De Mello and Alex Trevithick},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2026}
}

@inproceedings{gao2022dycheck,
  title = {Monocular Dynamic View Synthesis: A Reality Check},
  author = {Hang Gao and Ruilong Li and Shubham Tulsiani and Bryan Russell and Angjoo Kanazawa},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2022}
}
```
