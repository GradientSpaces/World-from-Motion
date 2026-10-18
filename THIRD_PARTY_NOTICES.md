# Third-party notices

This repository vendors pinned source snapshots. Each component remains under its
own license, and its license file is retained with it.
The original World from Motion code is licensed under [Apache-2.0](LICENSE).
That license does not replace the terms of bundled third-party components.

| Component | Snapshot | License location |
| --- | --- | --- |
| MoSca with project integrations | Integration snapshot `0328637b07eb048d293d5c0a89595d451393e6c9` | `vendor/mosca/LICENSE` |
| DiffSynth-Studio with project modifications | Integration snapshot `e46fcedf7b111d7bbe64ee520a9e4369373f78e4` | `vendor/diffsynth/LICENSE` |
| SAM3 source | Integration snapshot `6ae999d49baf887c99f0016c8eb72e229189b9fc` | `vendor/mosca/lib_prior/sam3/LICENSE` |
| VGGT-Omega | MoSca snapshot above | `vendor/mosca/lib_prior/depth_models/vggt-omega/LICENSE` |
| CoWTracker | MoSca snapshot above | `vendor/mosca/lib_prior/tracking/cowtracker/LICENSE` |
| WAFT | MoSca snapshot above | `vendor/mosca/lib_prior/optical_flow/WAFT/LICENSE` |
| PyTorch3D stable | `75ebeeaea0908c5527e7b1e305fbc7681382db47` | `vendor/pytorch3d/LICENSE` |
| Gaussian rasterizers | MoSca snapshot above | License files under `vendor/mosca/lib_render/` |
| simple-knn (Inria GRAPHDECO) | MoSca snapshot above | `vendor/mosca/lib_render/simple-knn/LICENSE.md` (Gaussian-Splatting License) |
| BootsTAPIR/TAPNet PyTorch port (DeepMind) | MoSca snapshot above | `vendor/mosca/lib_prior/tracking/tapnet_pt/LICENSE` (Apache-2.0) |
| RAFT, via Meta robust-dynrf | MoSca snapshot above | `vendor/mosca/lib_prior/optical_flow/RAFT/LICENSE` (BSD-3-Clause) and `LICENSE-robust-dynrf` (MIT) |
| Epipolar-error priors, from Meta robust-dynrf | MoSca snapshot above | `vendor/mosca/lib_prior/epi_error/LICENSE` (MIT) |
| Depth-Anything-V2 modules used by WAFT | MoSca snapshot above | `vendor/mosca/lib_prior/optical_flow/WAFT/thirdparty/DepthAnythingV2/LICENSE` (Apache-2.0) |
| Table 1 MoSca and WorldTree research backends | Per-file hashes in `reproduce/table1/backends/provenance.json` | `reproduce/table1/backends/ms/LICENSE` and `reproduce/table1/backends/wt/LICENSE`; existing per-file headers |
| CoTracker visualization helper (Meta) | `vendor/mosca/lib_prior/tracking/cotracker_visualizer.py` and its copies under `reproduce/table1/backends/{ms,wt}/lib_prior/tracking/` | CC BY-NC 4.0; text in `reproduce/from_raw/runtime/cotracker3/LICENSE.md` |
| CoTracker3 for DyCheck priors | Per-file hashes in `reproduce/from_raw/runtime/provenance.json` | `reproduce/from_raw/runtime/cotracker3/LICENSE.md` (CC BY-NC 4.0) |
| Gen4DGS generation/export and WorldTree buffer renderer | Selected research modules; hashes in `reproduce/from_raw/runtime/provenance.json` | Apache-2.0 for original WfM contributions; preserved WorldTree and other third-party terms apply to their portions |

The MoSca upstream README explicitly limits its MIT grant to author-written portions;
bundled prior and renderer code must be reviewed under the original component terms.
Model weights are distributed separately. The WfM step-12500 checkpoint is
licensed under Apache-2.0 and derives from Wan2.1-VACE-14B, also Apache-2.0;
see [NOTICE](NOTICE) for the pinned model identities. Other pretrained models
retain their original terms.

## Components not included

SpaTracker (CC BY-NC 4.0, with its bundled CoTracker v2 and DINOv2 snapshots) and
DepthCrafter (research-only license) are not distributed. MoSca's optional
`tap_mode: spatracker` and `dep_mode: depthcrafter` settings require installing them
separately under their own terms. The CoWTracker paper PDF and Depth-Anything-V2
metric-depth training splits are also omitted.

The source package omits MoSca's optional DyCheck PCK ground-truth packs. The
arbitrary-video pipeline does not use those benchmark assets.
