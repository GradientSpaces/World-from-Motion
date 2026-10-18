# **CoWTracker: Tracking by Warping instead of Correlation**


<p align="center">
  <a href="https://arxiv.org/abs/2602.04877"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b" alt="arXiv"></a>
  <a href="https://cowtracker.github.io/"><img src="https://img.shields.io/badge/🌐-Project_Page-orange" alt="Project Page"></a>
  <a href="https://github.com/facebookresearch/cowtracker"><img src="https://img.shields.io/badge/GitHub-Repo-blue" alt="GitHub"></a>
  <a href="https://huggingface.co/spaces/facebook/cowtracker"><img src="https://img.shields.io/badge/🤗-Demo-yellow" alt="Hugging Face Demo"></a>
  <a href="https://youtu.be/QQP8TZPMZMw"><img src="https://img.shields.io/badge/🎬-Video-red" alt="Video"></a>
</p>

<p align="center">
  Zihang Lai<sup>1,2</sup>, Eldar Insafutdinov<sup>1</sup>, Edgar Sucar<sup>1</sup>, Andrea Vedaldi<sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup>Visual Geometry Group (VGG), University of Oxford &nbsp;&nbsp; <sup>2</sup>Meta AI
</p>

<p align="center">
  <img src="docs/cowtracker_long.jpg" alt="CoWTracker dense tracking visualization" width="100%"/>
</p>

**CoWTracker** is a state-of-the-art dense point tracker that eschews traditional cost volumes in favor of an iterative warping mechanism. By warping target features to the query frame and refining tracks with a joint spatio-temporal transformer, CoWTracker achieves state-of-the-art performance on **TAP-Vid** (DAVIS, Kinetics), **RoboTAP**, and demonstrates strong zero-shot transfer to optical flow benchmarks like **Sintel** and **KITTI**.

## 🚀 Key Features

<p align="center">
  <img src="docs/teaser.jpg" alt="CoWTracker example" width="800"/>
</p>

* **No Cost Volumes:** Replaces memory-heavy correlation volumes with a lightweight warping operation, scaling linearly with spatial resolution.
* **High-Resolution Tracking:** Processes features at high resolution (stride 2) to capture fine details and thin structures, unlike the stride 8 used by most prior methods.
* **Unified Architecture:** A single model that excels at both long-range point tracking and optical flow estimation.
* **Robustness:** Handles occlusions and rapid motion effectively using a video transformer with interleaved spatial and temporal attention.

## ⚡ Quick Start

### 1. Clone the Repository

Clone with submodules to get all required dependencies:

```bash
git clone --recurse-submodules https://github.com/facebookresearch/cowtracker.git
cd cowtracker
```

If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

Create the conda environment:

```bash
conda env create -f environments.yml
conda activate cowtracker
```

### 3. Run Inference

Model weights are automatically downloaded from HuggingFace Hub on first run:

```bash
python demo.py --video videos/bmx-bumps.mp4 --output output.mp4
```

To use a local checkpoint instead:

```bash
python demo.py --video videos/bmx-bumps.mp4 --output output.mp4 --checkpoint ./cow_tracker_model.pth
```


## ⚙️ Command Line Options

```
python demo.py --help

Options:
  --video             Path to input video (required)
  --output            Path to output video (default: {input_name}_tracked.mp4)
  --checkpoint        Path to model checkpoint (optional, auto-downloads from HuggingFace if not provided)
  --rate              Subsampling rate for visualization (default: 8)
  --max_frames        Maximum number of frames to process (default: 200)
  --no_bkg            Hide video and show only tracks on black background
```


## 🐍 Python API

```python
import torch
from cowtracker import CoWTracker

# Load model (auto-downloads from HuggingFace Hub)
model = CoWTracker.from_checkpoint(
    device="cuda",
    dtype=torch.float16,
)

# Or load from a local checkpoint
model = CoWTracker.from_checkpoint(
    "./cow_tracker_model.pth",
    device="cuda",
    dtype=torch.float16,
)

# Prepare video tensor [T, 3, H, W] in float32, range [0, 255]
video = ...

# Run inference
with torch.no_grad():
    predictions = model(video)

# Get outputs
tracks = predictions["track"]  # [B, T, H, W, 2] - tracked point coordinates
vis = predictions["vis"]       # [B, T, H, W] - visibility scores
conf = predictions["conf"]     # [B, T, H, W] - confidence scores
```

For long videos, use `CoWTrackerWindowed`:

```python
from cowtracker import CoWTrackerWindowed

# Auto-downloads from HuggingFace Hub
model = CoWTrackerWindowed.from_checkpoint(
    device="cuda",
    dtype=torch.float16,
)
```


## 📖 Citation

```bibtex
@article{lai2026a,
  title   = {CoWTracker: Tracking by Warping instead of Correlation},
  author  = {Lai, Zihang and Insafutdinov, Eldar and Sucar, Edgar and Vedaldi, Andrea},
  journal = {arXiv preprint arXiv:2602.04877},
  year    = {2026},
}}
```

## 🙏 Acknowledgements

Thanks to these great repositories: [VGGT](https://github.com/facebookresearch/vggt), [WAFT](https://github.com/princeton-vl/WAFT), [CoTracker](https://github.com/facebookresearch/co-tracker), [AllTracker](https://github.com/aharley/alltracker), [DUSt3R](https://github.com/naver/dust3r), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and many other inspiring works in the community.

## 📄 License

See the [LICENSE](LICENSE) file for details about the license under which this code is made available.

This release is intended to support the open-source research community and fundamental research. Users are expected to leverage the artifacts for research purposes and make research findings arising from the artifacts publicly available for the benefit of the research community.
