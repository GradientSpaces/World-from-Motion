"""Record the source and runtime behind a fresh fitting or evaluation result."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def allocation_label(environ=None):
    """Opaque label shared by records from one scheduler allocation.

    Records can be paired by allocation without storing job IDs or host names.
    """
    environ = os.environ if environ is None else environ
    job = environ.get("SLURM_JOB_ID")
    if not job:
        return None
    node = environ.get("SLURMD_NODENAME", "")
    return "alloc-" + hashlib.sha256(f"{job}@{node}".encode()).hexdigest()[:12]


def portable_platform():
    """OS, architecture and C library, without the host's kernel release."""
    libc, version = platform.libc_ver()
    parts = [platform.system(), platform.machine()]
    return "-".join(parts + ([f"with-{libc}{version}"] if libc else []))


def capture():
    import torch

    root = Path(__file__).resolve().parent
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "jax", "jaxlib",
                 "lpips", "pytorch3d", "kornia", "openpyxl", "opencv-python",
                 "imageio", "imageio-ffmpeg", "omegaconf", "evo",
                 "torch-geometric", "torch-cluster", "torch-scatter", "torch-sparse",
                 "triton", "sympy", "ninja",
                 "pyg-lib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    extensions = {}
    for name in ("diff_gaussian_rasterization_add3", "simple_knn", "pytorch3d"):
        spec = importlib.util.find_spec(name)
        if spec:
            locations = spec.submodule_search_locations or (
                [str(Path(spec.origin).parent)] if spec.origin else []
            )
            extensions[name] = {
                str(path): sha256(path)
                for location in locations for path in sorted(Path(location).glob("*.so"))
            }
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "interpreter": sys.executable,
        "platform": portable_platform(),
        "allocation": allocation_label(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [
            {"index": index, "name": torch.cuda.get_device_name(index),
             "capability": list(torch.cuda.get_device_capability(index))}
            for index in range(torch.cuda.device_count())
        ],
        "environment": {
            key: os.environ[key] for key in (
                "CUDA_VISIBLE_DEVICES", "GS_BACKEND", "JAX_PLATFORMS",
                "SLURM_ARRAY_TASK_ID",
                "WFM_TABLE1_RENDER_CACHE", "WFM_TABLE1_KEEP_ALLOCATOR",
                "WFM_TABLE1_RESIDENT_TOPOLOGY", "WFM_TABLE1_MATRIX_MATH",
                "WFM_TABLE1_DEVICE_MASKS",
                "WFM_TABLE1_FUSED_GAUSSIAN", "WFM_TABLE1_BATCH_LPIPS",
                "WFM_TABLE1_GRAPH_LPIPS", "WFM_TABLE1_SSIM",
            ) if key in os.environ
        },
        "source_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml", ".sbatch", ".json", ".txt")
        },
        "extension_sha256": extensions,
    }


def write_provenance(output):
    with (output / "runtime.json").open("x") as handle:
        handle.write(json.dumps(capture(), indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_provenance(args.output)
