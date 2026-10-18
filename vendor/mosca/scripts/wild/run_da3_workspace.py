#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_intrinsics(path: Path, count: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "K" in payload:
        k = np.asarray(payload["K"], dtype=np.float32)
    else:
        k = np.array(
            [
                [float(payload["fx"]), 0.0, float(payload["cx"])],
                [0.0, float(payload["fy"]), float(payload["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    if k.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 K matrix in {path}, got {k.shape}")
    return np.repeat(k[None], count, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser("Run DepthAnything3 on an existing image workspace.")
    parser.add_argument("--ws", required=True, type=Path)
    parser.add_argument("--rgb-dirname", default="images")
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weights",
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )
    args = parser.parse_args()

    from lib_prior.depth_models.depthanything3_wrapper import (
        depthanything3_process_folder,
        get_depthanything3_model,
    )
    from mosca_precompute import load_imgs_from_dir

    ws = args.ws.expanduser().resolve()
    img_list, img_fns = load_imgs_from_dir(str(ws), rgb_dirname=args.rgb_dirname)
    intrinsics = None
    if args.intrinsics_json is not None:
        intrinsics_path = args.intrinsics_json
        if not intrinsics_path.is_absolute():
            intrinsics_path = ws / intrinsics_path
        intrinsics = load_intrinsics(intrinsics_path, len(img_fns))

    model = get_depthanything3_model(device=args.device, weights=args.weights)
    depthanything3_process_folder(
        model=model,
        img_list=img_list,
        fn_list=img_fns,
        dst=str(ws),
        intrinsics=intrinsics,
    )


if __name__ == "__main__":
    main()
