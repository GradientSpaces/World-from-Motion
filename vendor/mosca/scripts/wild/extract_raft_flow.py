#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_PRIOR_ROOT = REPO_ROOT / "lib_prior"
for path in (REPO_ROOT, LIB_PRIOR_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RAFT optical flow for an existing wild MoSca workspace."
    )
    parser.add_argument("--ws", required=True, help="Workspace containing images/.")
    parser.add_argument("--rgb_dirname", default="images")
    parser.add_argument("--flow_dirname", default="flow_raft")
    parser.add_argument("--steps", default="1,3", help="Comma-separated frame intervals.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="RAFT checkpoint. Defaults to weights/raft_models/raft-things.pth.",
    )
    return parser.parse_args()


def main() -> None:
    import torch

    from lib_prior.optical_flow.raft_wrapper import get_raft_model, raft_process_folder
    from mosca_precompute import load_imgs_from_dir

    args = parse_args()
    ws = Path(args.ws).expanduser().resolve()
    if not ws.is_dir():
        raise FileNotFoundError(f"Workspace not found: {ws}")

    img_list, img_fns = load_imgs_from_dir(str(ws), rgb_dirname=args.rgb_dirname)
    if len(img_list) < 2:
        raise ValueError(f"Need at least two frames for flow, got {len(img_list)}")

    ckpt = (
        Path(args.ckpt).expanduser().resolve()
        if args.ckpt is not None
        else REPO_ROOT / "weights" / "raft_models" / "raft-things.pth"
    )
    if not ckpt.is_file():
        raise FileNotFoundError(f"RAFT checkpoint not found: {ckpt}")

    model = get_raft_model(str(ckpt), args.device)
    try:
        raft_process_folder(
            model=model,
            img_list=img_list,
            img_name_list=img_fns,
            dst_dir=osp.join(str(ws), args.flow_dirname),
            step_list=parse_int_csv(args.steps),
        )
    finally:
        model.cpu()
        del model
        torch.cuda.empty_cache()

    print(f"Saved RAFT flow under {ws / args.flow_dirname}")


if __name__ == "__main__":
    main()
