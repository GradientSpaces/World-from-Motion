from __future__ import annotations

import os.path as osp
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_relative(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((REPO_ROOT / path_obj).resolve())


def read_commandline_tokens(path: str) -> list[str]:
    line = Path(path).read_text(encoding="utf-8").strip()
    if not line:
        raise ValueError(f"Command line file is empty: {path}")
    return shlex.split(line)


def get_flag_value(tokens: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def infer_cfg_path(saved_dir: str, cfg_override: str | None = None) -> str:
    if cfg_override is not None:
        resolved = resolve_repo_relative(cfg_override)
        return resolved if osp.exists(resolved) else cfg_override
    args_path = osp.join(saved_dir, "fit_commandline_args.txt")
    if not osp.isfile(args_path):
        raise FileNotFoundError(
            f"Missing --cfg override and fit_commandline_args.txt was not found in {saved_dir}"
        )
    tokens = read_commandline_tokens(args_path)
    cfg_path = get_flag_value(tokens, "--cfg") or get_flag_value(tokens, "--fit_cfg")
    if cfg_path is None:
        raise ValueError(f"Could not infer --cfg from {args_path}")
    resolved = resolve_repo_relative(cfg_path)
    return resolved if osp.exists(resolved) else cfg_path


def infer_workspace_path(saved_dir: str, ws_override: str | None = None) -> str:
    if ws_override is not None:
        return osp.abspath(ws_override)
    args_path = osp.join(saved_dir, "fit_commandline_args.txt")
    if not osp.isfile(args_path):
        raise FileNotFoundError(
            f"Missing --ws override and fit_commandline_args.txt was not found in {saved_dir}"
        )
    tokens = read_commandline_tokens(args_path)
    ws_path = get_flag_value(tokens, "--ws")
    if ws_path is None:
        raise ValueError(f"Could not infer --ws from {args_path}")
    resolved = resolve_repo_relative(ws_path)
    return osp.abspath(resolved if osp.exists(resolved) else ws_path)
