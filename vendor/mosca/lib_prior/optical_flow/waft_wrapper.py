import argparse
import json
import os
import os.path as osp
import sys
from contextlib import contextmanager
from copy import deepcopy

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

from .flow_io import save_flow_bundle


DEFAULT_WAFT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "WAFT"))
DEFAULT_WAFT_CFG = osp.join(DEFAULT_WAFT_ROOT, "config", "a1", "tar-c-t-kitti.json")
DEFAULT_WAFT_CKPT = osp.abspath(
    osp.join(
        osp.dirname(__file__),
        "..",
        "..",
        "weights",
        "waft_models",
        "waft-a1-downstream.pth",
    )
)
DEFAULT_WAFT_LONG_DIM = 768


@contextmanager
def _working_directory(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _ensure_waft_import_paths(waft_root=DEFAULT_WAFT_ROOT):
    waft_root = osp.abspath(waft_root)
    if not osp.exists(osp.join(waft_root, "model", "__init__.py")):
        raise FileNotFoundError(f"WAFT repo root not found at {waft_root}")

    if waft_root in sys.path:
        sys.path.remove(waft_root)
    sys.path.insert(0, waft_root)

    # WAFT uses top-level imports named `model`, `utils`, and `config`.
    # Drop conflicting modules from other packages before importing WAFT.
    for module_name in ["model", "utils", "config", "inference_tools", "thirdparty"]:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", "") if module is not None else ""
        if module is not None and module_file and not osp.abspath(module_file).startswith(waft_root):
            del sys.modules[module_name]
    return waft_root


def _namespace_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return argparse.Namespace(**data)


def _require_file(path, description):
    if not osp.exists(path):
        raise FileNotFoundError(f"{description} not found at {path}")
    return path


def _load_image(input_img, long_dim=DEFAULT_WAFT_LONG_DIM):
    if isinstance(input_img, str):
        img = imageio.imread(input_img)[..., :3]
    else:
        assert isinstance(input_img, np.ndarray)
        assert input_img.ndim == 3 and input_img.shape[2] == 3, f"{input_img.shape}"
        img = input_img[..., :3]
    img = img.astype(np.uint8).copy()
    ori_shape = img.shape[:2]

    if long_dim is not None:
        if img.shape[0] > img.shape[1]:
            input_h = long_dim
            input_w = int(round(float(input_h) / img.shape[0] * img.shape[1]))
        else:
            input_w = long_dim
            input_h = int(round(float(input_w) / img.shape[1] * img.shape[0]))
        img = cv2.resize(img, (input_w, input_h), cv2.INTER_LINEAR)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None], ori_shape


def _resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    return cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)


def _warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]
    return cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = _warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1 * (
        np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1)
    ) + alpha_2

    fwd2bwd_flow = _warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)
    bwd_mask = bwd_lr_error < alpha_1 * (
        np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1)
    ) + alpha_2

    return fwd_mask, bwd_mask


def get_dense_pair_list(name_list, jump_steps=[1]):
    names = deepcopy(name_list)
    names.sort()
    pair_list = []
    for step in jump_steps:
        for i in range(len(names) - step):
            pair_list.append((names[i], names[i + step]))
    return pair_list


class WAFTFlowModel:
    def __init__(self, model, wrapped_model, args, waft_root, long_dim=DEFAULT_WAFT_LONG_DIM):
        self.model = model
        self.wrapped_model = wrapped_model
        self.args = args
        self.waft_root = waft_root
        self.long_dim = long_dim

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def calc_flow(self, image1, image2, iters=None):
        old_iters = getattr(self.args, "iters", None)
        if iters is not None:
            self.args.iters = iters
        try:
            output = self.wrapped_model.calc_flow(image1, image2)
        finally:
            if iters is not None:
                self.args.iters = old_iters
        return output["flow"][-1]


def get_waft_model(
    ckpt_path=DEFAULT_WAFT_CKPT,
    device="cuda",
    cfg_path=DEFAULT_WAFT_CFG,
    scale=None,
    waft_root=DEFAULT_WAFT_ROOT,
    long_dim=DEFAULT_WAFT_LONG_DIM,
    pad_to_train_size=False,
    tiling=False,
):
    waft_root = _ensure_waft_import_paths(waft_root)
    ckpt_path = _require_file(ckpt_path, "WAFT checkpoint")
    cfg_path = _require_file(cfg_path, "WAFT config")

    from inference_tools import InferenceWrapper
    from model import fetch_model
    from utils.utils import load_ckpt

    args = _namespace_from_json(cfg_path)
    args.cfg = cfg_path
    args.ckpt = ckpt_path
    if scale is not None:
        args.scale = scale

    with _working_directory(waft_root):
        model = fetch_model(args)
        load_ckpt(model, ckpt_path)

    model.to(device)
    model.eval()
    wrapped_model = InferenceWrapper(
        model,
        scale=getattr(args, "scale", 0),
        train_size=getattr(args, "image_size", None),
        pad_to_train_size=pad_to_train_size,
        tiling=tiling,
    )
    return WAFTFlowModel(model, wrapped_model, args, waft_root, long_dim=long_dim)


@torch.no_grad()
def waft_process_folder(
    model,
    img_list,
    img_name_list,
    dst_dir,
    pair_list=None,
    iters=None,
    step_list=[1],
    save_pair_files=True,
):
    if pair_list is None:
        pair_list = get_dense_pair_list(img_name_list, jump_steps=step_list)
    device = next(model.parameters()).device
    os.makedirs(dst_dir, exist_ok=True)

    _ensure_waft_import_paths(model.waft_root)
    from utils.flow_viz import flow_to_image

    flow_viz_list, flow_mask_viz_list = [], []
    flow_records = []
    for vi, vj in tqdm(pair_list):
        image1, img_shape = _load_image(img_list[img_name_list.index(vi)], long_dim=model.long_dim)
        image2, _ = _load_image(img_list[img_name_list.index(vj)], long_dim=model.long_dim)
        image1, image2 = image1.to(device), image2.to(device)

        flow_fwd = model.calc_flow(image1, image2, iters=iters)
        flow_bwd = model.calc_flow(image2, image1, iters=iters)

        flow_fwd = flow_fwd[0].cpu().numpy().transpose(1, 2, 0)
        flow_bwd = flow_bwd[0].cpu().numpy().transpose(1, 2, 0)

        flow_fwd = _resize_flow(flow_fwd, img_shape[0], img_shape[1])
        flow_bwd = _resize_flow(flow_bwd, img_shape[0], img_shape[1])

        mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

        flow_records.append(
            dict(src_name=vi, dst_name=vj, flow=flow_fwd, mask=mask_fwd)
        )
        flow_records.append(
            dict(src_name=vj, dst_name=vi, flow=flow_bwd, mask=mask_bwd)
        )

        if vi < vj:
            flow_viz_list.append(flow_to_image(flow_fwd))
            flow_mask_viz_list.append(mask_fwd.astype(np.uint8) * 255)

    save_flow_bundle(
        dst_dir,
        flow_records,
        frame_names=img_name_list,
        save_pair_files=save_pair_files,
    )
    if flow_viz_list:
        imageio.mimsave(osp.join(osp.dirname(dst_dir), "flow_viz.mp4"), flow_viz_list)
        imageio.mimsave(
            osp.join(osp.dirname(dst_dir), "flow_mask_viz.mp4"), flow_mask_viz_list
        )
