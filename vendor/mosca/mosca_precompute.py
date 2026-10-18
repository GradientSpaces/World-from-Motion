import torch
import imageio
import os, os.path as osp
import numpy as np
import sys

from lib_prior.moca_processor import MoCaPrep
from lib_prior.preprocessor_utils import load_imgs, convert_from_mp4
from lib_prior.prior_loading import Saved2D, visualize_track
from lib_prior.moca_processor import mark_dynamic_region

from lib_render.render_helper import GS_BACKEND

from lib_moca.moca import moca_solve
from lib_moca.epi_helpers import analyze_track_epi, identify_tracks
from lib_moca.camera import MonocularCameras

from viz_utils import viz_list_of_colored_points_in_cam_frame
import logging
from lib_prior.moca_processor import *
from omegaconf import OmegaConf
from lib_moca.moca_misc import make_pair_list
import random

from data_utils.iphone_helpers import load_iphone_multiview_manifest


def seed_everything(seed):
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")


def get_moca_processor(pre_cfg):
    moca_processor = MoCaPrep(
        dep_mode=getattr(
            pre_cfg, "dep_mode", "sensor"
        ),  # "depthcrafter", "metric3d", "uni"
        tap_mode=getattr(
            pre_cfg, "tap_mode", "bootstapir"
        ),  # "spatracker", "cotracker"
        flow_mode=getattr(pre_cfg, "flow_mode", "raft"),
        flow_checkpoint=getattr(pre_cfg, "flow_checkpoint", None),
        cowtracker_checkpoint=getattr(pre_cfg, "cowtracker_checkpoint", None),
        align_metric_flag=getattr(pre_cfg, "align_metric_flag", True),
    )
    return moca_processor


def normalize_gpu_spec(gpus):
    if gpus is None:
        return None
    if isinstance(gpus, str):
        return None if gpus.strip() == "" else gpus
    if isinstance(gpus, (list, tuple)):
        return ",".join(str(it) for it in gpus)
    return str(gpus)


def load_imgs_from_dir(src, rgb_dirname="images"):
    img_dir = osp.join(src, rgb_dirname)
    img_fns = sorted(
        [it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")]
    )
    img_list = [imageio.imread(osp.join(img_dir, it))[..., :3] for it in img_fns]
    return img_list, img_fns


def load_imgs_from_mp4():
    raise RuntimeError("Not implemented yet")
    return


def _get_multiview_tap_camera_infos(ws, pre_cfg):
    manifest_name = getattr(pre_cfg, "multiview_manifest", "multiview_manifest.json")
    manifest_path = osp.join(ws, manifest_name)
    if not osp.exists(manifest_path):
        logging.warning(
            f"Skip multiview TAP priors because manifest is missing: {manifest_path}"
        )
        return []
    manifest = load_iphone_multiview_manifest(manifest_path)
    selected_cameras = getattr(pre_cfg, "multiview_tap_cameras", None)
    if selected_cameras is None:
        selected_cameras = [
            camera["camera_name"]
            for camera in manifest["cameras"]
            if camera["camera_name"] != manifest["anchor_camera"]
        ]
    else:
        selected_cameras = list(selected_cameras)
    selected_cameras = set(selected_cameras)
    ret = []
    for camera in manifest["cameras"]:
        camera_name = camera["camera_name"]
        if camera_name not in selected_cameras:
            continue
        tap_prior_dir = camera.get(
            "tap_prior_dir", osp.join("multiview", "tap_priors", camera_name)
        )
        ret.append((camera_name, camera["rgb_dir"], tap_prior_dir))
    return ret


def compute_multiview_tap_priors(ws, moca_processor: MoCaPrep, pre_cfg: OmegaConf):
    if not getattr(pre_cfg, "use_multiview_tap_priors", False):
        return
    if moca_processor.tap_mode == "spatracker":
        logging.warning(
            "Multiview TAP priors skip spatracker because auxiliary extraction is RGB-only"
        )
        return

    camera_infos = _get_multiview_tap_camera_infos(ws, pre_cfg)
    if len(camera_infos) == 0:
        logging.warning("No multiview cameras selected for auxiliary TAP priors")
        return

    chunk_size = getattr(pre_cfg, "multiview_tap_chunk_size", None)
    if chunk_size is None:
        chunk_size = getattr(pre_cfg, "tap_chunk_size", 5000)
    n_track = getattr(
        pre_cfg, "multiview_n_track_uniform", getattr(pre_cfg, "n_track_uniform", 8192)
    )
    save_name = getattr(pre_cfg, "multiview_tap_save_name", "uniform_dep=sensor")

    for camera_name, rgb_dirname, tap_prior_dir in camera_infos:
        logging.info(
            f"Extract multiview TAP priors for {camera_name} from {rgb_dirname} into {tap_prior_dir}"
        )
        img_list, _ = load_imgs_from_dir(ws, rgb_dirname=rgb_dirname)
        sample_mask = np.ones((len(img_list), img_list[0].shape[0], img_list[0].shape[1]))
        moca_processor.compute_tap(
            ws=osp.join(ws, tap_prior_dir),
            save_name=save_name,
            n_track=n_track,
            img_list=img_list,
            mask_list=sample_mask > 0,
            dep_list=None,
            chunk_size=chunk_size,
            max_viz_cnt=getattr(pre_cfg, "max_viz_cnt", 512),
        )


def preprocess(
    img_list: list,
    img_fns: list,
    ws: str,
    moca_processor: MoCaPrep,
    pre_cfg: OmegaConf,
    resample_for_dynamic=True,
):
    seed_everything(getattr(pre_cfg, "seed", 12345))
    start_t = time.time()
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)
    logging.info(f"Working on {ws}, start phase-1 preprocessing")
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)

    BOUNDARY_EHNAHCE_TH = getattr(pre_cfg, "boundary_enhance_th", -1)
    DEPTH_DIR_POSTFIX = "_depth_sharp" if BOUNDARY_EHNAHCE_TH > 0 else "_depth"
    RGB_DIR = getattr(
        pre_cfg, "rgb_dirname", getattr(pre_cfg, "anchor_rgb_dirname", "images")
    )

    EPI_TH = getattr(pre_cfg, "epi_th", 1e-3)
    DEPTH_BOUNDARY_TH = getattr(
        pre_cfg, "depth_boundary_th", 1.0
    )  # this is in the median=1.0 space

    TAP_CHUNK_SIZE = getattr(pre_cfg, "tap_chunk_size", 5000)

    moca_processor.process(
        t_list=None,
        img_list=img_list,
        img_name_list=img_fns,
        save_dir=ws,
        n_track=getattr(pre_cfg, "n_track_uniform", 8192),
        # depth crafter
        depthcrafter_denoising_steps=getattr(
            pre_cfg, "depthcrafter_denoising_steps", 25
        ),
        metric_alignment_frames=getattr(pre_cfg, "metric_alignment_frames", 10),
        metric_alignment_first_quantil=getattr(
            pre_cfg, "metric_alignment_first_quantil", 0.7
        ),
        metric_alignment_bias_flag=getattr(pre_cfg, "metric_alignment_bias_flag", True),
        metric_alignment_kernel=getattr(pre_cfg, "metric_alignment_kernel", "cauchy"),
        metric_alignment_fscale=getattr(pre_cfg, "metric_alignment_fscale", 0.001),
        # TAP
        compute_tap=True,
        tap_chunk_size=TAP_CHUNK_SIZE,
        # Flow
        flow_steps=getattr(pre_cfg, "flow_steps", [1, 3]),
        epi_num_threads=getattr(pre_cfg, "epi_num_threads", 64),
        dynamic_mask_mode=getattr(pre_cfg, "dynamic_mask_mode", "raft_epi"),
        sam3_empty_fallback_to_flow=getattr(
            pre_cfg, "sam3_empty_fallback_to_flow", False
        ),
        sam3_text=getattr(pre_cfg, "sam3_text", None),
        sam3_gpus=normalize_gpu_spec(getattr(pre_cfg, "sam3_gpus", None)),
        sam3_checkpoint=getattr(pre_cfg, "sam3_checkpoint", None),
        sam3_bpe=getattr(pre_cfg, "sam3_bpe", None),
        sam3_lib_prior_root=getattr(pre_cfg, "sam3_lib_prior_root", None),
        sam3_fps=getattr(pre_cfg, "sam3_fps", 10.0),
        sam3_seed_frame_indices=getattr(pre_cfg, "sam3_seed_frame_indices", None),
        # Dep enhance for spatracker
        boundary_enhance_th=BOUNDARY_EHNAHCE_TH,  # if > 0 will create a sharp dir
        # boost
        compute_flow=getattr(pre_cfg, "compute_flow", True),
    )

    if resample_for_dynamic:
        logging.info("*" * 20 + " Preprocessing " + "*" * 20)
        logging.info(f"Working on {ws}, start phase-2 preprocessing, densify the fg TAP")

        s2d = (
            Saved2D(ws, rgb_dirname=RGB_DIR)
            .load_epi()
            .load_dep(f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}", DEPTH_BOUNDARY_TH)
            .normalize_depth(median_depth=1.0)
            .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
            .load_track(f"*uniform*{moca_processor.tap_mode}", min_valid_cnt=4)
            .load_vos()
        )

        if hasattr(s2d, "epi"):
            sample_mask = s2d.epi > EPI_TH
        else:
            continuous_pair_list = make_pair_list(s2d.T, interval=[1, 4], dense_flag=True)
            F_list, epierr_list, _ = analyze_track_epi(
                continuous_pair_list, s2d.track, s2d.track_mask, H=s2d.H, W=s2d.W
            )
            track_static_selection, _ = identify_tracks(epierr_list, EPI_TH)
            sample_mask = mark_dynamic_region(
                s2d.track[:, ~track_static_selection],
                s2d.track_mask[:, ~track_static_selection],
                s2d.H,
                s2d.W,
                0.1,
            )
        resampling_mask_dilate_ksize = getattr(pre_cfg, "resampling_mask_dilate_ksize", 7)
        sample_mask = (
            torch.nn.functional.max_pool2d(
                sample_mask[:, None].float(),
                kernel_size=resampling_mask_dilate_ksize,
                stride=1,
                padding=(resampling_mask_dilate_ksize - 1) // 2,
            )[:, 0]
            > 0.5
        )
        imageio.mimsave(
            osp.join(ws, "epi_resample_mask.gif"),
            sample_mask.cpu().numpy().astype(np.uint8) * 255,
        )

        moca_processor.compute_tap(
            ws=ws,
            save_name=f"dynamic_dep={moca_processor.dep_mode}",
            n_track=getattr(pre_cfg, "n_track_pdynamic", 8192 * 3),
            img_list=img_list,
            mask_list=sample_mask.detach().cpu().numpy() > 0,
            dep_list=moca_processor.load_dep_list(
                ws, f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}"
            ),
            max_viz_cnt=getattr(pre_cfg, "max_viz_cnt", 512),
            chunk_size=TAP_CHUNK_SIZE,
        )
    else:
        logging.info("Skip dynamic resample phase")

    compute_multiview_tap_priors(ws, moca_processor, pre_cfg)

    duration = (time.time() - start_t) / 60.0
    logging.info(f"Preprocessing done, time cost: {duration:.3f}min")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoSca-V2 Preprocessing")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument(
        "--skip_dynamic_resample", action="store_true", help="skip dynamic resample"
    )
    args, unknown = parser.parse_known_args()

    prep_cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip('--') for arg in unknown])
    prep_cfg = OmegaConf.merge(prep_cfg, cli_cfg)
    img_list, img_fns = load_imgs_from_dir(
        args.ws,
        rgb_dirname=getattr(
            prep_cfg, "rgb_dirname", getattr(prep_cfg, "anchor_rgb_dirname", "images")
        ),
    )

    moca_processor = get_moca_processor(prep_cfg)

    preprocess(
        img_list=img_list,
        img_fns=img_fns,
        ws=args.ws,
        moca_processor=moca_processor,
        pre_cfg=prep_cfg,
        resample_for_dynamic=not args.skip_dynamic_resample,
    )
