import numpy as np
import torch
import imageio
import sys, os, os.path as osp
import logging
import kornia
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D

from lib_render.render_helper import GS_BACKEND

from lib_moca.camera import MonocularCameras

from lib_mosca.mosca import MoSca
from lib_mosca.dynamic_solver import get_dynamic_curves
from lib_mosca.dynamic_solver import geometry_scf_init
from lib_mosca.photo_recon_utils import OptimCFG, GSControlCFG
from lib_mosca.mosca import MoSca
from lib_mosca.photo_recon import DynReconstructionSolver
from lib_mosca.static_gs import StaticGaussian
from lib_mosca.misc import seed_everything
from lib_mosca.dynamic_solver_utils import (
    round_int_coordinates,
    query_image_buffer_by_pix_int_coord,
)

from mosca_viz import viz_main, viz_list_of_colored_points_in_cam_frame
from mosca_evaluate_ours import (
    test_fps,
    test_main_ours_v3,
    test_main_ours_v3_per_test_track_joint,
    test_main_ours_v3_post_stats,
)
from lite_moca_reconstruct import static_reconstruct

from recon_utils import (
    SEED,
    seed_everything,
    setup_recon_ws,
    setup_recon_ours,
    resume_recon_ours, 
    auto_get_depth_dir_tap_mode,
    update_s2d_track_identification,
    viz_mosca_curves_before_optim,
    set_epi_mask_to_s2d_for_bg_render,
)
from lib_mosca.photo_recon_utils import DynamicSegTreeBFS
from recon_utils import parse_interval_from_dir, find_root, find_leaves, imgs_to_video, find_all_renders, find_all_nodes_and_intervals

from multiprocessing import Pool
from functools import partial
from lib_mosca.photo_recon_utils import get_available_gpus
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import copy
import signal
from threading import Event, Lock

from lib_mosca.photo_recon_utils import split_cams

def get_all_leaves(tree_node):
    if tree_node.is_leaf():
        rst_list = []
        rst_list.append(tree_node)
        return rst_list
    left_list = get_all_leaves(tree_node.left_child)
    right_list = get_all_leaves(tree_node.right_child)
    rst_list = []
    rst_list.extend(left_list)
    rst_list.extend(right_list)
    return rst_list

def process_layer(process_height, logdir, args, cfg, datamode, gpu_id):
    '''
    need to get:
        all involved nodes list, len = N
        the corresponding time interval of all nodes (list), len = N
    '''
    torch.cuda.set_device(gpu_id)
    node_device = torch.device(f"cuda:{gpu_id}")
    # 设置当前进程使用的GPU
    print(f"gpu-id:{gpu_id} initializing.\n")

    use_fachain = getattr(cfg, "use_fachain", True)
    eval_tto = getattr(cfg, "eval_tto", True)
    eval_tto_mode = getattr(cfg, "eval_tto_mode", "per_frame")
    print(f"gpu-id:{gpu_id}. use fachain: {use_fachain}.\n")
    
    all_cams_list = []

    dir_list = sorted(os.listdir(logdir))
    all_nodename_list = []
    for dir_name in dir_list:
        interval = parse_interval_from_dir(dir_name)
        if interval:
            all_nodename_list.append(dir_name)
    root_name = find_root(all_nodename_list)
    interval = parse_interval_from_dir(root_name)
    assert interval
    start_t, end_t = interval
    segtree = ["None" for i in range(1000)]
    leaves_list = find_leaves(logdir, all_nodename_list, segtree, 1) # NOTE: construct segtree
    all_nodes_and_intervals_list = find_all_nodes_and_intervals(logdir, segtree, 1, process_height, post_cut=getattr(cfg, "min_interval", 12))
    assert all_nodes_and_intervals_list and len(all_nodes_and_intervals_list) > 0

    for idx in range(0, len(all_nodes_and_intervals_list)):
        node_name = all_nodes_and_intervals_list[idx]["node_name"]
        interval = parse_interval_from_dir(node_name)
        assert interval
        left_bound, right_bound = interval
        left_t, right_t = all_nodes_and_intervals_list[idx]["left_t"], all_nodes_and_intervals_list[idx]["right_t"]
        node_cam_dir = os.path.join(logdir, node_name, "photometric_cam.pth")
        if left_bound == left_t and right_bound == right_t:
            node_cam = MonocularCameras.load_from_ckpt(
                torch.load(node_cam_dir, map_location=torch.device('cpu'))
            ).to(node_device)
        else:
            if left_bound == left_t:
                node_cam = split_cams(node_cam_dir, right_t+1, "left", device=node_device)
            elif right_bound == right_t:
                node_cam = split_cams(node_cam_dir, left_t, "right", device=node_device)
            else:
                print(f"gpuid: {gpu_id}. bound: [{left_bound}, {right_bound}]. t: [{left_t}, {right_t}]. no such case.")
                exit(-1)
        all_cams_list.append(node_cam)
    
    # assert check
    for idx in range(0, len(all_nodes_and_intervals_list)):
        node_cam = all_cams_list[idx]
        node_name = all_nodes_and_intervals_list[idx]["node_name"]
        interval = parse_interval_from_dir(node_name)
        assert interval
        left_bound, right_bound = interval
        left_t, right_t = all_nodes_and_intervals_list[idx]["left_t"], all_nodes_and_intervals_list[idx]["right_t"]
        # start assert
        assert node_cam.T == right_t - left_t + 1
        assert left_bound <= left_t and right_t <= right_bound
        if idx != len(all_nodes_and_intervals_list) - 1:
            assert right_t + 1 == all_nodes_and_intervals_list[idx+1]["left_t"]

    rendering_dir_name = getattr(cfg, "eval_rendering_dir_name", None)
    if rendering_dir_name is None:
        rendering_dir_name = f"rendering_results_max_height_{str(process_height)}"
    else:
        rendering_dir_name = rendering_dir_name.format(height=process_height)
    rendering_dir = os.path.join(logdir, rendering_dir_name)
    if os.path.exists(rendering_dir):
        os.system(f"rm -r {rendering_dir}")
    os.makedirs(rendering_dir)
    all_frames = []
    valid_tto_modes = {"per_frame", "per_test_track_joint"}
    if eval_tto_mode not in valid_tto_modes:
        raise ValueError(
            f"Unknown eval_tto_mode={eval_tto_mode}, expected one of {sorted(valid_tto_modes)}"
        )
    if eval_tto and eval_tto_mode == "per_test_track_joint":
        saved_dir_list = []
        interval_bound_list = []
        for idx in range(0, len(all_nodes_and_intervals_list)):
            node_name = all_nodes_and_intervals_list[idx]["node_name"]
            left_t = all_nodes_and_intervals_list[idx]["left_t"]
            right_t = all_nodes_and_intervals_list[idx]["right_t"]
            saved_dir_list.append(os.path.join(logdir, node_name))
            interval_bound_list.append((left_t, right_t))
        print(
            f"gpu-id:{gpu_id}: evaluate {rendering_dir} with per_test_track_joint TTO.\n"
        )
        if datamode in [
            "nvidia_ours",
            "nvidia_ours_v2",
            "nvidia_ours_v3",
            "iphone_ours_v3",
            "nerfds_ours_v3",
        ]:
            all_frames = test_main_ours_v3_per_test_track_joint(
                cfg,
                saved_dir_list=saved_dir_list,
                interval_bound_list=interval_bound_list,
                data_root=args.ws,
                device=torch.device(f"cuda:{gpu_id}"),
                rendering_dir=rendering_dir,
                layer_cams_list=all_cams_list,
                use_fachain=use_fachain,
                segtree=segtree,
            )
    else:
        for idx in range(0, len(all_nodes_and_intervals_list)):
            node_name = all_nodes_and_intervals_list[idx]["node_name"]
            left_t, right_t = all_nodes_and_intervals_list[idx]["left_t"], all_nodes_and_intervals_list[idx]["right_t"]

            leaf_dir = os.path.join(logdir, node_name)

            print(f"gpu-id:{gpu_id}: evaluate {rendering_dir}.\n")
            if datamode in ["nvidia_ours", "nvidia_ours_v2", "nvidia_ours_v3", "iphone_ours_v3", "nerfds_ours_v3"]:
                node_frames = test_main_ours_v3(
                    cfg,
                    saved_dir=leaf_dir,
                    data_root=args.ws,
                    device=torch.device(f"cuda:{gpu_id}"),
                    tto_flag=eval_tto,
                    eval_also_dyncheck_non_masked=False,
                    skip_test_gen=False,
                    left_bound=left_t, 
                    right_bound=right_t,
                    node_depth=process_height, 
                    layer_cams_list=all_cams_list, 
                    rendering_dir=rendering_dir, 
                    use_fachain=use_fachain, 
                    segtree=segtree, 
                )
                if len(all_frames) == 0:
                    all_frames = node_frames
                else:
                    for test_cam_i in range(len(all_frames)):
                        if (all_frames[test_cam_i] is not None) and (node_frames[test_cam_i] is not None):
                            all_frames[test_cam_i].extend(node_frames[test_cam_i])
                        elif (all_frames[test_cam_i] is None) and (node_frames[test_cam_i] is not None):
                            all_frames[test_cam_i] = node_frames[test_cam_i]
    if datamode in ["nvidia_ours", "nvidia_ours_v2", "nvidia_ours_v3", "iphone_ours_v3", "nerfds_ours_v3"]:
        test_main_ours_v3_post_stats(
            cfg,
            data_root=args.ws,
            device=torch.device(f"cuda:{gpu_id}"),
            tto_flag=eval_tto,
            eval_also_dyncheck_non_masked=False,
            skip_test_gen=False,
            left_bound=-1, # stats all frames
            right_bound=-1, # stats all frames
            layer_cams_list=all_cams_list, 
            rendering_dir=rendering_dir, 
            all_frames=all_frames, 
        )
    for test_cam_i in range(len(all_frames)):
        imageio.mimsave(osp.join(rendering_dir, f"test_cam{test_cam_i}.mp4"), all_frames[test_cam_i])
    print(f"gpu-id:{gpu_id}: done for evaluate layer {process_height}.\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoSca-V2 Reconstruction")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--model_path", type=str, help="Model folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument("--no_viz", action="store_true", help="no viz")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    logdir = resume_recon_ours(args.model_path)

    dir_list = sorted(os.listdir(logdir))
    nodename_list = []
    for dir_name in dir_list:
        interval = parse_interval_from_dir(dir_name)
        if interval:
            nodename_list.append(dir_name)
    root_name = find_root(nodename_list)
    interval = parse_interval_from_dir(root_name)
    assert interval
    start_t, end_t = interval

    tree_height = getattr(cfg, "stack_depth_max", 3)
    eval_tree_height_start = getattr(cfg, "eval_tree_height_start", 0)

    datamode = getattr(cfg, "mode", "iphone")

    # TODO: height parallel
    gpu_list = get_available_gpus()
    gpu_list = [gpu_id for gpu_id in range(len(gpu_list))]
    for max_tree_height in reversed(range(eval_tree_height_start, tree_height+1, len(gpu_list))):
        try:
            h_start = max_tree_height
            h_end = max_tree_height + len(gpu_list)
            h_end = min(tree_height+1, h_end)
            mx_wks = h_end - h_start
            with ThreadPoolExecutor(max_workers=mx_wks) as executor:
                futures = []
                for process_height in range(h_start, h_end):
                    gpu_id = gpu_list[process_height % len(gpu_list)]
                    futures.append(
                        executor.submit(
                            process_layer, 
                            process_height, logdir, args, cfg, datamode, gpu_id
                        )
                    )
            for future in futures:
                result = future.result()
        except Exception as e:
            torch.cuda.empty_cache()
            raise e
