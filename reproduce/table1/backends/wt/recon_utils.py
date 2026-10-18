import torch
import json
import logging, glob, sys, os, shutil, os.path as osp
from datetime import datetime
import numpy as np, random, kornia
import imageio

from lib_render.render_helper import GS_BACKEND
from mosca_viz import viz_main, viz_list_of_colored_points_in_cam_frame
from lib_moca.bundle import query_buffers_by_track
from lib_moca.epi_helpers import analyze_track_epi, identify_tracks

SEED = 12345


def seed_everything(seed=SEED):
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_recon_ws(ws, fit_cfg, logdir="logs"):
    seed_everything(SEED)
    # get datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    name = getattr(fit_cfg, "exp_name", "default")
    name = f"{name}_{GS_BACKEND.lower()}_{dt_string}"
    log_path = osp.join(ws, logdir, name)
    os.makedirs(log_path, exist_ok=True)
    logging.info(f"Log dir set to: {log_path}")
    # backup the files
    logging.info(f"Backup files")
    backup_dir = osp.join(log_path, "src_backup")
    os.makedirs(backup_dir, exist_ok=True)
    for path in [
        "profile",
        "lib_prior",
        "lib_moca",
        "lib_mosca",
        "lite_moca_reconstruct.py",
        "mosca_reconstruct.py",
        "mosca_evaluate.py",
        "mosca_precompute.py",
        "mosca_viz.py",
    ]:
        os.system(f"cp -r {path} {backup_dir}")
    # reduce the backup size
    shutil.rmtree(osp.join(backup_dir, "lib_prior", "seg"))
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            if file.endswith(".pth") or file.endswith(".ckpt"):
                if osp.isfile(osp.join(root, file)):
                    os.remove(osp.join(root, file))
                else:
                    shutil.rmtree(osp.join(root, file))
    # backu the commandline args
    with open(osp.join(log_path, "fit_commandline_args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    return log_path

def setup_recon_ours(logdir, fit_cfg):
    seed_everything(SEED)
    # get datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    name = getattr(fit_cfg, "exp_name", "default")
    name = f"{name}_{GS_BACKEND.lower()}_{dt_string}"
    log_path = osp.join(logdir, name)
    os.makedirs(log_path, exist_ok=True)
    logging.info(f"Log dir set to: {log_path}")
    # backup the files
    logging.info(f"Backup files")
    backup_dir = osp.join(log_path, "src_backup")
    os.makedirs(backup_dir, exist_ok=True)
    for path in [
        "profile",
        "lib_prior",
        "lib_moca",
        "lib_mosca",
        "scripts", 
        "lite_moca_reconstruct.py",
        "mosca_reconstruct.py",
        "mosca_reconstruct_debug.py",
        "mosca_evaluate.py",
        "mosca_evaluate_ours.py",
        "mosca_precompute.py",
        "mosca_viz.py",
        "mosca_calmetrics.py", 
        "mosca_precompute.py", 
        "mosca_onlyeval.py", 
    ]:
        os.system(f"cp -r {path} {backup_dir}")
    # reduce the backup size
    shutil.rmtree(osp.join(backup_dir, "lib_prior", "seg"))
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            if file.endswith(".pth") or file.endswith(".ckpt"):
                if osp.isfile(osp.join(root, file)):
                    os.remove(osp.join(root, file))
                else:
                    shutil.rmtree(osp.join(root, file))
    # backu the commandline args
    with open(osp.join(log_path, "fit_commandline_args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    return log_path

def resume_recon_ours(logdir):
    seed_everything(SEED)
    all_logdirs = sorted(os.listdir(logdir))
    latest_idx = -1
    while all_logdirs[latest_idx].endswith(".txt"):
        latest_idx -= 1
    log_path = os.path.join(logdir, all_logdirs[latest_idx])
    # get datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    logging.info(f"Log dir resume: {log_path}")
    # backup the files
    logging.info(f"Backup files")
    backup_dir = osp.join(log_path, f"src_backup_resume_{dt_string}")
    os.makedirs(backup_dir, exist_ok=True)
    for path in [
        "profile",
        "lib_prior",
        "lib_moca",
        "lib_mosca",
        "scripts", 
        "lite_moca_reconstruct.py",
        "mosca_reconstruct.py",
        "mosca_reconstruct_debug.py",
        "mosca_evaluate.py",
        "mosca_evaluate_ours.py",
        "mosca_precompute.py",
        "mosca_viz.py",
        "mosca_calmetrics.py", 
        "mosca_precompute.py", 
        "mosca_onlyeval.py", 
    ]:
        os.system(f"cp -r {path} {backup_dir}")
    # reduce the backup size
    shutil.rmtree(osp.join(backup_dir, "lib_prior", "seg"))
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            if file.endswith(".pth") or file.endswith(".ckpt"):
                if osp.isfile(osp.join(root, file)):
                    os.remove(osp.join(root, file))
                else:
                    shutil.rmtree(osp.join(root, file))
    # backu the commandline args
    with open(osp.join(log_path, f"fit_commandline_args_resume_{dt_string}.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    return log_path


def load_multiview_manifest(ws, fit_cfg):
    manifest_name = getattr(fit_cfg, "multiview_manifest", "multiview_manifest.json")
    manifest_path = osp.join(ws, manifest_name)
    if not osp.isfile(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _workspace_has_saved2d_rgb(ws, rgb_dirname="images"):
    img_dir = osp.join(ws, rgb_dirname)
    return osp.isdir(img_dir) or osp.isfile(img_dir + ".npz")


def get_anchor_rgb_dirname(ws, fit_cfg):
    manifest = load_multiview_manifest(ws, fit_cfg)
    default_rgb_dirname = "images"
    if manifest is not None:
        default_rgb_dirname = manifest.get(
            "anchor_saved2d_rgb_dir", default_rgb_dirname
        )
    return getattr(fit_cfg, "anchor_rgb_dirname", default_rgb_dirname)


def resolve_anchor_workspace(ws, fit_cfg):
    manifest = load_multiview_manifest(ws, fit_cfg)
    if manifest is None:
        return ws

    rgb_dirname = get_anchor_rgb_dirname(ws, fit_cfg)
    repo_root = osp.dirname(osp.abspath(__file__))
    candidate_paths = []

    scene_name = manifest.get("scene_name")
    if scene_name:
        candidate_paths.append(
            osp.join(
                repo_root,
                "data_extracted",
                "released_data",
                "dycheck",
                "iphone",
                scene_name,
            )
        )

    source_workspace = manifest.get("source_workspace")
    if source_workspace:
        candidate_paths.append(source_workspace)

    candidate_paths.append(ws)

    for candidate in candidate_paths:
        if candidate and osp.isdir(candidate) and _workspace_has_saved2d_rgb(
            candidate, rgb_dirname
        ):
            if candidate != ws:
                logging.info(
                    f"Resolve anchor workspace for {ws} -> {candidate} (rgb_dir={rgb_dirname})"
                )
            return candidate

    logging.warning(
        f"Unable to resolve anchor workspace for {ws}; fall back to workspace root"
    )
    return ws


def auto_get_depth_dir_tap_mode(ws, fit_cfg):
    dep_dir = getattr(fit_cfg, "depth_dirname", None)
    if dep_dir is None:
        logging.info("Auto get depth dir")
        pattern = "*_depth"
        candidates = glob.glob(osp.join(ws, pattern))
        # ensure is dir
        candidates = [it for it in candidates if osp.isdir(it)]
        if len(candidates) > 1:
            # have a default order
            priority_key = ["gt", "sensor", "sharp", "depthcrafter"]
            for priority_it in priority_key:
                _candidates = [it for it in candidates if priority_it in it]
                if len(_candidates) == 1:
                    logging.warning(f"Multiple depth dir, use {priority_it} depth dir")
                    candidates = _candidates
                    break
        assert len(candidates) == 1, f"Found {len(candidates)} depth dir"
        dep_dir = osp.basename(candidates[0])
    tap_mode = getattr(fit_cfg, "tap_mode", None)
    if tap_mode is None:
        logging.info("Auto get tap mode")
        pattern = "*uniform*tap.npz"
        candidates = glob.glob(osp.join(ws, pattern))
        assert len(candidates) == 1, f"Found {len(candidates)} tap mode"
        tap_mode = osp.basename(candidates[0])
        tap_mode = tap_mode.split("_tap.npz")[0].split("_")[-1]
    return dep_dir, tap_mode


def viz_mosca_curves_before_optim(curve_xyz, curve_rgb, curve_mask, cams, viz_dir):
    # * viz
    os.makedirs(viz_dir, exist_ok=True)
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [
            cams.trans_pts_to_cam(cams.T // 2, it).cpu()
            for t, it in enumerate(curve_xyz)
        ],
        curve_rgb,
        zoom_out_factor=1.0,
    )
    imageio.mimsave(osp.join(viz_dir, "curve.gif"), viz_list, loop=1000)
    if curve_mask.any(dim=1).all():
        viz_list = viz_list_of_colored_points_in_cam_frame(
            [
                cams.trans_pts_to_cam(t, it[curve_mask[t]]).cpu()
                for t, it in enumerate(curve_xyz)
            ],
            [curve_rgb[itm] for itm in curve_mask.cpu()],
            zoom_out_factor=0.2,
        )
        imageio.mimsave(osp.join(viz_dir, "cam_curve_masked.gif"), viz_list, loop=1000)
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [cams.trans_pts_to_cam(t, it).cpu() for t, it in enumerate(curve_xyz)],
        curve_rgb,
        zoom_out_factor=0.2,
    )
    imageio.mimsave(osp.join(viz_dir, "cam_curve.gif"), viz_list, loop=1000)
    viz_valid_color = torch.tensor([0.0, 1.0, 0.0]).to(curve_xyz.device)
    viz_invalid_color = torch.tensor([1.0, 0.0, 0.0]).to(curve_xyz.device)
    # T,N,3
    viz_mask_color = (
        viz_valid_color[None, None] * curve_mask.float()[..., None]
        + viz_invalid_color[None, None] * (1 - curve_mask.float())[..., None]
    )
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [cams.trans_pts_to_cam(t, it).cpu() for t, it in enumerate(curve_xyz)],
        [it for it in viz_mask_color],
        zoom_out_factor=0.2,
    )
    imageio.mimsave(osp.join(viz_dir, "cam_curve_valid.gif"), viz_list, loop=1000)
    return


def update_s2d_track_identification(
    s2d,
    log_path,
    epi_th,
    dyn_id_cnt,
    min_curve_num=32,
    photo_error_masks=None,
    photo_error_mode="only",
    photo_error_id_cnt=None,
):
    # identify the fg tack by EPI
    if s2d.has_epi:
        raft_epi = s2d.epi.clone()
        with torch.no_grad():
            epi_error_list = query_buffers_by_track(
                raft_epi[..., None], s2d.track, s2d.track_mask
            )
            epi_error_list = epi_error_list.squeeze(-1).cpu()
    else:
        epi_data = np.load(osp.join(log_path, "tracker_epi.npz"))
        pair_list = [tuple(it) for it in epi_data["continuous_pair_list"].tolist()]
        F_list = epi_data["F_list"]
        _, epi_error_list, _ = analyze_track_epi(
            pair_list, s2d.track, s2d.track_mask, s2d.H, s2d.W, F_list
        )
    epi_track_static_selection, epi_track_dynamic_selection = identify_tracks(
        epi_error_list, epi_th, static_cnt=1, dynamic_cnt=dyn_id_cnt
    )

    # * optionally: identify the fg track by photo error
    if photo_error_masks is not None:
        assert photo_error_mode in [
            "only",
            "and",
            "or",
        ], f"photo_error_mode={photo_error_mode}"
        with torch.no_grad():
            photo_error_list = query_buffers_by_track(
                photo_error_masks[..., None], s2d.track, s2d.track_mask
            )
            photo_error_list = photo_error_list.squeeze(
                -1
            ).cpu()  # ! this is 01 mask, not error
            if photo_error_id_cnt is None:
                photo_error_id_cnt = dyn_id_cnt
            photo_track_static_selection, photo_track_dynamic_selection = (
                identify_tracks(
                    photo_error_list, 0.5, static_cnt=1, dynamic_cnt=photo_error_id_cnt
                )
            )
        if photo_error_mode == "only":
            epi_track_static_selection = photo_track_static_selection
            epi_track_dynamic_selection = photo_track_dynamic_selection
        elif photo_error_mode == "and":
            epi_track_static_selection = (
                epi_track_static_selection & photo_track_static_selection
            )
            epi_track_dynamic_selection = (
                epi_track_dynamic_selection & photo_track_dynamic_selection
            )
        elif photo_error_mode == "or":
            epi_track_static_selection = (
                epi_track_static_selection | photo_track_static_selection
            )
            epi_track_dynamic_selection = (
                epi_track_dynamic_selection | photo_track_dynamic_selection
            )
        else:
            raise NotImplementedError(f"photo_error_mode={photo_error_mode}")

    if epi_track_dynamic_selection.sum() < min_curve_num:
        logging.warning(
            f"Dynamic curves identification get too few curves, select K={min_curve_num} highest epi curves"
        )
        epi_error_recover = epi_error_list.max(dim=0).values
        highest_k = epi_error_recover.topk(min_curve_num, largest=False).indices
        epi_track_dynamic_selection[highest_k] = True

    s2d.register_track_indentification(
        epi_track_static_selection, epi_track_dynamic_selection
    )
    return s2d  # ! warning, changed


def set_epi_mask_to_s2d_for_bg_render(s2d, epi_th, device):
    assert s2d.has_epi, "EPI is required for static warm"
    static_mask = s2d.epi < epi_th
    # erode the static mask
    kernel = torch.ones((3, 3), device=device)
    static_mask = kornia.morphology.erosion(
        static_mask.float().unsqueeze(1), kernel
    ).squeeze(1)
    dynamic_mask = kornia.morphology.erosion(
        (1.0 - static_mask).float().unsqueeze(1), kernel
    ).squeeze(1)
    s2d.register_2d_identification(
        static_2d_mask=static_mask > 0, dynamic_2d_mask=dynamic_mask > 0
    )
    logging.info(f"Set EPI maks to s2d with epi_th={epi_th}")
    return s2d

import re
def parse_interval_from_dir(dir_name):
    """从目录名解析L和R的值，例如dst_L100_R200 -> (100, 200)"""
    match = re.match(r'dst_L(\d+)_R(\d+)', dir_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def find_root(node_name_list):
    root_name = ""
    root_value = -1
    for node_name in node_name_list:
        interval = parse_interval_from_dir(node_name)
        if interval:
            if root_value < interval[1] - interval[0]:
                root_value = interval[1] - interval[0]
                root_name = node_name
    assert root_name != ""
    return root_name

def find_leaves(exp_dir, node_name_list, segtree, nodeidx):
    if len(node_name_list) == 0:
        return []
    rootname = find_root(node_name_list)
    segtree[nodeidx] = rootname
    if len(node_name_list) == 1:
        return [rootname]
    split_ind_dir = os.path.join(exp_dir, rootname, "split_ind.txt")
    split_ind_file = open(split_ind_dir, 'r')
    split_ind = int(split_ind_file.read()) # [left, split_ind), [split_ind, right]
    left_list = []
    right_list = []
    for node_name in node_name_list:
        if node_name == rootname: continue
        interval = parse_interval_from_dir(node_name)
        assert interval
        if interval[1] < split_ind: left_list.append(node_name)
        if interval[0] >= split_ind: right_list.append(node_name)
    left_leaves = find_leaves(exp_dir, left_list, segtree, nodeidx*2)
    right_leaves = find_leaves(exp_dir, right_list, segtree, nodeidx*2+1)
    return left_leaves + right_leaves

import cv2
def imgs_to_video(imgs_dir, video_dir):
    """
    将指定文件夹中的图片序列转换为视频文件
    
    参数:
        imgs_dir (str): 包含图片的文件夹路径
        video_dir (str): 输出视频文件路径(如output.mp4)
    """
    # 获取所有图片文件并按文件名排序[1,2,3](@ref)
    images = [img for img in os.listdir(imgs_dir) 
              if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # 确保按顺序处理
    
    if not images:
        raise ValueError("未找到任何图片文件(.png/.jpg/.jpeg)")
    
    # 读取第一张图片获取尺寸[1,3,5](@ref)
    first_img = cv2.imread(os.path.join(imgs_dir, images[0]))
    if first_img is None:
        raise ValueError("无法读取第一张图片，请检查图片格式")
    
    height, width, _ = first_img.shape
    
    # 根据视频扩展名确定编码格式[3,7,8](@ref)
    if video_dir.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif video_dir.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError("不支持的视频格式，请使用.avi或.mp4")
    
    # 创建视频写入对象(默认帧率30fps)[1,3,7](@ref)
    video = cv2.VideoWriter(video_dir, fourcc, 30.0, (width, height))
    
    # 逐帧写入视频[1,3,5](@ref)
    for image in images:
        img_path = os.path.join(imgs_dir, image)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"警告: 无法读取图片 {image}，已跳过")
            continue
            
        # 确保图片尺寸一致[3,4](@ref)
        if frame.shape != (height, width, 3):
            frame = cv2.resize(frame, (width, height))
            
        video.write(frame)
    
    # 释放资源[1,3,7](@ref)
    video.release()
    print(f"视频已成功生成: {video_dir}")

def find_all_renders(scene_dir, segtree, nodeidx, max_tree_height, post_cut=-1, renders_path="tto_test"):
    left_child = nodeidx * 2
    right_child = nodeidx * 2 + 1
    all_renders_list = []
    has_left = segtree[left_child] != 'None'
    has_right = segtree[right_child] != 'None'
    now_height = int(np.log2(nodeidx))
    if now_height == max_tree_height:
        has_left = False
        has_right = False
    if post_cut != -1:
        if has_left:
            left_child_interval = parse_interval_from_dir(segtree[left_child])
            assert left_child_interval
            left_interval = left_child_interval[1] - left_child_interval[0] + 1
            if left_interval < post_cut:
                has_left = False
        if has_right:
            right_child_interval = parse_interval_from_dir(segtree[right_child])
            assert right_child_interval
            right_interval = right_child_interval[1] - right_child_interval[0] + 1
            if right_interval < post_cut:
                has_right = False
    if has_left == False and has_right == False: # leaf
        leaf_dir = os.path.join(scene_dir, segtree[nodeidx])
        leaf_renders_dir = os.path.join(leaf_dir, renders_path)
        leaf_renders_list = sorted(os.listdir(leaf_renders_dir))
        for leaf_render_name in leaf_renders_list:
            all_renders_list.append(os.path.join(leaf_renders_dir, leaf_render_name))
        return all_renders_list
    if has_left == True and has_right == False: # right missing
        left_all_renders_list = find_all_renders(scene_dir, segtree, left_child, max_tree_height, post_cut, renders_path)
        all_renders_list.extend(left_all_renders_list)
        left_child_interval = parse_interval_from_dir(segtree[left_child])
        assert left_child_interval
        left_child_dir = os.path.join(scene_dir, segtree[left_child])
        left_child_renders_dir = os.path.join(left_child_dir, renders_path)
        left_child_renders_list = sorted(os.listdir(left_child_renders_dir))
        left_child_last_rendername = left_child_renders_list[-1]
        right_all_renders_list = []
        leaf_dir = os.path.join(scene_dir, segtree[nodeidx])
        leaf_renders_dir = os.path.join(leaf_dir, renders_path)
        leaf_renders_list = sorted(os.listdir(leaf_renders_dir))
        for leaf_render_name in leaf_renders_list:
            if leaf_render_name > left_child_last_rendername:
                all_renders_list.append(os.path.join(leaf_renders_dir, leaf_render_name))
        return all_renders_list
    if has_left == False and has_right == True: # left missing
        right_child_interval = parse_interval_from_dir(segtree[right_child])
        assert right_child_interval
        right_child_dir = os.path.join(scene_dir, segtree[right_child])
        right_child_renders_dir = os.path.join(right_child_dir, renders_path)
        right_child_renders_list = sorted(os.listdir(right_child_renders_dir))
        right_child_first_rendername = right_child_renders_list[0]
        right_all_renders_list = []
        leaf_dir = os.path.join(scene_dir, segtree[nodeidx])
        leaf_renders_dir = os.path.join(leaf_dir, renders_path)
        leaf_renders_list = sorted(os.listdir(leaf_renders_dir))
        for leaf_render_name in leaf_renders_list:
            if leaf_render_name < right_child_first_rendername:
                all_renders_list.append(os.path.join(leaf_renders_dir, leaf_render_name))
        right_all_renders_list = find_all_renders(scene_dir, segtree, right_child, max_tree_height, post_cut, renders_path)
        all_renders_list.extend(right_all_renders_list)
        return all_renders_list
    if has_left == True and has_right == True: # normal node
        left_all_renders_list = find_all_renders(scene_dir, segtree, left_child, max_tree_height, post_cut, renders_path)
        right_all_renders_list = find_all_renders(scene_dir, segtree, right_child, max_tree_height, post_cut, renders_path)
        all_renders_list.extend(left_all_renders_list)
        all_renders_list.extend(right_all_renders_list)
        return all_renders_list
    
def find_all_nodes_and_intervals(scene_dir, segtree, nodeidx, max_tree_height, post_cut=-1):
    left_child = nodeidx * 2
    right_child = nodeidx * 2 + 1
    all_renders_list = []
    has_left = segtree[left_child] != 'None'
    has_right = segtree[right_child] != 'None'
    now_height = int(np.log2(nodeidx))
    if now_height == max_tree_height:
        has_left = False
        has_right = False

    if post_cut != -1:
        if has_left:
            left_child_interval = parse_interval_from_dir(segtree[left_child])
            assert left_child_interval
            left_interval = left_child_interval[1] - left_child_interval[0] + 1
            if left_interval < post_cut:
                has_left = False
        if has_right:
            right_child_interval = parse_interval_from_dir(segtree[right_child])
            assert right_child_interval
            right_interval = right_child_interval[1] - right_child_interval[0] + 1
            if right_interval < post_cut:
                has_right = False

    if has_left == False and has_right == False: # leaf
        leaf_interval = parse_interval_from_dir(segtree[nodeidx])
        assert leaf_interval
        left_t, right_t = leaf_interval
        leaf_result = {
            "node_name": segtree[nodeidx], 
            "left_t": left_t, 
            "right_t": right_t, 
        }
        all_renders_list.append(leaf_result)
        return all_renders_list
    
    if has_left == True and has_right == False: # right missing
        left_all_renders_list = find_all_nodes_and_intervals(scene_dir, segtree, left_child, max_tree_height, post_cut)
        all_renders_list.extend(left_all_renders_list)
        left_child_last_result = left_all_renders_list[-1]
        left_t = left_child_last_result["right_t"] + 1
        leaf_interval = parse_interval_from_dir(segtree[nodeidx])
        assert leaf_interval
        right_t = leaf_interval[1]
        leaf_result = {
            "node_name": segtree[nodeidx], 
            "left_t": left_t, 
            "right_t": right_t, 
        }
        all_renders_list.append(leaf_result)
        return all_renders_list
    
    if has_left == False and has_right == True: # left missing
        right_all_renders_list = find_all_nodes_and_intervals(scene_dir, segtree, right_child, max_tree_height, post_cut)
        right_child_first_result = right_all_renders_list[0]
        right_t = right_child_first_result["left_t"] - 1
        leaf_interval = parse_interval_from_dir(segtree[nodeidx])
        assert leaf_interval
        left_t = leaf_interval[0]
        leaf_result = {
            "node_name": segtree[nodeidx], 
            "left_t": left_t, 
            "right_t": right_t, 
        }
        all_renders_list.append(leaf_result)
        all_renders_list.extend(right_all_renders_list)
        return all_renders_list
    if has_left == True and has_right == True: # normal node
        left_all_renders_list = find_all_nodes_and_intervals(scene_dir, segtree, left_child, max_tree_height, post_cut)
        right_all_renders_list = find_all_nodes_and_intervals(scene_dir, segtree, right_child, max_tree_height, post_cut)
        all_renders_list.extend(left_all_renders_list)
        all_renders_list.extend(right_all_renders_list)
        return all_renders_list

import threading
import time

class MultiGPUMonitor:
    """手动控制的多GPU监控器"""
    
    def __init__(self, gpu_ids=None, sampling_interval=0.05):
        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.threads = []
        self.samples = {gpu_id: [] for gpu_id in self.gpu_ids}
        self.start_time = None
        self.initial_memory = {}
    
    def start_monitoring(self):
        """开始监控"""
        if not self.available:
            return
        if self.monitoring:
            print("监控已在运行中")
            return
        
        self.monitoring = True
        self.samples = {gpu_id: [] for gpu_id in self.gpu_ids}
        self.start_time = time.time()
        
        # 重置所有GPU的峰值统计
        for gpu_id in self.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self.initial_memory[gpu_id] = torch.cuda.memory_allocated()
        
        # 启动监控线程
        self.threads = []
        for gpu_id in self.gpu_ids:
            thread = threading.Thread(target=self._monitor_gpu, args=(gpu_id,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        print(f"开始监控GPU {self.gpu_ids}，采样间隔: {self.sampling_interval*1000:.0f}ms")
    
    def _monitor_gpu(self, gpu_id):
        """监控单个GPU"""
        device = torch.device(f'cuda:{gpu_id}')
        
        while self.monitoring:
            try:
                memory = torch.cuda.memory_allocated(device)
                timestamp = time.time() - self.start_time
                self.samples[gpu_id].append((timestamp, memory))
                time.sleep(self.sampling_interval)
            except:
                break
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.available:
            return None
        if not self.monitoring:
            print("监控未启动")
            return None
        
        self.monitoring = False
        
        # 等待线程结束
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        return self._collect_results()
    
    def _collect_results(self):
        """收集结果"""
        results = {}
        total_peak = 0
        
        for gpu_id in self.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.synchronize(device)
            
            final_memory = torch.cuda.memory_allocated(device)
            peak_memory = torch.cuda.max_memory_allocated(device)
            
            # 计算平均内存（从采样数据）
            if self.samples[gpu_id]:
                avg_memory = sum(mem for _, mem in self.samples[gpu_id]) / len(self.samples[gpu_id])
            else:
                avg_memory = (self.initial_memory[gpu_id] + final_memory) / 2
            
            results[gpu_id] = {
                'peak_mb': peak_memory / 1024**2,
                'avg_mb': avg_memory / 1024**2,
                'final_mb': final_memory / 1024**2,
                'initial_mb': self.initial_memory[gpu_id] / 1024**2,
                'samples': len(self.samples[gpu_id]),
                'duration': time.time() - self.start_time,
                'peak_memory': peak_memory
            }
            
            total_peak += peak_memory
        
        results['total_peak_mb'] = total_peak / 1024**2
        return results
    
    def print_results(self, results):
        """打印结果"""
        if not self.available:
            print("pynvml not available, no GPU monitoring results")
            return
        if not results:
            print("无结果数据")
            return
        
        print(f"\n=== 多GPU监控结果 ===")
        print(f"总监控时长: {results[0]['duration']:.2f}秒")
        print(f"采样间隔: {self.sampling_interval*1000:.0f}ms")
        
        for gpu_id in self.gpu_ids:
            if gpu_id in results:
                data = results[gpu_id]
                print(f"GPU {gpu_id}:")
                print(f"  初始显存: {data['initial_mb']:.2f} MB")
                print(f"  最终显存: {data['final_mb']:.2f} MB")
                print(f"  峰值显存: {data['peak_mb']:.2f} MB")
                print(f"  平均显存: {data['avg_mb']:.2f} MB")
                print(f"  显存增量: {data['final_mb'] - data['initial_mb']:.2f} MB")
                print(f"  采样点数: {data['samples']}")
        
        print(f"总计峰值显存: {results['total_peak_mb']:.2f} MB")
        print("=" * 50)
    
    def get_current_stats(self):
        """获取当前统计（不停止监控）"""
        if not self.monitoring:
            return None
        
        current_results = {}
        for gpu_id in self.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            current_memory = torch.cuda.memory_allocated(device)
            peak_memory = torch.cuda.max_memory_allocated(device)
            
            current_results[gpu_id] = {
                'current_mb': current_memory / 1024**2,
                'peak_mb': peak_memory / 1024**2,
                'elapsed_time': time.time() - self.start_time
            }
        
        return current_results
    
    def export_results(self, results):
        vram_content = f"\n=== 多GPU实时监控结果 ===\n"
        vram_content += f"总监控时长: {results[0]['duration']:.2f}秒\n"
        vram_content += f"采样间隔: {self.sampling_interval*1000:.0f}ms\n"
        for gpu_id in self.gpu_ids:
            if gpu_id in results:
                data = results[gpu_id]
                vram_content += f"GPU {gpu_id}:\n"
                vram_content += f"  初始显存: {data['initial_mb']:.2f} MB\n"
                vram_content += f"  最终显存: {data['final_mb']:.2f} MB\n"
                vram_content += f"  峰值显存: {data['peak_mb']:.2f} MB\n"
                vram_content += f"  平均显存: {data['avg_mb']:.2f} MB\n"
                vram_content += f"  显存增量: {data['final_mb'] - data['initial_mb']:.2f} MB\n"
                vram_content += f"  采样点数: {data['samples']}\n"
        vram_content += f"总计峰值显存: {results['total_peak_mb']:.2f} MB\n"
        vram_content += "=" * 50 + "\n"
        return vram_content

import threading
import time
try:
    import pynvml
except ImportError:
    pynvml = None

class MultiGPUMonitorPYNVML:
    """手动控制的多GPU监控器（基于pynvml）"""
    
    def __init__(self, gpu_ids=None, sampling_interval=0.05):
        self.available = pynvml is not None
        self.gpu_ids = gpu_ids or []
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.threads = []
        self.samples = {gpu_id: [] for gpu_id in self.gpu_ids}
        self.start_time = None
        self.initial_memory = {}
        self.peak_memory = {}
        self.handles = {}
        if not self.available:
            print("pynvml not available, GPU memory monitoring is disabled")
            return
        # 初始化pynvml
        pynvml.nvmlInit()
        
        self.gpu_count = pynvml.nvmlDeviceGetCount()
        self.gpu_ids = gpu_ids or list(range(self.gpu_count))
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.threads = []
        self.samples = {gpu_id: [] for gpu_id in self.gpu_ids}
        self.start_time = None
        self.initial_memory = {}
        self.peak_memory = {}  # 用于手动跟踪峰值内存
        
        # 获取GPU句柄
        self.handles = {}
        for gpu_id in self.gpu_ids:
            try:
                self.handles[gpu_id] = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.peak_memory[gpu_id] = 0
            except pynvml.NVMLError as e:
                print(f"无法获取GPU {gpu_id}的句柄: {e}")
    
    def __del__(self):
        """析构函数，确保pynvml被正确关闭"""
        if not self.available:
            return
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    
    def _get_memory_info(self, gpu_id):
        """获取指定GPU的内存信息"""
        if not self.available:
            return 0
        try:
            handle = self.handles[gpu_id]
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used
        except pynvml.NVMLError as e:
            print(f"获取GPU {gpu_id}内存信息失败: {e}")
            return 0
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            print("监控已在运行中")
            return
        
        self.monitoring = True
        self.samples = {gpu_id: [] for gpu_id in self.gpu_ids}
        self.start_time = time.time()
        self.peak_memory = {gpu_id: 0 for gpu_id in self.gpu_ids}
        
        # 记录初始内存并重置峰值统计
        for gpu_id in self.gpu_ids:
            initial_mem = self._get_memory_info(gpu_id)
            self.initial_memory[gpu_id] = initial_mem
            self.peak_memory[gpu_id] = initial_mem
        
        # 启动监控线程
        self.threads = []
        for gpu_id in self.gpu_ids:
            thread = threading.Thread(target=self._monitor_gpu, args=(gpu_id,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        print(f"开始监控GPU {self.gpu_ids}，采样间隔: {self.sampling_interval*1000:.0f}ms")
    
    def _monitor_gpu(self, gpu_id):
        """监控单个GPU"""
        while self.monitoring:
            try:
                memory = self._get_memory_info(gpu_id)
                timestamp = time.time() - self.start_time
                self.samples[gpu_id].append((timestamp, memory))
                
                # 更新峰值内存
                if memory > self.peak_memory[gpu_id]:
                    self.peak_memory[gpu_id] = memory
                
                time.sleep(self.sampling_interval)
            except:
                break
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            print("监控未启动")
            return None
        
        self.monitoring = False
        
        # 等待线程结束
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        return self._collect_results()
    
    def _collect_results(self):
        """收集结果"""
        results = {}
        total_peak = 0
        
        for gpu_id in self.gpu_ids:
            final_memory = self._get_memory_info(gpu_id)
            peak_memory = self.peak_memory[gpu_id]
            
            # 计算平均内存（从采样数据）
            if self.samples[gpu_id]:
                avg_memory = sum(mem for _, mem in self.samples[gpu_id]) / len(self.samples[gpu_id])
            else:
                avg_memory = (self.initial_memory[gpu_id] + final_memory) / 2
            
            results[gpu_id] = {
                'peak_mb': peak_memory / 1024**2,
                'avg_mb': avg_memory / 1024**2,
                'final_mb': final_memory / 1024**2,
                'initial_mb': self.initial_memory[gpu_id] / 1024**2,
                'samples': len(self.samples[gpu_id]),
                'duration': time.time() - self.start_time,
                'peak_memory': peak_memory
            }
            
            total_peak += peak_memory
        
        results['total_peak_mb'] = total_peak / 1024**2
        return results
    
    def print_results(self, results):
        """打印结果"""
        if not results:
            print("无结果数据")
            return
        
        print(f"\n=== 多GPU监控结果 ===")
        print(f"总监控时长: {results[0]['duration']:.2f}秒")
        print(f"采样间隔: {self.sampling_interval*1000:.0f}ms")
        
        for gpu_id in self.gpu_ids:
            if gpu_id in results:
                data = results[gpu_id]
                print(f"GPU {gpu_id}:")
                print(f"  初始显存: {data['initial_mb']:.2f} MB")
                print(f"  最终显存: {data['final_mb']:.2f} MB")
                print(f"  峰值显存: {data['peak_mb']:.2f} MB")
                print(f"  平均显存: {data['avg_mb']:.2f} MB")
                print(f"  显存增量: {data['final_mb'] - data['initial_mb']:.2f} MB")
                print(f"  采样点数: {data['samples']}")
        
        print(f"总计峰值显存: {results['total_peak_mb']:.2f} MB")
        print("=" * 50)
    
    def get_current_stats(self):
        """获取当前统计（不停止监控）"""
        if not self.monitoring:
            return None
        
        current_results = {}
        for gpu_id in self.gpu_ids:
            current_memory = self._get_memory_info(gpu_id)
            peak_memory = self.peak_memory[gpu_id]
            
            current_results[gpu_id] = {
                'current_mb': current_memory / 1024**2,
                'peak_mb': peak_memory / 1024**2,
                'elapsed_time': time.time() - self.start_time
            }
        
        return current_results
    
    def export_results(self, results):
        """导出结果到字符串"""
        if not self.available:
            return "pynvml not available, GPU memory monitoring is disabled\n"
        if not results:
            return "无结果数据"
            
        vram_content = f"\n=== 多GPU实时监控结果 ===\n"
        vram_content += f"总监控时长: {results[0]['duration']:.2f}秒\n"
        vram_content += f"采样间隔: {self.sampling_interval*1000:.0f}ms\n"
        for gpu_id in self.gpu_ids:
            if gpu_id in results:
                data = results[gpu_id]
                vram_content += f"GPU {gpu_id}:\n"
                vram_content += f"  初始显存: {data['initial_mb']:.2f} MB\n"
                vram_content += f"  最终显存: {data['final_mb']:.2f} MB\n"
                vram_content += f"  峰值显存: {data['peak_mb']:.2f} MB\n"
                vram_content += f"  平均显存: {data['avg_mb']:.2f} MB\n"
                vram_content += f"  显存增量: {data['final_mb'] - data['initial_mb']:.2f} MB\n"
                vram_content += f"  采样点数: {data['samples']}\n"
        vram_content += f"总计峰值显存: {results['total_peak_mb']:.2f} MB\n"
        vram_content += "=" * 50 + "\n"
        return vram_content
