# from origial reconstruction.py import all rendering related stuff
import sys, os, os.path as osp
import torch
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from misc import configure_logging, get_timestamp
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import kornia
import colorsys
import torch.nn.functional as F

sys.path.append(osp.dirname(osp.abspath(__file__)))

from pytorch3d.ops import knn_points
from dynamic_gs import DynSCFGaussian
from static_gs import StaticGaussian
from mosca import MoSca
from camera import MonocularCameras
from photo_recon_utils import fetch_leaves_in_world_frame, estimate_normal_map, move_module_device, move_mosca_device, move_mosca_device_lite
from lib_render.render_helper import GS_BACKEND, render, RGB2SH
from lib_prior.prior_loading import Saved2D
from gs_utils.gs_optim_helper import update_learning_rate
from gs_utils.loss_helper import (
    build_lpips_model,
    compute_lpips_loss,
    compute_rgb_loss,
    compute_dep_loss,
    compute_normal_loss,
    compute_normal_reg_loss,
    compute_dep_reg_loss,
)
from dynamic_solver import get_dynamic_curves

from photo_recon_utils import (
    fetch_leaves_in_world_frame,
    GSControlCFG,
    OptimCFG,
    apply_gs_control,
    apply_gs_control_param_interval, 
    error_grow_dyn_model,
    identify_traj_id,
    get_colorplate,
)
from photo_recon_viz_utils import (
    viz_hist,
    viz_dyn_hist,
    viz2d_total_video,
    viz3d_total_video,
    viz_curve,
)
from photo_recon_viz_recur_utils import viz2d_total_video_interval, viz2d_total_video_interval_fachain
from photo_recon_utils import DynamicSegTree, DynamicSegTreeBFS, split_cams, split_d_model, split_s_model, load_s_model, release_model_memory, get_available_gpus
from concurrent.futures import ThreadPoolExecutor
import copy
import signal
from threading import Event, Lock

def get_recon_cfg(cfg_fn=None):
    if cfg_fn is None:
        logging.info("No cfg_fn provided, use dummy cfg")
        # get dummy
        cfg = OmegaConf.create()
        cfg.s_ctrl = OmegaConf.create()
        cfg.d_ctrl = OmegaConf.create()
        return cfg
    cfg = OmegaConf.load(cfg_fn)
    for key in ["photometric", "geometric"]:
        if not hasattr(cfg, key):
            setattr(cfg, key, OmegaConf.create())
    for key in ["d_ctrl", "s_ctrl"]:
        if not hasattr(cfg.photometric, key):
            setattr(cfg.photometric, key, None)
    OmegaConf.set_readonly(cfg, True)
    logging.info(f"Load cfg from {cfg_fn}: {cfg}")
    return cfg


def _normalize_camera_weight_dict(weight_dict):
    if weight_dict is None:
        return {}
    if not isinstance(weight_dict, dict):
        weight_dict = OmegaConf.to_container(weight_dict, resolve=True)
    return dict(weight_dict)


def _get_lpips_camera_weight(
    camera_name,
    multiview_obs,
    multiview_lpips_camera_weights,
):
    if camera_name in multiview_lpips_camera_weights:
        return float(multiview_lpips_camera_weights[camera_name])
    if multiview_obs is not None and camera_name in multiview_obs.camera_name_to_index:
        return float(multiview_obs.get_weight(camera_name).item())
    return 1.0


def _resolve_anchor_photo_target(
    base_rgb_target,
    base_sup_mask,
    obs_view_ind,
    multiview_obs=None,
    multiview_use_masks=False,
):
    anchor_rgb_target = base_rgb_target.detach().clone()
    rgb_sup_mask = base_sup_mask
    anchor_camera_name = "anchor"
    anchor_rgb_weight = 1.0
    if multiview_obs is None:
        return anchor_camera_name, anchor_rgb_target, rgb_sup_mask, anchor_rgb_weight

    anchor_camera_name = multiview_obs.anchor_camera
    anchor_rgb_weight = float(multiview_obs.get_weight(anchor_camera_name).item())
    if multiview_obs.override_anchor_rgb:
        anchor_rgb_target = (
            multiview_obs.get_rgb(multiview_obs.anchor_rgb_camera, obs_view_ind)
            .to(base_rgb_target)
            .detach()
            .clone()
        )
        if multiview_use_masks:
            rgb_sup_mask = rgb_sup_mask * multiview_obs.get_mask(
                multiview_obs.anchor_rgb_camera,
                obs_view_ind,
                use_mask=True,
            ).to(rgb_sup_mask)
    return anchor_camera_name, anchor_rgb_target, rgb_sup_mask, anchor_rgb_weight


def _accumulate_multiview_photo_losses(
    render_dict_list,
    gs5,
    bg_color,
    camera_view_ind,
    obs_view_ind,
    multiview_obs,
    multiview_cams,
    multiview_use_masks=False,
    multiview_aux_ssim_lambda=0.1,
    lpips_model=None,
    multiview_lpips_camera_weights=None,
):
    if multiview_obs is None or multiview_cams is None:
        render_device = gs5[0][0].device
        zero = torch.zeros((), device=render_device)
        return zero, zero

    render_device = gs5[0][0].device
    loss_rgb = torch.zeros((), device=render_device)
    loss_lpips = torch.zeros((), device=render_device)
    if multiview_lpips_camera_weights is None:
        multiview_lpips_camera_weights = {}
    batch_aux_lpips = os.environ.get("WFM_TABLE1_BATCH_LPIPS") == "1" and lpips_model is not None
    pending_aux_lpips = []
    if batch_aux_lpips:
        from efficiency.batched_losses import lpips_for_views
    for camera_name in multiview_obs.auxiliary_cameras:
        aux_H, aux_W = multiview_obs.get_hw(camera_name)
        aux_render_dict = render(
            gs5,
            aux_H,
            aux_W,
            multiview_cams.K(camera_name, aux_H, aux_W),
            multiview_cams.T_cw(camera_name, camera_view_ind),
            bg_color=bg_color,
            add_buffer=None,
        )
        render_dict_list.append(aux_render_dict)
        if os.environ.get("WFM_TABLE1_DEVICE_MASKS") == "1":
            aux_sup_mask = multiview_obs.get_mask(
                camera_name, obs_view_ind, use_mask=multiview_use_masks,
                device=render_device,
            )
        else:
            aux_sup_mask = multiview_obs.get_mask(
                camera_name, obs_view_ind, use_mask=multiview_use_masks
            ).to(render_device)
        aux_target = multiview_obs.get_rgb(camera_name, obs_view_ind).to(render_device)
        _l_aux_rgb, _, _, _ = compute_rgb_loss(
            aux_target.detach().clone(),
            aux_render_dict,
            aux_sup_mask,
            ssim_lambda=multiview_aux_ssim_lambda,
        )
        loss_rgb = loss_rgb + _l_aux_rgb * float(
            multiview_obs.get_weight(camera_name).item()
        )
        if lpips_model is not None:
            if batch_aux_lpips:
                pending_aux_lpips.append((aux_render_dict["rgb"], aux_target.detach().clone(), aux_sup_mask,
                                          _get_lpips_camera_weight(camera_name, multiview_obs,
                                                                   multiview_lpips_camera_weights)))
            else:
                _l_aux_lpips, _, _, _ = compute_lpips_loss(
                    aux_target.detach().clone(),
                    aux_render_dict,
                    aux_sup_mask,
                    lpips_model,
                )
                loss_lpips = loss_lpips + _l_aux_lpips * _get_lpips_camera_weight(
                    camera_name,
                    multiview_obs,
                    multiview_lpips_camera_weights,
                )
    if pending_aux_lpips:
        auxiliary_losses = lpips_for_views(
            [entry[:3] for entry in pending_aux_lpips], lpips_model, compute_lpips_loss
        )
        for entry, value in zip(pending_aux_lpips, auxiliary_losses):
            loss_lpips = loss_lpips + value * entry[3]
    return loss_rgb, loss_lpips


class DynReconstructionSolver:
    def __init__(
        self,
        working_dir,
        device=torch.device("cuda:0"),
        # cfg
        radius_init_factor=4.0,  # ! 1.0
        opacity_init_factor=0.95,
    ):
        self.src = working_dir
        self.device = device
        timestamp = get_timestamp()
        configure_logging(
            osp.join(self.src, f"dynamic_reconstruction_{timestamp}.log"),
            debug=False,
        )

        self.log_dir = self.src
        self.viz_dir = osp.join(self.src, f"mosca_photo_viz_{timestamp}")
        os.makedirs(self.viz_dir, exist_ok=True)

        self.radius_init_factor = radius_init_factor
        self.opacity_init_factor = opacity_init_factor
        return

    @torch.no_grad()
    def identify_fg_mask_by_nearest_curve(
        self, s2d: Saved2D, cams: MonocularCameras, viz_fname=None
    ):
        # get global anchor
        curve_xyz, curve_mask, _, _ = get_dynamic_curves(
            s2d, cams, return_all_curves=True
        )
        assert curve_xyz.shape[1] == len(s2d.dynamic_track_mask)

        # only consider the valid case
        static_curve_mean = (
            curve_xyz[:, ~s2d.dynamic_track_mask]
            * curve_mask[:, ~s2d.dynamic_track_mask, None]
        ).sum(0, keepdim=True) / curve_mask[:, ~s2d.dynamic_track_mask, None].sum(
            0, keepdim=True
        ).expand(
            len(curve_xyz), -1, -1
        )
        curve_xyz[:, ~s2d.dynamic_track_mask] = static_curve_mean
        np.savetxt(
            osp.join(self.viz_dir, "fg_id_non_dyn_curve_meaned.xyz"),
            static_curve_mean.reshape(-1, 3).cpu().numpy(),
            fmt="%.4f",
        )

        with torch.no_grad():
            fg_mask_list = []
            for query_tid in tqdm(range(s2d.T)):
                query_dep = s2d.dep[query_tid].clone()  # H,W
                query_xyz_cam = cams.backproject(
                    cams.get_homo_coordinate_map(), query_dep
                )
                query_xyz_world = cams.trans_pts_to_world(
                    query_tid, query_xyz_cam
                )  # H,W,3

                # find the nearest distance and acc sk weight
                # use all the curve at this position to id the fg and bg
                _, knn_id, _ = knn_points(
                    query_xyz_world.reshape(1, -1, 3), curve_xyz[query_tid][None], K=1
                )
                knn_id = knn_id[0, :, 0]
                fg_mask = s2d.dynamic_track_mask[knn_id].reshape(s2d.H, s2d.W)
                fg_mask_list.append(fg_mask.cpu())
            fg_mask_list = torch.stack(fg_mask_list, 0)
        if viz_fname is not None:
            viz_rgb = s2d.rgb.clone().cpu()
            viz_fg_mask_list = fg_mask_list * s2d.dep_mask.to(fg_mask_list)
            viz_rgb = viz_rgb * viz_fg_mask_list.float()[..., None] + viz_rgb * 0.1 * (
                1 - viz_fg_mask_list.float()[..., None]
            )
            imageio.mimsave(osp.join(self.viz_dir, viz_fname), viz_rgb.numpy())

        fg_mask_list = fg_mask_list.to(cams.rel_focal.device).bool()
        s2d.register_2d_identification(
            static_2d_mask=~fg_mask_list, dynamic_2d_mask=fg_mask_list
        )
        return fg_mask_list

    @torch.no_grad()
    def compute_normals_for_s2d(
        self,
        s2d,
        cams,
        patch_size=7,
        nn_dist_th=0.03,
        nn_min_cnt=4,
        viz_fn=None,
        viz_subsample=4,
    ):
        # compute normal maps for s2d
        logging.info(f"Computing normal maps from depth maps using local SVD...")
        # the computed normals are always pointing backward, on -z direction
        ray_direction = cams.backproject(
            s2d.homo_map.clone(), torch.ones_like(s2d.homo_map[..., 0])
        )
        ray_direction = F.normalize(ray_direction, dim=-1)
        normal_map_list = []
        new_mask_list = []
        for t in tqdm(range(s2d.T)):
            dep = s2d.dep[t].clone()
            dep_mask = s2d.dep_mask[t].clone()
            normal_map = torch.zeros(*dep.shape, 3).to(dep)
            xyz = cams.backproject(s2d.homo_map[dep_mask].clone(), dep[dep_mask])
            vtx_map = torch.zeros_like(normal_map).float()
            vtx_map[dep_mask] = xyz
            normal_map, mask = estimate_normal_map(
                vtx_map, dep_mask, patch_size, nn_dist_th, nn_min_cnt
            )

            normal = normal_map[mask]
            inner = (normal * ray_direction[mask]).sum(-1)
            correct_orient = inner < 0
            sign = torch.ones_like(normal[..., :1])
            sign[~correct_orient] = -1.0
            normal = normal.clone() * sign
            normal_map[mask] = normal
            new_mask = dep_mask * mask
            normal_map_list.append(normal_map)
            new_mask_list.append(new_mask)
        ret_nrm = torch.stack(normal_map_list, 0)
        ret_mask = torch.stack(new_mask_list, 0)

        if viz_fn is not None:
            viz_fn = osp.join(self.viz_dir, viz_fn)
            logging.info(f"Viz normal maps to {viz_fn}")
            viz_frames = (
                ((-ret_nrm + 1) / 2.0 * 255).detach().cpu().numpy().astype(np.uint8)
            )
            if len(viz_frames) > 50:
                _step = max(1, len(viz_frames) // 50)
            else:
                _step = 1
            # skip to boost viz
            viz_frames = viz_frames[:, ::viz_subsample, ::viz_subsample, :]
            imageio.mimsave(viz_fn, viz_frames[::_step])

        s2d.register_buffer("nrm", ret_nrm.detach().clone())
        s2d.dep_mask = ret_mask.detach().clone()
        return

    @torch.no_grad()
    def get_static_model(
        self,
        s2d: Saved2D,
        cams,
        n_init=30000,
        radius_max=0.05,
        max_sph_order=0,
        image_stride=1,
        viz_fn=None,
        mask_type="static_depth",
    ):
        device = self.device

        if mask_type == "static_depth":
            gather_mask = s2d.sta_mask * s2d.dep_mask
        elif mask_type == "depth":
            gather_mask = s2d.dep_mask
        else:
            raise ValueError(f"Unknown mask_type={mask_type}")

        mu_init, q_init, s_init, o_init, rgb_init, id_init, _ = (
            fetch_leaves_in_world_frame(
                cams=cams,
                n_attach=n_init,
                input_mask_list=gather_mask,
                input_dep_list=s2d.dep,
                input_rgb_list=s2d.rgb,
                save_xyz_fn=viz_fn,
                subsample=image_stride,
            )
        )
        s_model: StaticGaussian = StaticGaussian(
            init_mean=mu_init.clone().to(device),
            init_q=q_init,
            init_s=s_init * self.radius_init_factor,
            init_o=o_init * self.opacity_init_factor,
            init_rgb=rgb_init,
            init_id=id_init,
            max_scale=radius_max,
            min_scale=0.0,
            max_sph_order=max_sph_order,
        )
        s_model.to(device)
        return s_model


    @torch.no_grad()
    def get_dynamic_model(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        scf: MoSca,
        n_init=10000,
        image_stride=1,
        radius_max=0.05,
        max_sph_order=0,
        leaf_local_flag=True,
        topo_th_ratio=None,
        dyn_o_flag=True,
        additional_mask=None,
        nn_fusion=-1,
        max_node_num=100000,
    ):
        device = self.device
        collect_t_list = torch.arange(0, s2d.T, 1)
        logging.info(f"Collect GS at t={collect_t_list}")
        input_mask_list = s2d.dyn_mask * s2d.dep_mask
        if additional_mask is not None:
            assert additional_mask.shape == s2d.dep_mask.shape
            input_mask_list = input_mask_list * additional_mask
        mu_init, q_init, s_init, o_init, rgb_init, id_init, time_init = (
            fetch_leaves_in_world_frame(
                cams=cams,
                n_attach=n_init,
                input_mask_list=input_mask_list,
                input_dep_list=s2d.dep,
                input_rgb_list=s2d.rgb,
                subsample=image_stride,
            )
        )
        # * Reset SCF topo th!
        if topo_th_ratio is not None:
            old_th_ratio = scf.topo_th_ratio
            scf.topo_th_ratio = torch.ones_like(scf.topo_th_ratio) * topo_th_ratio
            logging.info(
                f"Reset SCF topo th ratio from {old_th_ratio} to {scf.topo_th_ratio}"
            )

        # * Init the scf
        d_model: DynSCFGaussian = DynSCFGaussian(
            scf=scf,
            max_scale=radius_max,
            min_scale=0.0,
            max_sph_order=max_sph_order,
            device=device,
            leaf_local_flag=leaf_local_flag,
            dyn_o_flag=dyn_o_flag,
            nn_fusion=nn_fusion,
            max_node_num=max_node_num,
        )
        d_model.to(device)

        # * Init the leaves
        optimizer = torch.optim.Adam(d_model.get_optimizable_list())
        unique_tid = time_init.unique()
        logging.info("Attach to Dynamic Scaffold ...")
        for tid in tqdm(unique_tid):
            t_mask = time_init == tid
            d_model.append_new_gs(
                optimizer,
                tid=tid,
                mu_w=mu_init[t_mask],
                quat_w=q_init[t_mask],
                scales=s_init[t_mask] * self.radius_init_factor,
                opacity=o_init[t_mask] * self.opacity_init_factor,
                rgb=rgb_init[t_mask],
            )
        # d_model.scf.update_topology()
        d_model.summary()
        return d_model


    @torch.no_grad()
    def get_dynamic_model_interval(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        scf: MoSca,
        n_init=10000,
        image_stride=1,
        radius_max=0.05,
        max_sph_order=0,
        leaf_local_flag=True,
        topo_th_ratio=None,
        dyn_o_flag=True,
        additional_mask=None,
        nn_fusion=-1,
        max_node_num=100000,
        left_bound=-1, 
        right_bound=-1, 
        device=torch.device("cuda"), 
    ):
        if left_bound == -1 and right_bound == -1:
            start_t = 0
            end_t = s2d.T
        else:
            start_t = left_bound
            end_t = right_bound
        assert start_t != -1 and end_t != -1
        # collect_t_list = torch.arange(0, s2d.T, 1)
        collect_t_list = torch.arange(start_t, end_t+1, 1)
        logging.info(f"Collect GS at t={collect_t_list}")
        input_mask_list = s2d.dyn_mask[start_t:end_t+1] * s2d.dep_mask[start_t:end_t+1]
        if additional_mask is not None:
            assert additional_mask.shape == s2d.dep_mask[start_t:end_t+1].shape
            input_mask_list = input_mask_list * additional_mask
        mu_init, q_init, s_init, o_init, rgb_init, id_init, time_init = (
            fetch_leaves_in_world_frame(
                cams=cams,
                n_attach=n_init,
                input_mask_list=input_mask_list,
                input_dep_list=s2d.dep[start_t:end_t+1],
                input_rgb_list=s2d.rgb[start_t:end_t+1],
                subsample=image_stride,
            )
        )
        # * Reset SCF topo th!
        if topo_th_ratio is not None:
            old_th_ratio = scf.topo_th_ratio
            scf.topo_th_ratio = torch.ones_like(scf.topo_th_ratio) * topo_th_ratio
            logging.info(
                f"Reset SCF topo th ratio from {old_th_ratio} to {scf.topo_th_ratio}"
            )

        # * Init the scf
        d_model: DynSCFGaussian = DynSCFGaussian(
            scf=scf,
            max_scale=radius_max,
            min_scale=0.0,
            max_sph_order=max_sph_order,
            device=device,
            leaf_local_flag=leaf_local_flag,
            dyn_o_flag=dyn_o_flag,
            nn_fusion=nn_fusion,
            max_node_num=max_node_num,
        )
        d_model.to(device)

        # * Init the leaves
        optimizer = torch.optim.Adam(d_model.get_optimizable_list())
        unique_tid = time_init.unique()
        logging.info("Attach to Dynamic Scaffold ...")
        for tid in tqdm(unique_tid):
            t_mask = time_init == tid
            d_model.append_new_gs(
                optimizer,
                tid=tid,
                mu_w=mu_init[t_mask],
                quat_w=q_init[t_mask],
                scales=s_init[t_mask] * self.radius_init_factor,
                opacity=o_init[t_mask] * self.opacity_init_factor,
                rgb=rgb_init[t_mask],
            )
        # d_model.scf.update_topology()
        d_model.summary()
        return d_model

    def photometric_fit(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        s_model: StaticGaussian,
        d_model: DynSCFGaussian = None,
        optim_cam_after_steps=0,
        total_steps=8000,
        topo_update_feq=50,
        skinning_corr_start_steps=1e10,
        s_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00025,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        d_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00015,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        s_gs_ctrl_start_ratio=0.2,
        s_gs_ctrl_end_ratio=0.9,
        d_gs_ctrl_start_ratio=0.2,
        d_gs_ctrl_end_ratio=0.9,
        # optim
        optimizer_cfg: OptimCFG = OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.0001,
            lr_cam_t=0.0001,
            lr_p=0.0003,
            lr_q=0.002,
            lr_s=0.01,
            lr_o=0.1,
            lr_sph=0.005,
            # dyn
            lr_np=0.001,
            lr_nq=0.01,
            lr_w=0.3,
        ),
        # cfg loss
        lambda_rgb=1.0,
        lambda_lpips=0.0,
        lambda_dep=1.0,
        lambda_mask=0.5,
        dep_st_invariant=True,
        lambda_normal=1.0,
        lambda_depth_normal=0.05,  # from GOF
        lambda_distortion=100.0,  # from GOF
        lambda_arap_coord=3.0,
        lambda_arap_len=0.0,
        lambda_vel_xyz_reg=0.0,
        lambda_vel_rot_reg=0.0,
        lambda_acc_xyz_reg=0.5,
        lambda_acc_rot_reg=0.5,
        lambda_small_w_reg=0.0,
        #
        lambda_track=0.0,
        track_flow_chance=0.0,
        track_flow_interval_candidates=[1],
        track_loss_clamp=100.0,
        track_loss_protect_steps=100,
        track_loss_interval=3,  # 1/3 steps are used for track loss and does not count the grad
        track_loss_start_step=-1,
        track_loss_end_step=100000,
        ######
        reg_radius=None,
        geo_reg_start_steps=0,
        viz_interval=1000,
        viz_cheap_interval=1000,
        viz_skip_t=5,
        viz_move_angle_deg=30.0,
        phase_name="photometric",
        use_decay=True,
        decay_start=2000,
        temporal_diff_shift=[1, 3, 6],
        temporal_diff_weight=[0.6, 0.3, 0.1],
        # * error grow
        dyn_error_grow_steps=[],
        dyn_error_grow_th=0.2,
        dyn_error_grow_num_frames=4,
        dyn_error_grow_subsample=1,
        # ! warning, the corr loss is in pixel int unit!!
        dyn_node_densify_steps=[],
        dyn_node_densify_grad_th=0.2,
        dyn_node_densify_record_start_steps=2000,
        dyn_node_densify_max_gs_per_new_node=100000,
        # * scf pruning
        dyn_scf_prune_steps=[],
        dyn_scf_prune_sk_th=0.02,
        # other
        random_bg=False,
        default_bg_color=[1.0, 1.0, 1.0],
        # * DynGS cleaning
        photo_s2d_trans_steps=[],
        multiview_obs=None,
        multiview_cams=None,
        multiview_use_masks=False,
        multiview_aux_ssim_lambda=0.1,
        multiview_lpips_camera_weights=None,
        lpips_net="alex",
    ):
        logging.info(f"Finetune with GS-BACKEND={GS_BACKEND.lower()}")

        torch.cuda.empty_cache()
        n_frame = 1

        d_flag = d_model is not None
        multiview_flag = multiview_obs is not None and multiview_cams is not None
        lpips_flag = lambda_lpips > 0.0
        multiview_lpips_camera_weights = _normalize_camera_weight_dict(
            multiview_lpips_camera_weights
        )
        if multiview_flag:
            logging.info(
                f"Use multiview auxiliary RGB SSIM lambda={multiview_aux_ssim_lambda}"
            )
        if lpips_flag:
            lpips_model = build_lpips_model(
                device=s2d.rgb.device,
                net=lpips_net,
                spatial=True,
            )
            logging.info(
                f"Enable LPIPS photo supervision with lambda={lambda_lpips}, "
                f"net={lpips_net}, multiview_camera_weights="
                f"{multiview_lpips_camera_weights if multiview_flag else {'anchor': 1.0}}"
            )
        else:
            lpips_model = None

        corr_flag = lambda_track > 0.0 and d_flag
        if corr_flag:
            logging.info(
                f"Enabel Flow/Track backing with supervision interval={track_loss_interval}"
            )

        optimizer_static = torch.optim.Adam(
            s_model.get_optimizable_list(**optimizer_cfg.get_static_lr_dict)
        )
        if d_flag:
            optimizer_dynamic = torch.optim.Adam(
                d_model.get_optimizable_list(**optimizer_cfg.get_dynamic_lr_dict)
            )
            if reg_radius is None:
                reg_radius = int(np.array(temporal_diff_shift).max()) * 2
            logging.info(f"Set reg_radius={reg_radius} for dynamic model")
            sup_mask_type = "all"
        else:
            sup_mask_type = "static"
        cam_param_list = cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
        if len(cam_param_list) > 0:
            optimizer_cam = torch.optim.Adam(
                cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
            )
        else:
            optimizer_cam = None
        if use_decay:
            gs_scheduling_func_dict, cam_scheduling_func_dict = (
                optimizer_cfg.get_scheduler(total_steps=total_steps - decay_start)
            )
        else:
            gs_scheduling_func_dict, cam_scheduling_func_dict = {}, {}

        loss_rgb_list, loss_lpips_list, loss_dep_list, loss_nrm_list = [], [], [], []
        loss_mask_list = []
        loss_dep_nrm_reg_list, loss_distortion_reg_list = [], []

        loss_arap_coord_list, loss_arap_len_list = [], []
        loss_vel_xyz_reg_list, loss_vel_rot_reg_list = [], []
        loss_acc_xyz_reg_list, loss_acc_rot_reg_list = [], []
        s_n_count_list, d_n_count_list = [], []
        d_m_count_list = []
        loss_sds_list = []
        loss_small_w_list = []
        loss_track_list = []

        s_gs_ctrl_start = int(total_steps * s_gs_ctrl_start_ratio)
        d_gs_ctrl_start = int(total_steps * d_gs_ctrl_start_ratio)
        s_gs_ctrl_end = int(total_steps * s_gs_ctrl_end_ratio)
        d_gs_ctrl_end = int(total_steps * d_gs_ctrl_end_ratio)
        assert s_gs_ctrl_start >= 0
        assert d_gs_ctrl_start >= 0

        latest_track_event = 0
        base_u, base_v = np.meshgrid(np.arange(s2d.W), np.arange(s2d.H))
        base_uv = np.stack([base_u, base_v], -1)
        base_uv = torch.tensor(base_uv, device=s2d.rgb.device).long()

        # prepare a color-plate for seamntic rendering
        if d_flag:
            # ! for now the group rendering only works for dynamic joitn mode
            n_group_static = len(s_model.group_id.unique())
            n_group_dynamic = len(d_model.scf.unique_grouping)
            color_plate = get_colorplate(n_group_static + n_group_dynamic)
            # random permute
            color_permute = torch.randperm(len(color_plate))
            color_plate = color_plate[color_permute]
            s_model.get_cate_color(
                color_plate=color_plate[:n_group_static].to(s_model.device)
            )
            d_model.get_cate_color(
                color_plate=color_plate[n_group_static:].to(d_model.device)
            )

        for step in tqdm(range(total_steps)):
            # * control the w correction
            if d_flag and step == skinning_corr_start_steps:
                logging.info(
                    f"at {step} stop all the topology update and add skinning weight correction"
                )
                d_model.set_surface_deform()
            corr_exe_flag = (
                corr_flag
                and step > latest_track_event + track_loss_protect_steps
                and step % track_loss_interval == 0
                and step >= track_loss_start_step
                and step < track_loss_end_step
            )
            optimizer_static.zero_grad()
            if optimizer_cam is not None:
                optimizer_cam.zero_grad()
            cams.zero_grad()
            s_model.zero_grad()
            s2d.zero_grad()
            if d_flag:
                optimizer_dynamic.zero_grad()
                d_model.zero_grad()
                if step % topo_update_feq == 0:
                    d_model.scf.update_topology()

            if step > decay_start:
                for k, v in gs_scheduling_func_dict.items():
                    update_learning_rate(v(step), k, optimizer_static)
                    if d_flag:
                        update_learning_rate(v(step), k, optimizer_dynamic)
                if optimizer_cam is not None:
                    for k, v in cam_scheduling_func_dict.items():
                        update_learning_rate(v(step), k, optimizer_cam)

            view_ind_list = np.random.choice(cams.T, n_frame, replace=False).tolist()
            if corr_exe_flag:
                # select another ind different than the view_ind_list
                corr_dst_ind_list, corr_flow_flag_list = [], []
                for view_ind in view_ind_list:
                    flow_flag = np.random.rand() < track_flow_chance
                    corr_flow_flag_list.append(flow_flag)
                    if flow_flag:
                        corr_dst_ind_candidates = []
                        for flow_interval in track_flow_interval_candidates:
                            if view_ind + flow_interval < cams.T:
                                corr_dst_ind_candidates.append(view_ind + flow_interval)
                            if view_ind - flow_interval >= 0:
                                corr_dst_ind_candidates.append(view_ind - flow_interval)
                        corr_dst_ind = np.random.choice(corr_dst_ind_candidates)
                        corr_dst_ind_list.append(corr_dst_ind)
                    else:
                        corr_dst_ind = view_ind
                        while corr_dst_ind == view_ind:
                            corr_dst_ind = np.random.choice(cams.T)
                        corr_dst_ind_list.append(corr_dst_ind)
                corr_dst_ind_list = np.array(corr_dst_ind_list)
            else:
                corr_dst_ind_list = view_ind_list
                corr_flow_flag_list = [False] * n_frame

            render_dict_list, corr_render_dict_list = [], []
            loss_rgb, loss_dep, loss_nrm = 0.0, 0.0, 0.0
            loss_lpips = torch.zeros((), device=s2d.rgb.device)
            loss_dep_nrm_reg, loss_distortion_reg = 0.0, 0.0
            loss_mask = 0.0
            loss_track = 0.0
            for _inner_loop_i, view_ind in enumerate(view_ind_list):
                dst_ind = corr_dst_ind_list[_inner_loop_i]
                flow_flag = corr_flow_flag_list[_inner_loop_i]
                gs5 = [list(s_model())]

                add_buffer = None
                if corr_exe_flag:
                    # ! detach bg pts
                    dst_xyz = torch.cat([gs5[0][0].detach(), d_model(dst_ind)[0]], 0)
                    dst_xyz_cam = cams.trans_pts_to_cam(dst_ind, dst_xyz)
                    if GS_BACKEND in ["native_add3"]:
                        add_buffer = dst_xyz_cam

                if d_flag:
                    gs5.append(list(d_model(view_ind)))
                if random_bg:
                    bg_color = np.random.rand(3).tolist()
                else:
                    bg_color = default_bg_color  # [1.0, 1.0, 1.0]
                if GS_BACKEND in ["natie_add3"]:
                    # the render internally has another protection, because if not set, the grad has bug
                    bg_color += [0.0, 0.0, 0.0]

                render_dict = render(
                    gs5,
                    s2d.H,
                    s2d.W,
                    cams.K(s2d.H, s2d.W),
                    cams.T_cw(view_ind),
                    bg_color=bg_color,
                    add_buffer=add_buffer,
                )
                render_dict_list.append(render_dict)

                # compute losses
                rgb_sup_mask = s2d.get_mask_by_key(sup_mask_type)[view_ind]
                (
                    anchor_camera_name,
                    anchor_rgb_target,
                    rgb_sup_mask,
                    anchor_rgb_weight,
                ) = _resolve_anchor_photo_target(
                    s2d.rgb[view_ind],
                    rgb_sup_mask,
                    view_ind,
                    multiview_obs=multiview_obs if multiview_flag else None,
                    multiview_use_masks=multiview_use_masks,
                )
                _l_rgb, _, _, _ = compute_rgb_loss(
                    anchor_rgb_target, render_dict, rgb_sup_mask
                )
                if lpips_flag:
                    _l_lpips, _, _, _ = compute_lpips_loss(
                        anchor_rgb_target,
                        render_dict,
                        rgb_sup_mask,
                        lpips_model,
                    )
                    loss_lpips = loss_lpips + _l_lpips * _get_lpips_camera_weight(
                        anchor_camera_name,
                        multiview_obs if multiview_flag else None,
                        multiview_lpips_camera_weights,
                    )
                dep_sup_mask = rgb_sup_mask * s2d.dep_mask[view_ind]
                _l_dep, _, _, _ = compute_dep_loss(
                    s2d.dep[view_ind].detach().clone(),
                    render_dict,
                    dep_sup_mask,
                    st_invariant=dep_st_invariant,
                )
                loss_rgb = loss_rgb + _l_rgb * anchor_rgb_weight
                _loss_multiview_rgb, _loss_multiview_lpips = (
                    _accumulate_multiview_photo_losses(
                    render_dict_list=render_dict_list,
                    gs5=gs5,
                    bg_color=bg_color,
                    camera_view_ind=view_ind,
                    obs_view_ind=view_ind,
                    multiview_obs=multiview_obs,
                    multiview_cams=multiview_cams,
                    multiview_use_masks=multiview_use_masks,
                    multiview_aux_ssim_lambda=multiview_aux_ssim_lambda,
                    lpips_model=lpips_model,
                    multiview_lpips_camera_weights=multiview_lpips_camera_weights,
                )
                )
                loss_rgb = loss_rgb + _loss_multiview_rgb
                loss_lpips = loss_lpips + _loss_multiview_lpips
                loss_dep = loss_dep + _l_dep

                if corr_exe_flag:
                    # * Track Loss
                    if GS_BACKEND in ["native_add3"]:
                        corr_render_dict = render_dict
                        rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3
                    else:
                        corr_render_dict = render(
                            # # ! use detached bg gs
                            # [[it.detach() for it in gs5[0]], gs5[1]],
                            # ! debug, align wiht .bck old version
                            gs5,
                            s2d.H,
                            s2d.W,
                            cams.K(s2d.H, s2d.W),
                            cams.T_cw(view_ind),
                            bg_color=[0.0, 0.0, 0.0],
                            colors_precomp=dst_xyz_cam,
                        )
                        rendered_xyz_map = corr_render_dict["rgb"].permute(
                            1, 2, 0
                        )  # H,W,3
                    corr_render_dict_list.append(corr_render_dict)
                    # get the flow
                    with torch.no_grad():
                        if flow_flag:
                            flow_ind = s2d.flow_ij_to_listind_dict[(view_ind, dst_ind)]
                            flow = s2d.flow[flow_ind].detach().clone()
                            flow_mask = s2d.flow_mask[flow_ind].detach().clone().bool()
                            track_src = base_uv.clone().detach()[flow_mask]
                            flow = flow[flow_mask]
                            track_dst = track_src.float() + flow
                        else:
                            # contruct target by track
                            track_valid = (
                                s2d.track_mask[view_ind] & s2d.track_mask[dst_ind]
                            )
                            track_src = s2d.track[view_ind][track_valid][..., :2]
                            track_dst = s2d.track[dst_ind][track_valid][..., :2]
                        src_fetch_index = (
                            track_src[:, 1].long() * s2d.W + track_src[:, 0].long()
                        )
                    if len(track_src) == 0:
                        _loss_track = torch.zeros_like(_l_rgb)
                    else:
                        warped_xyz_cam = rendered_xyz_map.reshape(-1, 3)[
                            src_fetch_index
                        ]
                        # filter the pred, only add loss to points that are infront of the camera
                        track_loss_mask = warped_xyz_cam[:, 2] > 1e-4
                        if track_loss_mask.sum() == 0:
                            _loss_track = torch.zeros_like(_l_rgb)
                        else:
                            pred_track_dst = cams.project(warped_xyz_cam)
                            L = min(s2d.W, s2d.H)
                            pred_track_dst[:, :1] = (
                                (pred_track_dst[:, :1] + s2d.W / L) / 2.0 * L
                            )
                            pred_track_dst[:, 1:] = (
                                (pred_track_dst[:, 1:] + s2d.H / L) / 2.0 * L
                            )
                            _loss_track = (pred_track_dst - track_dst).norm(dim=-1)[
                                track_loss_mask
                            ]
                            _loss_track = torch.clamp(
                                _loss_track, 0.0, track_loss_clamp
                            )
                            _loss_track = _loss_track.mean()
                else:
                    _loss_track = torch.zeros_like(_l_rgb)
                loss_track = loss_track + _loss_track

                # * GOF normal and regularization
                if GS_BACKEND == "gof":
                    _l_nrm, _, _, _ = compute_normal_loss(
                        s2d.nrm[view_ind].detach().clone(), render_dict, dep_sup_mask
                    )
                    loss_nrm = loss_nrm + _l_nrm
                    if step > geo_reg_start_steps:
                        _l_reg_nrm, _, _, _ = compute_normal_reg_loss(
                            s2d, cams, render_dict
                        )
                        _l_reg_distortion, _ = compute_dep_reg_loss(
                            anchor_rgb_target, render_dict
                        )
                    else:
                        _l_reg_nrm = torch.zeros_like(_l_rgb)
                        _l_reg_distortion = torch.zeros_like(_l_rgb)
                    loss_dep_nrm_reg = loss_dep_nrm_reg + _l_reg_nrm
                    loss_distortion_reg = loss_distortion_reg + _l_reg_distortion
                else:
                    loss_nrm = torch.zeros_like(loss_rgb)
                    loss_dep_nrm_reg = torch.zeros_like(loss_rgb)
                    loss_distortion_reg = torch.zeros_like(loss_rgb)

                ############
                if d_flag and lambda_mask > 0.0:
                    # * do the mask loss, including the background
                    s_cate_sph, s_gid2color = s_model.get_cate_color(
                        perm=torch.randperm(len(s_model.group_id.unique()))
                    )
                    d_cate_sph, d_gid2color = d_model.get_cate_color(
                        perm=torch.randperm(len(d_model.scf.unique_grouping))
                    )
                    with torch.no_grad():
                        inst_map = s2d.inst[view_ind]
                        gt_mask = torch.zeros_like(s2d.rgb[0])
                        for gid, color in d_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                        for gid, color in s_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                    gs5[1][-1] = d_cate_sph
                    gs5[0][-1] = s_cate_sph
                    render_dict = render(
                        gs5,
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                    )
                    pred_mask = render_dict["rgb"].permute(1, 2, 0)
                    l_mask = torch.nn.functional.mse_loss(pred_mask, gt_mask)
                    loss_mask = loss_mask + l_mask
                    # imageio.imsave(f"./debug/mask.jpg", pred_mask.detach().cpu())
                    # imageio.imsave(f"./debug/gt_mask.jpg", gt_mask.detach().cpu())
                else:
                    loss_mask = torch.zeros_like(loss_rgb)

            if d_flag:
                _l = max(0, view_ind_list[0] - reg_radius)
                _r = min(cams.T, view_ind_list[0] + 1 + reg_radius)
                reg_tids = torch.arange(_l, _r, device=s_model.device)
            if (lambda_arap_coord > 0.0 or lambda_arap_len > 0.0) and d_flag:
                loss_arap_coord, loss_arap_len = d_model.scf.compute_arap_loss(
                    reg_tids,
                    temporal_diff_shift=temporal_diff_shift,
                    temporal_diff_weight=temporal_diff_weight,
                )
                assert torch.isnan(loss_arap_coord).sum() == 0
                assert torch.isnan(loss_arap_len).sum() == 0
            else:
                loss_arap_coord = torch.zeros_like(loss_rgb)
                loss_arap_len = torch.zeros_like(loss_rgb)

            if (
                lambda_vel_xyz_reg > 0.0
                or lambda_vel_rot_reg > 0.0
                or lambda_acc_xyz_reg > 0.0
                or lambda_acc_rot_reg > 0.0
            ) and d_flag:
                (
                    loss_vel_xyz_reg,
                    loss_vel_rot_reg,
                    loss_acc_xyz_reg,
                    loss_acc_rot_reg,
                ) = d_model.scf.compute_vel_acc_loss(reg_tids)
            else:
                loss_vel_xyz_reg = loss_vel_rot_reg = loss_acc_xyz_reg = (
                    loss_acc_rot_reg
                ) = torch.zeros_like(loss_rgb)

            if d_flag:
                loss_small_w = abs(d_model._skinning_weight).mean()
            else:
                loss_small_w = torch.zeros_like(loss_rgb)

            loss = (
                loss_rgb * lambda_rgb
                + loss_lpips * lambda_lpips
                + loss_dep * lambda_dep
                + loss_mask * lambda_mask
                + loss_nrm * lambda_normal
                + loss_dep_nrm_reg * lambda_depth_normal
                + loss_distortion_reg * lambda_distortion
                + loss_arap_coord * lambda_arap_coord
                + loss_arap_len * lambda_arap_len
                + loss_vel_xyz_reg * lambda_vel_xyz_reg
                + loss_vel_rot_reg * lambda_vel_rot_reg
                + loss_acc_xyz_reg * lambda_acc_xyz_reg
                + loss_acc_rot_reg * lambda_acc_rot_reg
                + loss_small_w * lambda_small_w_reg
                + loss_track * lambda_track
            )

            loss.backward()

            optimizer_static.step()
            if d_flag:
                optimizer_dynamic.step()
            if step >= optim_cam_after_steps and optimizer_cam is not None:
                optimizer_cam.step()

            # d_model to s_model transfer [1] copy the d_gs5
            dynamic_to_static_transfer_flag = step in photo_s2d_trans_steps and d_flag
            if dynamic_to_static_transfer_flag:
                with torch.no_grad():
                    # before the gs control to append full opacity GS
                    random_select_t = np.random.choice(cams.T)
                    trans_d_gs5 = d_model(random_select_t)
                    logging.info(f"Transfer dynamic to static at step={step}")

            # gs control
            if (
                (
                    step in d_gs_ctrl_cfg.reset_steps
                    and step >= d_gs_ctrl_start
                    and step < d_gs_ctrl_end
                )
                or (
                    step in s_gs_ctrl_cfg.reset_steps
                    and step >= s_gs_ctrl_start
                    and step < s_gs_ctrl_end
                )
                or dynamic_to_static_transfer_flag
            ):
                if corr_flag:
                    logging.info(f"Reset event happened, protect tracking loss")
                    latest_track_event = step

            if (
                s_gs_ctrl_cfg is not None
                and step >= s_gs_ctrl_start
                and step < s_gs_ctrl_end
            ):
                apply_gs_control(
                    render_list=render_dict_list,
                    model=s_model,
                    gs_control_cfg=s_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_static,
                    first_N=s_model.N,
                    record_flag=(not corr_exe_flag)
                    or (GS_BACKEND not in ["native_add3"]),
                )
            if (
                d_gs_ctrl_cfg is not None
                and step >= d_gs_ctrl_start
                and step < d_gs_ctrl_end
                and d_flag
            ):
                apply_gs_control(
                    render_list=render_dict_list,
                    model=d_model,
                    gs_control_cfg=d_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_dynamic,
                    last_N=d_model.N,
                    record_flag=(not corr_exe_flag)
                    or (GS_BACKEND not in ["native_add3"]),
                )

                if corr_exe_flag and step > dyn_node_densify_record_start_steps:
                    # record the geo gradient
                    for corr_render_dict in corr_render_dict_list:
                        d_model.record_corr_grad(
                            # ! normalize the gradient by loss weight.
                            corr_render_dict["viewspace_points"].grad[-d_model.N :]
                            / lambda_track,
                            corr_render_dict["visibility_filter"][-d_model.N :],
                        )

            # d_model to s_model transfer [2] append to static model
            if dynamic_to_static_transfer_flag:
                s_model.append_gs(optimizer_static, *trans_d_gs5, new_group_id=None)

            if d_flag and step in dyn_node_densify_steps:
                d_model.gradient_based_node_densification(
                    optimizer_dynamic,
                    gradient_th=dyn_node_densify_grad_th,
                    max_gs_per_new_node=dyn_node_densify_max_gs_per_new_node,
                )

            # error grow
            if d_flag and step in dyn_error_grow_steps:
                error_grow_dyn_model(
                    s2d,
                    cams,
                    s_model,
                    d_model,
                    optimizer_dynamic,
                    step,
                    dyn_error_grow_th,
                    dyn_error_grow_num_frames,
                    dyn_error_grow_subsample,
                    viz_dir=self.viz_dir,
                    opacity_init_factor=self.opacity_init_factor,
                )
            if d_flag and step in dyn_scf_prune_steps:
                d_model.prune_nodes(
                    optimizer_dynamic,
                    prune_sk_th=dyn_scf_prune_sk_th,
                    viz_fn=osp.join(self.viz_dir, f"scf_node_prune_at_step={step}"),
                )

            loss_rgb_list.append(loss_rgb.item())
            loss_lpips_list.append(loss_lpips.item())
            loss_dep_list.append(loss_dep.item())
            loss_nrm_list.append(loss_nrm.item())
            loss_mask_list.append(loss_mask.item())

            loss_dep_nrm_reg_list.append(loss_dep_nrm_reg.item())
            loss_distortion_reg_list.append(loss_distortion_reg.item())

            loss_arap_coord_list.append(loss_arap_coord.item())
            loss_arap_len_list.append(loss_arap_len.item())
            loss_vel_xyz_reg_list.append(loss_vel_xyz_reg.item())
            loss_vel_rot_reg_list.append(loss_vel_rot_reg.item())
            loss_acc_xyz_reg_list.append(loss_acc_xyz_reg.item())
            loss_acc_rot_reg_list.append(loss_acc_rot_reg.item())
            s_n_count_list.append(s_model.N)
            d_n_count_list.append(d_model.N if d_flag else 0)
            d_m_count_list.append(d_model.M if d_flag else 0)

            loss_small_w_list.append(loss_small_w.item())
            loss_track_list.append(loss_track.item())

            # viz
            viz_flag = viz_interval > 0 and (step % viz_interval == 0)
            if viz_flag:

                if d_flag:
                    viz_hist(d_model, self.viz_dir, f"{phase_name}_step={step}_dynamic")
                    viz_dyn_hist(
                        d_model.scf,
                        self.viz_dir,
                        f"{phase_name}_step={step}_dynamic",
                    )
                    viz_path = osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_3dviz.mp4"
                    )
                    viz3d_total_video(
                        cams,
                        d_model,
                        0,
                        cams.T - 1,
                        save_path=viz_path,
                        res=480,  # 240
                        s_model=s_model,
                    )

                    # * viz grouping
                    if lambda_mask > 0.0:
                        d_model.return_cate_colors_flag = True
                        viz_path = osp.join(
                            self.viz_dir, f"{phase_name}_step={step}_3dviz_group.mp4"
                        )
                        viz3d_total_video(
                            cams,
                            d_model,
                            0,
                            cams.T - 1,
                            save_path=viz_path,
                            res=480,  # 240
                            s_model=s_model,
                        )
                        viz2d_total_video(
                            viz_vid_fn=osp.join(
                                self.viz_dir,
                                f"{phase_name}_step={step}_2dviz_group.mp4",
                            ),
                            s2d=s2d,
                            start_from=0,
                            end_at=cams.T - 1,
                            skip_t=viz_skip_t,
                            cams=cams,
                            s_model=s_model,
                            d_model=d_model,
                            subsample=1,
                            mask_type=sup_mask_type,
                            move_around_angle_deg=viz_move_angle_deg,
                        )
                        d_model.return_cate_colors_flag = False

                viz_hist(s_model, self.viz_dir, f"{phase_name}_step={step}_static")
                viz2d_total_video(
                    viz_vid_fn=osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_2dviz.mp4"
                    ),
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    subsample=1,
                    mask_type=sup_mask_type,
                    move_around_angle_deg=viz_move_angle_deg,
                )

            if viz_cheap_interval > 0 and (
                step % viz_cheap_interval == 0 or step == total_steps - 1
            ):
                # viz the accumulated grad
                with torch.no_grad():
                    photo_grad = [
                        s_model.xyz_gradient_accum
                        / torch.clamp(s_model.xyz_gradient_denom, min=1e-6)
                    ]
                    corr_grad = [torch.zeros_like(photo_grad[0])]
                    if d_flag:
                        photo_grad.append(
                            d_model.xyz_gradient_accum
                            / torch.clamp(d_model.xyz_gradient_denom, min=1e-6)
                        )
                        corr_grad.append(
                            d_model.corr_gradient_accum
                            / torch.clamp(d_model.corr_gradient_denom, min=1e-6)
                        )

                    photo_grad = torch.cat(photo_grad, 0)
                    viz_grad_color = (
                        torch.clamp(photo_grad, 0.0, d_gs_ctrl_cfg.densify_max_grad)
                        / d_gs_ctrl_cfg.densify_max_grad
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(photo_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_photo_grad_step={step}.jpg"
                        ),
                        viz_grad,
                    )

                    corr_grad = torch.cat(corr_grad, 0)
                    viz_grad_color = (
                        torch.clamp(corr_grad, 0.0, dyn_node_densify_grad_th)
                        / dyn_node_densify_grad_th
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(corr_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_corr_grad_step={step}.jpg"
                        ),
                        viz_grad,
                    )

                fig = plt.figure(figsize=(30, 8))
                for plt_i, plt_pack in enumerate(
                    [
                        ("loss_rgb", loss_rgb_list),
                        ("loss_lpips", loss_lpips_list),
                        ("loss_dep", loss_dep_list),
                        ("loss_nrm", loss_nrm_list),
                        ("loss_mask", loss_mask_list),
                        ("loss_sds", loss_sds_list),
                        ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                        ("loss_distortion_reg", loss_distortion_reg_list),
                        ("loss_arap_coord", loss_arap_coord_list),
                        ("loss_arap_len", loss_arap_len_list),
                        ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                        ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                        ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                        ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                        ("loss_small_w", loss_small_w_list),
                        ("loss_track", loss_track_list),
                        ("S-N", s_n_count_list),
                        ("D-N", d_n_count_list),
                        ("D-M", d_m_count_list),
                    ]
                ):
                    plt.subplot(2, 10, plt_i + 1)
                    value_end = 0 if len(plt_pack[1]) == 0 else plt_pack[1][-1]
                    plt.plot(plt_pack[1]), plt.title(
                        plt_pack[0] + f" End={value_end:.4f}"
                    )
                plt.savefig(
                    osp.join(self.viz_dir, f"{phase_name}_optim_loss_step={step}.jpg")
                )
                plt.close()

        # save static, camera and dynamic model
        s_save_fn = osp.join(
            self.log_dir, f"{phase_name}_s_model_{GS_BACKEND.lower()}.pth"
        )
        torch.save(s_model.state_dict(), s_save_fn)
        torch.save(cams.state_dict(), osp.join(self.log_dir, f"{phase_name}_cam.pth"))

        if d_model is not None:
            d_save_fn = osp.join(
                self.log_dir, f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth"
            )
            torch.save(d_model.state_dict(), d_save_fn)

        # viz
        fig = plt.figure(figsize=(30, 8))
        for plt_i, plt_pack in enumerate(
            [
                ("loss_rgb", loss_rgb_list),
                ("loss_lpips", loss_lpips_list),
                ("loss_dep", loss_dep_list),
                ("loss_nrm", loss_nrm_list),
                ("loss_mask", loss_mask_list),
                ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                ("loss_distortion_reg", loss_distortion_reg_list),
                ("loss_arap_coord", loss_arap_coord_list),
                ("loss_arap_len", loss_arap_len_list),
                ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                ("loss_small_w", loss_small_w_list),
                ("loss_track", loss_track_list),
                ("S-N", s_n_count_list),
                ("D-N", d_n_count_list),
                ("D-M", d_m_count_list),
            ]
        ):
            plt.subplot(2, 10, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
        plt.savefig(osp.join(self.log_dir, f"{phase_name}_optim_loss.jpg"))
        plt.close()
        viz2d_total_video(
            viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz.mp4"),
            s2d=s2d,
            start_from=0,
            end_at=cams.T - 1,
            skip_t=viz_skip_t,
            cams=cams,
            s_model=s_model,
            d_model=d_model,
            move_around_angle_deg=viz_move_angle_deg,
            print_text=False,
        )
        viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz.mp4")
        if d_flag:
            try:
                viz3d_total_video(
                    cams,
                    d_model,
                    0,
                    cams.T - 1,
                    save_path=viz_path,
                    res=480,
                    s_model=s_model,
                )
            except:
                logging.info(f"Failing to visualize the photometric_3Dviz_0.mp4, probably due to OOM Error.")
            if lambda_mask > 0.0:
                # * viz grouping
                d_model.return_cate_colors_flag = True
                s_model.return_cate_colors_flag = True
                viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz_group.mp4")
                viz3d_total_video(
                    cams,
                    d_model,
                    0,
                    cams.T - 1,
                    save_path=viz_path,
                    res=480,
                    s_model=s_model,
                )
                viz2d_total_video(
                    viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz_group.mp4"),
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    move_around_angle_deg=viz_move_angle_deg,
                    print_text=False,
                )
                d_model.return_cate_colors_flag = False
                s_model.return_cate_colors_flag = False
        torch.cuda.empty_cache()
        return

    def photometric_fit_bfs_fachain(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        s_model: StaticGaussian,
        d_model: DynSCFGaussian = None,
        optim_cam_after_steps=0,
        total_steps=8000,
        topo_update_feq=50,
        skinning_corr_start_steps=1e10,
        s_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00025,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        d_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00015,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        s_gs_ctrl_start_ratio=0.2,
        s_gs_ctrl_end_ratio=0.9,
        d_gs_ctrl_start_ratio=0.2,
        d_gs_ctrl_end_ratio=0.9,
        # optim
        optimizer_cfg: OptimCFG = OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.0001,
            lr_cam_t=0.0001,
            lr_p=0.0003,
            lr_q=0.002,
            lr_s=0.01,
            lr_o=0.1,
            lr_sph=0.005,
            # dyn
            lr_np=0.001,
            lr_nq=0.01,
            lr_w=0.3,
        ),
        # cfg loss
        lambda_rgb=1.0,
        lambda_lpips=0.0,
        lambda_dep=1.0,
        lambda_mask=0.5,
        dep_st_invariant=True,
        lambda_normal=1.0,
        lambda_depth_normal=0.05,  # from GOF
        lambda_distortion=100.0,  # from GOF
        lambda_arap_coord=3.0,
        lambda_arap_len=0.0,
        lambda_vel_xyz_reg=0.0,
        lambda_vel_rot_reg=0.0,
        lambda_acc_xyz_reg=0.5,
        lambda_acc_rot_reg=0.5,
        lambda_small_w_reg=0.0,
        #
        lambda_track=0.0,
        track_flow_chance=0.0,
        track_flow_interval_candidates=[1],
        track_loss_clamp=100.0,
        track_loss_protect_steps=100,
        track_loss_interval=3,  # 1/3 steps are used for track loss and does not count the grad
        track_loss_start_step=-1,
        track_loss_end_step=100000,
        ######
        reg_radius=None, 
        geo_reg_start_steps=0,
        viz_interval=1000,
        viz_cheap_interval=1000,
        viz_skip_t=5,
        viz_move_angle_deg=30.0,
        phase_name="photometric",
        use_decay=True,
        decay_start=2000,
        temporal_diff_shift=[1, 3, 6],
        temporal_diff_weight=[0.6, 0.3, 0.1],
        # * error grow
        dyn_error_grow_steps=[],
        dyn_error_grow_th=0.2,
        dyn_error_grow_num_frames=4,
        dyn_error_grow_subsample=1,
        # ! warning, the corr loss is in pixel int unit!!
        dyn_node_densify_steps=[],
        dyn_node_densify_grad_th=0.2,
        dyn_node_densify_record_start_steps=2000,
        dyn_node_densify_max_gs_per_new_node=100000,
        # * scf pruning
        dyn_scf_prune_steps=[],
        dyn_scf_prune_sk_th=0.02,
        # other
        random_bg=False,
        default_bg_color=[1.0, 1.0, 1.0],
        # * DynGS cleaning
        photo_s2d_trans_steps=[],
        # additional params for recur
        stack_depth_max=3, 
        dyn_gs_num_max=30000, 
        additional_mask=None, 
        image_stride=1, 
        mid_split=False, 
        min_interval=12, 
        max_depth_for_static=2, 
        max_depth_for_cams=100, 
        total_steps_list=[], 
        decay_start_list=[], 
        optim_cam_after_steps_list=[], 
        use_fachain=True, 
        multiview_obs=None,
        multiview_cams=None,
        multiview_use_masks=False,
        multiview_aux_ssim_lambda=0.1,
        multiview_lpips_camera_weights=None,
        lpips_net="alex",
        fit_cfg=None, 
    ):
        assert fit_cfg is not None
        multiview_flag = multiview_obs is not None and multiview_cams is not None
        lpips_flag = lambda_lpips > 0.0
        multiview_lpips_camera_weights = _normalize_camera_weight_dict(
            multiview_lpips_camera_weights
        )
        if multiview_flag:
            logging.info(
                f"Use multiview auxiliary RGB SSIM lambda in fachain={multiview_aux_ssim_lambda}"
            )
        if lpips_flag:
            logging.info(
                f"Enable LPIPS photo supervision in fachain with lambda={lambda_lpips}, "
                f"net={lpips_net}, multiview_camera_weights="
                f"{multiview_lpips_camera_weights if multiview_flag else {'anchor': 1.0}}"
            )

        if len(total_steps_list) != 0:
            total_steps_list = total_steps_list + [total_steps_list[-1] for idx in range(1000)]
        if len(decay_start_list) != 0:
            decay_start_list = decay_start_list + [decay_start_list[-1] for idx in range(1000)]
        if len(optim_cam_after_steps_list) != 0:
            optim_cam_after_steps_list = optim_cam_after_steps_list + [optim_cam_after_steps_list[-1] for idx in range(1000)]

        if d_model:
            self.sync_event = Event()  # 全局同步事件
            self.sync_counter = 0
            self.sync_lock = Lock()  # 保护计数器
            self.shared_s_models = {}  # 存储各GPU上的模型引用
            self.param_buffer = {}  # 用于存储梯度平均的缓冲区
            self.gpu_streams = {}  # 各GPU的通信流
            self.gpu_events = {}   # 各GPU的同步事件
            self.grad_sync_event = Event()
            self.grad_sync_counter = 0
            self.final_sync_event = Event()
            self.final_sync_counter = 0
            self.batch_nodes_len = 1

        def optimizing_bfs_interval_fachain(
            treenode: DynamicSegTreeBFS, 
            left_bound: int, 
            right_bound: int, 
            treenode_depth: int, 
            fa_dir: str, 
            fa_split_ind: int, 
            fa_left_bound: int, 
            son_name: str, 
            root_cams=None, 
            root_s_model=None, 
            root_d_model=None, 
            gpu_id=0,
            fachain_d_model_list=[], 
        ):
            '''
            fachain_d_model_list: list
                element: dict, keys: "d_model_dir", "left_bound", "right_bound"
                if len(fachain_d_model_list) == 0 -> means do not use fachain_d_model
            '''
            torch.cuda.empty_cache()
            if (mid_split is False) and (right_bound - left_bound + 1 < min_interval):
                return -1
            
            node_device = torch.device(f"cuda:{str(gpu_id)}")

            node_dir_name = f"dst_L{str(left_bound).zfill(3)}_R{str(right_bound).zfill(3)}"
            # if has trained, skip training, goto childs
            if osp.exists(osp.join(self.log_dir, node_dir_name, "photometric_d_model_native_add3.pth")):
                if not mid_split and treenode_depth == 0: # NOTE: inherit the pre-trained tree-root, but use the custom_split
                    custom_split = getattr(fit_cfg, "custom_split", "gradsplit")
                    if custom_split == "gradsplit":
                        split_grads = torch.load(os.path.join(self.log_dir, node_dir_name, "split_grads.pt"), map_location='cpu')
                        split_grads_sum = split_grads.sum()
                        min_delta_value = 4e8
                        min_delta_tid = -1
                        split_grads_accum = 0.
                        tmp_cams = MonocularCameras.load_from_ckpt(
                            torch.load(osp.join(self.log_dir, node_dir_name, "photometric_cam.pth"), map_location=torch.device('cpu'))
                        )
                        tmp_cams_T = tmp_cams.T
                        tmp_d_model = DynSCFGaussian.load_from_ckpt(
                            torch.load(
                                osp.join(self.log_dir, node_dir_name, f"photometric_d_model_{GS_BACKEND.lower()}.pth"), 
                                map_location=torch.device('cpu'),
                            ), 
                            device=torch.device('cpu'), 
                        )
                        tmp_d_model_ref_time = tmp_d_model.ref_time
                        for tid in range(tmp_cams_T):
                            _d_model_ref_time_max = tmp_d_model_ref_time.max()
                            _d_model_ref_time_min = tmp_d_model_ref_time.min()
                            tid_mask = (tmp_d_model_ref_time == tid)
                            tid_grads = split_grads[tid_mask]
                            split_grads_accum += tid_grads.sum()
                            if torch.abs(split_grads_accum - 0.5 * split_grads_sum) < min_delta_value:
                                min_delta_value = torch.abs(split_grads_accum - 0.5 * split_grads_sum)
                                min_delta_tid = tid
                        assert min_delta_tid != -1
                    if custom_split == "gradsplit":
                        if min_delta_tid == tmp_cams_T - 1:
                            split_ind = left_bound + min_delta_tid
                        else:
                            split_ind = left_bound + min_delta_tid + 1
                    elif custom_split == "flowsplit":
                        flow_split_infos = getattr(fit_cfg, "flow_split_infos", [])
                        assert len(flow_split_infos) > 0
                        if son_name == "left":
                            split_ind = flow_split_infos[2 * treenode_depth]
                        elif son_name == "right":
                            split_ind = flow_split_infos[2 * treenode_depth + 1]
                        elif son_name == "":
                            split_ind = flow_split_infos[1]
                        else:
                            print("No support for son_name:", son_name)
                            exit(-1)
                    else:
                        print("No support for custom_split:", custom_split)
                        exit(-1)
                    assert left_bound < split_ind <= right_bound
                    # NOTE: update split_ind for custom_split
                    with open(osp.join(self.log_dir, node_dir_name, "split_ind.txt"), "w") as f:
                        f.write(str(split_ind))
                        f.close()
                    stored_split_ind = split_ind
                    return stored_split_ind
                else:
                    stored_split_ind = -1
                    assert osp.exists(osp.join(self.log_dir, node_dir_name, "split_ind.txt"))
                    with open(osp.join(self.log_dir, node_dir_name, "split_ind.txt"), 'r') as f:
                        stored_split_ind_str = f.read()
                        stored_split_ind = int(stored_split_ind_str)
                        f.close()
                    return stored_split_ind

            _s2d = copy.deepcopy(s2d).to(node_device)
            _s2d._apply(lambda tt: tt.to(node_device))
            torch.cuda.synchronize(node_device)
            _additional_mask = additional_mask.to(node_device)[left_bound:right_bound+1, ...] if additional_mask is not None else None

            if root_s_model is not None:
                assert root_cams is not None
                assert root_d_model is not None
                _cams = root_cams
                _s_model = root_s_model
                _d_model = root_d_model
                assert _cams is not None
                assert _s_model is not None
                assert _d_model is not None
                _cams.to(node_device)
                _s_model.to(node_device)
                _d_model.to(node_device)
            else:
                treenode_saved_cams_dir = osp.join(fa_dir, f"{phase_name}_cam.pth")
                _cams = split_cams(treenode_saved_cams_dir, fa_split_ind-fa_left_bound, son_name, device=node_device)
                treenode_saved_smodel_dir = osp.join(fa_dir, f"{phase_name}_s_model_{GS_BACKEND.lower()}.pth")
                _s_model = load_s_model(treenode_saved_smodel_dir, device=node_device)
                assert _cams is not None
                assert _s_model is not None
                _cams.to(node_device)
                _s_model.to(node_device)

                inherit_father = getattr(fit_cfg, "inherit_father", True)
                if inherit_father:
                    treenode_saved_dmodel_dir = osp.join(fa_dir, f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth")
                    n_init=getattr(fit_cfg, "child_layer_dynamic_n_init", [30000, 10000, 5000, 2500, 1000])[treenode_depth]
                    _d_model = split_d_model(treenode_saved_dmodel_dir, fa_split_ind-fa_left_bound, son_name, left_bound=fa_left_bound, n_init=n_init, device=node_device)
                    assert _d_model is not None
                    _d_model.to(node_device)
                else:
                    scaffold = MoSca.load_from_ckpt(
                        torch.load(osp.join(osp.dirname(fa_dir), "mosca", "mosca.pth"), map_location="cpu"), 
                        node_device, 
                    )
                    scaffold.set_multi_level(
                        mlevel_arap_flag=True,
                        mlevel_list=getattr(fit_cfg, "photo_mlevel_list", [1, 6]),
                        mlevel_k_list=getattr(fit_cfg, "photo_mlevel_k_list", [16, 8]),
                        mlevel_w_list=getattr(fit_cfg, "photo_mlevel_w_list", [0.4, 0.3]),
                    )
                    scaffold.resample_time(torch.arange(left_bound, right_bound+1), scaffold._node_certain.data[left_bound:right_bound+1, :])
                    _d_model = self.get_dynamic_model_interval(
                        s2d=_s2d, 
                        cams=_cams, 
                        scf=scaffold, # splited_d_model_scf, 
                        n_init=getattr(fit_cfg, "child_layer_dynamic_n_init", [30000, 10000, 5000, 2500, 1000])[treenode_depth],
                        radius_max=getattr(fit_cfg, "gs_radius_max", 0.1),
                        max_sph_order=getattr(fit_cfg, "gs_max_sph_order", 0),
                        leaf_local_flag=getattr(fit_cfg, "gs_leaf_local_flag", True),
                        additional_mask=_additional_mask,
                        nn_fusion=getattr(fit_cfg, "gs_nn_fusion", -1),
                        # ! below is set to dyn_gs_model becaues it controls the densification
                        max_node_num=getattr(fit_cfg, "gs_max_node_num", 100000),
                        left_bound=left_bound,
                        right_bound=right_bound, 
                        device=node_device, 
                    )
                    with torch.no_grad():
                        DYNAMIC_GS_START_OPA = getattr(fit_cfg, "gs_dynamic_start_opacity", 0.02)
                        if DYNAMIC_GS_START_OPA > 0:
                            _d_model._opacity.data = _d_model.o_inv_act(
                                torch.min(
                                    _d_model.o_act(_d_model._opacity),
                                    torch.ones_like(_d_model._opacity) * DYNAMIC_GS_START_OPA,
                                )
                            )

            _optimizer_cfg = copy.deepcopy(optimizer_cfg)
            if getattr(fit_cfg, "optim_stage_decay", False) == True:
                _optimizer_cfg = copy.deepcopy(optimizer_cfg)
                optim_stage_decay_rate = np.exp((np.log(optimizer_cfg.lr_p) - np.log(optimizer_cfg.lr_p_final)) / (stack_depth_max + 1.))
                _optimizer_cfg.lr_p = optimizer_cfg.lr_p / (optim_stage_decay_rate ** treenode_depth)
                _optimizer_cfg.lr_p_final = optimizer_cfg.lr_p / (optim_stage_decay_rate ** (treenode_depth + 1.))
                optim_stage_decay_rate = np.exp((np.log(optimizer_cfg.lr_np) - np.log(optimizer_cfg.lr_np_final)) / (stack_depth_max + 1.))
                _optimizer_cfg.lr_np = optimizer_cfg.lr_np / (optim_stage_decay_rate ** treenode_depth)
                _optimizer_cfg.lr_np_final = optimizer_cfg.lr_np / (optim_stage_decay_rate ** (treenode_depth + 1.))
                optim_stage_decay_rate = np.exp((np.log(optimizer_cfg.lr_nq) - np.log(optimizer_cfg.lr_nq_final)) / (stack_depth_max + 1.))
                _optimizer_cfg.lr_nq = optimizer_cfg.lr_nq / (optim_stage_decay_rate ** treenode_depth)
                _optimizer_cfg.lr_nq_final = optimizer_cfg.lr_nq / (optim_stage_decay_rate ** (treenode_depth + 1.))

            if len(fachain_d_model_list) > 0:
                # NOTE: load to gpu: gpu_id
                fachain_length = 0
                fachain_optim_activate_length = getattr(fit_cfg, "fachain_optim_activate_length", 2)
                for fa_d_model_infodict in fachain_d_model_list:
                    fachain_length += 1
                    fa_d_model_dir = fa_d_model_infodict["d_model_dir"]
                    fa_d_model_infodict["d_model"] = DynSCFGaussian.load_from_ckpt(
                        torch.load(fa_d_model_dir, map_location=torch.device('cpu')),
                        device=node_device, 
                    )
                    if not fa_d_model_infodict["do_fachain_optim"] or fachain_length > fachain_optim_activate_length:
                        fa_d_model_infodict["d_model"].to(node_device)
                        fa_d_model_infodict["d_model"].eval()
                        for param in fa_d_model_infodict["d_model"].parameters():
                            param.requires_grad_(False)
                        for param in fa_d_model_infodict["d_model"].scf.parameters():
                            param.requires_grad_(False)
                    else:
                        fa_d_model_infodict["d_model"].to(node_device)
                        fa_d_model_infodict["d_model"].train()
                        for param in fa_d_model_infodict["d_model"].parameters():
                            param.requires_grad_(True)
                        for param in fa_d_model_infodict["d_model"].scf.parameters():
                            param.requires_grad_(True)
                        fa_d_model_infodict["d_model"].unset_surface_deform()
                        fa_d_model_infodict["optimizer_dynamic"] = torch.optim.Adam(
                            fa_d_model_infodict["d_model"].get_optimizable_list(**_optimizer_cfg.get_dynamic_lr_dict)
                        )
            
            os.makedirs(osp.join(self.log_dir, node_dir_name), exist_ok=True)
            os.makedirs(osp.join(self.log_dir, node_dir_name, "mosca_photo_viz"), exist_ok=True)

            assert _cams is not None
            assert (right_bound - left_bound + 1) == _cams.T
            n_frame = 1
            d_flag = _d_model is not None
            corr_flag = lambda_track > 0.0 and d_flag
            if corr_flag:
                logging.info(
                    f"Enabel Flow/Track backing with supervision interval={track_loss_interval}"
                )

            if treenode_depth <= max_depth_for_static or max_depth_for_static == -1:
                optimizer_static = torch.optim.Adam(
                    _s_model.get_optimizable_list(**_optimizer_cfg.get_static_lr_dict)
                )
            else:
                optimizer_static = None
                _s_model.eval()
                for param in _s_model.parameters():
                    param.requires_grad_(False)
            if d_flag:
                optimizer_dynamic = torch.optim.Adam(
                    _d_model.get_optimizable_list(**_optimizer_cfg.get_dynamic_lr_dict)
                )
                if reg_radius is None:
                    _reg_radius = int(np.array(temporal_diff_shift).max()) * 2
                else:
                    _reg_radius = reg_radius
                logging.info(f"Set reg_radius={_reg_radius} for dynamic model")
                sup_mask_type = "all"
                if root_s_model is None:
                    inherit_father = getattr(fit_cfg, "inherit_father", True)
                    if inherit_father:
                        _d_model.reset_opacity(optimizer_dynamic, d_gs_ctrl_cfg.reset_opacity)
            else:
                sup_mask_type = "static"
            _multiview_cams = None
            if multiview_obs is not None and multiview_cams is not None:
                _multiview_cams = multiview_cams.slice_time(
                    left_bound, right_bound + 1, anchor_cams=_cams
                )
            _lpips_model = None
            if lpips_flag:
                _lpips_model = build_lpips_model(
                    device=node_device,
                    net=lpips_net,
                    spatial=True,
                )

            if len(total_steps_list) == 0:
                _total_steps = total_steps # total_steps
            else:
                assert treenode_depth < len(total_steps_list)
                _total_steps = total_steps_list[treenode_depth]
            if len(decay_start_list) == 0:
                _decay_start = decay_start # decay_start
            else:
                assert treenode_depth < len(decay_start_list)
                _decay_start = decay_start_list[treenode_depth]
            if len(optim_cam_after_steps_list) == 0:
                _optim_cam_after_steps = optim_cam_after_steps # optim_cam_after_steps
            else:
                assert treenode_depth < len(optim_cam_after_steps_list)
                _optim_cam_after_steps = optim_cam_after_steps_list[treenode_depth]

            cam_param_list = _cams.get_optimizable_list(**_optimizer_cfg.get_cam_lr_dict)[:2]
            if len(cam_param_list) > 0:
                if treenode_depth <= max_depth_for_cams:
                    optimizer_cam = torch.optim.Adam(
                        _cams.get_optimizable_list(**_optimizer_cfg.get_cam_lr_dict)[:2]
                    )
                else:
                    optimizer_cam = None
            else:
                optimizer_cam = None

            if use_decay:
                gs_scheduling_func_dict, cam_scheduling_func_dict = (
                    _optimizer_cfg.get_scheduler(total_steps=_total_steps - _decay_start)
                )
            else:
                gs_scheduling_func_dict, cam_scheduling_func_dict = {}, {}

            fachain_decay_start_list = getattr(fit_cfg, "fachain_decay_start_list", [])
            if len(fachain_decay_start_list) > 0:
                if fachain_decay_start_list[treenode_depth] != -1:
                    fachain_gs_scheduling_func_dict, fachain_cam_scheduling_func_dict = (
                        _optimizer_cfg.get_scheduler(total_steps=_total_steps - fachain_decay_start_list[treenode_depth])
                    )
                else:
                    fachain_gs_scheduling_func_dict, fachain_cam_scheduling_func_dict = {}, {}

            # if optimizer_static and treenode_depth > 0 and max_depth_for_static != -1:
            if getattr(fit_cfg, "share_s_model", False):
                # register _s_model
                with self.sync_lock:
                    self.sync_counter += 1
                    self.shared_s_models[gpu_id] = _s_model
                    if self.sync_counter == self.batch_nodes_len and self.sync_counter == len(self.shared_s_models):
                        self.sync_event.set()
                self.sync_event.wait()
                with self.sync_lock:
                    self.sync_counter -= 1
                    if self.sync_counter == 0:
                        self.sync_event.clear()

            loss_rgb_list, loss_lpips_list, loss_dep_list, loss_nrm_list = [], [], [], []
            loss_mask_list = []
            loss_dep_nrm_reg_list, loss_distortion_reg_list = [], []

            loss_arap_coord_list, loss_arap_len_list = [], []
            loss_vel_xyz_reg_list, loss_vel_rot_reg_list = [], []
            loss_acc_xyz_reg_list, loss_acc_rot_reg_list = [], []
            s_n_count_list, d_n_count_list = [], []
            d_m_count_list = []
            loss_sds_list = []
            loss_small_w_list = []
            loss_track_list = []

            s_gs_ctrl_start = int(_total_steps * s_gs_ctrl_start_ratio)
            d_gs_ctrl_start = int(_total_steps * d_gs_ctrl_start_ratio)
            s_gs_ctrl_end = int(_total_steps * s_gs_ctrl_end_ratio)
            d_gs_ctrl_end = int(_total_steps * d_gs_ctrl_end_ratio)
            assert s_gs_ctrl_start >= 0
            assert d_gs_ctrl_start >= 0

            latest_track_event = 0
            base_u, base_v = np.meshgrid(np.arange(_s2d.W), np.arange(_s2d.H))
            base_uv = np.stack([base_u, base_v], -1)
            base_uv = torch.tensor(base_uv, device=_s2d.rgb.device).long()

            # prepare a color-plate for seamntic rendering
            if d_flag:
                # ! for now the group rendering only works for dynamic joitn mode
                n_group_static = len(_s_model.group_id.unique())
                n_group_dynamic = len(_d_model.scf.unique_grouping)
                color_plate = get_colorplate(n_group_static + n_group_dynamic)
                # random permute
                color_permute = torch.randperm(len(color_plate))
                color_plate = color_plate[color_permute]
                _s_model.get_cate_color(
                    color_plate=color_plate[:n_group_static].to(_s_model.device)
                )
                _d_model.get_cate_color(
                    color_plate=color_plate[n_group_static:].to(_d_model.device)
                )

            for step in tqdm(range(_total_steps)):
                # * control the w correction
                if d_flag and step == skinning_corr_start_steps: 
                    logging.info(
                        f"at {step} stop all the topology update and add skinning weight correction"
                    )
                    _d_model.set_surface_deform()
                fachain_photo_skinning_corr_start_steps = getattr(fit_cfg, "fachain_photo_skinning_corr_start_steps", [])
                if len(fachain_photo_skinning_corr_start_steps) == 0:
                    if d_flag and step == skinning_corr_start_steps: 
                        if len(fachain_d_model_list) > 0:
                            for fa_d_model_infodict in fachain_d_model_list:
                                if "optimizer_dynamic" in fa_d_model_infodict.keys():
                                    fa_d_model_infodict["d_model"].set_surface_deform()
                else:
                    if step == fachain_photo_skinning_corr_start_steps[treenode_depth]:
                        if len(fachain_d_model_list) > 0:
                            for fa_d_model_infodict in fachain_d_model_list:
                                if "optimizer_dynamic" in fa_d_model_infodict.keys():
                                    fa_d_model_infodict["d_model"].set_surface_deform()
                corr_exe_flag = (
                    corr_flag
                    and step > latest_track_event + track_loss_protect_steps 
                    and step % track_loss_interval == 0
                    and step >= track_loss_start_step 
                    and step < track_loss_end_step 
                )
                if treenode_depth > 0:
                    corr_exe_flag = corr_exe_flag and (not getattr(fit_cfg, "child_deactivate_corr", False))
                if optimizer_static:
                    optimizer_static.zero_grad()
                if optimizer_cam is not None:
                    optimizer_cam.zero_grad()
                _cams.zero_grad()
                _s_model.zero_grad()
                _s2d.zero_grad() 
                if d_flag:
                    optimizer_dynamic.zero_grad()
                    _d_model.zero_grad()
                    if len(fachain_d_model_list) > 0:
                        for fa_d_model_infodict in fachain_d_model_list:
                            if "optimizer_dynamic" in fa_d_model_infodict.keys():
                                fa_d_model_infodict["optimizer_dynamic"].zero_grad(set_to_none=True)
                                fa_d_model_infodict["d_model"].zero_grad(set_to_none=True)
                        if os.environ.get("WFM_TABLE1_KEEP_ALLOCATOR") != "1":
                            torch.cuda.empty_cache()
                    topo_update_depth_max = getattr(fit_cfg, "topo_update_depth_max", 2)
                    if step % topo_update_feq == 0 and treenode_depth <= topo_update_depth_max: 
                        start_depth_for_skip_gpu = getattr(fit_cfg, "start_depth_for_skip_gpu", 1000) 
                        skip_gpu = getattr(fit_cfg, "skip_gpu", -1)
                        scf_tmp_gpu = getattr(fit_cfg, "scf_tmp_gpu", -1)

                        if scf_tmp_gpu == -1 or treenode_depth < start_depth_for_skip_gpu:
                            if os.environ.get("WFM_TABLE1_RESIDENT_TOPOLOGY") == "1":
                                # Use the same update on its current GPU without moving all
                                # scene observations and Gaussian parameters through CPU RAM.
                                _d_model.scf.update_topology()
                                for fa_d_model_infodict in fachain_d_model_list:
                                    if "optimizer_dynamic" in fa_d_model_infodict:
                                        fa_d_model_infodict["d_model"].scf.update_topology()
                            else:
                                _s_model.to('cpu')
                                _s2d._apply(lambda tt: tt.to('cpu'))
                                move_module_device(_d_model, 'cpu')
                                if len(fachain_d_model_list) > 0:
                                    for fa_d_model_infodict in fachain_d_model_list:
                                        move_module_device(fa_d_model_infodict["d_model"], 'cpu')
                                _d_model.scf.to(node_device)
                                _d_model.scf.update_topology()
                                move_module_device(_d_model.scf, 'cpu')
                                if len(fachain_d_model_list) > 0:
                                    for fa_d_model_infodict in fachain_d_model_list:
                                        if "optimizer_dynamic" in fa_d_model_infodict.keys():
                                            start_depth_for_skip_gpu = getattr(fit_cfg, "start_depth_for_skip_gpu", 1000)
                                            skip_gpu = getattr(fit_cfg, "skip_gpu", -1)
                                            if skip_gpu != -1 and treenode_depth >= start_depth_for_skip_gpu:
                                                scf_tmp_device = torch.device(f"cuda:{skip_gpu}")
                                            else:
                                                scf_tmp_device = node_device
                                            fa_d_model_infodict["d_model"].scf.to(scf_tmp_device)
                                            fa_d_model_infodict["d_model"].scf.update_topology()
                                            move_module_device(fa_d_model_infodict["d_model"].scf, 'cpu')
                                _s_model.to(node_device)
                                _s2d._apply(lambda tt: tt.to(node_device))
                                move_module_device(_d_model, node_device)
                                if len(fachain_d_model_list) > 0:
                                    for fa_d_model_infodict in fachain_d_model_list:
                                        move_module_device(fa_d_model_infodict["d_model"], node_device)
                        else: 
                            assert scf_tmp_gpu == skip_gpu
                            scf_tmp_device = torch.device(f"cuda:{scf_tmp_gpu}")
                            with torch.cuda.device(scf_tmp_device):
                                torch.cuda.empty_cache()

                            move_mosca_device(_d_model.scf, scf_tmp_device)
                            torch.cuda.empty_cache()
                            with torch.cuda.device(scf_tmp_device):
                                torch.cuda.empty_cache()

                            _d_model.scf.update_topology()
                            
                            move_mosca_device(_d_model.scf, node_device)
                            torch.cuda.empty_cache()
                            with torch.cuda.device(scf_tmp_device):
                                torch.cuda.empty_cache()

                            if len(fachain_d_model_list) > 0:
                                for fa_d_model_infodict in fachain_d_model_list:
                                    move_mosca_device(fa_d_model_infodict["d_model"].scf, scf_tmp_device)
                                    torch.cuda.empty_cache()
                                    with torch.cuda.device(scf_tmp_device):
                                        torch.cuda.empty_cache()
                                    
                                    fa_d_model_infodict["d_model"].scf.update_topology()

                                    move_mosca_device(fa_d_model_infodict["d_model"].scf, node_device)
                                    torch.cuda.empty_cache()
                                    with torch.cuda.device(scf_tmp_device):
                                        torch.cuda.empty_cache()

                if step > _decay_start:
                    for k, v in gs_scheduling_func_dict.items():
                        if optimizer_static:
                            update_learning_rate(v(step), k, optimizer_static)
                        if d_flag:
                            update_learning_rate(v(step), k, optimizer_dynamic)
                    if optimizer_cam is not None:
                        for k, v in cam_scheduling_func_dict.items():
                            update_learning_rate(v(step), k, optimizer_cam)

                fachain_decay_start_list = getattr(fit_cfg, "fachain_decay_start_list", [])
                if len(fachain_decay_start_list) > 0:
                    for k, v in fachain_gs_scheduling_func_dict.items():
                        if len(fachain_d_model_list) > 0:
                            for fa_d_model_infodict in fachain_d_model_list:
                                if "optimizer_dynamic" in fa_d_model_infodict.keys():
                                    if step > fachain_decay_start_list[treenode_depth]:
                                        update_learning_rate(v(step), k, fa_d_model_infodict["optimizer_dynamic"])

                view_ind_list = np.random.choice(_cams.T, n_frame, replace=False).tolist()
                if corr_exe_flag:
                    # select another ind different than the view_ind_list
                    corr_dst_ind_list, corr_flow_flag_list = [], []
                    for view_ind in view_ind_list:
                        flow_flag = np.random.rand() < track_flow_chance
                        corr_flow_flag_list.append(flow_flag)
                        if flow_flag:
                            corr_dst_ind_candidates = []
                            for flow_interval in track_flow_interval_candidates:
                                if view_ind + flow_interval < _cams.T:
                                    corr_dst_ind_candidates.append(view_ind + flow_interval)
                                if view_ind - flow_interval >= 0:
                                    corr_dst_ind_candidates.append(view_ind - flow_interval)
                            corr_dst_ind = np.random.choice(corr_dst_ind_candidates)
                            corr_dst_ind_list.append(corr_dst_ind)
                        else:
                            corr_dst_ind = view_ind
                            while corr_dst_ind == view_ind:
                                corr_dst_ind = np.random.choice(_cams.T)
                            corr_dst_ind_list.append(corr_dst_ind)
                    corr_dst_ind_list = np.array(corr_dst_ind_list)
                else:
                    corr_dst_ind_list = view_ind_list
                    corr_flow_flag_list = [False] * n_frame
                render_dict_list, corr_render_dict_list = [], []
                loss_rgb, loss_dep, loss_nrm = 0.0, 0.0, 0.0
                loss_lpips = torch.zeros((), device=node_device)
                loss_dep_nrm_reg, loss_distortion_reg = 0.0, 0.0
                loss_mask = 0.0
                loss_track = 0.0
                for _inner_loop_i, view_ind in enumerate(view_ind_list):
                    _s2d_view_ind = left_bound + view_ind
                    dst_ind = corr_dst_ind_list[_inner_loop_i]
                    _s2d_dst_ind = left_bound + dst_ind
                    flow_flag = corr_flow_flag_list[_inner_loop_i]
                    gs5 = [list(_s_model())]

                    add_buffer = None
                    if corr_exe_flag:
                        # ! detach bg pts
                        if len(fachain_d_model_list) == 0:
                            dst_xyz = torch.cat([gs5[0][0].detach(), _d_model(dst_ind)[0]], 0)
                        else:
                            dst_xyz_list = [gs5[0][0].detach()]
                            dst_xyz_list.append(_d_model(dst_ind)[0])
                            for fa_d_model_infodict_idx in range(0, len(fachain_d_model_list)):
                                fa_d_model_infodict = fachain_d_model_list[fa_d_model_infodict_idx]
                                ref_time_mask = None
                                if fa_d_model_infodict["do_fachain_ref_time_mask"]:
                                    if fa_d_model_infodict_idx == 0:
                                        ref_time_mask = torch.logical_or(
                                            fa_d_model_infodict["d_model"].ref_time < fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], 
                                            fa_d_model_infodict["d_model"].ref_time > fa_d_model_infodict["leaf_right_bound"] - fa_d_model_infodict["left_bound"], 
                                        )
                                    else:
                                        ref_time_mask = torch.logical_or(
                                            fa_d_model_infodict["d_model"].ref_time < fachain_d_model_list[fa_d_model_infodict_idx-1]["left_bound"] - fa_d_model_infodict["left_bound"], 
                                            fa_d_model_infodict["d_model"].ref_time > fachain_d_model_list[fa_d_model_infodict_idx-1]["right_bound"] - fa_d_model_infodict["left_bound"], 
                                        )
                                    ref_time_extend_mask = torch.logical_and(
                                        fa_d_model_infodict["d_model"].ref_time >= fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["extend_length"], 
                                        fa_d_model_infodict["d_model"].ref_time <= fa_d_model_infodict["leaf_right_bound"] + fa_d_model_infodict["extend_length"], 
                                    )
                                    ref_time_mask = torch.logical_and(ref_time_extend_mask, ref_time_mask)
                                if ref_time_mask is not None:
                                    fa_d_model_infodict["ref_time_mask"] = ref_time_mask
                                if (ref_time_mask is None) or (ref_time_mask is not None and ref_time_mask.sum() > 0):
                                    dst_xyz_list.append(fa_d_model_infodict["d_model"](dst_ind + fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], ref_time_mask=ref_time_mask)[0]) 
                                '''
                                NOTE: the idx calculation
                                dst_ind = real_dst_ind - left_bound -> real_dst_ind = dst_ind + left_bound
                                fa_dst_ind = real_dst_ind - fa_d_model_infodict["left_bound"]
                                fa_dst_ind = dst_ind + left_bound - fa_d_model_infodict["left_bound"]
                                '''
                            dst_xyz = torch.cat(dst_xyz_list, 0)

                        dst_xyz_cam = _cams.trans_pts_to_cam(dst_ind, dst_xyz)
                        if GS_BACKEND in ["native_add3"]:
                            add_buffer = dst_xyz_cam

                    if d_flag:
                        if len(fachain_d_model_list) == 0:
                            gs5.append(list(_d_model(view_ind)))
                        else:
                            gs5.append(list(_d_model(view_ind)))
                            for fa_d_model_infodict_idx in range(0, len(fachain_d_model_list)):
                                fa_d_model_infodict = fachain_d_model_list[fa_d_model_infodict_idx]
                                ref_time_mask = None
                                if fa_d_model_infodict["do_fachain_ref_time_mask"]:
                                    if fa_d_model_infodict_idx == 0:
                                        ref_time_mask = torch.logical_or(
                                            fa_d_model_infodict["d_model"].ref_time < fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], 
                                            fa_d_model_infodict["d_model"].ref_time > fa_d_model_infodict["leaf_right_bound"] - fa_d_model_infodict["left_bound"], 
                                        )
                                    else:
                                        ref_time_mask = torch.logical_or(
                                            fa_d_model_infodict["d_model"].ref_time < fachain_d_model_list[fa_d_model_infodict_idx-1]["left_bound"] - fa_d_model_infodict["left_bound"], 
                                            fa_d_model_infodict["d_model"].ref_time > fachain_d_model_list[fa_d_model_infodict_idx-1]["right_bound"] - fa_d_model_infodict["left_bound"], 
                                        )
                                    ref_time_extend_mask = torch.logical_and(
                                        fa_d_model_infodict["d_model"].ref_time >= fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["extend_length"], 
                                        fa_d_model_infodict["d_model"].ref_time <= fa_d_model_infodict["leaf_right_bound"] + fa_d_model_infodict["extend_length"], 
                                    )
                                    ref_time_mask = torch.logical_and(ref_time_extend_mask, ref_time_mask)
                                if ref_time_mask is not None:
                                    fa_d_model_infodict["ref_time_mask"] = ref_time_mask
                                if (ref_time_mask is None) or (ref_time_mask is not None and ref_time_mask.sum() > 0):
                                    gs5.append(list(fa_d_model_infodict["d_model"](dst_ind + fa_d_model_infodict["leaf_left_bound"] - fa_d_model_infodict["left_bound"], ref_time_mask=ref_time_mask)))
                    if random_bg:
                        bg_color = np.random.rand(3).tolist()
                    else:
                        bg_color = default_bg_color  # [1.0, 1.0, 1.0]
                    if GS_BACKEND in ["natie_add3"]:
                        # the render internally has another protection, because if not set, the grad has bug
                        bg_color += [0.0, 0.0, 0.0]
                    try:
                        render_dict = render(
                            gs5,
                            _s2d.H,
                            _s2d.W,
                            _cams.K(_s2d.H, _s2d.W),
                            _cams.T_cw(view_ind),
                            bg_color=bg_color,
                            add_buffer=add_buffer,
                        )
                    except:
                        if add_buffer is not None:
                            print(f"fachain debug: gpu_id:{gpu_id}: {gs5[1][0].shape}, {_d_model._xyz.shape}, {add_buffer.shape}")
                        else:
                            print(f"fachain debug: gpu_id:{gpu_id}: {gs5[1][0].shape}, {_d_model._xyz.shape}")
                        import pdb; pdb.set_trace()
                    render_dict_list.append(render_dict)

                    # compute losses
                    rgb_sup_mask = _s2d.get_mask_by_key(sup_mask_type)[_s2d_view_ind] 
                    (
                        anchor_camera_name,
                        anchor_rgb_target,
                        rgb_sup_mask,
                        anchor_rgb_weight,
                    ) = _resolve_anchor_photo_target(
                        _s2d.rgb[_s2d_view_ind],
                        rgb_sup_mask,
                        _s2d_view_ind,
                        multiview_obs=multiview_obs if multiview_flag else None,
                        multiview_use_masks=multiview_use_masks,
                    )
                    _l_rgb, _, _, _ = compute_rgb_loss(
                        anchor_rgb_target, render_dict, rgb_sup_mask
                    )
                    if lpips_flag:
                        _l_lpips, _, _, _ = compute_lpips_loss(
                            anchor_rgb_target,
                            render_dict,
                            rgb_sup_mask,
                            _lpips_model,
                        )
                        loss_lpips = loss_lpips + _l_lpips * _get_lpips_camera_weight(
                            anchor_camera_name,
                            multiview_obs if multiview_flag else None,
                            multiview_lpips_camera_weights,
                        )
                    dep_sup_mask = rgb_sup_mask * _s2d.dep_mask[_s2d_view_ind] 
                    _l_dep, _, _, _ = compute_dep_loss(
                        _s2d.dep[_s2d_view_ind].detach().clone(),
                        render_dict,
                        dep_sup_mask,
                        st_invariant=dep_st_invariant,
                    )
                    loss_rgb = loss_rgb + _l_rgb * anchor_rgb_weight
                    _loss_multiview_rgb, _loss_multiview_lpips = (
                        _accumulate_multiview_photo_losses(
                        render_dict_list=render_dict_list,
                        gs5=gs5,
                        bg_color=bg_color,
                        camera_view_ind=view_ind,
                        obs_view_ind=_s2d_view_ind,
                        multiview_obs=multiview_obs,
                        multiview_cams=_multiview_cams,
                        multiview_use_masks=multiview_use_masks,
                        multiview_aux_ssim_lambda=multiview_aux_ssim_lambda,
                        lpips_model=_lpips_model,
                        multiview_lpips_camera_weights=multiview_lpips_camera_weights,
                    )
                    )
                    loss_rgb = loss_rgb + _loss_multiview_rgb
                    loss_lpips = loss_lpips + _loss_multiview_lpips
                    loss_dep = loss_dep + _l_dep

                    if corr_exe_flag:
                        # * Track Loss
                        if GS_BACKEND in ["native_add3"]:
                            corr_render_dict = render_dict
                            rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3
                        else:
                            corr_render_dict = render(
                                # # ! use detached bg gs
                                # [[it.detach() for it in gs5[0]], gs5[1]],
                                # ! debug, align wiht .bck old version
                                gs5,
                                _s2d.H,
                                _s2d.W,
                                _cams.K(_s2d.H, _s2d.W),
                                _cams.T_cw(view_ind),
                                bg_color=[0.0, 0.0, 0.0],
                                colors_precomp=dst_xyz_cam,
                            )
                            rendered_xyz_map = corr_render_dict["rgb"].permute(
                                1, 2, 0
                            )  # H,W,3
                        corr_render_dict_list.append(corr_render_dict)
                        # get the flow
                        with torch.no_grad():
                            if flow_flag:
                                flow_ind = _s2d.flow_ij_to_listind_dict[(_s2d_view_ind, _s2d_dst_ind)]
                                flow = _s2d.flow[flow_ind].detach().clone()
                                flow_mask = _s2d.flow_mask[flow_ind].detach().clone().bool()
                                track_src = base_uv.clone().detach()[flow_mask]
                                flow = flow[flow_mask]
                                track_dst = track_src.float() + flow
                            else:
                                # contruct target by track
                                track_valid = (
                                    _s2d.track_mask[_s2d_view_ind] & _s2d.track_mask[_s2d_dst_ind]
                                )
                                track_src = _s2d.track[_s2d_view_ind][track_valid][..., :2]
                                track_dst = _s2d.track[_s2d_dst_ind][track_valid][..., :2]
                            src_fetch_index = (
                                track_src[:, 1].long() * _s2d.W + track_src[:, 0].long()
                            )
                        if len(track_src) == 0:
                            _loss_track = torch.zeros_like(_l_rgb)
                        else:
                            warped_xyz_cam = rendered_xyz_map.reshape(-1, 3)[
                                src_fetch_index
                            ]
                            # filter the pred, only add loss to points that are infront of the camera
                            track_loss_mask = warped_xyz_cam[:, 2] > 1e-4
                            if track_loss_mask.sum() == 0:
                                _loss_track = torch.zeros_like(_l_rgb)
                            else:
                                pred_track_dst = _cams.project(warped_xyz_cam)
                                L = min(_s2d.W, _s2d.H)
                                pred_track_dst[:, :1] = (
                                    (pred_track_dst[:, :1] + _s2d.W / L) / 2.0 * L
                                )
                                pred_track_dst[:, 1:] = (
                                    (pred_track_dst[:, 1:] + _s2d.H / L) / 2.0 * L
                                )
                                _loss_track = (pred_track_dst - track_dst).norm(dim=-1)[
                                    track_loss_mask
                                ]
                                _loss_track = torch.clamp(
                                    _loss_track, 0.0, track_loss_clamp
                                )
                                _loss_track = _loss_track.mean()
                    else:
                        _loss_track = torch.zeros_like(_l_rgb)
                    loss_track = loss_track + _loss_track

                    # * GOF normal and regularization
                    if GS_BACKEND == "gof": # NOTE: GS_BACKEND is always defaut native_add3
                        _l_nrm, _, _, _ = compute_normal_loss(
                            _s2d.nrm[_s2d_view_ind].detach().clone(), render_dict, dep_sup_mask
                        )
                        loss_nrm = loss_nrm + _l_nrm
                        if step > geo_reg_start_steps: 
                            _l_reg_nrm, _, _, _ = compute_normal_reg_loss(
                                _s2d, _cams, render_dict
                            )
                            _l_reg_distortion, _ = compute_dep_reg_loss(
                                anchor_rgb_target, render_dict
                            )
                        else:
                            _l_reg_nrm = torch.zeros_like(_l_rgb)
                            _l_reg_distortion = torch.zeros_like(_l_rgb)
                        loss_dep_nrm_reg = loss_dep_nrm_reg + _l_reg_nrm
                        loss_distortion_reg = loss_distortion_reg + _l_reg_distortion
                    else:
                        loss_nrm = torch.zeros_like(loss_rgb)
                        loss_dep_nrm_reg = torch.zeros_like(loss_rgb)
                        loss_distortion_reg = torch.zeros_like(loss_rgb)

                    ############
                    if d_flag and lambda_mask > 0.0:
                        # * do the mask loss, including the background
                        s_cate_sph, s_gid2color = _s_model.get_cate_color(
                            perm=torch.randperm(len(_s_model.group_id.unique()))
                        )
                        d_cate_sph, d_gid2color = _d_model.get_cate_color(
                            perm=torch.randperm(len(_d_model.scf.unique_grouping))
                        )
                        with torch.no_grad():
                            inst_map = _s2d.inst[_s2d_view_ind]
                            gt_mask = torch.zeros_like(_s2d.rgb[0])
                            for gid, color in d_gid2color.items():
                                gt_mask[inst_map == gid] = color[None]
                            for gid, color in s_gid2color.items():
                                gt_mask[inst_map == gid] = color[None]
                        gs5[1][-1] = d_cate_sph
                        gs5[0][-1] = s_cate_sph
                        render_dict = render(
                            gs5,
                            _s2d.H,
                            _s2d.W,
                            _cams.K(_s2d.H, _s2d.W),
                            _cams.T_cw(view_ind),
                            bg_color=[0.0, 0.0, 0.0],
                        )
                        pred_mask = render_dict["rgb"].permute(1, 2, 0)
                        l_mask = torch.nn.functional.mse_loss(pred_mask, gt_mask)
                        loss_mask = loss_mask + l_mask
                    else:
                        loss_mask = torch.zeros_like(loss_rgb)

                if d_flag:
                    _l = max(0, view_ind_list[0] - _reg_radius)
                    _r = min(_cams.T, view_ind_list[0] + 1 + _reg_radius)
                    reg_tids = torch.arange(_l, _r, device=_s_model.device)
                if (lambda_arap_coord > 0.0 or lambda_arap_len > 0.0) and d_flag:
                    loss_arap_coord, loss_arap_len = _d_model.scf.compute_arap_loss(
                        reg_tids,
                        temporal_diff_shift=temporal_diff_shift,
                        temporal_diff_weight=temporal_diff_weight,
                    )
                    assert torch.isnan(loss_arap_coord).sum() == 0
                    assert torch.isnan(loss_arap_len).sum() == 0
                else:
                    loss_arap_coord = torch.zeros_like(loss_rgb)
                    loss_arap_len = torch.zeros_like(loss_rgb)

                if (
                    lambda_vel_xyz_reg > 0.0
                    or lambda_vel_rot_reg > 0.0
                    or lambda_acc_xyz_reg > 0.0
                    or lambda_acc_rot_reg > 0.0
                ) and d_flag:
                    (
                        loss_vel_xyz_reg,
                        loss_vel_rot_reg,
                        loss_acc_xyz_reg,
                        loss_acc_rot_reg,
                    ) = _d_model.scf.compute_vel_acc_loss(reg_tids)
                else:
                    loss_vel_xyz_reg = loss_vel_rot_reg = loss_acc_xyz_reg = (
                        loss_acc_rot_reg
                    ) = torch.zeros_like(loss_rgb)

                if d_flag:
                    loss_small_w = abs(_d_model._skinning_weight).mean()
                else:
                    loss_small_w = torch.zeros_like(loss_rgb)

                loss = (
                    loss_rgb * lambda_rgb
                    + loss_lpips * lambda_lpips
                    + loss_dep * lambda_dep
                    + loss_mask * lambda_mask
                    + loss_nrm * lambda_normal
                    + loss_dep_nrm_reg * lambda_depth_normal
                    + loss_distortion_reg * lambda_distortion
                    + loss_arap_coord * lambda_arap_coord
                    + loss_arap_len * lambda_arap_len
                    + loss_vel_xyz_reg * lambda_vel_xyz_reg
                    + loss_vel_rot_reg * lambda_vel_rot_reg
                    + loss_acc_xyz_reg * lambda_acc_xyz_reg
                    + loss_acc_rot_reg * lambda_acc_rot_reg
                    + loss_small_w * lambda_small_w_reg
                    + loss_track * lambda_track
                )

                loss.backward()

                if optimizer_static:
                    if not getattr(fit_cfg, "share_s_model", False):
                        optimizer_static.step()
                    else: # NOTE: easily causing deadlock
                        with self.sync_lock:
                            self.sync_counter += 1
                            if self.sync_counter == len(self.shared_s_models):
                                self.sync_event.set()
                        self.sync_event.wait()
                        with self.sync_lock:
                            self.sync_counter -= 1
                            if self.sync_counter == 0:
                                self.sync_event.clear()  # 重置事件，为下次同步做准备
                        if gpu_id == 0:
                            self.grad_sync_counter += 1
                            while True:
                                if self.grad_sync_counter == len(self.shared_s_models):
                                    break
                            with self.sync_lock:
                                for name, param in _s_model.named_parameters():
                                    if param.grad is not None:
                                        grads = []
                                        for gid, model in self.shared_s_models.items():
                                            other_param = dict(model.named_parameters())[name]
                                            if other_param.grad is not None:
                                                grads.append(other_param.grad.to(node_device))
                                        if grads:
                                            self.param_buffer[name] = torch.mean(torch.stack(grads), dim=0)
                            self.grad_sync_event.set()
                        else:
                            self.grad_sync_counter += 1
                            self.grad_sync_event.wait()
                        with self.sync_lock:
                            self.grad_sync_counter -= 1
                            if self.grad_sync_counter == 0:
                                self.grad_sync_event.clear()
                        # update param on gpu 0 and broad to other gpus
                        if gpu_id == 0:
                            assert self.param_buffer is not None
                            self.final_sync_counter += 1
                            while True:
                                if self.final_sync_counter == len(self.shared_s_models):
                                    break
                            with self.sync_lock:
                                for name, param in _s_model.named_parameters():
                                    if name in self.param_buffer:
                                        param.grad = self.param_buffer[name].to(node_device)
                                if optimizer_static:
                                    optimizer_static.step()
                                main_stream = torch.cuda.Stream(device=node_device)
                                with torch.cuda.stream(main_stream):
                                    # get params of gpu 0 (updated)
                                    state_dict = {k: v for k, v in _s_model.state_dict().items()}
                                    # broad to other gpus
                                    for gid, model in self.shared_s_models.items():
                                        if gid != 0:
                                            if gid not in self.gpu_streams:
                                                self.gpu_streams[gid] = torch.cuda.Stream(device=f"cuda:{gid}")
                                                self.gpu_events[gid] = torch.cuda.Event(enable_timing=False)
                                            # gpu-to-gpu for params transport
                                            with torch.cuda.stream(self.gpu_streams[gid]):
                                                for name, param in model.named_parameters():
                                                    if name in state_dict:
                                                        param.data.copy_(state_dict[name].to(f"cuda:{gid}", non_blocking=True))
                                                self.gpu_events[gid].record()
                                # wait for params transport
                                for gid in self.shared_s_models:
                                    if gid != 0:
                                        self.gpu_events[gid].synchronize()
                            self.final_sync_event.set()
                        else:
                            self.final_sync_counter += 1
                            self.final_sync_event.wait()
                        with self.sync_lock:
                            self.final_sync_counter -= 1
                            if self.final_sync_counter == 0:
                                self.final_sync_event.clear()

                if d_flag:
                    optimizer_dynamic.step()
                    if len(fachain_d_model_list) > 0:
                        for fa_d_model_infodict in fachain_d_model_list:
                            if "optimizer_dynamic" in fa_d_model_infodict.keys():
                                fa_d_model_infodict["optimizer_dynamic"].step()
                if step >= _optim_cam_after_steps and optimizer_cam is not None: 
                    optimizer_cam.step()

                # d_model to s_model transfer [1] copy the d_gs5
                dynamic_to_static_transfer_flag = step in photo_s2d_trans_steps and d_flag and treenode_depth == 0 
                if dynamic_to_static_transfer_flag: # NOTE: we set the nodes except of root do not dynamic_to_static_transfer_flag == True by the above setting, so no fachain_d_model
                    with torch.no_grad():
                        # before the gs control to append full opacity GS
                        random_select_t = np.random.choice(_cams.T)
                        trans_d_gs5 = _d_model(random_select_t)
                        logging.info(f"Transfer dynamic to static at step={step}")
                # gs control
                if (
                    (
                        step in d_gs_ctrl_cfg.reset_steps 
                        and step >= d_gs_ctrl_start
                        and step < d_gs_ctrl_end
                    )
                    or (
                        step in s_gs_ctrl_cfg.reset_steps
                        and step >= s_gs_ctrl_start
                        and step < s_gs_ctrl_end
                    )
                    or dynamic_to_static_transfer_flag
                ):
                    if corr_flag:
                        logging.info(f"Reset event happened, protect tracking loss")
                        latest_track_event = step

                fachain_gs_control = getattr(fit_cfg, "fachain_gs_control", False)
                rec_s_model_N = copy.deepcopy(_s_model.N)
                rec_d_model_N = copy.deepcopy(_d_model.N)
                rec_fachain_d_model_N_list = [copy.deepcopy(fachain_info_dict["d_model"].N) for fachain_info_dict in fachain_d_model_list]
                if (
                    s_gs_ctrl_cfg is not None
                    and step >= s_gs_ctrl_start
                    and step < s_gs_ctrl_end
                    and ((root_s_model is not None )or fachain_gs_control)
                    and optimizer_static
                ): 
                    apply_gs_control(
                        render_list=render_dict_list,
                        model=_s_model,
                        gs_control_cfg=s_gs_ctrl_cfg,
                        step=step,
                        optimizer_gs=optimizer_static,
                        first_N=_s_model.N,
                        record_flag=(not corr_exe_flag)
                        or (GS_BACKEND not in ["native_add3"]),
                    )
                if (
                    d_gs_ctrl_cfg is not None
                    and step >= d_gs_ctrl_start
                    and step < d_gs_ctrl_end
                    and d_flag
                    and ((root_d_model is not None) or fachain_gs_control)
                ):
                    if len(fachain_d_model_list) == 0:
                        apply_gs_control(
                            render_list=render_dict_list,
                            model=_d_model,
                            gs_control_cfg=d_gs_ctrl_cfg,
                            step=step,
                            optimizer_gs=optimizer_dynamic,
                            last_N=_d_model.N,
                            record_flag=(not corr_exe_flag)
                            or (GS_BACKEND not in ["native_add3"]),
                        )
                    else:
                        last_N_start = rec_s_model_N
                        last_N_end = last_N_start + rec_d_model_N
                        apply_gs_control_param_interval(
                            render_list=render_dict_list,
                            model=_d_model,
                            gs_control_cfg=d_gs_ctrl_cfg,
                            step=step,
                            optimizer_gs=optimizer_dynamic,
                            last_N_start=last_N_start,
                            last_N_end=last_N_end, 
                            record_flag=(not corr_exe_flag)
                            or (GS_BACKEND not in ["native_add3"]),
                        )
                        for fa_d_model_infodict_idx in range(len(fachain_d_model_list)):
                            fa_d_model_infodict = fachain_d_model_list[fa_d_model_infodict_idx]
                            last_N_start = last_N_end
                            if "ref_time_mask" in fa_d_model_infodict.keys():
                                if fa_d_model_infodict["ref_time_mask"].sum() == 0:
                                    continue
                                last_N_end = last_N_start + fa_d_model_infodict["ref_time_mask"].sum()
                            else:
                                last_N_end = last_N_start + rec_fachain_d_model_N_list[fa_d_model_infodict_idx]
                            assert last_N_start < last_N_end
                            for each_render_dict in render_dict_list:
                                assert last_N_end <= each_render_dict["viewspace_points"].grad.shape[0]
                            if "optimizer_dynamic" in fa_d_model_infodict.keys():
                                if "ref_time_mask" in fa_d_model_infodict.keys():
                                    apply_gs_control_param_interval(
                                        render_list=render_dict_list,
                                        model=fa_d_model_infodict["d_model"],
                                        gs_control_cfg=d_gs_ctrl_cfg,
                                        step=step,
                                        optimizer_gs=fa_d_model_infodict["optimizer_dynamic"],
                                        last_N_start=last_N_start,
                                        last_N_end=last_N_end, 
                                        record_flag=(not corr_exe_flag)
                                        or (GS_BACKEND not in ["native_add3"]),
                                        ref_time_mask=fa_d_model_infodict["ref_time_mask"], 
                                    )
                                else:
                                    apply_gs_control_param_interval(
                                        render_list=render_dict_list,
                                        model=fa_d_model_infodict["d_model"],
                                        gs_control_cfg=d_gs_ctrl_cfg,
                                        step=step,
                                        optimizer_gs=fa_d_model_infodict["optimizer_dynamic"],
                                        last_N_start=last_N_start,
                                        last_N_end=last_N_end, 
                                        record_flag=(not corr_exe_flag)
                                        or (GS_BACKEND not in ["native_add3"]),
                                        ref_time_mask=None, 
                                    )
                        for each_render_dict in render_dict_list:
                            grad_N = each_render_dict["viewspace_points"].grad.shape[0]
                            try:
                                assert last_N_end == grad_N, f"{last_N_end}, {grad_N}"
                            except:
                                import pdb; pdb.set_trace()

                    if corr_exe_flag and step > dyn_node_densify_record_start_steps:
                        # record the geo gradient
                        for corr_render_dict in corr_render_dict_list:
                            if len(fachain_d_model_list) == 0:
                                _d_model.record_corr_grad(
                                    # ! normalize the gradient by loss weight.
                                    corr_render_dict["viewspace_points"].grad[-_d_model.N :]
                                    / lambda_track,
                                    corr_render_dict["visibility_filter"][-_d_model.N :],
                                )

                # d_model to s_model transfer [2] append to static model
                if dynamic_to_static_transfer_flag and optimizer_static:
                    _s_model.append_gs(optimizer_static, *trans_d_gs5, new_group_id=None)

                if d_flag and step in dyn_node_densify_steps:
                    _d_model.gradient_based_node_densification(
                        optimizer_dynamic,
                        gradient_th=dyn_node_densify_grad_th,
                        max_gs_per_new_node=dyn_node_densify_max_gs_per_new_node,
                    )

                # error grow
                if d_flag and step in dyn_error_grow_steps: 
                    error_grow_dyn_model(
                        _s2d,
                        _cams,
                        _s_model,
                        _d_model,
                        optimizer_dynamic,
                        step,
                        dyn_error_grow_th,
                        dyn_error_grow_num_frames,
                        dyn_error_grow_subsample,
                        viz_dir=osp.join(self.log_dir, node_dir_name),
                        opacity_init_factor=self.opacity_init_factor,
                    )
                if d_flag and step in dyn_scf_prune_steps: 
                    _d_model.prune_nodes(
                        optimizer_dynamic,
                        prune_sk_th=dyn_scf_prune_sk_th,
                        viz_fn=osp.join(self.log_dir, node_dir_name, "mosca_photo_viz", f"scf_node_prune_at_step={step}"),
                    )


                loss_rgb_list.append(loss_rgb.item())
                loss_lpips_list.append(loss_lpips.item())
                loss_dep_list.append(loss_dep.item())
                loss_nrm_list.append(loss_nrm.item())
                loss_mask_list.append(loss_mask.item())

                loss_dep_nrm_reg_list.append(loss_dep_nrm_reg.item())
                loss_distortion_reg_list.append(loss_distortion_reg.item())

                loss_arap_coord_list.append(loss_arap_coord.item())
                loss_arap_len_list.append(loss_arap_len.item())
                loss_vel_xyz_reg_list.append(loss_vel_xyz_reg.item())
                loss_vel_rot_reg_list.append(loss_vel_rot_reg.item())
                loss_acc_xyz_reg_list.append(loss_acc_xyz_reg.item())
                loss_acc_rot_reg_list.append(loss_acc_rot_reg.item())
                s_n_count_list.append(_s_model.N)
                d_n_count_list.append(_d_model.N if d_flag else 0)
                d_m_count_list.append(_d_model.M if d_flag else 0)

                loss_small_w_list.append(loss_small_w.item())
                loss_track_list.append(loss_track.item())

                # train-loop viz
                viz_flag = viz_interval > 0 and (step % viz_interval == 0) 
                if viz_flag:

                    if d_flag:
                        viz_hist(_d_model, osp.join(self.log_dir, node_dir_name, "mosca_photo_viz"), f"{phase_name}_step={step}_dynamic")
                        viz_dyn_hist(
                            _d_model.scf,
                            osp.join(self.log_dir, node_dir_name, "mosca_photo_viz"),
                            f"{phase_name}_step={step}_dynamic",
                        )
                        viz_path = osp.join(
                            self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_step={step}_3dviz.mp4"
                        )
                        try:
                            viz3d_total_video(
                                _cams,
                                _d_model,
                                0,
                                _cams.T - 1,
                                save_path=viz_path,
                                res=480,  # 240
                                s_model=_s_model,
                            )
                        except Exception as e:
                            logging.info(f"Failing to visualize the {phase_name}_step={step}_3dviz.mp4, probably due to OOM Error.")

                        # * viz grouping
                        if lambda_mask > 0.0:
                            _d_model.return_cate_colors_flag = True
                            viz_path = osp.join(
                                self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_step={step}_3dviz_group.mp4"
                            )
                            try:
                                viz3d_total_video(
                                    _cams,
                                    _d_model,
                                    0,
                                    _cams.T - 1,
                                    save_path=viz_path,
                                    res=480,  # 240
                                    s_model=_s_model,
                                )
                            except Exception as e:
                                logging.info(f"Failing to visualize the {phase_name}_step={step}_3dviz_group.mp4, probably due to OOM Error.")
                            viz2d_total_video_interval_fachain(
                                viz_vid_fn=osp.join(
                                    self.log_dir, node_dir_name, "mosca_photo_viz",
                                    f"{phase_name}_step={step}_2dviz_group.mp4",
                                ),
                                s2d=_s2d,
                                start_from=0,
                                end_at=_cams.T - 1,
                                skip_t=viz_skip_t,
                                cams=_cams,
                                s_model=_s_model,
                                d_model=_d_model,
                                subsample=1,
                                mask_type=sup_mask_type,
                                move_around_angle_deg=viz_move_angle_deg,
                                left_bound=left_bound, 
                                fachain_d_model_list=fachain_d_model_list, 
                            )
                            _d_model.return_cate_colors_flag = False

                    viz_hist(_s_model, osp.join(self.log_dir, node_dir_name, "mosca_photo_viz"), f"{phase_name}_step={step}_static")
                    viz2d_total_video_interval_fachain(
                        viz_vid_fn=osp.join(
                            self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_step={step}_2dviz.mp4"
                        ),
                        s2d=_s2d,
                        start_from=0,
                        end_at=_cams.T - 1,
                        skip_t=viz_skip_t,
                        cams=_cams,
                        s_model=_s_model,
                        d_model=_d_model,
                        subsample=1,
                        mask_type=sup_mask_type,
                        move_around_angle_deg=viz_move_angle_deg,
                        left_bound=left_bound, 
                        fachain_d_model_list=fachain_d_model_list, 
                    )

                if viz_cheap_interval > 0 and (
                    step % viz_cheap_interval == 0 or step == _total_steps - 1
                ):
                    try:
                        # viz the accumulated grad
                        with torch.no_grad():
                            photo_grad = [
                                _s_model.xyz_gradient_accum
                                / torch.clamp(_s_model.xyz_gradient_denom, min=1e-6)
                            ]
                            corr_grad = [torch.zeros_like(photo_grad[0])]
                            if d_flag:
                                photo_grad.append(
                                    _d_model.xyz_gradient_accum
                                    / torch.clamp(_d_model.xyz_gradient_denom, min=1e-6)
                                )
                                corr_grad.append(
                                    _d_model.corr_gradient_accum
                                    / torch.clamp(_d_model.corr_gradient_denom, min=1e-6)
                                )

                            photo_grad = torch.cat(photo_grad, 0)
                            viz_grad_color = (
                                torch.clamp(photo_grad, 0.0, d_gs_ctrl_cfg.densify_max_grad)
                                / d_gs_ctrl_cfg.densify_max_grad
                            )
                            viz_grad_color = viz_grad_color.detach().cpu().numpy()
                            viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                            viz_render_dict = render(
                                [_s_model(), _d_model(view_ind)],
                                _s2d.H,
                                _s2d.W,
                                _cams.K(_s2d.H, _s2d.W),
                                _cams.T_cw(view_ind),
                                bg_color=[0.0, 0.0, 0.0],
                                colors_precomp=torch.from_numpy(viz_grad_color).to(photo_grad),
                            )
                            viz_grad = (
                                viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                            )
                            imageio.imsave(
                                osp.join(
                                    self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_photo_grad_step={step}.jpg"
                                ),
                                viz_grad,
                            )

                            corr_grad = torch.cat(corr_grad, 0)
                            viz_grad_color = (
                                torch.clamp(corr_grad, 0.0, dyn_node_densify_grad_th)
                                / dyn_node_densify_grad_th
                            )
                            viz_grad_color = viz_grad_color.detach().cpu().numpy()
                            viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                            viz_render_dict = render(
                                [_s_model(), _d_model(view_ind)],
                                _s2d.H,
                                _s2d.W,
                                _cams.K(_s2d.H, _s2d.W),
                                _cams.T_cw(view_ind),
                                bg_color=[0.0, 0.0, 0.0],
                                colors_precomp=torch.from_numpy(viz_grad_color).to(corr_grad),
                            )
                            viz_grad = (
                                viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                            )
                            imageio.imsave(
                                osp.join(
                                    self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_corr_grad_step={step}.jpg"
                                ),
                                viz_grad,
                            )
                        # NOTE: matplotlib is not safe for threading.
                    except:
                        logging.info(f"Failing to visualizing of viz_cheap_interval")
            ########################### save static, camera and dynamic model ###########################
            s_save_fn = osp.join(
                self.log_dir, node_dir_name, f"{phase_name}_s_model_{GS_BACKEND.lower()}.pth"
            )
            torch.save(_s_model.state_dict(), s_save_fn)
            torch.save(_cams.state_dict(), osp.join(self.log_dir, node_dir_name, f"{phase_name}_cam.pth"))

            if _d_model is not None:
                d_save_fn = osp.join(
                    self.log_dir, node_dir_name, f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth"
                )
                torch.save(_d_model.state_dict(), d_save_fn)
            
            if len(fachain_d_model_list) > 0:
                for fa_d_model_infodict in fachain_d_model_list:
                    os.makedirs(os.path.join(self.log_dir, node_dir_name, fa_d_model_infodict["dyn_node"].node_dir_name()), exist_ok=True)
                    fa_d_save_fn = os.path.join(self.log_dir, node_dir_name, fa_d_model_infodict["dyn_node"].node_dir_name(), f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth")
                    torch.save(fa_d_model_infodict["d_model"].state_dict(), fa_d_save_fn)
            ########################################## end save ##########################################

            ########################################## start viz ##########################################
            try:
                viz2d_total_video_interval_fachain(
                    viz_vid_fn=osp.join(self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_2dviz.mp4"),
                    s2d=_s2d,
                    start_from=0,
                    end_at=_cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=_cams,
                    s_model=_s_model,
                    d_model=_d_model,
                    move_around_angle_deg=viz_move_angle_deg,
                    print_text=False,
                    left_bound=left_bound, 
                    fachain_d_model_list=fachain_d_model_list, 
                )
            except:
                logging.info(f"Failing to visualize the photometric_2dviz.mp4, probably due to OOM Error.")
            if d_flag:
                try:
                    viz_path = osp.join(self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_3Dviz.mp4")
                    viz3d_total_video(
                        _cams,
                        _d_model,
                        0,
                        _cams.T - 1,
                        save_path=viz_path,
                        res=480,
                        s_model=_s_model,
                    )
                except:
                    logging.info(f"Failing to visualize the photometric_3Dviz_0.mp4, probably due to OOM Error.")
                if lambda_mask > 0.0:
                    # * viz grouping
                    _d_model.return_cate_colors_flag = True
                    _s_model.return_cate_colors_flag = True
                    try:
                        viz_path = osp.join(self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_3Dviz_group.mp4")
                        viz3d_total_video(
                            _cams,
                            _d_model,
                            0,
                            _cams.T - 1,
                            save_path=viz_path,
                            res=480,
                            s_model=_s_model,
                        )
                    except Exception as e:
                        logging.info(f"Failing to visualize the photometric_3Dviz_group.mp4, probably due to OOM Error.")
                    try:
                        viz2d_total_video_interval_fachain(
                            viz_vid_fn=osp.join(self.log_dir, node_dir_name, "mosca_photo_viz", f"{phase_name}_2dviz_group.mp4"),
                            s2d=_s2d,
                            start_from=0,
                            end_at=_cams.T - 1,
                            skip_t=viz_skip_t,
                            cams=_cams,
                            s_model=_s_model,
                            d_model=_d_model,
                            move_around_angle_deg=viz_move_angle_deg,
                            print_text=False,
                            left_bound=left_bound, 
                            fachain_d_model_list=fachain_d_model_list, 
                        )
                    except Exception as e:
                        logging.info(f"Failing to visualize the photometric_2dviz_group.mp4, probably due to OOM Error.")
                    _d_model.return_cate_colors_flag = False
                    _s_model.return_cate_colors_flag = False
            ########################################## end viz ##########################################

            ######################################## start split judge ##########################################
            

            split_ind = -1
            split_grads = _d_model.xyz_gradient_accum / _d_model.xyz_gradient_denom
            split_grads[split_grads.isnan()] = 0.0

            split_grads_cl = split_grads.detach().cpu().clone()
            torch.save(split_grads_cl, os.path.join(self.log_dir, node_dir_name, "split_grads.pt"))

            split_grads_sum = split_grads.sum()
            min_delta_value = 4e8
            min_delta_tid = -1
            split_grads_accum = 0.
            custom_split = getattr(fit_cfg, "custom_split", "gradsplit")
            for tid in range(_cams.T):
                if not mid_split and custom_split == "gradsplit":
                    _d_model_ref_time_max = _d_model.ref_time.max()
                    _d_model_ref_time_min = _d_model.ref_time.min()
                tid_mask = (_d_model.ref_time == tid)
                tid_grads = split_grads[tid_mask]
                split_grads_accum += tid_grads.sum()
                if torch.abs(split_grads_accum - 0.5 * split_grads_sum) < min_delta_value:
                    min_delta_value = torch.abs(split_grads_accum - 0.5 * split_grads_sum)
                    min_delta_tid = tid

            assert min_delta_tid != -1
            if mid_split:
                split_ind = (left_bound + right_bound) // 2
            else:
                print(f"custom_split: {custom_split}")
                if custom_split == "gradsplit":
                    if min_delta_tid == _cams.T - 1:
                        split_ind = left_bound + min_delta_tid
                    else:
                        split_ind = left_bound + min_delta_tid + 1
                    print("GradSplit split_ind:", split_ind)
                elif custom_split == "flowsplit":
                    flow_split_infos = getattr(fit_cfg, "flow_split_infos", [])
                    assert len(flow_split_infos) > 0
                    if son_name == "left":
                        father_node_idx = flow_split_infos.index(fa_split_ind)
                        split_ind = flow_split_infos[2 * father_node_idx]
                    elif son_name == "right":
                        father_node_idx = flow_split_infos.index(fa_split_ind)
                        split_ind = flow_split_infos[2 * father_node_idx + 1]
                    elif son_name == "":
                        split_ind = flow_split_infos[1]
                    else:
                        print("No support for son_name:", son_name)
                        exit(-1)
                    print("FlowSplit left-right:", left_bound, right_bound)
                    print("FlowSplit flow_split_infos:", flow_split_infos)
                    print("FlowSplit split_ind:", split_ind)
                else:
                    print("No support for custom_split:", custom_split)
                    exit(-1)
            assert left_bound < split_ind <= right_bound

            release_model_memory(_s2d)
            _s2d.cpu()
            del _s2d
            release_model_memory(_cams, optimizer_cam)
            _cams.cpu()
            del _cams
            release_model_memory(_s_model, optimizer_static)
            _s_model.cpu()
            del _s_model
            release_model_memory(_d_model, optimizer_dynamic)
            _d_model.cpu()
            del _d_model
            if len(fachain_d_model_list) > 0:
                for fa_d_model_infodict in fachain_d_model_list:
                    if "optimizer_dynamic" in fa_d_model_infodict.keys():
                        release_model_memory(fa_d_model_infodict["d_model"], fa_d_model_infodict["optimizer_dynamic"])
                    else:
                        release_model_memory(fa_d_model_infodict["d_model"], None)
                    fa_d_model_infodict["d_model"].cpu()
                    del fa_d_model_infodict["d_model"]
                del fachain_d_model_list

            if getattr(fit_cfg, "share_s_model", False):
                # 清理注册的模型
                with self.sync_lock:
                    if gpu_id in self.shared_s_models:
                        del self.shared_s_models[gpu_id]

            torch.cuda.empty_cache()

            with open(osp.join(self.log_dir, node_dir_name, "split_ind.txt"), "w") as f:
                f.write(str(split_ind))
                f.close()

            ######################################## end split judge ##########################################

            return split_ind

        logging.info(f"Finetune with GS-BACKEND={GS_BACKEND.lower()}")

        def process_node_fachain(node_idx, bfs_list, tree_height, stack_depth_max, cams, s_model, d_model, log_dir, gpu_id):
            tree_node = bfs_list[node_idx]
            if tree_node.left_bound == -1 and tree_node.right_bound == -1 and tree_node.depth == -1:
                return None  # node doesn't exist
            
            # Set the GPU for this node
            torch.cuda.set_device(gpu_id)
            
            fa_tree_node = bfs_list[node_idx//2]
            son_name = "left" if node_idx % 2 == 0 else "right"
            
            if tree_height == 0:
                split_ind = optimizing_bfs_interval_fachain(
                    treenode=tree_node, 
                    left_bound=tree_node.left_bound, 
                    right_bound=tree_node.right_bound, 
                    treenode_depth=tree_node.depth, 
                    fa_dir="", 
                    fa_split_ind=-1, 
                    fa_left_bound=-1, 
                    son_name="", 
                    root_cams=cams, 
                    root_s_model=s_model, 
                    root_d_model=d_model,
                    gpu_id=gpu_id,
                )
            else:
                '''
                fachain_d_model_list: list
                    element: dict, keys: "d_model_dir", "left_bound", "right_bound"
                    if len(fachain_d_model_list) == 0 -> means do not use fachain_d_model
                '''
                fachain_d_model_list = []
                if use_fachain:
                    fa_node_idx = node_idx // 2
                    while fa_node_idx != 0: # NOTE: 1 at bfs_list is root
                        direct_fa_node = bfs_list[node_idx // 2]
                        founded_fa_node = bfs_list[fa_node_idx]
                        assert founded_fa_node.left_bound != -1 and founded_fa_node.right_bound != -1 and founded_fa_node.depth != -1
                        if fa_node_idx == node_idx // 2: # direct father
                            d_model_dir = os.path.join(log_dir, direct_fa_node.node_dir_name(), f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth")
                        else:
                            d_model_dir = os.path.join(log_dir, direct_fa_node.node_dir_name(), founded_fa_node.node_dir_name(), f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth")
                        extend_length = getattr(fit_cfg, "extend_length", 0)
                        do_fachain_optim = getattr(fit_cfg, "fachain_optim", False)
                        do_fachain_ref_time_mask = getattr(fit_cfg, "fachain_ref_time_mask", False)
                        founded_fa_d_model_infodict = {
                            "d_model_dir": d_model_dir, 
                            "dyn_node": founded_fa_node, 
                            "left_bound": founded_fa_node.left_bound, 
                            "right_bound": founded_fa_node.right_bound, 
                            "leaf_left_bound": tree_node.left_bound, 
                            "leaf_right_bound": tree_node.right_bound, 
                            "extend_length": extend_length, 
                            "do_fachain_optim": do_fachain_optim, 
                            "do_fachain_ref_time_mask": do_fachain_ref_time_mask, 
                        }
                        fachain_d_model_list.append(founded_fa_d_model_infodict)
                        fa_node_idx = fa_node_idx // 2
                split_ind = optimizing_bfs_interval_fachain(
                    treenode=tree_node, 
                    left_bound=tree_node.left_bound, 
                    right_bound=tree_node.right_bound, 
                    treenode_depth=tree_node.depth, 
                    fa_dir=osp.join(log_dir, fa_tree_node.node_dir_name()), 
                    fa_split_ind=fa_tree_node.split_ind, 
                    fa_left_bound=fa_tree_node.left_bound, 
                    son_name=son_name,
                    gpu_id=gpu_id,
                    fachain_d_model_list=fachain_d_model_list, 
                )
            
            if split_ind == -1 or tree_node.depth == stack_depth_max:
                return None
            
            tree_node.split_ind = split_ind
            tree_node.insert_left(left_bound=tree_node.left_bound, right_bound=split_ind-1, min_interval=min_interval)
            tree_node.insert_right(left_bound=split_ind, right_bound=tree_node.right_bound, min_interval=min_interval)
            
            if tree_node.left_child is not None:
                left_child_node = DynamicSegTreeBFS(left_bound=tree_node.left_bound, right_bound=split_ind-1, depth=tree_node.depth+1)
            else:
                left_child_node = DynamicSegTreeBFS(left_bound=-1, right_bound=-1, depth=-1)
            if tree_node.right_child is not None:
                right_child_node = DynamicSegTreeBFS(left_bound=split_ind, right_bound=tree_node.right_bound, depth=tree_node.depth+1)
            else:
                right_child_node = DynamicSegTreeBFS(left_bound=-1, right_bound=-1, depth=-1)
            return (
                node_idx * 2, 
                left_child_node,
                node_idx * 2 + 1,
                right_child_node, 
            )

        def parallel_bfs():
            tree_root = DynamicSegTreeBFS(
                left_bound=0, 
                right_bound=cams.T - 1, 
                depth=0, 
            )
            tree_root_idx = 1
            bfs_list = [DynamicSegTreeBFS(left_bound=-1, right_bound=-1, depth=-1) for idx in range(4 * 2**(stack_depth_max+1))]
            bfs_list[tree_root_idx] = tree_root
            
            real_gpu_list = get_available_gpus()  # Available GPUs
            if not real_gpu_list:
                raise RuntimeError("No available GPUs found in CUDA_VISIBLE_DEVICES")
            else:
                gpu_list = [gpu_idx for gpu_idx in range(len(real_gpu_list))]

            for tree_height in range(stack_depth_max+1):
                if tree_height == 0:
                    # Process root node
                    result = process_node_fachain(tree_root_idx, bfs_list, tree_height, stack_depth_max, cams, s_model, d_model, self.log_dir, gpu_list[0])
                    if result:
                        left_idx, left_node, right_idx, right_node = result
                        bfs_list[left_idx] = left_node
                        bfs_list[right_idx] = right_node
                else:
                    
                    del self.sync_event
                    del self.sync_counter
                    del self.sync_lock
                    del self.shared_s_models
                    del self.param_buffer
                    del self.gpu_streams
                    del self.gpu_events
                    del self.grad_sync_event
                    del self.grad_sync_counter
                    del self.final_sync_event
                    del self.final_sync_counter
                    torch.cuda.empty_cache()
                    self.sync_event = Event()  # 全局同步事件
                    self.sync_counter = 0
                    self.sync_lock = Lock()  # 保护计数器
                    self.shared_s_models = {}  # 存储各GPU上的模型引用
                    self.param_buffer = {}  # 用于存储梯度平均的缓冲区
                    self.gpu_streams = {}  # 各GPU的通信流
                    self.gpu_events = {}   # 各GPU的同步事件
                    self.grad_sync_event = Event()
                    self.grad_sync_counter = 0
                    self.final_sync_event = Event()
                    self.final_sync_counter = 0

                    layer_list = [layer_idx for layer_idx in range(2**tree_height, 2**(tree_height+1))]
                    num_nodes = len(layer_list)
                    
                    start_depth_for_skip_gpu = getattr(fit_cfg, "start_depth_for_skip_gpu", 1000)
                    if tree_height >= start_depth_for_skip_gpu:
                        all_gpu_list = get_available_gpus()
                        all_gpu_list = [each_gpu_idx for each_gpu_idx in range(len(all_gpu_list))]
                        skip_gpu = getattr(fit_cfg, "skip_gpu", -1)
                        if skip_gpu != -1:
                            gpu_list = []
                            for this_gpu_id in all_gpu_list:
                                if this_gpu_id != skip_gpu:
                                    gpu_list.append(this_gpu_id)
                        else:
                            gpu_list = all_gpu_list

                    # Split nodes into batches that can be processed with available GPUs
                    batch_start = 0
                    parallel_nodes_num = len(gpu_list)
                    while batch_start < num_nodes:
                        batch_nodes = []
                        new_batch_start = batch_start
                        for node_index in range(batch_start, num_nodes):
                            node_need_judged = bfs_list[layer_list[node_index]]
                            if node_need_judged.left_bound != -1 and node_need_judged.right_bound != -1 and node_need_judged.depth != -1:
                                if len(batch_nodes) < parallel_nodes_num:
                                    batch_nodes.append(layer_list[node_index])
                                    new_batch_start = node_index
                            if len(batch_nodes) == parallel_nodes_num: break
                        self.batch_nodes_len = len(batch_nodes)
                        if len(batch_nodes) == 0: break # NOTE: no more nodes for processing -> next height layer.
                        try:
                            with ThreadPoolExecutor(max_workers=len(batch_nodes)) as executor:
                                futures = []
                                for i, node_idx in enumerate(batch_nodes):
                                    gpu_id = gpu_list[i % len(gpu_list)]
                                    futures.append(
                                        executor.submit(
                                            process_node_fachain, 
                                            node_idx, bfs_list, tree_height, stack_depth_max, 
                                            cams, s_model, d_model, self.log_dir, gpu_id
                                        )
                                    )
                                
                                for future in futures:
                                    result = future.result()
                                    if result:
                                        left_idx, left_node, right_idx, right_node = result
                                        bfs_list[left_idx] = left_node
                                        bfs_list[right_idx] = right_node
                        except Exception as e:
                            cleanup()
                            raise e
                        batch_start = new_batch_start + 1

                print("one layer done.")
            return tree_root, bfs_list
        
        def cleanup():
            print("\nCleaning up GPU memory...")
            torch.cuda.empty_cache()

        def signal_handler(sig, frame):
            print("\nReceived Ctrl+C, shutting down gracefully...")
            cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)

        tree_root, bfs_list = parallel_bfs()
        
        return tree_root, bfs_list

    @torch.no_grad()
    def render_all(self, cams: MonocularCameras, s_model=None, d_model=None):
        ret = []
        assert s_model is not None or d_model is not None, "No model to render"
        s_gs5 = s_model()
        for t in tqdm(range(cams.T)):
            gs5 = [s_gs5]
            if d_model is not None:
                gs5.append(d_model(t))
            render_dict = render(
                gs5,
                cams.default_H,
                cams.default_W,
                cams.K(),
                cams.T_cw(t),
                bg_color=[1.0, 1.0, 1.0],
            )
            ret.append(render_dict)
        rgb = torch.stack([r["rgb"] for r in ret], 0)
        dep = torch.stack([r["dep"] for r in ret], 0)
        alp = torch.stack([r["alpha"] for r in ret], 0)
        return rgb, dep, alp
