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
from photo_recon_utils import fetch_leaves_in_world_frame, estimate_normal_map
from lib_render.render_helper import GS_BACKEND, render, RGB2SH
from lib_prior.prior_loading import Saved2D
from gs_utils.gs_optim_helper import update_learning_rate
from gs_utils.loss_helper import (
    build_lpips_model,
    compute_rgb_loss,
    compute_lpips_loss,
    compute_dep_loss,
    compute_normal_loss,
    compute_normal_reg_loss,
    compute_dep_reg_loss,
)
from dynamic_solver import get_dynamic_curves, get_multiview_dynamic_curves

from photo_recon_utils import (
    fetch_leaves_in_world_frame,
    _count_seed_mask_points,
    _append_seed_pool_from_camera_source,
    _finalize_seed_pool,
    GSControlCFG,
    OptimCFG,
    apply_gs_control,
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
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        viz_fname=None,
        multiview_obs=None,
        multiview_cams=None,
        min_dynamic_valid_cnt=1,
    ):
        # get global anchor
        curve_xyz, curve_mask, _, _ = get_dynamic_curves(
            s2d, cams, return_all_curves=True
        )
        curve_dynamic_mask = s2d.dynamic_track_mask.clone()

        # only consider the valid case
        static_curve_mean = (
            curve_xyz[:, ~curve_dynamic_mask]
            * curve_mask[:, ~curve_dynamic_mask, None]
        ).sum(0, keepdim=True) / curve_mask[:, ~curve_dynamic_mask, None].sum(
            0, keepdim=True
        ).expand(
            len(curve_xyz), -1, -1
        )
        curve_xyz[:, ~curve_dynamic_mask] = static_curve_mean
        np.savetxt(
            osp.join(self.viz_dir, "fg_id_non_dyn_curve_meaned.xyz"),
            static_curve_mean.reshape(-1, 3).cpu().numpy(),
            fmt="%.4f",
        )
        if multiview_obs is not None and multiview_cams is not None:
            aux_curve_xyz, _, _, aux_curve_counts, _ = get_multiview_dynamic_curves(
                multiview_obs,
                multiview_cams,
                t_list=torch.arange(s2d.T, device=s2d.rgb.device),
                min_dynamic_valid_cnt=min_dynamic_valid_cnt,
                refilter_2d_track_flag=True,
                refilter_min_valid_cnt=max(2, min_dynamic_valid_cnt),
            )
            if aux_curve_xyz is not None:
                logging.info(
                    f"Append multiview dynamic curves for FG identification from {aux_curve_counts}"
                )
                curve_xyz = torch.cat([curve_xyz, aux_curve_xyz], dim=1)
                curve_dynamic_mask = torch.cat(
                    [
                        curve_dynamic_mask,
                        torch.ones(aux_curve_xyz.shape[1], device=curve_dynamic_mask.device)
                        .bool(),
                    ],
                    dim=0,
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
                fg_mask = curve_dynamic_mask[knn_id].reshape(s2d.H, s2d.W)
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
        multiview_obs=None,
        multiview_cams=None,
    ):
        device = self.device

        if mask_type not in ["static_depth", "depth"]:
            raise ValueError(f"Unknown mask_type={mask_type}")

        if multiview_obs is not None and multiview_cams is not None:
            t_list = list(range(s2d.T))
            seed_pool = ([], [], [], [], [])
            anchor_camera_name = multiview_obs.anchor_camera
            anchor_rgb_list = (
                [multiview_obs.get_rgb(anchor_camera_name, t) for t in t_list]
                if multiview_obs.override_anchor_rgb
                else [s2d.rgb[t] for t in t_list]
            )
            if multiview_obs.has_depth(anchor_camera_name):
                anchor_dep_list = [
                    multiview_obs.get_depth(anchor_camera_name, t) for t in t_list
                ]
                anchor_dep_mask_list = [
                    multiview_obs.get_depth_mask(anchor_camera_name, t) for t in t_list
                ]
            else:
                logging.warning(
                    "No multiview anchor depth for %s; fall back to Saved2D anchor depth for static GS init",
                    anchor_camera_name,
                )
                anchor_dep_list = [s2d.dep[t] for t in t_list]
                anchor_dep_mask_list = [s2d.dep_mask[t] for t in t_list]

            anchor_mask_list = []
            for t in t_list:
                mask = anchor_dep_mask_list[t].bool()
                if mask_type == "static_depth":
                    mask = mask & s2d.sta_mask[t].bool()
                anchor_mask_list.append(mask)

            source_specs = [
                dict(
                    camera_name=None,
                    cams_obj=cams,
                    mask_list=anchor_mask_list,
                    dep_list=anchor_dep_list,
                    rgb_list=anchor_rgb_list,
                )
            ]

            aux_camera_cnt = 0
            for camera_name in multiview_obs.auxiliary_cameras:
                if not multiview_obs.has_depth(camera_name):
                    continue
                aux_depth_list = [multiview_obs.get_depth(camera_name, t) for t in t_list]
                aux_rgb_list = [multiview_obs.get_rgb(camera_name, t) for t in t_list]
                aux_mask_list = []
                use_dynamic_mask = multiview_obs.has_explicit_mask(camera_name)
                if mask_type == "static_depth" and not use_dynamic_mask:
                    logging.warning(
                        "No explicit dynamic mask for %s; use depth mask only for static GS init",
                        camera_name,
                    )
                for t in t_list:
                    aux_mask = multiview_obs.get_depth_mask(camera_name, t).bool()
                    if mask_type == "static_depth" and use_dynamic_mask:
                        aux_mask = aux_mask & (~multiview_obs.get_mask(camera_name, t).bool())
                    aux_mask_list.append(aux_mask)
                source_specs.append(
                    dict(
                        camera_name=camera_name,
                        cams_obj=multiview_cams,
                        mask_list=aux_mask_list,
                        dep_list=aux_depth_list,
                        rgb_list=aux_rgb_list,
                    )
                )
                aux_camera_cnt += 1

            reference_count = _count_seed_mask_points(anchor_mask_list, t_list)
            fused_count = sum(
                _count_seed_mask_points(spec["mask_list"], t_list) for spec in source_specs
            )
            uniform_keep_ratio = 1.0
            if fused_count > 0 and reference_count > 0:
                uniform_keep_ratio = min(1.0, reference_count / float(fused_count))

            logging.info(
                "Initialize static GS from fused multiview depth using anchor + %d auxiliary cameras "
                "(reference_count=%d, fused_count=%d, keep_ratio=%.4f)",
                aux_camera_cnt,
                reference_count,
                fused_count,
                uniform_keep_ratio,
            )
            for spec in source_specs:
                _append_seed_pool_from_camera_source(
                    seed_pool,
                    spec["cams_obj"],
                    spec["mask_list"],
                    spec["dep_list"],
                    spec["rgb_list"],
                    t_list=t_list,
                    subsample=image_stride,
                    camera_name=spec["camera_name"],
                    uniform_keep_ratio=uniform_keep_ratio,
                )
            mu_init, q_init, s_init, o_init, rgb_init, id_init, _ = _finalize_seed_pool(
                seed_pool, n_init, device, save_xyz_fn=viz_fn
            )
        else:
            if mask_type == "static_depth":
                gather_mask = s2d.sta_mask * s2d.dep_mask
            else:
                gather_mask = s2d.dep_mask

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
        temporal_opacity_flag=False,
        temporal_opacity_init_scale=8.0,
        max_node_num=100000,
        multiview_obs=None,
        multiview_cams=None,
    ):
        device = self.device
        collect_t_list = torch.arange(0, s2d.T, 1)
        logging.info(f"Collect GS at t={collect_t_list}")
        if multiview_obs is not None and multiview_cams is not None:
            t_list = collect_t_list.tolist()
            seed_pool = ([], [], [], [], [])
            anchor_camera_name = multiview_obs.anchor_camera
            anchor_rgb_list = (
                [multiview_obs.get_rgb(anchor_camera_name, t) for t in t_list]
                if multiview_obs.override_anchor_rgb
                else [s2d.rgb[t] for t in t_list]
            )
            if multiview_obs.has_depth(anchor_camera_name):
                anchor_dep_list = [
                    multiview_obs.get_depth(anchor_camera_name, t) for t in t_list
                ]
                anchor_dep_mask_list = [
                    multiview_obs.get_depth_mask(anchor_camera_name, t) for t in t_list
                ]
            else:
                logging.warning(
                    "No multiview anchor depth for %s; fall back to Saved2D anchor depth for dynamic GS init",
                    anchor_camera_name,
                )
                anchor_dep_list = [s2d.dep[t] for t in t_list]
                anchor_dep_mask_list = [s2d.dep_mask[t] for t in t_list]

            anchor_mask_list = []
            for t in t_list:
                mask = s2d.dyn_mask[t].bool() & anchor_dep_mask_list[t].bool()
                if additional_mask is not None:
                    mask = mask & additional_mask[t].bool()
                anchor_mask_list.append(mask)
            source_specs = [
                dict(
                    camera_name=None,
                    cams_obj=cams,
                    mask_list=anchor_mask_list,
                    dep_list=anchor_dep_list,
                    rgb_list=anchor_rgb_list,
                )
            ]

            aux_camera_cnt = 0
            for camera_name in multiview_obs.auxiliary_cameras:
                if not multiview_obs.has_depth(camera_name):
                    continue
                if not multiview_obs.has_explicit_mask(camera_name):
                    logging.warning(
                        "Skip multiview dynamic GS init for %s because no explicit dynamic mask is available",
                        camera_name,
                    )
                    continue
                aux_depth_list = [multiview_obs.get_depth(camera_name, t) for t in t_list]
                aux_rgb_list = [multiview_obs.get_rgb(camera_name, t) for t in t_list]
                aux_mask_list = [
                    multiview_obs.get_depth_mask(camera_name, t).bool()
                    & multiview_obs.get_mask(camera_name, t).bool()
                    for t in t_list
                ]
                source_specs.append(
                    dict(
                        camera_name=camera_name,
                        cams_obj=multiview_cams,
                        mask_list=aux_mask_list,
                        dep_list=aux_depth_list,
                        rgb_list=aux_rgb_list,
                    )
                )
                aux_camera_cnt += 1

            reference_count = _count_seed_mask_points(anchor_mask_list, t_list)
            fused_count = sum(
                _count_seed_mask_points(spec["mask_list"], t_list) for spec in source_specs
            )
            uniform_keep_ratio = 1.0
            if fused_count > 0 and reference_count > 0:
                uniform_keep_ratio = min(1.0, reference_count / float(fused_count))

            logging.info(
                "Initialize dynamic GS from fused multiview depth using anchor + %d auxiliary cameras "
                "(reference_count=%d, fused_count=%d, keep_ratio=%.4f)",
                aux_camera_cnt,
                reference_count,
                fused_count,
                uniform_keep_ratio,
            )
            for spec in source_specs:
                _append_seed_pool_from_camera_source(
                    seed_pool,
                    spec["cams_obj"],
                    spec["mask_list"],
                    spec["dep_list"],
                    spec["rgb_list"],
                    t_list=t_list,
                    subsample=image_stride,
                    camera_name=spec["camera_name"],
                    uniform_keep_ratio=uniform_keep_ratio,
                )
            mu_init, q_init, s_init, o_init, rgb_init, id_init, time_init = (
                _finalize_seed_pool(seed_pool, n_init, device)
            )
        else:
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
            temporal_opacity_flag=temporal_opacity_flag,
            temporal_opacity_init_scale=temporal_opacity_init_scale,
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
        rgb_ssim_lambda=0.1,
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
        # * optional synchronized multiview supervision
        multiview_use_rgb=True,
        multiview_use_depth=False,
        multiview_lambda_dep=0.0,
        multiview_depth_camera_weights=None,
        multiview_use_track=False,
        multiview_track_camera_weights=None,
        multiview_dep_st_invariant=None,
        multiview_obs=None,
        multiview_cams=None,
        multiview_use_masks=False,
        multiview_lpips_camera_weights=None,
        multiview_rgb_ssim_lambda=None,
        multiview_log_interval=100,
        lpips_net="alex",
    ):
        logging.info(f"Finetune with GS-BACKEND={GS_BACKEND.lower()}")

        torch.cuda.empty_cache()
        n_frame = 1

        d_flag = d_model is not None
        multiview_flag = multiview_obs is not None and multiview_cams is not None
        multiview_rgb_flag = multiview_flag and multiview_use_rgb
        multiview_depth_flag = (
            multiview_flag and multiview_use_depth and multiview_lambda_dep > 0.0
        )
        multiview_track_flag = (
            multiview_flag and multiview_use_track and lambda_track > 0.0
        )
        lpips_flag = lambda_lpips > 0.0
        if multiview_dep_st_invariant is None:
            multiview_dep_st_invariant = dep_st_invariant
        if multiview_rgb_ssim_lambda is None:
            multiview_rgb_ssim_lambda = rgb_ssim_lambda
        if multiview_lpips_camera_weights is None:
            multiview_lpips_camera_weights = {}
        elif not isinstance(multiview_lpips_camera_weights, dict):
            multiview_lpips_camera_weights = OmegaConf.to_container(
                multiview_lpips_camera_weights, resolve=True
            )
        multiview_lpips_camera_weights = dict(multiview_lpips_camera_weights)
        if multiview_depth_camera_weights is None:
            multiview_depth_camera_weights = {}
        elif not isinstance(multiview_depth_camera_weights, dict):
            multiview_depth_camera_weights = OmegaConf.to_container(
                multiview_depth_camera_weights, resolve=True
            )
        multiview_depth_camera_weights = dict(multiview_depth_camera_weights)
        if multiview_track_camera_weights is None:
            multiview_track_camera_weights = {}
        elif not isinstance(multiview_track_camera_weights, dict):
            multiview_track_camera_weights = OmegaConf.to_container(
                multiview_track_camera_weights, resolve=True
            )
        multiview_track_camera_weights = dict(multiview_track_camera_weights)

        corr_flag = lambda_track > 0.0 and d_flag
        if corr_flag:
            logging.info(
                f"Enabel Flow/Track backing with supervision interval={track_loss_interval}"
            )
        if multiview_rgb_flag:
            logging.info(
                "Enable synchronized multiview RGB supervision with "
                f"{multiview_obs.camera_names}"
            )
        if multiview_depth_flag:
            logging.info(
                "Enable synchronized multiview depth supervision with "
                f"{multiview_obs.auxiliary_cameras}, lambda={multiview_lambda_dep}, "
                f"camera_weights={multiview_depth_camera_weights}"
            )
        if multiview_track_flag:
            logging.info(
                "Enable synchronized multiview track supervision with "
                f"{multiview_obs.track_cameras}, camera_weights="
                f"{multiview_track_camera_weights}"
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

        def get_lpips_camera_weight(camera_name):
            if camera_name in multiview_lpips_camera_weights:
                return float(multiview_lpips_camera_weights[camera_name])
            if multiview_rgb_flag:
                return float(multiview_obs.get_weight(camera_name).item())
            return 1.0

        def get_multiview_depth_camera_weight(camera_name):
            if camera_name in multiview_depth_camera_weights:
                return float(multiview_depth_camera_weights[camera_name])
            return float(multiview_obs.get_weight(camera_name).item())

        def get_multiview_track_camera_weight(camera_name):
            if camera_name in multiview_track_camera_weights:
                return float(multiview_track_camera_weights[camera_name])
            return float(multiview_obs.get_weight(camera_name).item())

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

        loss_rgb_list, loss_dep_list, loss_nrm_list = [], [], []
        loss_multiview_dep_list = []
        loss_lpips_list = []
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
        multiview_loss_dict = (
            {camera_name: [] for camera_name in multiview_obs.camera_names}
            if multiview_rgb_flag
            else None
        )
        multiview_lpips_dict = (
            {camera_name: [] for camera_name in multiview_obs.camera_names}
            if multiview_rgb_flag and lpips_flag
            else None
        )
        multiview_depth_dict = (
            {camera_name: [] for camera_name in multiview_obs.auxiliary_cameras}
            if multiview_depth_flag
            else None
        )
        multiview_track_dict = (
            {camera_name: [] for camera_name in multiview_obs.auxiliary_cameras}
            if multiview_track_flag
            else None
        )

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
            step_camera_rgb_losses = {} if multiview_rgb_flag else None
            step_camera_lpips_losses = (
                {} if multiview_rgb_flag and lpips_flag else None
            )
            step_camera_depth_losses = {} if multiview_depth_flag else None
            step_camera_track_losses = {} if multiview_track_flag else None
            loss_rgb, loss_dep, loss_nrm = 0.0, 0.0, 0.0
            loss_multiview_dep = torch.zeros((), device=s2d.rgb.device)
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
                anchor_rgb_target = s2d.rgb[view_ind].detach().clone()
                anchor_camera_name = "anchor"
                anchor_rgb_weight = 1.0
                if multiview_rgb_flag:
                    anchor_camera_name = multiview_obs.anchor_camera
                    anchor_rgb_weight = multiview_obs.get_weight(
                        multiview_obs.anchor_camera
                    ).item()
                    if multiview_obs.override_anchor_rgb:
                        anchor_rgb_target = multiview_obs.get_rgb(
                            multiview_obs.anchor_rgb_camera, view_ind
                        ).detach().clone()
                        if multiview_use_masks:
                            rgb_sup_mask = rgb_sup_mask * multiview_obs.get_mask(
                                multiview_obs.anchor_rgb_camera,
                                view_ind,
                                use_mask=True,
                            )
                _l_rgb, _, _, _ = compute_rgb_loss(
                    anchor_rgb_target,
                    render_dict,
                    rgb_sup_mask,
                    ssim_lambda=rgb_ssim_lambda,
                )
                if lpips_flag:
                    anchor_lpips_weight = get_lpips_camera_weight(anchor_camera_name)
                    _l_lpips, _, _, _ = compute_lpips_loss(
                        anchor_rgb_target,
                        render_dict,
                        rgb_sup_mask,
                        lpips_model,
                    )
                    loss_lpips = loss_lpips + _l_lpips * anchor_lpips_weight
                    if multiview_rgb_flag:
                        step_camera_lpips_losses[anchor_camera_name] = (
                            step_camera_lpips_losses.get(anchor_camera_name, 0.0)
                            + _l_lpips.detach().item() * anchor_lpips_weight
                        )
                dep_sup_mask = rgb_sup_mask * s2d.dep_mask[view_ind]
                _l_dep, _, _, _ = compute_dep_loss(
                    s2d.dep[view_ind].detach().clone(),
                    render_dict,
                    dep_sup_mask,
                    st_invariant=dep_st_invariant,
                )
                loss_rgb = loss_rgb + _l_rgb * anchor_rgb_weight
                loss_dep = loss_dep + _l_dep
                if multiview_rgb_flag:
                    step_camera_rgb_losses[anchor_camera_name] = (
                        step_camera_rgb_losses.get(anchor_camera_name, 0.0)
                        + _l_rgb.detach().item() * anchor_rgb_weight
                    )

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

                if multiview_flag:
                    for camera_name in multiview_obs.auxiliary_cameras:
                        # Sparse aux cameras (e.g. a new-trajectory clip) only cover
                        # a subset of the anchor timeline; skip times they don't see.
                        if not multiview_obs.has_view(camera_name, view_ind):
                            continue
                        aux_H, aux_W = multiview_obs.get_hw(camera_name)
                        aux_render_dict = render(
                            gs5,
                            aux_H,
                            aux_W,
                            multiview_cams.K(camera_name, aux_H, aux_W),
                            multiview_cams.T_cw(camera_name, view_ind),
                            bg_color=bg_color,
                            add_buffer=None,
                        )
                        render_dict_list.append(aux_render_dict)
                        aux_sup_mask = multiview_obs.get_mask(
                            camera_name,
                            view_ind,
                            use_mask=multiview_use_masks,
                        )
                        if multiview_rgb_flag:
                            aux_target = multiview_obs.get_rgb(
                                camera_name, view_ind
                            ).detach().clone()
                            aux_weight = multiview_obs.get_weight(camera_name).item()
                            _l_aux_rgb, _, _, _ = compute_rgb_loss(
                                aux_target,
                                aux_render_dict,
                                aux_sup_mask,
                                ssim_lambda=multiview_rgb_ssim_lambda,
                            )
                            loss_rgb = loss_rgb + _l_aux_rgb * aux_weight
                            step_camera_rgb_losses[camera_name] = (
                                step_camera_rgb_losses.get(camera_name, 0.0)
                                + _l_aux_rgb.detach().item() * aux_weight
                            )
                            if lpips_flag:
                                aux_lpips_weight = get_lpips_camera_weight(camera_name)
                                _l_aux_lpips, _, _, _ = compute_lpips_loss(
                                    aux_target,
                                    aux_render_dict,
                                    aux_sup_mask,
                                    lpips_model,
                                )
                                loss_lpips = (
                                    loss_lpips + _l_aux_lpips * aux_lpips_weight
                                )
                                step_camera_lpips_losses[camera_name] = (
                                    step_camera_lpips_losses.get(camera_name, 0.0)
                                    + _l_aux_lpips.detach().item()
                                    * aux_lpips_weight
                                )
                        if multiview_depth_flag and multiview_obs.has_depth(camera_name):
                            aux_dep_sup_mask = multiview_obs.get_depth_mask(
                                camera_name, view_ind
                            )
                            if multiview_use_masks:
                                aux_dep_sup_mask = aux_dep_sup_mask * aux_sup_mask
                            if aux_dep_sup_mask.sum() > 0:
                                aux_depth_weight = get_multiview_depth_camera_weight(
                                    camera_name
                                )
                                _l_aux_dep, _, _, _ = compute_dep_loss(
                                    multiview_obs.get_depth(camera_name, view_ind)
                                    .detach()
                                    .clone(),
                                    aux_render_dict,
                                    aux_dep_sup_mask,
                                    st_invariant=multiview_dep_st_invariant,
                                )
                                loss_multiview_dep = (
                                    loss_multiview_dep
                                    + _l_aux_dep * aux_depth_weight
                                )
                                step_camera_depth_losses[camera_name] = (
                                    step_camera_depth_losses.get(camera_name, 0.0)
                                    + _l_aux_dep.detach().item() * aux_depth_weight
                                )
                        if (
                            multiview_track_flag
                            and corr_exe_flag
                            and multiview_obs.has_track(camera_name)
                        ):
                            aux_track = multiview_obs.get_track(camera_name)
                            aux_track_mask = multiview_obs.get_track_mask(camera_name)
                            with torch.no_grad():
                                aux_track_valid = (
                                    aux_track_mask[view_ind] & aux_track_mask[dst_ind]
                                )
                                aux_track_src = aux_track[view_ind][aux_track_valid][
                                    ..., :2
                                ]
                                aux_track_dst = aux_track[dst_ind][aux_track_valid][
                                    ..., :2
                                ]
                                aux_src_fetch_index = (
                                    aux_track_src[:, 1].long() * aux_W
                                    + aux_track_src[:, 0].long()
                                )
                            if len(aux_track_src) > 0:
                                aux_dst_xyz_cam = multiview_cams.trans_pts_to_cam(
                                    camera_name, dst_ind, dst_xyz
                                )
                                if GS_BACKEND in ["native_add3"]:
                                    aux_corr_render_dict = render(
                                        gs5,
                                        aux_H,
                                        aux_W,
                                        multiview_cams.K(camera_name, aux_H, aux_W),
                                        multiview_cams.T_cw(camera_name, view_ind),
                                        bg_color=[0.0, 0.0, 0.0],
                                        add_buffer=aux_dst_xyz_cam,
                                    )
                                    aux_rendered_xyz_map = aux_corr_render_dict[
                                        "buf"
                                    ].permute(1, 2, 0)
                                else:
                                    aux_corr_render_dict = render(
                                        gs5,
                                        aux_H,
                                        aux_W,
                                        multiview_cams.K(camera_name, aux_H, aux_W),
                                        multiview_cams.T_cw(camera_name, view_ind),
                                        bg_color=[0.0, 0.0, 0.0],
                                        colors_precomp=aux_dst_xyz_cam,
                                    )
                                    aux_rendered_xyz_map = aux_corr_render_dict[
                                        "rgb"
                                    ].permute(1, 2, 0)
                                corr_render_dict_list.append(aux_corr_render_dict)
                                aux_warped_xyz_cam = aux_rendered_xyz_map.reshape(
                                    -1, 3
                                )[aux_src_fetch_index]
                                aux_track_loss_mask = (
                                    aux_warped_xyz_cam[:, 2] > 1e-4
                                )
                                if aux_track_loss_mask.sum() > 0:
                                    pred_aux_track_dst = multiview_cams.project(
                                        camera_name, aux_warped_xyz_cam
                                    )
                                    aux_L = min(aux_W, aux_H)
                                    pred_aux_track_dst[:, :1] = (
                                        (pred_aux_track_dst[:, :1] + aux_W / aux_L)
                                        / 2.0
                                        * aux_L
                                    )
                                    pred_aux_track_dst[:, 1:] = (
                                        (pred_aux_track_dst[:, 1:] + aux_H / aux_L)
                                        / 2.0
                                        * aux_L
                                    )
                                    _l_aux_track = (
                                        pred_aux_track_dst - aux_track_dst
                                    ).norm(dim=-1)[aux_track_loss_mask]
                                    _l_aux_track = torch.clamp(
                                        _l_aux_track, 0.0, track_loss_clamp
                                    ).mean()
                                    aux_track_weight = get_multiview_track_camera_weight(
                                        camera_name
                                    )
                                    loss_track = (
                                        loss_track + _l_aux_track * aux_track_weight
                                    )
                                    step_camera_track_losses[camera_name] = (
                                        step_camera_track_losses.get(camera_name, 0.0)
                                        + _l_aux_track.detach().item()
                                        * aux_track_weight
                                    )

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
                + loss_multiview_dep * multiview_lambda_dep
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
            loss_multiview_dep_list.append(loss_multiview_dep.item())
            if multiview_rgb_flag:
                for camera_name in multiview_obs.camera_names:
                    multiview_loss_dict[camera_name].append(
                        step_camera_rgb_losses.get(camera_name, 0.0)
                    )
                    if multiview_lpips_dict is not None:
                        multiview_lpips_dict[camera_name].append(
                            step_camera_lpips_losses.get(camera_name, 0.0)
                        )
                if (
                    step == 0
                    or step == total_steps - 1
                    or (
                        multiview_log_interval > 0
                        and step % multiview_log_interval == 0
                    )
                ):
                    logging.info(
                        f"[{phase_name}] step={step} weighted multiview rgb losses="
                        f"{step_camera_rgb_losses}"
                    )
                    if multiview_lpips_dict is not None:
                        logging.info(
                            f"[{phase_name}] step={step} weighted multiview lpips losses="
                            f"{step_camera_lpips_losses}"
                        )
            if multiview_depth_flag:
                for camera_name in multiview_obs.auxiliary_cameras:
                    multiview_depth_dict[camera_name].append(
                        step_camera_depth_losses.get(camera_name, 0.0)
                    )
                if (
                    step == 0
                    or step == total_steps - 1
                    or (
                        multiview_log_interval > 0
                        and step % multiview_log_interval == 0
                    )
                ):
                    logging.info(
                        f"[{phase_name}] step={step} weighted multiview depth losses="
                        f"{step_camera_depth_losses}"
                    )
            if multiview_track_flag:
                for camera_name in multiview_obs.auxiliary_cameras:
                    multiview_track_dict[camera_name].append(
                        step_camera_track_losses.get(camera_name, 0.0)
                    )
                if (
                    step == 0
                    or step == total_steps - 1
                    or (
                        multiview_log_interval > 0
                        and step % multiview_log_interval == 0
                    )
                ):
                    logging.info(
                        f"[{phase_name}] step={step} weighted multiview track losses="
                        f"{step_camera_track_losses}"
                    )

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
                        ("loss_mv_dep", loss_multiview_dep_list),
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
                ("loss_mv_dep", loss_multiview_dep_list),
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
            viz3d_total_video(
                cams,
                d_model,
                0,
                cams.T - 1,
                save_path=viz_path,
                res=480,
                s_model=s_model,
            )
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
        if multiview_rgb_flag:
            np.savez_compressed(
                osp.join(self.log_dir, f"{phase_name}_multiview_rgb_loss.npz"),
                **{
                    camera_name: np.asarray(loss_values, dtype=np.float32)
                    for camera_name, loss_values in multiview_loss_dict.items()
                },
            )
            if multiview_lpips_dict is not None:
                np.savez_compressed(
                    osp.join(self.log_dir, f"{phase_name}_multiview_lpips_loss.npz"),
                    **{
                        camera_name: np.asarray(loss_values, dtype=np.float32)
                        for camera_name, loss_values in multiview_lpips_dict.items()
                    },
                )
        if multiview_depth_flag:
            np.savez_compressed(
                osp.join(self.log_dir, f"{phase_name}_multiview_depth_loss.npz"),
                **{
                    camera_name: np.asarray(loss_values, dtype=np.float32)
                    for camera_name, loss_values in multiview_depth_dict.items()
                },
            )
        torch.cuda.empty_cache()
        return

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
