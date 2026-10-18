# This processor is to do Easy Video Camera Solving, the minimal pipeline here.
# The segmentation is outside this function
import ast
import os, sys, shutil, os.path as osp
import torch, numpy as np
import imageio, cv2
import logging, time
from tqdm import tqdm

sys.path.append(osp.dirname(osp.abspath(__file__)))

from preprocessor_utils import configure_logging, seed_everything, make_video
from prior_loading import laplacian_filter_depth
from epi_error.epi_analysis import analyze_epi
import kornia
from torch import nn
from scipy.interpolate import griddata

from preprocessor_utils import align_disparity_to_metric_depth
from depth_models.depth_utils import viz_depth_list, save_depth_list
from sam3_wrapper import generate_sam3_epi, save_sam3_epi_artifacts
from wfm.masks import mask_pixel_count, should_use_flow_fallback, union_mask_sequences


def normalize_sam3_prompts(text_prompt):
    if isinstance(text_prompt, str):
        stripped = text_prompt.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = [part.strip().strip("'\"") for part in stripped[1:-1].split(",")]
            else:
                if isinstance(parsed, str):
                    parsed = [parsed]
            return [str(prompt).strip() for prompt in parsed if str(prompt).strip()]
        return [stripped]
    try:
        return [str(prompt).strip() for prompt in list(text_prompt) if str(prompt).strip()]
    except TypeError:
        prompt = str(text_prompt).strip()
        return [prompt] if prompt else []


def fill_depth_boundaries(depth_map, boundary_mask):
    """
    Fill in the boundary masked values in the depth map using interpolation.

    Parameters:
    depth_map (np.ndarray): The input depth map.
    boundary_mask (np.ndarray): The boundary mask where True indicates boundary pixels.

    Returns:
    np.ndarray: The depth map with boundary values filled in.
    """
    # Ensure the boundary mask and depth map have the same shape
    assert (
        depth_map.shape == boundary_mask.shape
    ), "Depth map and boundary mask must have the same shape."

    # Get the coordinates of the non-boundary pixels
    non_boundary_coords = np.array(np.nonzero(~boundary_mask)).T
    non_boundary_values = depth_map[~boundary_mask]

    # Get the coordinates of the boundary pixels
    boundary_coords = np.array(np.nonzero(boundary_mask)).T

    # Interpolate the boundary values
    filled_depth_map = depth_map.copy()
    filled_depth_map[boundary_mask] = griddata(
        non_boundary_coords, non_boundary_values, boundary_coords, method="nearest"
    )

    return filled_depth_map


@torch.no_grad()
def mark_dynamic_region(dyn_track, dyn_track_mask, H, W, radius_ratio=0.05):
    # T,N,2
    # draw a circle with radius 0.05*max(H,W) around the dynamic track when it's visible
    T, N = dyn_track.shape[:2]
    radius = int(max(H, W) * radius_ratio)
    logging.info(
        f"Marking dynamic region with radius ratio {radius_ratio} (R={radius})"
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    kernel = torch.tensor(kernel, dtype=torch.float32).to(dyn_track.device)
    ret = []
    for t in tqdm(range(T)):
        uv = dyn_track[t][dyn_track_mask[t]][..., :2]
        int_uv = uv.round().long()
        uv_mask = (
            (uv[..., 0] >= 0) * (uv[..., 0] < W) * (uv[..., 1] >= 0) * (uv[..., 1] < H)
        )
        int_uv = int_uv[uv_mask]
        # use a convolution to draw the circle
        buffer = torch.zeros((H, W)).to(dyn_track.device)
        buffer[int_uv[:, 1], int_uv[:, 0]] = 1
        # make the circle with kornia
        # buffer = cv2.dilate(buffer.detach().cpu().numpy(), kernel, iterations=1)
        buffer = kornia.morphology.dilation(buffer[None, None], kernel)[0, 0]
        ret.append(buffer.cpu())
    return torch.stack(ret, 0)


class MoCaPrep:
    def __init__(
        self,
        seed=12345,
        viz_flag=True,
        flow_mode="raft",
        flow_checkpoint=None,
        dep_mode="metric3d",
        tap_mode="spatracker",
        cowtracker_checkpoint=None,
        depth_exclude_sky=False,
        # specific cfgs
        metric3d_depth_version="giant2",
        spatracker_S_lenth=12,
        cotrakcer_version="3",
        cotracker_online_flag=True,
        # align model cfg
        align_metric_flag=True,
        align_metric_model="metric3d",
    ):
        device = "cuda:0"  # ! warning, because a bug in SpaTracker soft-splat, must use cuda:0, set the device outside globally
        self.viz_flag = viz_flag
        self.device = torch.device(device)
        self.seed = seed

        self.__init_flow__(flow_mode, device, flow_checkpoint=flow_checkpoint)

        self.__init_depth__(
            dep_mode,
            device,
            depth_exclude_sky=depth_exclude_sky,
            metric3d_depth_version=metric3d_depth_version,
            align_metric_flag=align_metric_flag,
            align_metric_model=align_metric_model,
        )
        self.__init_tap__(
            tap_mode,
            device,
            spatracker_S_lenth,
            cotrakcer_version,
            cotracker_online_flag,
            cowtracker_checkpoint,
        )
        return

    def __init_flow__(self, flow_mode, device, flow_checkpoint=None):
        self.flow_mode = flow_mode
        self.flow_dirname = None
        print(f"loading {flow_mode} ...")
        if flow_mode == "raft":
            from optical_flow.raft_wrapper import get_raft_model, raft_process_folder

            self.flow_dirname = "flow_raft"
            self.flow_model = get_raft_model(
                osp.join(
                    osp.dirname(__file__), "../weights/raft_models/raft-things.pth"
                ),
                device,
            )
            self.flow_model.cpu(), torch.cuda.empty_cache()
            self.flow_process_func = raft_process_folder
        elif flow_mode == "waft":
            from optical_flow.waft_wrapper import get_waft_model, waft_process_folder

            self.flow_dirname = "flow_waft"
            kwargs = {"device": device}
            if flow_checkpoint is not None:
                kwargs["ckpt_path"] = flow_checkpoint
            self.flow_model = get_waft_model(**kwargs)
            self.flow_model.cpu(), torch.cuda.empty_cache()
            self.flow_process_func = waft_process_folder
        elif flow_mode in ["none", "skip", None]:
            logging.info("Flow model disabled")
        else:  # todo: add GMFlow, recent realtime RAFT
            logging.warning(f"Unknown flow_mode: {flow_mode}, Flow Model not load")
        return

    def __init_depth__(
        self,
        dep_mode,
        device,
        depth_exclude_sky=False,
        metric3d_depth_version="giant2",
        align_metric_flag=True,
        align_metric_model="metric3d",
    ):

        print(f"loading {dep_mode} ...")
        self.dep_mode = dep_mode
        self.align_to_metric = False  # when the main depth model is not metric, have to align to a metric model, by default we use unidepth because it's easy to use

        if self.dep_mode == "metric3d":
            from depth_models.metric3d_wrapper import metric3d_process_folder
            from depth_models.metric3d_wrapper import get_metric3dv2_model

            self.depth_model = get_metric3dv2_model(
                device=device, version=metric3d_depth_version
            )
            self.depth_model.cpu(), torch.cuda.empty_cache()
            self.depth_process_func = metric3d_process_folder
        elif self.dep_mode == "uni":
            from depth_models.unidepth_wrapper import unidepth_process_folder
            from depth_models.unidepth_wrapper import get_unidepth_model

            self.depth_model = get_unidepth_model(device)
            self.depth_model.cpu(), torch.cuda.empty_cache()
            self.depth_process_func = unidepth_process_folder
        elif self.dep_mode == "depthcrafter":
            try:
                from depth_models.depthcrafter_wrapper import depthcrafter_process_folder
                from depth_models.depthcrafter_wrapper import get_depthcrafter_model
            except ImportError as error:
                raise ImportError(
                    "DepthCrafter is not bundled (its license is non-commercial). Install "
                    "Tencent/DepthCrafter as the `depthcrafter` package under its own terms, "
                    "or choose another dep_mode."
                ) from error

            self.depth_model = get_depthcrafter_model(modify_scheduler=True)
            torch.cuda.empty_cache()
            self.depth_process_func = depthcrafter_process_folder
            self.align_to_metric = True
            logging.warning(
                f"DepthCrafter is not a metric model, align to metric model"
            )
        elif self.dep_mode == "depthanything3":
            # Depth is precomputed by lib_prior/depth_models/agibot_da3_joint_precompute.py
            # and materialized under {ws}/depthanything3_depth/. Skip loading any model
            # and skip metric alignment (DA3 is already in metric/aligned scale).
            logging.info(
                "dep_mode=depthanything3: depth model will not be loaded; "
                "expecting precomputed npz files under {ws}/depthanything3_depth/"
            )
        elif self.dep_mode == "vggt_omega":
            logging.info(
                "dep_mode=vggt_omega: depth model will not be loaded; "
                "expecting precomputed npz files under {ws}/vggt_omega_depth/"
            )
        else:  # todo: add unidepth and maybe zoe depth
            print(
                f"Unknown dep_mode: {dep_mode}, Depth Model not load, will try to find depth from dir {dep_mode}_depth or {dep_mode}_depth.npz"
            )

        # * manually overwrite the align_to_metric flag, because if there is only a scale correction, in real setting the scale normalization will wash out the alignment!
        self.align_to_metric = self.align_to_metric * align_metric_flag

        if self.align_to_metric:
            logging.info(f"Aligning to metric depth with {align_metric_model}")
            self.align_mode = align_metric_model
            if self.align_mode == "metric3d":
                from depth_models.metric3d_wrapper import metric3d_process_folder
                from depth_models.metric3d_wrapper import get_metric3dv2_model

                self.align_metric_depth_model = get_metric3dv2_model(
                    device=device, version=metric3d_depth_version
                )
                self.align_metric_depth_model.cpu(), torch.cuda.empty_cache()
                self.align_metric_depth_process_func = metric3d_process_folder
            elif self.align_mode == "uni":
                from depth_models.unidepth_wrapper import unidepth_process_folder
                from depth_models.unidepth_wrapper import get_unidepth_model

                self.align_metric_depth_model = get_unidepth_model(device)
                self.align_metric_depth_model.cpu(), torch.cuda.empty_cache()
                self.align_metric_depth_process_func = unidepth_process_folder
            else:
                logging.warning(f"Unknown {self.align_mode}, skip loading")

        self.depth_exclude_sky = depth_exclude_sky
        if depth_exclude_sky:
            from seg.segformer_wrapper import get_segformer_model, get_mask

            # load sky model
            self.sky_feature_extractor, self.sky_model = get_segformer_model("cpu")
            self.sky_func = lambda img: get_mask(
                img,
                self.sky_feature_extractor,
                self.sky_model,
                self.device,
                class_id=[2],
            )
        return

        cotracker_online_flag = (True,)

    def __init_tap__(
        self,
        tap_mode,
        device,
        spatracker_S_lenth,
        cotrakcer_version,
        cotracker_online_flag=True,
        cowtracker_checkpoint=None,
    ):
        self.tap_mode = tap_mode
        if tap_mode == "spatracker":
            try:
                from tracking.spatracker_wrapper import (
                    spatracker_process_folder,
                    get_spatracker,
                )
            except ImportError as error:
                raise ImportError(
                    "SpaTracker is not bundled (CC BY-NC 4.0). Install it under "
                    "lib_prior/tracking/spatracker under its own terms, or choose another tap_mode."
                ) from error

            print("loading spatracker...")
            self.tap = get_spatracker(device, S_lenth=spatracker_S_lenth)
            self.tap_process_func = spatracker_process_folder
        elif tap_mode == "cotracker":
            from tracking.cotracker_wrapper import (
                cotracker_process_folder,
                get_cotracker,
            )

            print(
                f"loading cotracker v={cotrakcer_version} online={cotracker_online_flag}..."
            )
            self.tap = get_cotracker(
                device,
                cotrakcer_version=cotrakcer_version,
                online_flag=cotracker_online_flag,
            )
            self.cotracker_online_flag = cotracker_online_flag
            self.tap_process_func = cotracker_process_folder
        elif tap_mode == "cowtracker":
            from tracking.cowtracker_wrapper import (
                cowtracker_process_folder,
                get_cowtracker,
            )

            print("loading cowtracker...")
            self.tap = get_cowtracker(device, checkpoint_path=cowtracker_checkpoint)
            self.tap_process_func = cowtracker_process_folder
        elif tap_mode == "bootstapir":
            from tracking.bootstapir_wrapper import (
                bootstapir_process_folder,
                get_bootstapir_model,
            )

            print("loading bootstapir...")
            self.tap = get_bootstapir_model(device=device)
            self.tap_process_func = bootstapir_process_folder
        else:
            raise NotImplementedError(f"Unknown tap_mode: {tap_mode}")
        self.tap.cpu()

    def create_workspace(
        self, save_dir, t_list, fn_list, img_list, imgdirname="images"
    ):
        # create workspace
        os.makedirs(save_dir, exist_ok=True)
        imageio.mimsave(osp.join(save_dir, "input.mp4"), img_list)
        configure_logging(osp.join(save_dir, "preprocess.log"), debug=False)
        images_dir = osp.join(save_dir, "images")
        if osp.exists(images_dir):
            # logging.warning(
            #     f"images_dir {images_dir} already exists, remove it and resave everything!"
            # )
            # shutil.rmtree(images_dir)
            logging.warning(
                f"images_dir {images_dir} already exists, skip saving images!"
            )
            assert len(os.listdir(images_dir)) == len(fn_list), "Image number mismatch"
        else:
            os.makedirs(images_dir, exist_ok=False)
            for i, img in enumerate(img_list):
                imageio.imwrite(osp.join(images_dir, osp.basename(fn_list[i])), img)
        if t_list is not None:
            np.savez(osp.join(save_dir, "t_list.npz"), t_list=t_list)
        logging.info(f"Processor process img_list with shape {img_list.shape} ...")
        # also backup the command line arguments
        with open(osp.join(save_dir, "precompute_commandline_args.txt"), "w") as f:
            f.write(" ".join(sys.argv))
        return

    def compute_depth(
        self,
        save_dir,
        fn_list,
        img_list,
        K=None,
        fallback_fov=53.13,
        # enhacne depth boundary
        enhance_depth_boudary_th=1.0,
        enhance_depth_boudary_ksize=5,
        enhance_depth_boudary_open_ksize=3,
        # depthcrafter
        depthcrafter_denoising_steps=25,
        metric_alignment_frames=10,
        metric_alignment_min_dep=0.001,
        metric_alignment_max_dep=100.0,
        metric_alignment_first_quantil=0.5,
        metric_alignment_bias_flag=True,
        metric_alignment_kernel="cauchy",
        metric_alignment_fscale=0.001,
    ):
        if not hasattr(self, "depth_model"):
            logging.warning(f"Depth model not loaded, skip depth")
            return
        start_t = time.time()
        seed_everything(self.seed)
        logging.info(f"Generating Depth")
        try:
            self.depth_model.to(self.device)
        except:
            pass  # for depthcrafter pipeline

        if self.depth_exclude_sky:
            self.sky_model.to(self.device)
            invalid_mask_list = [self.sky_func(img) for img in img_list]
            self.sky_model.to("cpu"), torch.cuda.empty_cache()
        else:
            invalid_mask_list = None

        depth_save_dir = osp.join(save_dir, f"{self.dep_mode}_depth")

        if self.dep_mode == "metric3d":
            # * metric 3d needs the K
            if K is None:
                default_fov = fallback_fov
                fxfycxcy_pixel = None
            else:
                default_fov = None
                fxfycxcy_pixel = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
            self.depth_process_func(
                self.depth_model,
                img_list=img_list,
                fn_list=fn_list,
                dst=depth_save_dir,
                fxfycxcy_pixel=fxfycxcy_pixel,
                default_fov_deg=default_fov,
                invalid_mask_list=invalid_mask_list,
            )
        elif self.dep_mode == "depthcrafter":
            self.depth_process_func(
                self.depth_model,
                img_list=img_list,
                fn_list=fn_list,
                dst=depth_save_dir + "_raw_disp",
                invalid_mask_list=invalid_mask_list,
                n_steps=depthcrafter_denoising_steps,
            )
        else:
            self.depth_process_func(
                self.depth_model,
                img_list=img_list,
                fn_list=fn_list,
                dst=depth_save_dir,
                invalid_mask_list=invalid_mask_list,
            )

        if self.dep_mode in ["depthcrafter"]:
            if self.align_to_metric:
                # * first infer metric depth with subsampled frames
                align_dir = osp.join(save_dir, f"alignment_{self.align_mode}_depth")
                if self.align_mode in ["metric3d", "uni"]:
                    self.align_metric_depth_model.to(self.device)
                    T = len(fn_list)
                    sampled_align_t = np.arange(
                        0, T, T // metric_alignment_frames
                    ).tolist()
                    align_fns = [fn_list[i] for i in sampled_align_t]
                    self.align_metric_depth_process_func(
                        self.align_metric_depth_model,
                        img_list=img_list[sampled_align_t],
                        fn_list=align_fns,
                        dst=align_dir,
                        invalid_mask_list=(
                            [invalid_mask_list[t] for t in sampled_align_t]
                            if invalid_mask_list is not None
                            else None
                        ),
                    )
                    self.align_metric_depth_model.to("cpu")
                else:
                    align_fns = sorted(os.listdir(align_dir))
                    _name_index_list = [".".join(fn.split(".")[:-1]) for fn in fn_list]
                    sampled_align_t = [
                        _name_index_list.index(".".join(fn.split(".")[:-1]))
                        for fn in align_fns
                    ]

                # * align the loaded depth to the metric depth with a global s,t
                aligned_dep, viz_align = align_disparity_to_metric_depth(
                    disp_map=self.load_dep_list(
                        save_dir, name=f"{self.dep_mode}_depth_raw_disp"
                    ),
                    metric_depth=self.load_dep_list(
                        save_dir, name=f"alignment_{self.align_mode}_depth"
                    ),
                    metric_depth_t=sampled_align_t,
                    quantil=metric_alignment_first_quantil,
                    min_dep=metric_alignment_min_dep,
                    max_dep=metric_alignment_max_dep,
                    bias=metric_alignment_bias_flag,
                    robust_loss=metric_alignment_kernel,
                    robust_f_scale=metric_alignment_fscale,
                )
                imageio.imsave(osp.join(save_dir, "alignment_viz.jpg"), viz_align)
            else:
                disp_list = self.load_dep_list(
                    save_dir, name=f"{self.dep_mode}_depth_raw_disp"
                )
                disp_list = torch.clamp(torch.as_tensor(disp_list), 1e-6, 1e10)
                aligned_dep = 1.0 / disp_list
            # * save final depth
            save_depth_list(aligned_dep, fn_list, depth_save_dir, invalid_mask_list)
            viz_depth_list(aligned_dep, depth_save_dir + ".mp4")
        try:
            self.depth_model.to("cpu")
        except:
            pass

        if enhance_depth_boudary_th > 0:
            assert (
                enhance_depth_boudary_open_ksize >= 3
            ), "To have reasonable boundary enhancement, open_ksize should be >=3"
            original_dep = self.load_dep_list(save_dir)
            fixed_dep, fix_mask = self.enhance_depth_boundary(
                original_dep,
                boundary_th_ratio=enhance_depth_boudary_th,
                ksize=enhance_depth_boudary_ksize,
                open_ksize=enhance_depth_boudary_open_ksize,
            )
            viz_depth_list(fixed_dep, depth_save_dir + "_sharp.mp4")
            save_depth_list(fixed_dep, fn_list, depth_save_dir + "_sharp")
            imageio.mimsave(
                osp.join(
                    save_dir,
                    f"sharper_boundary_mask_th={enhance_depth_boudary_th}_ks={enhance_depth_boudary_ksize}_oks={enhance_depth_boudary_open_ksize}.mp4",
                ),
                fix_mask.astype(np.uint8) * 255,
            )
            logging.info(f"Enhanced depth saved to {depth_save_dir}_sharp")

        torch.cuda.empty_cache()
        logging.info(f"Depth done in {(time.time()-start_t)/60.0:.2f}min")
        return

    def enhance_depth_boundary(
        self, dep_list, boundary_th_ratio=1.0, ksize=5, open_ksize=3
    ):
        dep_list = np.asarray(dep_list)
        fixed_dep = []
        dep_boundary_mask, _ = laplacian_filter_depth(
            dep_list,
            threshold_ratio=boundary_th_ratio,
            ksize=ksize,
            open_ksize=open_ksize,
        )
        for dep, mask in tqdm(zip(dep_list, dep_boundary_mask)):
            fixed_dep.append(fill_depth_boundaries(dep, ~mask))
        fixed_dep = np.stack(fixed_dep, 0)
        return fixed_dep, dep_boundary_mask

    def compute_flow(
        self,
        save_dir,
        fn_list,
        img_list,
        step_list=[1],
        epi_num_threads=16,
        analyze_epi_flag=True,
    ):
        start_t = time.time()
        logging.info(f"Generating Flow")
        seed_everything(self.seed)
        self.flow_model.to(self.device)
        flow_save_dir = osp.join(save_dir, self.flow_dirname)
        self.flow_process_func(
            self.flow_model, img_list, fn_list, flow_save_dir, step_list=step_list
        )
        self.flow_model.to("cpu"), torch.cuda.empty_cache()
        logging.info(f"Flow done in {(time.time()-start_t)/60.0:.2f}min")

        if analyze_epi_flag:
            logging.info(f"Analysis the Epi Error")
            analyze_epi(
                save_dir,
                num_threads=epi_num_threads,
                step_list=step_list,
                flow_dirname=self.flow_dirname,
            )
        return

    def compute_sam3_epi(
        self,
        save_dir,
        fn_list,
        img_list,
        text_prompt,
        gpus=None,
        checkpoint_path=None,
        bpe_path=None,
        sam3_lib_prior_root=None,
        fps=10.0,
        seed_frame_indices=None,
    ):
        start_t = time.time()
        logging.info("Generating SAM3 dynamic masks into epi/error")
        seed_everything(self.seed)
        prompt_list = normalize_sam3_prompts(text_prompt)
        if len(prompt_list) == 0:
            raise ValueError("SAM3 text prompt is required when dynamic_mask_mode=sam3")

        combined_mask_list = None
        for prompt in prompt_list:
            logging.info(f"Running SAM3 prompt: {prompt}")
            mask_list = generate_sam3_epi(
                save_dir=save_dir,
                img_name_list=fn_list,
                img_list=img_list,
                text_prompt=prompt,
                gpus=gpus,
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                sam3_lib_prior_root=sam3_lib_prior_root,
                fps=fps,
                save_mask_png=len(prompt_list) == 1,
                seed_frame_indices=seed_frame_indices,
            )
            combined_mask_list = union_mask_sequences(combined_mask_list, mask_list)
        if len(prompt_list) > 1:
            logging.info(f"Saving combined SAM3 mask for prompts: {prompt_list}")
            save_sam3_epi_artifacts(
                save_dir=save_dir,
                img_name_list=fn_list,
                img_list=img_list,
                mask_list=combined_mask_list,
                fps=fps,
            )
        combined_pixel_count = mask_pixel_count(combined_mask_list)
        logging.info(f"SAM3 dynamic mask pixels: {combined_pixel_count}")
        logging.info(f"SAM3 EPI done in {(time.time()-start_t)/60.0:.2f}min")
        return combined_mask_list

    def compute_tap(
        self,
        ws,
        save_name,
        n_track,
        img_list,
        mask_list,  # todo: Tensor or [[Tensor], [Tensor, Tensor]]
        dep_list=None,
        K=None,
        chunk_size=8192,
        # depth cfg for spatracker
        depth_boundary_filter_th=0.3,
        max_viz_cnt=512,
    ):
        start_t = time.time()
        if isinstance(img_list, list):
            if isinstance(img_list[0], np.ndarray):
                img_list = np.stack(img_list, 0)
            elif isinstance(img_list[0], torch.Tensor):
                img_list = torch.stack(img_list, 0)
        logging.info(f"Query Long-track")
        seed_everything(self.seed)
        self.tap.to(self.device)

        if self.tap_mode == "spatracker":
            # load depth
            logging.warning(f"Warning, for spatracker safty, filter the depth boundary")
            dep_mask, _ = laplacian_filter_depth(
                dep_list, depth_boundary_filter_th, 5
            )  # todo: this args may need some
            dep_list = dep_list * dep_mask.astype(np.float32)
            self.tap_process_func(
                working_dir=ws,
                img_list=img_list,
                dep_list=dep_list,
                sample_mask_list=mask_list,
                model=self.tap,
                K=K,
                total_n_pts=n_track,
                save_name=save_name,
                chunk_size=chunk_size,
                max_viz_cnt=max_viz_cnt,
            )
        elif self.tap_mode == "cotracker":
            # raise RuntimeError(
            #     "Need to remove the manaully forward, backward inference."
            # )
            logging.warning(f"SHOULD UPGRADE THE FWD BWD INFERENCE")
            self.tap_process_func(
                working_dir=ws,
                img_list=img_list,
                sample_mask_list=mask_list,
                model=self.tap,
                total_n_pts=n_track,
                save_name=save_name,
                online_flag=self.cotracker_online_flag,
                chunk_size=5000 if not self.cotracker_online_flag else chunk_size,
                max_viz_cnt=max_viz_cnt,
            )
        else:
            self.tap_process_func(
                working_dir=ws,
                img_list=img_list,
                sample_mask_list=mask_list,
                model=self.tap,
                total_n_pts=n_track,
                save_name=save_name,
                chunk_size=chunk_size,
                max_viz_cnt=max_viz_cnt,
            )
        self.tap.cpu(), torch.cuda.empty_cache()
        logging.info(f"Long-track done in {(time.time()-start_t)/60.0:.2f}min")
        return

    def process(
        self,
        t_list,
        img_list: list,
        img_name_list: list,
        save_dir: str,
        # proc cfg
        n_track: int = 8192,
        known_camera_K: np.ndarray = None,
        #
        epi_num_threads=16,
        compute_flow=True,
        flow_steps=[1],
        dynamic_mask_mode="raft_epi",
        sam3_empty_fallback_to_flow=False,
        sam3_text=None,
        sam3_gpus=None,
        sam3_checkpoint=None,
        sam3_bpe=None,
        sam3_lib_prior_root=None,
        sam3_fps=10.0,
        sam3_seed_frame_indices=None,
        #
        compute_tap=True,
        tap_chunk_size=8192,
        #
        depthcrafter_denoising_steps=25,
        # metric alignment config
        metric_alignment_frames=10,
        metric_alignment_min_dep=0.001,
        metric_alignment_max_dep=100.0,
        metric_alignment_first_quantil=0.5,
        metric_alignment_bias_flag=False,
        metric_alignment_kernel="cauchy",
        metric_alignment_fscale=0.001,
        # boundary enhancement
        boundary_enhance_th=1.0,
        boundary_enhance_ksize=5,
        boundary_enhance_open_ksize=3,
    ):
        assert dynamic_mask_mode in [
            "raft_epi",
            "sam3",
        ], f"Unknown dynamic_mask_mode={dynamic_mask_mode}"

        # todo: modify the epi interface
        # todo: modify the tap interface

        ########################################################################
        img_name_list = [osp.basename(fn) for fn in img_name_list]
        img_list = np.asarray(img_list)

        ########################################################################
        # * 0. Init
        self.create_workspace(save_dir, t_list, img_name_list, img_list)
        # ########################################################################
        # * 2. Depth
        self.compute_depth(
            save_dir,
            img_name_list,
            img_list,
            K=known_camera_K,
            depthcrafter_denoising_steps=depthcrafter_denoising_steps,
            # * boundary enhancement
            enhance_depth_boudary_th=boundary_enhance_th,
            enhance_depth_boudary_ksize=boundary_enhance_ksize,
            enhance_depth_boudary_open_ksize=boundary_enhance_open_ksize,
            # * metric alignment (not used, because we won't use bias correction)
            metric_alignment_frames=metric_alignment_frames,
            metric_alignment_min_dep=metric_alignment_min_dep,
            metric_alignment_max_dep=metric_alignment_max_dep,
            metric_alignment_first_quantil=metric_alignment_first_quantil,
            metric_alignment_bias_flag=metric_alignment_bias_flag,
            metric_alignment_kernel=metric_alignment_kernel,
            metric_alignment_fscale=metric_alignment_fscale,
        )
        ########################################################################
        # # * 3. Flow and dynamic mask generation [Opt]
        if self.flow_dirname is not None and compute_flow:
            logging.info(f"Generating Flow")
            self.compute_flow(
                save_dir,
                img_name_list,
                img_list,
                step_list=flow_steps,
                epi_num_threads=epi_num_threads,
                analyze_epi_flag=dynamic_mask_mode == "raft_epi",
            )
        elif dynamic_mask_mode == "raft_epi":
            logging.warning(
                "dynamic_mask_mode=raft_epi without flow generation will skip epi export"
            )

        if dynamic_mask_mode == "sam3":
            sam3_mask_list = self.compute_sam3_epi(
                save_dir=save_dir,
                fn_list=img_name_list,
                img_list=img_list,
                text_prompt=sam3_text,
                gpus=sam3_gpus,
                checkpoint_path=sam3_checkpoint,
                bpe_path=sam3_bpe,
                sam3_lib_prior_root=sam3_lib_prior_root,
                fps=sam3_fps,
                seed_frame_indices=sam3_seed_frame_indices,
            )
            if should_use_flow_fallback(
                sam3_mask_list, fallback_enabled=sam3_empty_fallback_to_flow
            ):
                if self.flow_dirname is None:
                    raise RuntimeError(
                        "SAM3 returned an empty mask and WAFT/flow fallback was requested, "
                        "but flow_mode is disabled."
                    )
                logging.warning(
                    "SAM3 returned an empty mask; falling back to %s optical-flow epipolar masks.",
                    self.flow_mode,
                )
                self.compute_flow(
                    save_dir,
                    img_name_list,
                    img_list,
                    step_list=flow_steps,
                    epi_num_threads=epi_num_threads,
                    analyze_epi_flag=True,
                )

        # ########################################################################
        # * 4. Long-track
        if compute_tap:
            dep_name = self.dep_mode + "_depth"
            if osp.exists(osp.join(save_dir, dep_name + "_sharp")):
                dep_name = dep_name + "_sharp"
                logging.info(f"TAP will use the sharpened depth {dep_name}")
            dep_list = self.load_dep_list(save_dir, dep_name)
            uniform_sample_list = np.ones_like(dep_list) > 0
            self.compute_tap(
                ws=save_dir,
                save_name=f"uniform_dep={self.dep_mode}",
                n_track=n_track,
                img_list=img_list,
                mask_list=uniform_sample_list,
                dep_list=dep_list,
                K=known_camera_K,
                chunk_size=tap_chunk_size,
            )
        ########################################################################
        return True

    def load_dep_list(self, ws, name=None):
        if name is None:
            name = f"{self.dep_mode}_depth"
        dep_dir = osp.join(ws, name)
        if osp.isdir(dep_dir):
            dep_fns = sorted([osp.join(dep_dir, fn) for fn in os.listdir(dep_dir)])
            dep_list = np.stack([np.load(fn)["dep"] for fn in dep_fns], 0)
        else:
            # try the npz
            depth_fn = dep_dir + ".npz"
            assert osp.exists(depth_fn), f"Depth not found in {dep_dir} or {depth_fn}"
            dep_list = np.load(depth_fn)["dep"]
        return dep_list


if __name__ == "__main__":

    from preprocessor_utils import load_imgs, convert_from_mp4

    img_list, img_fns = load_imgs(f"../data/davis/train/images")
    t_list = [i for i in range(len(img_fns))]

    processor = MoCaPrep()

    processor.process(
        t_list=t_list,
        img_list=img_list,
        img_name_list=img_fns,
        save_dir=f"../data/debug/train",
    )

    # name="breakdance-flare"
    # img_list, img_fns = processor.load_imgs(f"../data/davis/{name}/images")
    # processor.process(img_list, img_fns, f"../data/davis/{name}/")

    # img_list, img_fns = processor.load_imgs(f"../data/dragon/images")
    # processor.process(img_list, img_fns, f"../data/dragon")
