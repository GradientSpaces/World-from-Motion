import math
import os
import os.path as osp
import sys
import time
import logging

import imageio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(osp.dirname(osp.abspath(__file__)))
_COWTRACKER_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "cowtracker")
if _COWTRACKER_ROOT not in sys.path:
    sys.path.append(_COWTRACKER_ROOT)

from cotracker_visualizer import Visualizer
from tracking_utils import (
    convert_img_list_to_cotracker_input,
    get_uniform_random_queries,
    viz_queries,
)
from cowtracker import CoWTracker, CoWTrackerWindowed
from cowtracker.utils.padding import apply_padding, compute_padding_params


DEFAULT_INF_SHORT_SIDE = 336
DEFAULT_INF_LONG_SIDE = 560
DEFAULT_MAX_QUERY_FRAMES = 8
DEFAULT_VIS_CONF_THRESHOLD = 0.1
DEFAULT_WINDOW_LEN = 100
DEFAULT_WINDOW_STRIDE = 100


@torch.no_grad()
def get_cowtracker(
    device,
    checkpoint_path=None,
    dtype=None,
    use_windowed=True,
    window_len=DEFAULT_WINDOW_LEN,
    stride=DEFAULT_WINDOW_STRIDE,
    num_memory_frames=10,
):
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if dtype is None:
        dtype = torch.float16 if use_cuda else torch.float32
    if use_windowed:
        model = CoWTrackerWindowed.from_checkpoint(
            checkpoint_path=checkpoint_path,
            window_len=window_len,
            stride=stride,
            device=str(device),
            dtype=dtype,
        )
        model.windowed.num_memory_frames = num_memory_frames
    else:
        model = CoWTracker.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=str(device),
            dtype=dtype,
        )
    return model


def _get_inference_hw(orig_H, orig_W, short_side, long_side):
    if orig_H >= orig_W:
        return long_side, short_side
    return short_side, long_side


def _map_queries_to_inference(query_xy, padding_info):
    query_xy = query_xy.clone().float()
    query_xy[:, 0] = (
        query_xy[:, 0] * padding_info["scaled_W"] / max(padding_info["orig_W"], 1)
        + padding_info["pad_left"]
    )
    query_xy[:, 1] = (
        query_xy[:, 1] * padding_info["scaled_H"] / max(padding_info["orig_H"], 1)
        + padding_info["pad_top"]
    )
    return query_xy


def _build_sampling_grid(query_xy, H, W, T):
    query_xy = query_xy.float()
    denom_W = max(W - 1, 1)
    denom_H = max(H - 1, 1)
    grid_x = 2.0 * query_xy[:, 0] / denom_W - 1.0
    grid_y = 2.0 * query_xy[:, 1] / denom_H - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.view(1, -1, 1, 2).expand(T, -1, -1, -1)


def _sample_dense_predictions(dense_tracks, dense_vis, dense_conf, query_xy):
    T, H, W, _ = dense_tracks.shape
    sampling_grid = _build_sampling_grid(query_xy.to(dense_tracks.device), H, W, T)

    sampled_tracks = F.grid_sample(
        dense_tracks.permute(0, 3, 1, 2),
        sampling_grid,
        mode="bilinear",
        align_corners=True,
    )
    sampled_tracks = sampled_tracks.squeeze(-1).permute(0, 2, 1)

    sampled_vis = F.grid_sample(
        dense_vis.unsqueeze(1),
        sampling_grid,
        mode="bilinear",
        align_corners=True,
    )
    sampled_vis = sampled_vis.squeeze(1).squeeze(-1)

    sampled_conf = F.grid_sample(
        dense_conf.unsqueeze(1),
        sampling_grid,
        mode="bilinear",
        align_corners=True,
    )
    sampled_conf = sampled_conf.squeeze(1).squeeze(-1)
    return sampled_tracks, sampled_vis, sampled_conf


def _scale_tracks_to_original(tracks, padding_info):
    tracks = tracks.clone()
    tracks[..., 0] -= padding_info["pad_left"]
    tracks[..., 1] -= padding_info["pad_top"]
    if (
        padding_info["scaled_H"],
        padding_info["scaled_W"],
    ) != (
        padding_info["orig_H"],
        padding_info["orig_W"],
    ):
        tracks[..., 0] *= padding_info["orig_W"] / max(padding_info["scaled_W"], 1)
        tracks[..., 1] *= padding_info["orig_H"] / max(padding_info["scaled_H"], 1)
    return tracks


@torch.no_grad()
def _run_dense_cowtracker(model, clip):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    clip = clip.to(device)
    use_autocast = device.type == "cuda" and model_dtype in (
        torch.float16,
        torch.bfloat16,
    )
    if use_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=model_dtype):
            predictions = model.forward(video=clip, queries=None)
    else:
        predictions = model.forward(video=clip, queries=None)
    dense_tracks = predictions["track"][0].detach()
    dense_vis = predictions["vis"][0].detach()
    dense_conf = predictions["conf"][0].detach()
    del predictions, clip
    return dense_tracks, dense_vis, dense_conf


def _extract_sparse_tracks_from_dense(
    dense_tracks,
    dense_vis,
    dense_conf,
    query_xy_orig,
    padding_info,
    vis_conf_threshold,
):
    query_xy_inf = _map_queries_to_inference(query_xy_orig, padding_info).to(
        dense_tracks.device
    )
    sampled_tracks, sampled_vis, sampled_conf = _sample_dense_predictions(
        dense_tracks,
        dense_vis,
        dense_conf,
        query_xy_inf,
    )
    sampled_tracks = _scale_tracks_to_original(sampled_tracks, padding_info)
    visibility = (sampled_vis * sampled_conf) > vis_conf_threshold
    return sampled_tracks, visibility


@torch.no_grad()
def _track_queries_from_anchor(
    video_pt,
    query_frame,
    query_xy,
    model,
    padding_info,
    vis_conf_threshold,
):
    total_frames = len(video_pt)
    num_queries = len(query_xy)
    tracks = torch.zeros(total_frames, num_queries, 2)
    visibility = torch.zeros(total_frames, num_queries, dtype=torch.bool)

    if query_frame > 0:
        dense_tracks, dense_vis, dense_conf = _run_dense_cowtracker(
            model, video_pt[: query_frame + 1].flip(0)
        )
        bwd_tracks, bwd_visibility = _extract_sparse_tracks_from_dense(
            dense_tracks,
            dense_vis,
            dense_conf,
            query_xy,
            padding_info,
            vis_conf_threshold,
        )
        tracks[: query_frame + 1] = bwd_tracks.flip(0).cpu()
        visibility[: query_frame + 1] = bwd_visibility.flip(0).cpu()
        del dense_tracks, dense_vis, dense_conf, bwd_tracks, bwd_visibility
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if query_frame < total_frames - 1:
        dense_tracks, dense_vis, dense_conf = _run_dense_cowtracker(
            model, video_pt[query_frame:]
        )
        fwd_tracks, fwd_visibility = _extract_sparse_tracks_from_dense(
            dense_tracks,
            dense_vis,
            dense_conf,
            query_xy,
            padding_info,
            vis_conf_threshold,
        )
        tracks[query_frame:] = fwd_tracks.cpu()
        visibility[query_frame:] = fwd_visibility.cpu()
        del dense_tracks, dense_vis, dense_conf, fwd_tracks, fwd_visibility
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tracks[query_frame] = query_xy.float()
    visibility[query_frame] = True
    return tracks, visibility


@torch.no_grad()
def cowtracker_process_folder(
    working_dir,
    img_list,
    sample_mask_list,
    model,
    total_n_pts,
    chunk_size=10000,  # kept for API compatibility
    save_name="",
    max_viz_cnt=512,
    device=torch.device("cuda:0"),
    max_query_frames=DEFAULT_MAX_QUERY_FRAMES,
    vis_conf_threshold=DEFAULT_VIS_CONF_THRESHOLD,
    inf_short_side=DEFAULT_INF_SHORT_SIDE,
    inf_long_side=DEFAULT_INF_LONG_SIDE,
):
    del chunk_size, device

    viz_dir = osp.join(working_dir, "cowtracker_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)

    full_video_pt = convert_img_list_to_cotracker_input(img_list).cpu()
    _, total_frames, _, orig_H, orig_W = full_video_pt.shape
    logging.info(f"T=[{total_frames}], video shape: {full_video_pt.shape}")

    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,
        tracks_leave_trace=4,
    )

    sample_mask_list = torch.as_tensor(sample_mask_list).cpu() > 0
    viz_sample_mask = sample_mask_list[..., None] * full_video_pt[0].permute(0, 2, 3, 1)
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_sample_mask.mp4"),
        viz_sample_mask.numpy().astype(np.uint8),
    )

    inf_H, inf_W = _get_inference_hw(orig_H, orig_W, inf_short_side, inf_long_side)
    padding_info = compute_padding_params(
        orig_H,
        orig_W,
        inf_H,
        inf_W,
        skip_upscaling=True,
    )
    inference_video = apply_padding(full_video_pt[0], padding_info).cpu()
    logging.info(
        "CowTracker inference resolution %sx%s (scaled=%sx%s, padding=%s)",
        inf_H,
        inf_W,
        padding_info["scaled_H"],
        padding_info["scaled_W"],
        (
            padding_info["pad_top"],
            padding_info["pad_bottom"],
            padding_info["pad_left"],
            padding_info["pad_right"],
        ),
    )

    query_interval = 1
    if total_frames > max_query_frames:
        query_interval = int(math.ceil(total_frames / float(max_query_frames)))

    # CowTracker is first-frame anchored, so we only sample queries on a small set of
    # anchor frames and run dense tracking forward/backward from each anchor.
    queries = get_uniform_random_queries(
        full_video_pt,
        total_n_pts,
        mask_list=sample_mask_list,
        interval=query_interval,
    )
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_queries.mp4"),
        viz_queries(queries.squeeze(0), orig_H, orig_W, total_frames),
    )

    query_frames = queries[0, :, 0].long()
    query_xy = queries[0, :, 1:].float()
    tracks = torch.zeros(total_frames, query_xy.shape[0], 2)
    visibility = torch.zeros(total_frames, query_xy.shape[0], dtype=torch.bool)

    unique_query_frames = torch.unique(query_frames, sorted=True)
    logging.info(
        "Sampling %s CowTracker anchor frames with interval=%s",
        len(unique_query_frames),
        query_interval,
    )

    start_t = time.time()
    for anchor_idx, query_frame in enumerate(unique_query_frames.tolist()):
        anchor_mask = query_frames == query_frame
        anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).squeeze(1)
        anchor_xy = query_xy[anchor_indices]
        logging.info(
            "CowTracker anchor %s/%s at frame %s with %s queries",
            anchor_idx + 1,
            len(unique_query_frames),
            query_frame,
            len(anchor_indices),
        )
        anchor_tracks, anchor_visibility = _track_queries_from_anchor(
            inference_video,
            query_frame,
            anchor_xy,
            model,
            padding_info,
            vis_conf_threshold,
        )
        tracks[:, anchor_indices] = anchor_tracks
        visibility[:, anchor_indices] = anchor_visibility

    end_t = time.time()
    logging.info(f"Time cost: {(end_t - start_t) / 60.0:.3f}min")

    viz_cnt = min(tracks.shape[1], max_viz_cnt)
    viz_choice = np.random.choice(tracks.shape[1], viz_cnt, replace=False)
    vis.visualize(
        video=full_video_pt,
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None, :, viz_choice],
        filename=f"{save_name}_cowtracker_global",
    )
    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")

    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_cowtracker_tap.npz"),
        queries=queries.numpy(),  # kept for parity with other TAP backends
        tracks=tracks.numpy(),
        visibility=visibility.numpy(),
    )
    return
