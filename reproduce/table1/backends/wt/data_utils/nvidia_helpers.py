import numpy as np
import logging
import torch
import os

def load_nvidia_gt_pose(fn):
    poses_bounds = np.load(fn)  # (N_images, 17)
    gt_training_cam_T_wi = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    H, W, focal = gt_training_cam_T_wi[
        0, :, -1
    ]  # original intrinsics, same for all images
    # # Original poses has rotation in form "down right back", change to "right up back"
    # # See https://github.com/bmild/nerf/issues/34
    # poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # load the colmap recon scale so we can align the depth and translation scale!!
    # ! important: rescale the translation, now hardcoded
    gt_training_cam_T_wi = np.concatenate(
        [
            gt_training_cam_T_wi[..., 1:2],
            gt_training_cam_T_wi[..., :1],
            -gt_training_cam_T_wi[..., 2:3],
            gt_training_cam_T_wi[..., 3:4],
        ],
        -1,
    )
    short_side = min(H, W)
    fov = 2 * np.arctan(short_side / (2 * focal))
    gt_training_fov = np.rad2deg(fov)
    logging.info(f"Load GT poses from {fn}, fov: {gt_training_fov} deg")
    # the pose should be 4x4, not 3x4
    gt_training_cam_T_wi = torch.from_numpy(gt_training_cam_T_wi).float()
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=gt_training_cam_T_wi.device)[
        None
    ].expand(len(gt_training_cam_T_wi), -1, -1)
    gt_training_cam_T_wi = torch.cat([gt_training_cam_T_wi, bottom], dim=1)
    gt_training_cxcy_ratio = [[0.5, 0.5]]
    return (gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio)
    
class InputCamNameIterV3:
    def __init__(self):
        self.now_name = 2
        self.low_bound = 2
        self.up_bound = 12
        self.trend = 'UP'
    def sample_next_name(self):
        if self.trend == 'UP':
            if self.now_name == 12:
                self.now_name = 11
                self.trend = 'DOWN'
            else:
                self.now_name += 1
        else:
            if self.now_name == 2:
                self.now_name = 3
                self.trend = 'UP'
            else:
                self.now_name -= 1
        return self.now_name

def load_nvidia_ours_gt_pose_v3(fn, N_images=-1):

    poses_bounds = np.load(fn) # 12 cameras
    gt_training_cam_T_wi = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = gt_training_cam_T_wi[
        0, :, -1
    ]
    # gt_training_cam_T_wi = np.concatenate([gt_training_cam_T_wi[..., 1:2], gt_training_cam_T_wi[..., :1], -gt_training_cam_T_wi[..., 2:3], gt_training_cam_T_wi[..., 3:4]], -1)
    # only using camera 4 -> idx = 3
    assert N_images != -1

    gt_training_cam_T_wi_list = []
    cam_idx = 2
    input_cam_name_iter = InputCamNameIterV3()
    cam_idx_record_list = []
    for tidx in range(N_images):
        cam_idx_record_list.append(cam_idx)
        gt_training_cam_T_wi_each = np.concatenate([gt_training_cam_T_wi[cam_idx-1, :, 1:2], gt_training_cam_T_wi[cam_idx-1, :, :1], -gt_training_cam_T_wi[cam_idx-1, :, 2:3], gt_training_cam_T_wi[cam_idx-1, :, 3:4]], -1)
        gt_training_cam_T_wi_each = gt_training_cam_T_wi_each[None, ...]
        gt_training_cam_T_wi_list.append(gt_training_cam_T_wi_each)
        cam_idx = input_cam_name_iter.sample_next_name()
    gt_training_cam_T_wi = np.concatenate(gt_training_cam_T_wi_list, axis=0)

    print("CHECK(DEBUG): load_nvidia_ours_gt_pose_v3,", cam_idx_record_list)

    short_side = min(H, W)
    fov = 2 * np.arctan(short_side / (2 * focal))
    gt_training_fov = np.rad2deg(fov)
    logging.info(f"Load GT poses from {fn}, fov: {gt_training_fov} deg")
    # the pose should be 4x4, not 3x4
    gt_training_cam_T_wi = torch.from_numpy(gt_training_cam_T_wi).float()
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=gt_training_cam_T_wi.device)[
        None
    ].expand(len(gt_training_cam_T_wi), -1, -1)
    gt_training_cam_T_wi = torch.cat([gt_training_cam_T_wi, bottom], dim=1)
    gt_training_cxcy_ratio = [[0.5, 0.5]]
    return (gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio)

def get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov):
    gt_testing_cam_T_wi_list = [
        gt_training_cam_T_wi[:1].expand(len(gt_training_cam_T_wi), -1, -1)
    ]
    gt_testing_tids_list = [
        torch.arange(len(gt_training_cam_T_wi), device=gt_training_cam_T_wi.device)
    ]
    gt_testing_fns_list = [[f"v000_t{t:03d}" for t in range(len(gt_training_cam_T_wi))]]
    gt_testing_fov_list = [gt_training_fov]
    gt_testing_cxcy_ratio_list = [[0.5, 0.5]]
    return (
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_testing_fov_list,
        gt_testing_cxcy_ratio_list,
    )

def get_nvidia_ours_dummy_test_v3(gt_training_cam_T_wi, gt_training_fov, test_gt_dir, fn, N_images=-1):
    gt_testing_cam_T_wi_list = []
    gt_testing_tids_list = []
    gt_testing_fns_list = []
    gt_testing_fov_list = []
    gt_testing_cxcy_ratio_list = []

    gt_cam_idx = 1

    poses_bounds = np.load(fn) # 12 cameras
    tot_gt_training_cam_T_wi = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = tot_gt_training_cam_T_wi[
        0, :, -1
    ]
    assert N_images != -1
    tot_gt_training_cam_T_wi = np.concatenate([tot_gt_training_cam_T_wi[gt_cam_idx-1, :, 1:2], tot_gt_training_cam_T_wi[gt_cam_idx-1, :, :1], -tot_gt_training_cam_T_wi[gt_cam_idx-1, :, 2:3], tot_gt_training_cam_T_wi[gt_cam_idx-1, :, 3:4]], -1)
    tot_gt_training_cam_T_wi = tot_gt_training_cam_T_wi[None, ...]
    tot_gt_training_cam_T_wi = tot_gt_training_cam_T_wi.repeat(N_images, axis=0)

    tot_gt_training_cam_T_wi = torch.from_numpy(tot_gt_training_cam_T_wi).float()
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=tot_gt_training_cam_T_wi.device)[
        None
    ].expand(len(tot_gt_training_cam_T_wi), -1, -1)
    tot_gt_training_cam_T_wi = torch.cat([tot_gt_training_cam_T_wi, bottom], dim=1)

    gt_testing_cam_T_wi_list = [
        tot_gt_training_cam_T_wi[:1].expand(len(gt_training_cam_T_wi), -1, -1)
    ]
    gt_testing_tids_list = [
        torch.arange(len(gt_training_cam_T_wi), device=gt_training_cam_T_wi.device)
    ]

    gt_testing_fns_list = [[each_name.split('.')[0] for each_name in sorted(os.listdir(test_gt_dir))]]
    gt_testing_fov_list = [gt_training_fov]
    gt_testing_cxcy_ratio_list = [[0.5, 0.5]]
    return (
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_testing_fov_list,
        gt_testing_cxcy_ratio_list,
    )
