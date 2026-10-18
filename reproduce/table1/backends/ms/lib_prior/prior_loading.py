# helpers for loading saved priors
import json
import logging, sys, os, os.path as osp
from glob import glob
import torch, numpy as np
from torch import nn
import imageio, cv2
from tqdm import tqdm
import time

sys.path.append(osp.dirname(osp.abspath(__file__)))
from epi_error.epi_analysis import load_epi_error  # , load_vos
from tracking.cotracker_visualizer import Visualizer
from data_utils.iphone_helpers import filter_iphone_multiview_manifest


def RGB2INST(x):
    assert x.shape[-1] == 3
    y = x.astype(np.int32)
    y = y[..., 0] + y[..., 1] * 256 + y[..., 2] * 256**2
    return y


@torch.no_grad()
def gather_track_from_buffer(track, track_base_mask, buffer_list):
    # buffer_list: T, H, W, C
    if buffer_list.ndim == 3:
        buffer_list = buffer_list.unsqueeze(3)
    T, N = track_base_mask.shape
    # fill the ret with nan
    ret = torch.zeros(T, N, buffer_list.shape[-1], dtype=buffer_list.dtype)
    ret = ret.to(track.device)
    ret.fill_(np.nan)
    ret_mask = torch.zeros_like(track_base_mask) > 0
    for tid in range(T):
        _mask = track_base_mask[tid]
        _uv_int = track[tid][_mask]
        _value = query_image_buffer_by_pix_int_coord(buffer_list[tid], _uv_int)
        ret[tid][_mask] = _value
        ret_mask[tid][_mask] = True
    return ret, ret_mask


@torch.no_grad()
def filter_track(track, track_vis, dep_mask_list, min_valid_cnt=4):
    # query valid depth and visible mask
    track_dep_mask, _ = gather_track_from_buffer(
        track[..., :2].long(), track_vis, dep_mask_list
    )
    track_mask = track_dep_mask.squeeze(-1) * track_vis
    # check whether a track is visible and has valid depth more more thant min_valid_cnt times
    valid_cnt = track_mask.sum(0)
    filter_track_mask = valid_cnt >= min_valid_cnt
    logging.info(
        f"Valid check: min_cnt={min_valid_cnt} {(~filter_track_mask).sum()} tracks are removed!"
    )
    track = track[:, filter_track_mask]
    track_mask = track_mask[:, filter_track_mask].clone()
    return track, track_mask


@torch.no_grad()
def load_track_buffers(
    track_root,
    track_npz_key,
    H,
    W,
    dep_mask_list=None,
    min_valid_cnt=4,
):
    long_track_fns = sorted(glob(osp.join(track_root, f"*{track_npz_key}*.npz")))
    logging.info(f"Loading TAP from {long_track_fns}...")
    if len(long_track_fns) == 0:
        raise FileNotFoundError(
            f"No TAP priors matching *{track_npz_key}*.npz found in {track_root}"
        )

    track, track_mask = [], []
    for track_fn in long_track_fns:
        track_data = np.load(track_fn, allow_pickle=True)
        track.append(track_data["tracks"])
        track_mask.append(track_data["visibility"])

    track = torch.from_numpy(np.concatenate(track, 1)).float()
    track[:, :, :2] = track[:, :, :2].long().float()
    track_mask = torch.from_numpy(np.concatenate(track_mask, 1)).bool()
    track_mask = track_mask * (track[..., 0] >= 0) * (track[..., 1] >= 0)
    track_mask = track_mask * (track[..., 0] < W) * (track[..., 1] < H)
    assert track.shape[:2] == track_mask.shape

    if dep_mask_list is not None and min_valid_cnt > 0:
        track, track_mask = filter_track(
            track,
            track_mask,
            dep_mask_list.to(track.device),
            min_valid_cnt=min_valid_cnt,
        )

    return track.float(), track_mask.bool()


def laplacian_filter_depth(depths, threshold_ratio=0.5, ksize=5, open_ksize=3):
    # logging.info("Filtering depth maps...")
    # filter the depth changing boundary, they are not reliable
    dep_boundary_errors, dep_valid_masks = [], []
    ellip_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
    )
    for dep in depths:
        # detect the edge boundary of depth
        dep = dep.astype(np.float32)
        # ! to handle different scale, the threshold should be adaptive
        threshold = np.median(dep) * threshold_ratio
        mask, error = detect_depth_occlusion_boundaries(dep, threshold, ksize)
        mask = mask > 0.5
        mask = ~mask  # valid mask
        # ! do a morph operator to remove outliers
        mask_opened = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, ellip_kernel
        )
        mask_opened = mask_opened > 0
        # mask_opened = mask
        dep_valid_masks.append(mask_opened)
        dep_boundary_errors.append(error)
    dep_valid_masks = np.stack(dep_valid_masks, axis=0)
    dep_boundary_errors = np.stack(dep_boundary_errors, axis=0)
    return dep_valid_masks, dep_boundary_errors


def detect_depth_occlusion_boundaries(depth_map, threshold=10, ksize=5):
    error = cv2.Laplacian(depth_map, cv2.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv2.threshold(error, threshold, 255, cv2.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error


def get_homo_coordinate_map(H, W):
    # the grid take the short side has (-1,+1)
    if H > W:
        u_range = [-1.0, 1.0]
        v_range = [-float(H) / W, float(H) / W]
    else:  # H<=W
        u_range = [-float(W) / H, float(W) / H]
        v_range = [-1.0, 1.0]
    # make uv coordinate
    u, v = np.meshgrid(np.linspace(*u_range, W), np.linspace(*v_range, H))
    uv = np.stack([u, v], axis=-1)  # H,W,2
    return uv


def query_image_buffer_by_pix_int_coord(buffer, pixel_int_coordinate):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    if isinstance(ret, np.ndarray):
        ret = ret.copy()
    return ret


def visualize_track(save_fn, tracks, visibility, video_pt, max_viz_cnt=512):
    # video_pt (T,3,H,W)
    # (T,N,2)
    # (T,N)
    video_pt = torch.as_tensor(video_pt)[None]
    tracks = torch.as_tensor(tracks)[None]
    visibility = torch.as_tensor(visibility)[None]
    assert video_pt.ndim == 5 and video_pt.shape[0] == 1
    assert tracks.ndim == 4 and tracks.shape[0] == 1
    assert visibility.ndim == 3 and visibility.shape[0] == 1
    _step = max(1, tracks.shape[2] // max_viz_cnt)
    viz_choice = torch.arange(0, tracks.shape[2], _step)
    vis = Visualizer(
        save_dir=osp.dirname(save_fn),
        linewidth=2,
        draw_invisible=True,  # False
        tracks_leave_trace=4,
    )
    vis.visualize(
        video=video_pt,
        tracks=tracks[:, :, viz_choice, :2],
        visibility=visibility[:, :, viz_choice],
        filename=osp.basename(save_fn),
    )


def load_rgb_buffer_from_dir(img_dir):
    img_npz = img_dir + ".npz"
    if osp.exists(img_npz):
        images = np.load(img_npz)["images"]  # ! in [0,255]
        images = torch.from_numpy(images).float() / 255.0  # T,H,W,3
        img_names = [f"{i:05d}" for i in range(images.shape[0])]
    elif osp.exists(img_dir):
        img_fns = [
            f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")
        ]
        img_fns.sort()
        img_names = [osp.splitext(f)[0] for f in img_fns]
        images = [imageio.imread(osp.join(img_dir, img_fn)) for img_fn in img_fns]
        images = torch.Tensor(np.stack(images)) / 255.0  # T,H,W,3
    else:
        raise ValueError(f"Cannot find images in {img_dir}")
    return images[..., :3], img_names


def load_binary_mask_buffer_from_dir(mask_dir):
    if not osp.exists(mask_dir):
        raise ValueError(f"Cannot find masks in {mask_dir}")
    mask_fns = [
        f for f in os.listdir(mask_dir) if f.endswith(".jpg") or f.endswith(".png")
    ]
    mask_fns.sort()
    if len(mask_fns) == 0:
        raise ValueError(f"No mask images found in {mask_dir}")
    mask_names = [osp.splitext(f)[0] for f in mask_fns]
    masks = []
    for mask_fn in mask_fns:
        mask = imageio.imread(osp.join(mask_dir, mask_fn))
        if mask.ndim == 3:
            mask = mask[..., :3].sum(-1) > 0
        else:
            mask = mask > 0
        masks.append(mask)
    masks = torch.from_numpy(np.stack(masks)).bool()
    return masks, mask_names


def load_depth_buffer_from_dir(
    depth_dir,
    frame_names=None,
    depth_npz_key="dep",
    depth_min=1e-3,
    depth_max=1000.0,
):
    if osp.isdir(depth_dir):
        if frame_names is None:
            depth_fns = [f for f in os.listdir(depth_dir) if f.endswith(".npz")]
            depth_fns.sort()
            frame_names = [osp.splitext(f)[0] for f in depth_fns]
        dep = [
            np.load(osp.join(depth_dir, f"{frame_name}.npz"))[depth_npz_key]
            for frame_name in frame_names
        ]
        dep = np.stack(dep, axis=0)
    else:
        depth_fn = depth_dir + ".npz"
        if not osp.exists(depth_fn):
            raise ValueError(f"Cannot find depth in {depth_dir} or {depth_fn}")
        dep = np.load(depth_fn)[depth_npz_key]
    dep = np.clip(dep, depth_min, depth_max)
    return torch.from_numpy(dep).float()


class MultiViewObservations(nn.Module):
    def __init__(
        self,
        ws,
        manifest_name="multiview_manifest.json",
        selected_cameras=None,
    ) -> None:
        super().__init__()
        self.ws = ws
        manifest_path = osp.join(ws, manifest_name)
        if not osp.exists(manifest_path):
            raise FileNotFoundError(f"Cannot find multiview manifest: {manifest_path}")
        with open(manifest_path, "r") as f:
            self.manifest = filter_iphone_multiview_manifest(
                json.load(f),
                selected_cameras=selected_cameras,
            )

        self.anchor_camera = self.manifest["anchor_camera"]
        self.anchor_rgb_camera = self.manifest.get(
            "anchor_rgb_camera", self.anchor_camera
        )
        self.override_anchor_rgb = bool(self.manifest.get("override_anchor_rgb", False))
        self.camera_names = [it["camera_name"] for it in self.manifest["cameras"]]
        self.camera_name_to_index = {
            name: i for i, name in enumerate(self.camera_names)
        }

        frame_names_by_camera = {}
        camera_hw = {}
        camera_weights = []
        self.rgb_buffer_names = {}
        self.mask_buffer_names = {}
        self.depth_buffer_names = {}
        self.depth_mask_buffer_names = {}
        self.tap_prior_dirs = {}
        self.track_buffer_names = {}
        self.track_mask_buffer_names = {}
        self.camera_has_explicit_mask = {}
        for camera in self.manifest["cameras"]:
            camera_name = camera["camera_name"]
            rgb_dir = osp.join(ws, camera["rgb_dir"])
            rgb, frame_names = load_rgb_buffer_from_dir(rgb_dir)
            frame_names_by_camera[camera_name] = frame_names
            camera_hw[camera_name] = tuple(rgb.shape[1:3])
            rgb_buffer_name = f"rgb_{camera_name.replace('-', '_')}"
            self.rgb_buffer_names[camera_name] = rgb_buffer_name
            self.register_gradfree_buffer(rgb_buffer_name, rgb.float())

            mask_dir = camera.get("mask_dir")
            has_explicit_mask = False
            if mask_dir is not None:
                mask_dir_abs = osp.join(ws, mask_dir)
                if osp.isdir(mask_dir_abs):
                    has_explicit_mask = any(
                        f.endswith(".jpg") or f.endswith(".png")
                        for f in os.listdir(mask_dir_abs)
                    )
            if has_explicit_mask:
                masks, mask_names = load_binary_mask_buffer_from_dir(
                    osp.join(ws, mask_dir)
                )
                assert (
                    mask_names == frame_names
                ), f"Mask/image names mismatch for {camera_name}"
            else:
                masks = torch.ones(rgb.shape[:3], dtype=torch.bool)
            self.camera_has_explicit_mask[camera_name] = bool(has_explicit_mask)
            mask_buffer_name = f"mask_{camera_name.replace('-', '_')}"
            self.mask_buffer_names[camera_name] = mask_buffer_name
            self.register_gradfree_buffer(mask_buffer_name, masks.bool())
            camera_weights.append(float(camera.get("weight", 1.0)))

            depth_dir = camera.get("depth_dir")
            if depth_dir is None:
                derived_depth_dir = osp.join("DA3_depth", camera_name)
                derived_depth_path = osp.join(ws, derived_depth_dir)
                if osp.isdir(derived_depth_path) or osp.exists(derived_depth_path + ".npz"):
                    depth_dir = derived_depth_dir
            if depth_dir is not None:
                depth_path = osp.join(ws, depth_dir)
                if osp.isdir(depth_path) or osp.exists(depth_path + ".npz"):
                    depth_npz_key = camera.get(
                        "depth_npz_key", self.manifest.get("depth_npz_key", "dep")
                    )
                    depth_min = float(camera.get("depth_min", 1e-3))
                    depth_max = float(camera.get("depth_max", 1000.0))
                    depth = load_depth_buffer_from_dir(
                        depth_path,
                        frame_names=frame_names,
                        depth_npz_key=depth_npz_key,
                        depth_min=depth_min,
                        depth_max=depth_max,
                    )
                    depth_mask, _ = laplacian_filter_depth(
                        depth.cpu().numpy(),
                        float(
                            camera.get(
                                "depth_boundary_th",
                                self.manifest.get("multiview_depth_boundary_th", 1.0),
                            )
                        ),
                        5,
                    )
                    depth_mask = (
                        depth_mask
                        * (depth.cpu().numpy() > depth_min)
                        * (depth.cpu().numpy() < depth_max)
                    )
                    depth_mask_dir = camera.get("depth_mask_dir")
                    if depth_mask_dir is not None and osp.exists(osp.join(ws, depth_mask_dir)):
                        loaded_depth_mask, depth_mask_names = load_binary_mask_buffer_from_dir(
                            osp.join(ws, depth_mask_dir)
                        )
                        assert (
                            depth_mask_names == frame_names
                        ), f"Depth-mask/image names mismatch for {camera_name}"
                        depth_mask = depth_mask * loaded_depth_mask.cpu().numpy()
                    depth_buffer_name = f"dep_{camera_name.replace('-', '_')}"
                    depth_mask_buffer_name = (
                        f"dep_mask_{camera_name.replace('-', '_')}"
                    )
                    self.depth_buffer_names[camera_name] = depth_buffer_name
                    self.depth_mask_buffer_names[camera_name] = depth_mask_buffer_name
                    self.register_gradfree_buffer(depth_buffer_name, depth.float())
                    self.register_gradfree_buffer(
                        depth_mask_buffer_name, torch.from_numpy(depth_mask).bool()
                    )

            tap_prior_dir = camera.get("tap_prior_dir")
            if tap_prior_dir is None:
                derived_tap_dir = osp.join("multiview", "tap_priors", camera_name)
                if osp.isdir(osp.join(ws, derived_tap_dir)):
                    tap_prior_dir = derived_tap_dir
            elif not osp.isdir(osp.join(ws, tap_prior_dir)):
                tap_prior_dir = None
            if tap_prior_dir is not None:
                self.tap_prior_dirs[camera_name] = tap_prior_dir

        T = getattr(self, self.rgb_buffer_names[self.camera_names[0]]).shape[0]
        for camera_name in self.camera_names[1:]:
            camera_rgb = getattr(self, self.rgb_buffer_names[camera_name])
            assert camera_rgb.shape[0] == T, (
                f"Camera {camera_name} has {camera_rgb.shape[0]} frames, expected {T}"
            )
        for camera_name in self.camera_names:
            camera_rgb = getattr(self, self.rgb_buffer_names[camera_name])
            camera_mask = getattr(self, self.mask_buffer_names[camera_name])
            assert camera_mask.shape == camera_rgb.shape[:3], (
                f"Mask shape mismatch for {camera_name}: "
                f"{camera_mask.shape} vs {camera_rgb.shape[:3]}"
            )
            if camera_name in self.depth_buffer_names:
                camera_dep = getattr(self, self.depth_buffer_names[camera_name])
                camera_dep_mask = getattr(self, self.depth_mask_buffer_names[camera_name])
                assert camera_dep.shape == camera_rgb.shape[:3], (
                    f"Depth shape mismatch for {camera_name}: "
                    f"{camera_dep.shape} vs {camera_rgb.shape[:3]}"
                )
                assert camera_dep_mask.shape == camera_dep.shape, (
                    f"Depth-mask shape mismatch for {camera_name}: "
                    f"{camera_dep_mask.shape} vs {camera_dep.shape}"
                )

        self.frame_names_by_camera = frame_names_by_camera
        self.camera_hw = camera_hw
        self.register_gradfree_buffer(
            "camera_weights", torch.tensor(camera_weights).float()
        )

    def register_gradfree_buffer(self, name, buffer):
        buf = torch.as_tensor(buffer).detach().contiguous()
        buf.requires_grad = False
        self.register_buffer(name, buf)
        return

    @property
    def T(self):
        return getattr(self, self.rgb_buffer_names[self.anchor_rgb_camera]).shape[0]

    @property
    def H(self):
        return self.camera_hw[self.anchor_rgb_camera][0]

    @property
    def W(self):
        return self.camera_hw[self.anchor_rgb_camera][1]

    @property
    def V(self):
        return len(self.camera_names)

    @property
    def auxiliary_cameras(self):
        return [it for it in self.camera_names if it != self.anchor_camera]

    def validate_anchor_frame_names(self, frame_names):
        anchor_names = self.frame_names_by_camera[self.anchor_rgb_camera]
        if list(frame_names) != list(anchor_names):
            raise ValueError(
                f"Anchor frame names mismatch: {frame_names[:3]} vs {anchor_names[:3]}"
            )
        return self

    def get_camera_index(self, camera_name):
        if isinstance(camera_name, int):
            return camera_name
        if camera_name not in self.camera_name_to_index:
            raise KeyError(f"Unknown multiview camera: {camera_name}")
        return self.camera_name_to_index[camera_name]

    def get_camera_name(self, camera_name):
        return self.camera_names[camera_name] if isinstance(camera_name, int) else camera_name

    def get_rgb(self, camera_name, frame_ind):
        camera_name = self.get_camera_name(camera_name)
        return getattr(self, self.rgb_buffer_names[camera_name])[frame_ind]

    def get_hw(self, camera_name):
        camera_name = self.get_camera_name(camera_name)
        return self.camera_hw[camera_name]

    def get_mask(self, camera_name, frame_ind, use_mask=True):
        camera_name = self.get_camera_name(camera_name)
        if not use_mask:
            return torch.ones_like(
                getattr(self, self.mask_buffer_names[camera_name])[frame_ind]
            )
        return getattr(self, self.mask_buffer_names[camera_name])[frame_ind]

    def has_explicit_mask(self, camera_name):
        camera_name = self.get_camera_name(camera_name)
        return bool(self.camera_has_explicit_mask.get(camera_name, False))

    def has_depth(self, camera_name):
        camera_name = self.get_camera_name(camera_name)
        return camera_name in self.depth_buffer_names

    def get_depth(self, camera_name, frame_ind):
        camera_name = self.get_camera_name(camera_name)
        if camera_name not in self.depth_buffer_names:
            raise KeyError(f"No multiview depth loaded for {camera_name}")
        return getattr(self, self.depth_buffer_names[camera_name])[frame_ind]

    def get_depth_mask(self, camera_name, frame_ind, use_mask=True):
        camera_name = self.get_camera_name(camera_name)
        if camera_name not in self.depth_mask_buffer_names:
            raise KeyError(f"No multiview depth mask loaded for {camera_name}")
        if not use_mask:
            return torch.ones_like(
                getattr(self, self.depth_mask_buffer_names[camera_name])[frame_ind]
            )
        return getattr(self, self.depth_mask_buffer_names[camera_name])[frame_ind]

    def get_tap_prior_dir(self, camera_name):
        camera_name = self.get_camera_name(camera_name)
        if camera_name not in self.tap_prior_dirs:
            raise KeyError(f"No multiview TAP prior dir configured for {camera_name}")
        return self.tap_prior_dirs[camera_name]

    @torch.no_grad()
    def load_tap_track(
        self,
        camera_name,
        track_npz_key="spatracker",
        min_valid_cnt=4,
        require_depth=False,
    ):
        camera_name = self.get_camera_name(camera_name)
        if camera_name == self.anchor_camera:
            raise ValueError(
                "Anchor-camera tracks should be loaded from the root Saved2D workspace."
            )
        if require_depth and not self.has_depth(camera_name):
            raise KeyError(f"No multiview depth loaded for {camera_name}")

        H, W = self.get_hw(camera_name)
        if self.has_depth(camera_name):
            dep_mask = getattr(self, self.depth_mask_buffer_names[camera_name]).detach()
        else:
            dep_mask = torch.ones((self.T, H, W), dtype=torch.bool)

        track_root = osp.join(self.ws, self.get_tap_prior_dir(camera_name))
        track, track_mask = load_track_buffers(
            track_root=track_root,
            track_npz_key=track_npz_key,
            H=H,
            W=W,
            dep_mask_list=dep_mask,
            min_valid_cnt=min_valid_cnt,
        )
        target_device = getattr(self, self.rgb_buffer_names[camera_name]).device

        track_buffer_name = f"track_{camera_name.replace('-', '_')}"
        track_mask_buffer_name = f"track_mask_{camera_name.replace('-', '_')}"
        self.track_buffer_names[camera_name] = track_buffer_name
        self.track_mask_buffer_names[camera_name] = track_mask_buffer_name
        self.register_gradfree_buffer(track_buffer_name, track.to(target_device))
        self.register_gradfree_buffer(track_mask_buffer_name, track_mask.to(target_device))
        return self

    @torch.no_grad()
    def load_tap_tracks(
        self,
        track_npz_key="spatracker",
        min_valid_cnt=4,
        camera_names=None,
        require_depth=False,
    ):
        if camera_names is None:
            camera_names = self.auxiliary_cameras
        for camera_name in camera_names:
            self.load_tap_track(
                camera_name=camera_name,
                track_npz_key=track_npz_key,
                min_valid_cnt=min_valid_cnt,
                require_depth=require_depth,
            )
        return self

    def has_track(self, camera_name):
        camera_name = self.get_camera_name(camera_name)
        return camera_name in self.track_buffer_names

    def get_track(self, camera_name):
        camera_name = self.get_camera_name(camera_name)
        if camera_name not in self.track_buffer_names:
            raise KeyError(f"No multiview track prior loaded for {camera_name}")
        return getattr(self, self.track_buffer_names[camera_name])

    def get_track_mask(self, camera_name):
        camera_name = self.get_camera_name(camera_name)
        if camera_name not in self.track_mask_buffer_names:
            raise KeyError(f"No multiview track mask loaded for {camera_name}")
        return getattr(self, self.track_mask_buffer_names[camera_name])

    @property
    def track_cameras(self):
        return list(self.track_buffer_names.keys())

    def get_weight(self, camera_name):
        return self.camera_weights[self.get_camera_index(camera_name)]

    @torch.no_grad()
    def override_camera_weights(self, weight_dict):
        if not weight_dict:
            return self
        updated = self.camera_weights.clone()
        for camera_name, weight in weight_dict.items():
            updated[self.get_camera_index(camera_name)] = float(weight)
        self.camera_weights.copy_(updated)
        return self


class Saved2D(nn.Module):
    def __init__(self, ws, rgb_dirname="images") -> None:
        super().__init__()
        self.ws = ws

        # this init only load the RGBs and build basic attrs
        self.load_rgb(rgb_dirname)

        # * some old attr, not sure whether useful
        homo_map = torch.from_numpy(get_homo_coordinate_map(H=self.H, W=self.W))
        self.register_gradfree_buffer("homo_map", homo_map.float().detach())
        pixel_int_map = torch.from_numpy(
            np.stack(np.meshgrid(np.arange(self.W), np.arange(self.H)), -1)
        ).int()
        self.register_gradfree_buffer("pixel_int_map", pixel_int_map.detach())

        # * other flags
        self.has_epi = False
        self.has_vos = False
        return

    def register_gradfree_buffer(self, name, buffer):
        buf = torch.as_tensor(buffer).detach().contiguous()
        buf.requires_grad = False
        self.register_buffer(name, buf)
        return

    @torch.no_grad
    def load_rgb(self, rgb_dirname="images"):
        img_dir = osp.join(self.ws, rgb_dirname)
        images, img_names = load_rgb_buffer_from_dir(img_dir)
        # assign
        self.frame_names = img_names
        self.register_gradfree_buffer("rgb", images.detach())
        return self

    @property
    def T(self):
        return self.rgb.shape[0]

    @property
    def H(self):
        return self.rgb.shape[1]

    @property
    def W(self):
        return self.rgb.shape[2]

    # load more data
    # todo: register the dyn mask on 2D

    def load_epi(self, epi_dirname="epi"):
        epi_dir = osp.join(self.ws, epi_dirname)
        if not osp.exists(epi_dir):
            logging.warning(
                f"Calling 2D EPI loading, but {epi_dir} not found, usually this means use track epi not optical flow epi, so skip!"
            )
            return self
        epipolar_errs = load_epi_error(osp.join(self.ws, epi_dirname))
        epipolar_errs = torch.Tensor(np.stack(epipolar_errs)).float()  # T,H,W
        self.register_gradfree_buffer("epi", epipolar_errs)
        self.has_epi = True
        return self

    @torch.no_grad()
    def load_dep(
        self,
        depth_dirname="depth_metric3d",
        depth_boundary_th=0.1,
        depth_min=1e-3,
        depth_max=1000.0,
        mask_depth_flag=True,
    ):
        assert hasattr(self, "frame_names"), "Load RGB before DEP!"
        dep_dir = osp.join(self.ws, depth_dirname)
        if osp.isdir(dep_dir):
            dep = [
                np.load(osp.join(dep_dir, f"{img_name}.npz"))["dep"]
                for img_name in self.frame_names
            ]
            dep = np.stack(dep)  # T,H,W
        else:
            dep_fn = dep_dir + ".npz"
            assert osp.exists(dep_fn)
            dep = np.load(dep_fn)["dep"]
        if mask_depth_flag:
            dep_mask, _ = laplacian_filter_depth(dep, depth_boundary_th, 5)
            logging.info(f"Dep Boundary Mask {dep_mask.mean()*100:.2f}%")
            dep_mask = dep_mask * (dep > depth_min) * (dep < depth_max)
            dep = np.clip(dep, depth_min, depth_max)
        else:
            dep_mask = np.ones_like(dep) > 0

        # assign
        self.register_gradfree_buffer("dep", torch.Tensor(dep))
        self.register_gradfree_buffer("dep_mask", torch.Tensor(dep_mask).bool())
        return self

    @torch.no_grad()
    def replace_depth(self, dep, dep_mask):
        if hasattr(self, "dep"):
            assert dep.shape == self.dep.shape, f"{dep.shape} vs {self.dep.shape}"
            assert (
                dep_mask.shape == self.dep_mask.shape
            ), f"{dep_mask.shape} vs {self.dep_mask.shape}"
        self.register_gradfree_buffer("dep", dep)
        self.register_gradfree_buffer("dep_mask", dep_mask)
        return self

    @torch.no_grad()
    def recompute_dep_mask(
        self, depth_boundary_th=0.1, depth_min=1e-3, depth_max=1000.0
    ):
        dep = self.dep.cpu().numpy()
        dep_mask, _ = laplacian_filter_depth(dep, depth_boundary_th, 5)
        logging.info(f"Dep Boundary Mask {dep_mask.mean()*100:.2f}%")
        dep_mask = dep_mask * (dep > depth_min) * (dep < depth_max)
        # dep[~dep_mask] = np.inf
        dep = np.clip(dep, depth_min, depth_max)
        self.register_gradfree_buffer("dep", torch.Tensor(dep))
        self.register_gradfree_buffer("dep_mask", torch.Tensor(dep_mask).bool())
        return self

    @torch.no_grad()
    def load_track(self, track_npz_key="spatracker", min_valid_cnt=4):
        assert hasattr(self, "dep_mask"), "Load DEP before TAP!"
        track, track_mask = load_track_buffers(
            track_root=self.ws,
            track_npz_key=track_npz_key,
            H=self.H,
            W=self.W,
            dep_mask_list=self.dep_mask,
            min_valid_cnt=min_valid_cnt,
        )
        assert len(track) == self.T

        # append the tracks
        if hasattr(self, "track"):
            self.track = torch.cat([self.track, track.to(self.track.device)], 1)
            self.track_mask = torch.cat(
                [self.track_mask, track_mask.to(self.track_mask.device)], 1
            )
        else:
            self.register_gradfree_buffer("track", track.float())  # 2D/3D track
        self.register_gradfree_buffer("track_mask", track_mask)
        # ! warning, the track static mask is not saved here
        # assert (
        #     self.track.dtype == torch.long
        # ), "Must use Long type for track! to avoid later roudning error, otherwise the system will fail."

        # re-scale the 3rd depth if the depth is rescaled
        if self.track.shape[-1] == 3 and hasattr(self, "scale_nw"):
            logging.info(f"Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * self.scale_nw

        return self

    @torch.no_grad()
    def load_vos(self, vos_dirname="vos_deva/Annotations"):
        vos_dir = osp.join(self.ws, vos_dirname)
        if not osp.exists(vos_dir):
            logging.warning(
                f"Calling 2D VOS loading, but {vos_dir} not found, usually this means the data is not available, so skip!"
            )
            return self
        logging.info(f"loading vos results from {vos_dirname}...")
        id_mask_list = []
        fn_list = sorted(os.listdir(vos_dir))
        for fn in fn_list:
            seg_fn = osp.join(vos_dir, fn)
            seg = imageio.imread(seg_fn)
            id_map = RGB2INST(seg)
            id_mask_list.append(id_map)
        id_mask_list = np.stack(id_mask_list, 0)
        unique_id = np.unique(id_mask_list)
        # remove 0 from unique id, which is unknown
        unique_id = unique_id[unique_id != 0]
        logging.info(
            f"loaded {len(unique_id)} unique ids with {len(id_mask_list)} frames."
        )
        self.register_gradfree_buffer("vos", torch.from_numpy(id_mask_list))
        self.register_gradfree_buffer("vos_id", torch.from_numpy(unique_id))
        self.has_vos = True
        return self

    @torch.no_grad()
    def load_flow(self, flow_dirname="flow_raft"):
        flow_list, flow_mask_list, src_t_list, dst_t_list = [], [], [], []
        flow_ij_to_listind_dict = {}
        flow_dir = osp.join(self.ws, flow_dirname)
        if not osp.exists(flow_dir):
            logging.warning(
                f"Calling 2D Flow loading, but {flow_dir} not found, usually this means the data is not available, so skip!"
            )
            return self
        for fn in tqdm(sorted(os.listdir(flow_dir))):
            if not fn.endswith(".npz"):
                continue
            flow_fn = osp.join(flow_dir, fn)
            flow_data = np.load(flow_fn, allow_pickle=True)
            flow, mask = flow_data["flow"], flow_data["mask"]
            src_t, dst_t = fn[:-4].split("_to_")
            src_t = self.frame_names.index(src_t[:-4])
            dst_t = self.frame_names.index(dst_t[:-4])
            flow_list.append(flow)
            flow_mask_list.append(mask)
            src_t_list.append(src_t)
            dst_t_list.append(dst_t)
            flow_ij_to_listind_dict[(src_t, dst_t)] = len(flow_list) - 1
        flow_list = np.stack(flow_list, 0)
        flow_mask_list = np.stack(flow_mask_list, 0)
        src_t_list = np.array(src_t_list)
        dst_t_list = np.array(dst_t_list)
        self.register_gradfree_buffer("flow", torch.from_numpy(flow_list))
        self.register_gradfree_buffer("flow_mask", torch.from_numpy(flow_mask_list))
        self.register_gradfree_buffer("flow_src_t", torch.from_numpy(src_t_list))
        self.register_gradfree_buffer("flow_dst_t", torch.from_numpy(dst_t_list))
        self.flow_ij_to_listind_dict = flow_ij_to_listind_dict
        return self

    @torch.no_grad()
    def register_2d_identification(self, static_2d_mask, dynamic_2d_mask):
        device = self.rgb.device
        self.register_gradfree_buffer("sta_mask", static_2d_mask.float().to(device))
        self.register_gradfree_buffer("dyn_mask", dynamic_2d_mask.float().to(device))
        logging.info(
            f"Saved2d register 2d identification: {static_2d_mask.sum()} static, {dynamic_2d_mask.sum()} dynamic; Unused: {(~static_2d_mask & ~dynamic_2d_mask).sum()}"
        )
        return

    @torch.no_grad()
    def register_track_indentification(self, static_track_mask, dynamic_track_mask):
        assert hasattr(self, "track"), "Load track before identify!"
        assert len(static_track_mask) == self.track.shape[1]
        device = self.track.device
        self.register_gradfree_buffer(
            "static_track_mask", static_track_mask.bool().to(device)
        )
        self.register_gradfree_buffer(
            "dynamic_track_mask", dynamic_track_mask.bool().to(device)
        )
        logging.warning(
            f"Saved2d register track identification: {static_track_mask.sum()} static, {dynamic_track_mask.sum()} dynamic; Unused: {(~static_track_mask & ~dynamic_track_mask).sum()}"
        )
        return self

    @torch.no_grad()
    def get_mask_by_key(self, key):
        if key == "all":
            return torch.ones_like(self.dep_mask)
        elif key == "static":
            return self.sta_mask
        elif key == "dynamic":
            return self.dyn_mask
        elif key == "dep":
            return self.dep_mask
        else:
            raise ValueError(f"Unknown key={key}")

    @torch.no_grad()
    def rescale_depth(self, dep_scale):
        dep_scale = dep_scale.to(self.dep.device)
        assert len(dep_scale) == self.T, f"dep_scale:{len(dep_scale)} vs T:{self.T}"
        self.dep = self.dep.clone() * dep_scale[:, None, None]
        if self.track.shape[-1] == 3:
            logging.info(f"Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * dep_scale[:, None]
        return

    @torch.no_grad()
    def rescale_perframe_depth_from_bundle(self, bundle_pth_fn=None):
        if bundle_pth_fn is None:
            bundle_pth_fn = osp.join(self.ws, "bundle", "bundle.pth")
        bundle_data = torch.load(bundle_pth_fn)
        dep_scale = bundle_data["dep_scale"]
        logging.info(f"Rescale depth with {bundle_pth_fn} and scale={dep_scale}")
        self.rescale_depth(dep_scale)
        return self

    @torch.no_grad()
    def normalize_depth(self, median_depth: float = 1.0):
        # align the median of the depth to 1.0
        if median_depth < 0:
            scale_nw = 1.0
        else:
            world_depth = self.dep.clone()
            world_depth_fg = world_depth[self.dep_mask]
            world_depth_median = torch.median(world_depth_fg)
            scale_nw = (
                median_depth / world_depth_median
            )  # depth_normalized = scale_nw * depth_world
        self.scale_nw = float(scale_nw)
        assert hasattr(self, "dep_mask"), "Load DEP before noramlization!"
        # assert hasattr(self, "track"), "Load TAP before noramlization!"

        self.dep = self.dep.clone() * scale_nw
        if hasattr(self, "track") and self.track.shape[-1] == 3:
            logging.info(f"Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * scale_nw
        return self


if __name__ == "__main__":
    src = "../data/debug/C2_N11_S212_s03_T2/"

    s2d = Saved2D(src)
