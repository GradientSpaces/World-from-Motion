import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os, sys, os.path as osp
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import logging


class MonocularCameras(nn.Module):
    def __init__(
        self,
        n_time_steps,
        default_H,
        default_W,
        fxfycxcy: list = None,  # intr init 1, fxfy in deg, cxcy in ratio [53.1, 53.1, 0.5, 0.5]
        K=None,  # intr init 2, K is 3x3 mat
        #
        delta_flag=True,
        init_camera_pose=None,  # either T_wc in indep model; or T_i(i+1) in delta model
        # * cam flag
        iso_focal=False,
    ) -> None:
        super().__init__()

        self.T = n_time_steps
        self.register_buffer("delta_flag", torch.tensor(delta_flag))
        self.register_buffer("iso_focal", torch.tensor(iso_focal))

        rel_focal, cxcy_ratio = self.__get_init_fc__(
            fxfycxcy, None if K is None else [K, default_H, default_W]
        )
        self._rel_focal = nn.Parameter(rel_focal)  # ! both rel to short side=2
        self.cxcy_ratio = nn.Parameter(cxcy_ratio)  # ! separate ratio for H=1, W=1

        param_cam_q, param_cam_t = self.__get_init_qt__(init_camera_pose)

        self.q_wc = nn.Parameter(param_cam_q)
        self.t_wc = nn.Parameter(param_cam_t)

        assert self.q_wc.isnan().any() == False and self.t_wc.isnan().any() == False

        self.register_buffer("default_H", torch.tensor(default_H))
        self.register_buffer("default_W", torch.tensor(default_W))
        self.summary()
        return

    @property
    def rel_focal(self):
        if self.iso_focal:
            return self._rel_focal[0].repeat(2)
        else:
            return self._rel_focal

    def __len__(self):
        return self.T

    @classmethod
    def load_from_ckpt(cls, ckpt):
        logging.info("Load camera from checkpoint")
        H = ckpt["default_H"]
        W = ckpt["default_W"]
        delta_flag = ckpt["delta_flag"]
        if "iso_focal" not in ckpt.keys():
            logging.warning(
                f"Load the old ckpt, by default in the old version, iso-focal is False"
            )
            ckpt["iso_focal"] = torch.tensor(False)
        if "_rel_focal" not in ckpt.keys():
            logging.warning(
                f"Load the old ckpt, use rel_focal instead of _rel_focal"
            )
            ckpt["_rel_focal"] = ckpt["rel_focal"]
            del ckpt["rel_focal"]
        T = len(ckpt["q_wc"])
        cams = cls(n_time_steps=T, default_H=H, default_W=W, delta_flag=delta_flag)
        cams.load_state_dict(ckpt, strict=True)
        return cams

    def summary(self):
        logging.info(
            f"MonoCam Summary: T={self.T}, delta={self.delta_flag}, H={self.default_H}, W={self.default_W}"
        )

    def __get_init_fc__(self, fovxfovycxcy, KHW):
        if KHW is None:
            if fovxfovycxcy is None:
                fovxfovycxcy = [53.1, 53.1, 0.5, 0.5]
                logging.warning(
                    f"Both fxfycxcy and KHW are None, use default {fovxfovycxcy}"
                )
            rel_focal_x = 1.0 / np.tan(np.deg2rad(fovxfovycxcy[0]) / 2.0)
            rel_focal_y = 1.0 / np.tan(np.deg2rad(fovxfovycxcy[1]) / 2.0)
            rel_focal = torch.Tensor([rel_focal_x, rel_focal_y]).squeeze()
            cxcy_ratio = torch.Tensor([fovxfovycxcy[2], fovxfovycxcy[3]])
        else:
            assert fovxfovycxcy is None
            K, H, W = KHW
            H, W = float(H), float(W)
            L = min(H, W)
            rel_focal_x = K[0, 0] / L * 2.0
            rel_focal_y = K[1, 1] / L * 2.0
            cx_ratio = K[0, 2] / W
            cy_ratio = K[1, 2] / H
            rel_focal = torch.Tensor([rel_focal_x, rel_focal_y]).squeeze()
            cxcy_ratio = torch.Tensor([cx_ratio, cy_ratio])
        return rel_focal.float(), cxcy_ratio.float()

    def __get_init_qt__(self, init_camera_pose):
        if init_camera_pose is not None:
            if self.delta_flag:
                # T_w0, [T_01, T_12, ...]
                assert len(init_camera_pose) == self.T - 1
                delta_q0 = matrix_to_quaternion(torch.eye(3)[None])
                delta_q = matrix_to_quaternion(init_camera_pose[:, :3, :3])
                param_cam_q = torch.cat([delta_q0, delta_q], 0)
                delta_t0 = torch.zeros(3)[None]
                delta_t = init_camera_pose[:, :3, -1]
                param_cam_t = torch.cat([delta_t0, delta_t], 0)
            else:
                # construct independent: T_wc
                init_camera_pose = torch.as_tensor(init_camera_pose)
                param_cam_q = matrix_to_quaternion(init_camera_pose[:, :3, :3])
                param_cam_t = init_camera_pose[:, :3, -1]
        else:
            param_cam_q = torch.zeros(self.T, 4)
            param_cam_q[:, 0] = 1.0
            param_cam_t = torch.zeros(self.T, 3)
        return param_cam_q.float(), param_cam_t.float()

    ################################################################################
    # * Extrinsic
    ################################################################################
    def forward_T(self, until=None):
        # until is not included
        if until is None:
            until = self.T
        dR = quaternion_to_matrix(F.normalize(self.q_wc, dim=-1))
        dt = self.t_wc
        dT_list = torch.cat([dR, dt[..., None]], -1)
        bottom = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(dt)[None].expand(self.T, -1)
        dT_list = torch.cat([dT_list, bottom[:, None]], 1)
        ret = [dT_list[0]]  # ! the first frame can also be optimized
        for dT in dT_list[1:until]:
            # T = dT @ ret[-1] # old
            # ! different from the old version, here the order changed
            T = ret[-1] @ dT  # The saved is T[0] = T_w0, T[i] = T_(i-1)_i
            ret.append(T)
        ret = torch.stack(ret)
        return ret  # ->until,4,4

    def T_wc(self, ind):
        # * core function
        assert ind >= 0 and ind < self.T, f"Invalid index {ind} for T={self.T}"
        if self.delta_flag:
            T_list = self.forward_T(ind)
            assert len(T_list) == ind + 1
            return T_list[-1].float()
        else:
            R = quaternion_to_matrix(F.normalize(self.q_wc[ind : ind + 1], dim=-1))[0]
            t = self.t_wc[ind]
            T = torch.eye(4).to(R)
            T[:3, :3] = R
            T[:3, 3] = t
            return T.float()

    def T_cw(self, ind):
        T_wc = self.T_wc(ind)
        return torch.linalg.inv(T_wc)

    def Rt_wc(self, ind):
        T = self.T_wc(ind)
        return T[:3, :3], T[:3, -1]

    def Rt_cw(self, ind):
        R_wc, t_wc = self.Rt_wc(ind)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw

    def T_wc_list(self):
        # * core function
        if self.delta_flag:
            return self.forward_T().float()
        else:
            R = quaternion_to_matrix(F.normalize(self.q_wc, dim=-1))
            t = self.t_wc
            ret = torch.cat([R, t[..., None]], -1)
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(ret)
            ret = torch.cat([ret, bottom[None, None, :].expand(len(R), -1, -1)], -2)
            return ret.float()

    def T_cw_list(self):
        T_wc = self.T_wc_list()
        T_cw = torch.linalg.inv(T_wc)
        return T_cw

    def Rt_wc_list(self):
        T_wc_list = self.T_wc_list()
        R_wc = T_wc_list[:, :3, :3]
        t_wc = T_wc_list[:, :3, -1]
        return R_wc, t_wc

    def Rt_cw_list(self):
        R_wc, t_wc = self.Rt_wc_list()
        R_cw = R_wc.transpose(1, 2)
        t_cw = -torch.einsum("bij,bj->bi", R_cw, t_wc)
        return R_cw, t_cw

    def Rt_ij(self, i, j):
        T_wi = self.T_wc(i)
        T_wj = self.T_wc(j)
        T_ij = T_wi.inverse() @ T_wj
        R_ij = T_ij[:3, :3]
        t_ij = T_ij[:3, 3]
        return R_ij, t_ij

    @torch.no_grad()
    def disable_delta(self):
        if not self.delta_flag:
            logging.warning("Already in non-delta mode, skip")
            return

        T_wc_list = self.T_wc_list()
        param_cam_q = matrix_to_quaternion(T_wc_list[:, :3, :3])
        param_cam_t = T_wc_list[:, :3, -1]
        # ! directly modify ...
        self.q_wc.data = param_cam_q
        self.t_wc.data = param_cam_t
        self.delta_flag = ~self.delta_flag
        logging.info("Switch to independent mode")
        return

    ################################################################################
    # * Smoothness
    ################################################################################
    def smoothness_loss(
        self, t_list=None, shift_list=[1], weight_list=[1.0], square_flag=True
    ):
        # identify which frame to use
        if t_list is None:
            t_list = torch.arange(self.T).to(self.q_wc)
        else:
            t_list = torch.as_tensor(t_list).to(self.q_wc)
        t_list = t_list.sort()[0].long()
        T_wc_list = self.T_wc_list()
        T_wc_list = T_wc_list[t_list]
        assert len(shift_list) == len(weight_list)
        loss_rot, loss_trans = 0.0, 0.0
        for delta_t, w in zip(shift_list, weight_list):
            T_a = T_wc_list[:-delta_t]
            T_b = T_wc_list[delta_t:]
            T_a_inv = T_a.inverse()
            dT = torch.einsum("tij,tjk->tik", T_a_inv, T_b)
            dR = dT[:, :3, :3]
            ang_vel = matrix_to_axis_angle(dR).norm(dim=-1)
            transl_vel = dT[:, :3, -1].norm(dim=-1)
            if square_flag:
                ang_vel = ang_vel**2
                transl_vel = transl_vel**2
            loss_rot = loss_rot + w * ang_vel.sum()
            loss_trans = loss_trans + w * transl_vel.sum()
        return loss_rot, loss_trans

    ################################################################################
    # * Intrinsic
    ################################################################################
    def K(self, H=None, W=None):
        if H is None and W is None:
            return self.default_K
        else:
            assert H is not None and W is not None, "H and W must be both provided"
        L = min(H, W)  # ! the rel means to rel to the short side
        fx = self.rel_focal[0] * L / 2.0
        fy = self.rel_focal[1] * L / 2.0
        cx = W * self.cxcy_ratio[0]
        cy = H * self.cxcy_ratio[1]
        K = torch.eye(3).to(self.rel_focal)
        K[0, 0] = K[0, 0] * 0 + fx
        K[1, 1] = K[1, 1] * 0 + fy
        K[0, 2] = K[0, 2] * 0 + cx
        K[1, 2] = K[1, 2] * 0 + cy
        return K

    @property
    def default_K(self):
        return self.K(self.default_H, self.default_W)
    
    def homo(self, H=None, W=None):
        if H is None:
            H = int(self.default_H)
        if W is None:
            W = int(self.default_W)
        uv = __get_homo_coordinate_map__(H, W)
        uv = torch.from_numpy(uv).float().to(self.rel_focal.device)
        return uv

    @property
    def fov(self):
        focal = self.rel_focal.detach().cpu().numpy()
        half_angle = np.arctan(1.0 / focal)
        angle = np.rad2deg(half_angle * 2.0)
        return angle

    def get_optimizable_list(self, lr_q=1e-4, lr_t=1e-4, lr_f=1e-4, lr_c=0.0):
        ret = []
        if lr_q > 0:
            ret.append({"params": [self.q_wc], "lr": lr_q, "name": "R"})
        if lr_t > 0:
            ret.append({"params": [self.t_wc], "lr": lr_t, "name": "t"})
        if lr_f > 0:
            ret.append({"params": [self._rel_focal], "lr": lr_f, "name": "f"})
        if lr_c > 0:
            ret.append({"params": [self.cxcy_ratio], "lr": lr_c, "name": "cxcy"})
        return ret

    def backproject(self, uv, d):
        return __backproject__(uv, d, self)

    def project(self, xyz, th=1e-5):
        return __project__(xyz, self, th=th)

    def trans_pts_to_world(self, tid, pts_c):
        assert pts_c.shape[-1] == 3  # and pts_c.ndim == 2
        R, t = self.Rt_wc(tid)
        original_shape = pts_c.shape
        pts_w = torch.einsum("ij,nj->ni", R, pts_c.reshape(-1, 3)) + t
        return pts_w.reshape(*original_shape)

    def trans_pts_to_cam(self, tid, pts_w):
        assert pts_w.shape[-1] == 3  # and pts_w.ndim == 2
        R, t = self.Rt_cw(tid)
        original_shape = pts_w.shape
        pts_c = torch.einsum("ij,nj->ni", R, pts_w.reshape(-1, 3)) + t
        return pts_c.reshape(*original_shape)

    def get_homo_coordinate_map(self, H=None, W=None):
        if H is None:
            H = self.default_H.item()
        if W is None:
            W = self.default_W.item()
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
        return torch.from_numpy(uv).to(self.rel_focal).to(self.rel_focal.device)


class AnchorRelativeMultiViewCameras:
    def __init__(
        self,
        anchor_cams: MonocularCameras,
        camera_names,
        anchor_to_camera,
        anchor_camera="cam0",
        tie_anchor_intrinsics=True,
        rel_focals=None,
        cxcy_ratios=None,
    ):
        self.anchor_cams = anchor_cams
        self.camera_names = list(camera_names)
        self.camera_name_to_index = {
            name: i for i, name in enumerate(self.camera_names)
        }
        self.anchor_camera = anchor_camera
        self.tie_anchor_intrinsics = bool(tie_anchor_intrinsics)
        self.anchor_to_camera = torch.as_tensor(anchor_to_camera).float().to(
            anchor_cams.rel_focal
        )
        if self.anchor_to_camera.ndim != 4:
            raise ValueError(
                f"Expected anchor_to_camera with shape [V,T,4,4], got {self.anchor_to_camera.shape}"
            )
        if not self.tie_anchor_intrinsics:
            assert rel_focals is not None and cxcy_ratios is not None
            self.rel_focals = torch.as_tensor(rel_focals).float().to(
                anchor_cams.rel_focal
            )
            self.cxcy_ratios = torch.as_tensor(cxcy_ratios).float().to(
                anchor_cams.rel_focal
            )
        else:
            self.rel_focals = None
            self.cxcy_ratios = None

    @classmethod
    def from_workspace(
        cls,
        ws,
        anchor_cams: MonocularCameras,
        manifest_name="multiview_manifest.json",
        tie_anchor_intrinsics=True,
    ):
        manifest_path = osp.join(ws, manifest_name)
        if not osp.exists(manifest_path):
            raise FileNotFoundError(f"Cannot find multiview manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        camera_names = [it["camera_name"] for it in manifest["cameras"]]
        anchor_camera = manifest["anchor_camera"]
        reference_pose_list = []
        rel_focals = []
        cxcy_ratios = []
        anchor_ref = None
        for camera in manifest["cameras"]:
            pose_npz = np.load(osp.join(ws, camera["reference_c2w_path"]))
            T_wc = torch.from_numpy(pose_npz["c2w"]).float()
            reference_pose_list.append(T_wc)
            if camera["camera_name"] == anchor_camera:
                anchor_ref = T_wc

            if not tie_anchor_intrinsics:
                intrinsics_path = osp.join(ws, camera["camera_json_dir"])
                camera_jsons = sorted(
                    [it for it in os.listdir(intrinsics_path) if it.endswith(".json")]
                )
                with open(osp.join(intrinsics_path, camera_jsons[0]), "r") as jf:
                    camera_json = json.load(jf)
                image_size = np.asarray(camera_json["image_size"], dtype=np.float32)
                principal_point = np.asarray(
                    camera_json["principal_point"], dtype=np.float32
                )
                short_size = float(image_size.min())
                rel_focal = torch.tensor(
                    [
                        2.0 * float(camera_json["focal_length"]) / short_size,
                        2.0 * float(camera_json["focal_length"]) / short_size,
                    ],
                    dtype=torch.float32,
                )
                cxcy_ratio = torch.tensor(
                    [
                        float(principal_point[0] / image_size[0]),
                        float(principal_point[1] / image_size[1]),
                    ],
                    dtype=torch.float32,
                )
                rel_focals.append(rel_focal)
                cxcy_ratios.append(cxcy_ratio)

        if anchor_ref is None:
            raise KeyError(f"Anchor camera {anchor_camera} not found in manifest")
        anchor_ref_inv = torch.linalg.inv(anchor_ref)
        anchor_to_camera = torch.stack(
            [torch.matmul(anchor_ref_inv, T_wc) for T_wc in reference_pose_list], dim=0
        )
        return cls(
            anchor_cams=anchor_cams,
            camera_names=camera_names,
            anchor_to_camera=anchor_to_camera,
            anchor_camera=anchor_camera,
            tie_anchor_intrinsics=tie_anchor_intrinsics,
            rel_focals=None if tie_anchor_intrinsics else torch.stack(rel_focals, 0),
            cxcy_ratios=None if tie_anchor_intrinsics else torch.stack(cxcy_ratios, 0),
        )

    @property
    def T(self):
        return self.anchor_cams.T

    def get_camera_index(self, camera_name):
        if isinstance(camera_name, int):
            return camera_name
        if camera_name not in self.camera_name_to_index:
            raise KeyError(f"Unknown multiview camera: {camera_name}")
        return self.camera_name_to_index[camera_name]

    def _K_from_intrinsics(self, rel_focal, cxcy_ratio, H, W):
        L = min(H, W)
        fx = rel_focal[0] * L / 2.0
        fy = rel_focal[1] * L / 2.0
        cx = W * cxcy_ratio[0]
        cy = H * cxcy_ratio[1]
        K = torch.eye(3).to(rel_focal)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def K(self, camera_name, H=None, W=None):
        if self.tie_anchor_intrinsics:
            return self.anchor_cams.K(H=H, W=W)
        ind = self.get_camera_index(camera_name)
        if H is None:
            H = int(self.anchor_cams.default_H)
        if W is None:
            W = int(self.anchor_cams.default_W)
        return self._K_from_intrinsics(self.rel_focals[ind], self.cxcy_ratios[ind], H, W)

    def _get_rel_focal_cxcy_ratio(self, camera_name):
        if self.tie_anchor_intrinsics:
            return self.anchor_cams.rel_focal, self.anchor_cams.cxcy_ratio
        ind = self.get_camera_index(camera_name)
        return self.rel_focals[ind], self.cxcy_ratios[ind]

    def get_homo_coordinate_map(self, H=None, W=None):
        return self.anchor_cams.get_homo_coordinate_map(H=H, W=W)

    def backproject(self, camera_name, uv, d):
        assert uv.ndim == d.ndim + 1
        assert uv.shape[-1] == 2
        dep = d[..., None]
        rel_focal, cxcy_ratio = self._get_rel_focal_cxcy_ratio(camera_name)
        rel_focal = torch.as_tensor(rel_focal).to(uv)
        cxcy_ratio = torch.as_tensor(cxcy_ratio).to(uv) * 2.0 - 1.0
        xy = (uv - cxcy_ratio[None, :]) * dep / rel_focal[None]
        return torch.cat([xy, dep], dim=-1)

    def project(self, camera_name, xyz, th=1e-5):
        assert xyz.shape[-1] == 3
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        z_close_mask = abs(z) < th
        if z_close_mask.any():
            z_close_mask = z_close_mask.float()
            z = z * (1 - z_close_mask) + (1.0 * th) * z_close_mask
        rel_focal, cxcy_ratio = self._get_rel_focal_cxcy_ratio(camera_name)
        rel_focal = torch.as_tensor(rel_focal).to(xyz)
        cxcy_ratio = torch.as_tensor(cxcy_ratio).to(xyz) * 2.0 - 1.0
        uv = (xy * rel_focal[None] / z) + cxcy_ratio[None, :]
        return uv

    def T_wc(self, camera_name, ind):
        cam_ind = self.get_camera_index(camera_name)
        return self.anchor_cams.T_wc(ind) @ self.anchor_to_camera[cam_ind, ind]

    def T_cw(self, camera_name, ind):
        return torch.linalg.inv(self.T_wc(camera_name, ind))

    def Rt_wc(self, camera_name, ind):
        T_wc = self.T_wc(camera_name, ind)
        return T_wc[:3, :3], T_wc[:3, 3]

    def Rt_cw(self, camera_name, ind):
        T_cw = self.T_cw(camera_name, ind)
        return T_cw[:3, :3], T_cw[:3, 3]

    def T_wc_list(self, camera_name):
        cam_ind = self.get_camera_index(camera_name)
        return torch.matmul(self.anchor_cams.T_wc_list(), self.anchor_to_camera[cam_ind])

    def T_cw_list(self, camera_name):
        return torch.linalg.inv(self.T_wc_list(camera_name))

    def trans_pts_to_world(self, camera_name, tid, pts_c):
        assert pts_c.shape[-1] == 3
        R, t = self.Rt_wc(camera_name, tid)
        original_shape = pts_c.shape
        pts_w = torch.einsum("ij,nj->ni", R, pts_c.reshape(-1, 3)) + t
        return pts_w.reshape(*original_shape)

    def trans_pts_to_cam(self, camera_name, tid, pts_w):
        assert pts_w.shape[-1] == 3
        R, t = self.Rt_cw(camera_name, tid)
        original_shape = pts_w.shape
        pts_c = torch.einsum("ij,nj->ni", R, pts_w.reshape(-1, 3)) + t
        return pts_c.reshape(*original_shape)

    def slice_time(self, start, end, anchor_cams=None):
        if anchor_cams is None:
            anchor_cams = self.anchor_cams
        rel_focals = None if self.rel_focals is None else self.rel_focals.clone()
        cxcy_ratios = None if self.cxcy_ratios is None else self.cxcy_ratios.clone()
        return type(self)(
            anchor_cams=anchor_cams,
            camera_names=self.camera_names,
            anchor_to_camera=self.anchor_to_camera[:, start:end].clone(),
            anchor_camera=self.anchor_camera,
            tie_anchor_intrinsics=self.tie_anchor_intrinsics,
            rel_focals=rel_focals,
            cxcy_ratios=cxcy_ratios,
        )

    def to(self, device):
        self.anchor_to_camera = self.anchor_to_camera.to(device)
        if self.rel_focals is not None:
            self.rel_focals = self.rel_focals.to(device)
            self.cxcy_ratios = self.cxcy_ratios.to(device)
        return self


def __project__(xyz, cams, th=1e-5):
    # assert xyz.ndim == 2
    assert xyz.shape[-1] == 3
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    z_close_mask = abs(z) < th
    if z_close_mask.any():
        # logging.warning(
        #     f"Projection may create singularity with a point too close to the camera, detected [{z_close_mask.sum()}] points, clamp it"
        # )
        z_close_mask = z_close_mask.float()
        z = (
            z * (1 - z_close_mask) + (1.0 * th) * z_close_mask
        )  # ! always clamp to positive
        assert not (abs(z) < th).any()
    rel_f = torch.as_tensor(cams.rel_focal).to(xyz)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(xyz) * 2.0 - 1.0
    uv = (xy * rel_f[None] / z) + cxcy[None, :]
    return uv  # [-1,1]


def __backproject__(uv, d, cams):
    # assert uv.ndim == 2
    # uv: always be [-1,+1] on the short side
    assert uv.ndim == d.ndim + 1
    assert uv.shape[-1] == 2
    dep = d[..., None]
    rel_f = torch.as_tensor(cams.rel_focal).to(uv)
    # focal = rel_f / 2.0 * min(H, W)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(uv) * 2.0 - 1.0
    xy = (uv - cxcy[None, :]) * dep / rel_f[None]
    z = dep
    xyz = torch.cat([xy, z], dim=-1)
    return xyz


def __get_homo_coordinate_map__(H, W):
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