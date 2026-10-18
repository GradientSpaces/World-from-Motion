from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Sequence

import imageio.v3 as iio
import numpy as np
import torch

from gen4dgs.formats.worldtree_benchmark import camera_label, strip_camera_prefix

from .export_worldtree_layout import ExportFrameRecord, RenderedFrame
from .stationary_pose_sampler import CameraTrajectory, StationaryPose, TrajectoryPose

DEFAULT_WORLDTREE_ROOT = Path(__file__).resolve().parents[4] / "table1" / "backends" / "wt"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


@contextmanager
def prepend_sys_path(path: Path) -> Iterator[None]:
    text = str(path.expanduser().resolve())
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:  # pragma: no cover - defensive
            pass


def load_python_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_worldtree_saved_dir(
    *,
    template_pair_root: str | Path,
    saved_dir: str | Path | None = None,
) -> Path:
    if saved_dir is not None:
        resolved = Path(saved_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"WorldTree saved_dir not found: {resolved}")
        return resolved

    template_pair_root = Path(template_pair_root).expanduser().resolve()
    summary_path = template_pair_root / "metadata" / f"{template_pair_root.name}_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Cannot infer WorldTree saved_dir without pair summary: {summary_path}"
        )
    summary = _load_json(summary_path)
    result_dir_text = summary.get("result_dir")
    if result_dir_text is None:
        raise KeyError(f"Missing 'result_dir' in {summary_path}")

    result_dir = Path(str(result_dir_text)).expanduser().resolve()
    candidates = [result_dir, result_dir.parent]
    for candidate in candidates:
        if (candidate / "photometric_cam.pth").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not infer a WorldTree saved_dir with photometric checkpoints from "
        f"{summary_path}. Checked: {', '.join(str(path) for path in candidates)}"
    )


def _scale_intrinsics(
    K_raw: np.ndarray,
    src_img_wh: tuple[int, int],
    dst_img_wh: tuple[int, int],
) -> np.ndarray:
    if src_img_wh == dst_img_wh:
        return np.asarray(K_raw, dtype=np.float32)
    src_w, src_h = src_img_wh
    dst_w, dst_h = dst_img_wh
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source image size for intrinsics scaling: {src_img_wh}")
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    K_scaled = np.asarray(K_raw, dtype=np.float32).copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    return K_scaled


@dataclass(frozen=True)
class WorldTreeRenderConfig:
    worldtree_root: Path
    saved_dir: Path
    scene_root: Path
    source_cam: str = "cam0"
    device: str | None = None
    flow_step: int = 1
    scene_space: str = "world"
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    alpha_threshold: float = 1e-3
    process_height: int | None = None
    fit_cfg_path: Path | None = None
    use_fachain: bool | None = None


class WorldTreeStationaryRenderer:
    """Thin bridge around WorldTree's iPhone benchmark renderer."""

    def __init__(self, config: WorldTreeRenderConfig) -> None:
        self.config = WorldTreeRenderConfig(
            worldtree_root=config.worldtree_root.expanduser().resolve(),
            saved_dir=config.saved_dir.expanduser().resolve(),
            scene_root=config.scene_root.expanduser().resolve(),
            source_cam=camera_label(config.source_cam),
            device=config.device,
            flow_step=max(1, int(config.flow_step)),
            scene_space=str(config.scene_space),
            bg_color=tuple(float(value) for value in config.bg_color),
            alpha_threshold=float(config.alpha_threshold),
            process_height=(
                None if config.process_height is None else int(config.process_height)
            ),
            fit_cfg_path=(
                None
                if config.fit_cfg_path is None
                else config.fit_cfg_path.expanduser().resolve()
            ),
            use_fachain=(
                None if config.use_fachain is None else bool(config.use_fachain)
            ),
        )
        if not self.config.worldtree_root.is_dir():
            raise FileNotFoundError(f"WorldTree root not found: {self.config.worldtree_root}")
        if not self.config.saved_dir.is_dir():
            raise FileNotFoundError(f"WorldTree saved_dir not found: {self.config.saved_dir}")
        if not self.config.scene_root.is_dir():
            raise FileNotFoundError(f"WorldTree scene_root not found: {self.config.scene_root}")

        self.device = self._resolve_device(self.config.device)
        self.module = self._load_worldtree_render_module(self.config.worldtree_root)
        self.workspace_path = self._resolve_workspace_path()
        self.source_specs = self._load_source_specs()
        self.max_time_id = max(spec.time_id for spec in self.source_specs)
        self.source_spec_by_time = {int(spec.time_id): spec for spec in self.source_specs}
        self.source_spec_by_frame_name = {str(spec.frame_name): spec for spec in self.source_specs}
        self.fit_cfg_path: Path | None = None
        self.fit_cfg = None
        self.interval_contexts = ()
        self.cams = None
        self.s_model = None
        self.d_model = None
        if self.config.process_height is None:
            self.cams, self.s_model, self.d_model = self.module._load_eval_assets(
                str(self.config.saved_dir),
                self.device,
            )
            self.module._freeze_eval_scene_models(self.s_model, self.d_model, [])
        else:
            cfg_path = (
                str(self.config.fit_cfg_path)
                if self.config.fit_cfg_path is not None
                else self.module.infer_cfg_path(str(self.config.saved_dir), None)
            )
            self.fit_cfg_path = Path(cfg_path).expanduser().resolve()
            self.fit_cfg = self.module.load_fit_cfg(str(self.fit_cfg_path))
            if self.config.use_fachain is not None:
                self.module.OmegaConf.set_readonly(self.fit_cfg, False)
                self.fit_cfg.use_fachain = bool(self.config.use_fachain)
                self.module.OmegaConf.set_readonly(self.fit_cfg, True)
            self.interval_contexts = tuple(
                self.module.build_interval_contexts(
                    str(self.config.saved_dir),
                    int(self.config.process_height),
                    self.fit_cfg,
                    self.device,
                )
            )
            if not self.interval_contexts:
                raise RuntimeError(
                    f"No WorldTree interval contexts loaded for {self.config.saved_dir}"
                )

    @property
    def uses_interval_contexts(self) -> bool:
        return bool(self.interval_contexts)

    @property
    def process_height(self) -> int | None:
        return self.config.process_height

    @property
    def use_fachain(self) -> bool | None:
        if self.fit_cfg is not None and hasattr(self.fit_cfg, "use_fachain"):
            return bool(getattr(self.fit_cfg, "use_fachain"))
        return self.config.use_fachain

    @property
    def min_interval(self) -> int | None:
        if self.fit_cfg is not None and hasattr(self.fit_cfg, "min_interval"):
            return int(getattr(self.fit_cfg, "min_interval"))
        return None

    @property
    def fachain_ref_time_mask(self) -> bool | None:
        if self.fit_cfg is not None and hasattr(self.fit_cfg, "fachain_ref_time_mask"):
            return bool(getattr(self.fit_cfg, "fachain_ref_time_mask"))
        return None

    @property
    def extend_length(self) -> int | None:
        if self.fit_cfg is not None and hasattr(self.fit_cfg, "extend_length"):
            return int(getattr(self.fit_cfg, "extend_length"))
        return None

    def _locate_interval_context(self, time_id: int):
        if not self.uses_interval_contexts:
            raise RuntimeError("Interval contexts are not enabled for this renderer.")
        return self.module.locate_interval_context(self.interval_contexts, int(time_id))

    def _build_gs_for_time(self, time_id: int):
        if self.uses_interval_contexts:
            ctx, local_t = self._locate_interval_context(int(time_id))
            return (
                self.module._build_eval_gs_list(
                    ctx.s_model,
                    ctx.d_model,
                    ctx.fachain_d_model_list,
                    int(local_t),
                ),
                ctx,
                int(local_t),
            )
        return (
            self.module._build_eval_gs_list(
                self.s_model,
                self.d_model,
                [],
                int(time_id),
            ),
            None,
            int(time_id),
        )

    def _source_T_cw(self, time_id: int) -> np.ndarray:
        if self.uses_interval_contexts:
            ctx, local_t = self._locate_interval_context(int(time_id))
            return ctx.cams.T_cw(int(local_t)).detach().cpu().numpy().astype(np.float32)
        return self.cams.T_cw(int(time_id)).detach().cpu().numpy().astype(np.float32)

    @property
    def saved_dir(self) -> Path:
        return self.config.saved_dir

    @property
    def source_cam(self) -> str:
        return self.config.source_cam

    @property
    def source_img_wh(self) -> tuple[int, int]:
        if not self.source_specs:
            raise RuntimeError("No source camera frames were discovered.")
        first = self.source_specs[0]
        return int(first.img_wh[0]), int(first.img_wh[1])

    def load_source_camera_trajectory(self) -> CameraTrajectory:
        poses: list[TrajectoryPose] = []
        with torch.inference_mode():
            for spec in self.source_specs:
                T_cw = torch.from_numpy(self._source_T_cw(int(spec.time_id))).to(self.device)
                c2w = torch.linalg.inv(T_cw).detach().cpu().numpy()
                poses.append(
                    TrajectoryPose(
                        frame_name=spec.frame_name,
                        time_id=int(spec.time_id),
                        c2w=np.asarray(c2w, dtype=np.float32),
                    )
                )
        return CameraTrajectory(
            scene_root=self.workspace_path,
            camera_name=self.source_cam,
            poses=tuple(poses),
            pose_source_kind=(
                "worldtree_interval_cameras"
                if self.uses_interval_contexts
                else "photometric_cam"
            ),
            pose_source_path=(
                str(self.saved_dir)
                if self.uses_interval_contexts
                else str(self.saved_dir / "photometric_cam.pth")
            ),
        )

    def build_source_records(
        self,
        *,
        clip_start: int = 0,
        max_frames: int | None = None,
        reference_rgb_source: str = "gt",
    ) -> tuple[ExportFrameRecord, ...]:
        reference_rgb_source = str(reference_rgb_source).strip().lower()
        if reference_rgb_source not in {"gt", "render"}:
            raise ValueError(
                f"Unsupported reference_rgb_source {reference_rgb_source!r}; expected 'gt' or 'render'."
            )
        start = max(0, int(clip_start))
        source_specs = list(self.source_specs)[start:]
        if max_frames is not None:
            source_specs = source_specs[: int(max_frames)]
        records = []
        for frame_idx, spec in enumerate(source_specs):
            has_gt = spec.image_path is not None
            rgb_source_kind = (
                "gt_reference_image"
                if reference_rgb_source == "gt" and has_gt
                else "render"
            )
            records.append(
                ExportFrameRecord(
                    frame_idx=int(frame_idx),
                    time_id=int(spec.time_id),
                    frame_name=str(spec.frame_name),
                    has_gt=bool(has_gt),
                    has_test_dynamic=False,
                    has_dataset_frame_at_time=True,
                    has_exact_tto_pose=True,
                    render_pose_source_kind="solved_reference_camera",
                    render_pose_source_time_id=int(spec.time_id),
                    render_pose_source_frame_name=str(spec.frame_name),
                    render_pose_archive_path=str(self.saved_dir),
                    rgb_source_kind=rgb_source_kind,
                    scene_flow_valid=True,
                )
            )
        return tuple(records)

    def render_source_frames(
        self,
        *,
        records: Sequence[ExportFrameRecord],
        outputs: Sequence[str],
        reference_rgb_source: str = "gt",
    ):
        reference_rgb_source = str(reference_rgb_source).strip().lower()
        if reference_rgb_source not in {"gt", "render"}:
            raise ValueError(
                f"Unsupported reference_rgb_source {reference_rgb_source!r}; expected 'gt' or 'render'."
            )
        outputs_set = {str(name) for name in outputs}
        render_outputs = self._normalize_render_outputs(outputs)
        source_cam_id = strip_camera_prefix(self.source_cam)
        with torch.inference_mode():
            for record in records:
                spec = self.source_spec_by_frame_name.get(str(record.frame_name))
                if spec is None:
                    raise KeyError(
                        f"Source frame {record.frame_name} not found for {self.source_cam}."
                    )
                T_cw = self._source_T_cw(int(spec.time_id))
                pose_assignment = self.module.PoseAssignment(
                    T_cw=T_cw.copy(),
                    source_kind="solved_reference_camera",
                    source_time_id=int(spec.time_id),
                    source_frame_name=str(spec.frame_name),
                    source_camera_json_path=spec.camera_json_path,
                    pose_archive_path=str(self.saved_dir),
                    has_exact_tto_pose=True,
                )
                render_spec = self.module.CameraRenderSpec(
                    cam_id=source_cam_id,
                    time_id=int(spec.time_id),
                    frame_name=str(spec.frame_name),
                    camera_json_path=spec.camera_json_path,
                    image_path=spec.image_path,
                    K_raw=np.asarray(spec.K_raw, dtype=np.float32).copy(),
                    img_wh=(int(spec.img_wh[0]), int(spec.img_wh[1])),
                    T_cw=T_cw.copy(),
                    has_dataset_frame_at_time=True,
                    pose_assignment=pose_assignment,
                )
                gs_list, _, _ = self._build_gs_for_time(int(spec.time_id))
                next_gs_list = None
                next_t = min(int(spec.time_id) + self.config.flow_step, self.max_time_id)
                if next_t > int(spec.time_id) and "scene_flow" in render_outputs:
                    next_gs_list, _, _ = self._build_gs_for_time(next_t)
                frames, tensors, pose_meta = self.module.render_single_camera_frame(
                    spec=render_spec,
                    gs_list=gs_list,
                    next_gs_list=next_gs_list,
                    outputs=render_outputs,
                    scene_space=self.config.scene_space,
                    bg_color=list(self.config.bg_color),
                    alpha_threshold=self.config.alpha_threshold,
                    device=self.device,
                )
                images = {name: np.asarray(frame) for name, frame in frames.items()}
                if spec.image_path is not None:
                    gt_rgb_u8 = np.asarray(iio.imread(spec.image_path))
                    if "gt" in outputs_set:
                        images["gt"] = gt_rgb_u8
                    if "rgb" in outputs_set and reference_rgb_source == "gt":
                        images["rgb"] = gt_rgb_u8
                yield RenderedFrame(
                    frame_name=str(record.frame_name),
                    c2w=np.asarray(pose_meta["c2w"], dtype=np.float32),
                    images=images,
                    tensors={name: np.asarray(value) for name, value in tensors.items()},
                )

    def render_stationary_frames(
        self,
        *,
        target_cam: str,
        records: Sequence[ExportFrameRecord],
        selected_pose: StationaryPose,
        outputs: Sequence[str],
        target_img_wh: tuple[int, int],
    ):
        target_cam = camera_label(target_cam)
        target_cam_id = strip_camera_prefix(target_cam)
        selected_frame_spec = self.source_spec_by_time.get(int(selected_pose.selected_time_id))
        if selected_frame_spec is None:
            raise KeyError(
                f"Selected stationary source time {selected_pose.selected_time_id} "
                f"not found for {self.source_cam}"
            )

        T_cw = np.linalg.inv(np.asarray(selected_pose.c2w, dtype=np.float32)).astype(np.float32)
        K_raw = _scale_intrinsics(
            selected_frame_spec.K_raw,
            selected_frame_spec.img_wh,
            target_img_wh,
        )
        render_outputs = self._normalize_render_outputs(outputs)
        pose_assignment = self.module.PoseAssignment(
            T_cw=T_cw.copy(),
            source_kind=f"stationary_{selected_pose.side}_{selected_pose.axis_name}",
            source_time_id=int(selected_pose.selected_time_id),
            source_frame_name=selected_pose.selected_frame_name,
            source_camera_json_path=selected_frame_spec.camera_json_path,
            pose_archive_path=str(self.saved_dir),
            has_exact_tto_pose=True,
        )

        with torch.inference_mode():
            for record in records:
                spec = self.module.CameraRenderSpec(
                    cam_id=target_cam_id,
                    time_id=int(record.time_id),
                    frame_name=record.frame_name,
                    camera_json_path=selected_frame_spec.camera_json_path,
                    image_path=None,
                    K_raw=K_raw.copy(),
                    img_wh=(int(target_img_wh[0]), int(target_img_wh[1])),
                    T_cw=T_cw.copy(),
                    has_dataset_frame_at_time=False,
                    pose_assignment=pose_assignment,
                )
                gs_list, _, _ = self._build_gs_for_time(int(record.time_id))
                next_gs_list = None
                next_t = min(int(record.time_id) + self.config.flow_step, self.max_time_id)
                if next_t > int(record.time_id) and "scene_flow" in render_outputs:
                    next_gs_list, _, _ = self._build_gs_for_time(next_t)
                frames, tensors, pose_meta = self.module.render_single_camera_frame(
                    spec=spec,
                    gs_list=gs_list,
                    next_gs_list=next_gs_list,
                    outputs=render_outputs,
                    scene_space=self.config.scene_space,
                    bg_color=list(self.config.bg_color),
                    alpha_threshold=self.config.alpha_threshold,
                    device=self.device,
                )
                yield RenderedFrame(
                    frame_name=record.frame_name,
                    c2w=np.asarray(pose_meta["c2w"], dtype=np.float32),
                    images={name: np.asarray(frame) for name, frame in frames.items()},
                    tensors={name: np.asarray(value) for name, value in tensors.items()},
                )

    def _resolve_workspace_path(self) -> Path:
        explicit = self.config.scene_root.expanduser().resolve()
        if explicit.is_dir():
            return explicit
        inferred = Path(
            self.module.infer_workspace_path(str(self.config.saved_dir), None)
        ).expanduser().resolve()
        return inferred

    def _load_source_specs(self):
        source_cam_id = strip_camera_prefix(self.source_cam)
        train_groups = self.module.discover_camera_frames(
            str(self.workspace_path / "cameras"),
            str(self.workspace_path / "images"),
        )
        if source_cam_id not in train_groups:
            raise KeyError(
                f"Source camera {self.source_cam} not found in {self.workspace_path / 'cameras'}"
            )
        return tuple(train_groups[source_cam_id])

    def _normalize_render_outputs(self, outputs: Sequence[str]) -> set[str]:
        render_outputs = {str(name) for name in outputs if name not in {"gt", "test_dynamic"}}
        if "scene_flow_rgb" in render_outputs:
            render_outputs.add("scene_flow")
            render_outputs.remove("scene_flow_rgb")
        return render_outputs

    def _load_worldtree_render_module(self, worldtree_root: Path) -> ModuleType:
        script_path = (
            Path(__file__).resolve().parents[2] / "render_iphone_tto_benchmark_gbuffers.py"
        )
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing WorldTree render script: {script_path}")
        with prepend_sys_path(worldtree_root):
            return load_python_module("gen4dgs_worldtree_iphone_renderer", script_path)

    def _resolve_device(self, device_name: str | None) -> torch.device:
        if device_name is None or str(device_name).strip() == "":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(str(device_name))
