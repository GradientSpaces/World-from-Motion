from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from wfm.distillation import (
    DistillationConfig,
    _WorldTreeModules,
    distill_generated_views,
    load_distillation_dataset,
)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.full((4, 6, 3), value, dtype=np.uint8))


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.full((4, 6), 255, dtype=np.uint8))


def _build_layout(root: Path, *, omit_prediction: str | None = None) -> tuple[Path, Path]:
    benchmark = root / "benchmark"
    scene = benchmark / "scene"
    prediction = root / "inference" / "scene" / "pred_rgb" / "cam_virtual"
    metadata = scene / "metadata"
    metadata.mkdir(parents=True)
    frames = [
        {
            "frame_idx": index,
            "time_id": index,
            "frame_name": f"{index:05d}",
            "source_frame_name": f"{index:05d}",
            "reference_frame_name": f"{index:05d}",
        }
        for index in range(2)
    ]
    for camera in ("cam0", "cam_virtual"):
        (metadata / f"{camera}_frames.json").write_text(
            json.dumps({"num_frames": 2, "frames": frames})
        )
    cameras = scene / "cameras"
    cameras.mkdir()
    np.savez_compressed(
        cameras / "cam_virtual_c2w.npz",
        c2w=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
    )
    for index in range(2):
        name = f"{index:05d}.png"
        _write_rgb(scene / "gt" / "cam0" / name, 10 + index)
        _write_mask(scene / "alpha" / "cam0" / name)
        _write_mask(scene / "alpha" / "cam_virtual" / name)
        if omit_prediction != name:
            _write_rgb(prediction / name, 20 + index)
    _write_rgb(prediction / "00002.png", 99)
    return benchmark, root / "inference"


def test_distillation_layout_ignores_diffsynth_padding(tmp_path: Path) -> None:
    benchmark, inference = _build_layout(tmp_path)
    dataset = load_distillation_dataset(
        benchmark_root=benchmark,
        prediction_root=inference,
        scene_name="scene",
        reference_camera="cam0",
        target_cameras=("cam_virtual",),
    )

    assert dataset.num_frames == 2
    assert dataset.time_ids == (0, 1)
    assert dataset.targets["cam_virtual"].ignored_prediction_frames == ("00002",)


def test_distillation_layout_rejects_missing_real_prediction(tmp_path: Path) -> None:
    benchmark, inference = _build_layout(tmp_path, omit_prediction="00001.png")

    with pytest.raises(FileNotFoundError, match="Missing generated frames"):
        load_distillation_dataset(
            benchmark_root=benchmark,
            prediction_root=inference,
            scene_name="scene",
            reference_camera="cam0",
            target_cameras=("cam_virtual",),
        )


class _FakeCameras(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def K(self, height: int, width: int) -> torch.Tensor:
        return torch.eye(3) + self.dummy * 0

    def T_cw(self, time_id: int) -> torch.Tensor:
        return torch.eye(4) + self.dummy * 0


class _FakeStatic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.color = torch.nn.Parameter(torch.zeros(()))

    def forward(self) -> list[torch.Tensor]:
        return [self.color]

    def get_optimizable_list(self, **_: float) -> list[dict]:
        return [{"params": [self.color], "lr": 0.1}]


class _FakeDynamic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.color = torch.nn.Parameter(torch.zeros(()))
        self.scf = torch.nn.Linear(1, 1, bias=False)
        self._skinning_weight = torch.nn.Parameter(torch.zeros(()))
        self._dynamic_logit = torch.nn.Parameter(torch.zeros(()))

    def forward(self, time_id: int) -> list[torch.Tensor]:
        return [self.color + time_id * 0]

    def get_optimizable_list(self, **_: float | None) -> list[dict]:
        return [{"params": [self.color], "lr": 0.1}]


def _fake_render(
    gs5: list[list[torch.Tensor]],
    height: int,
    width: int,
    *_: object,
    **__: object,
) -> dict[str, torch.Tensor]:
    value = torch.sigmoid(gs5[0][0] + gs5[1][0])
    return {"rgb": value.expand(3, height, width)}


def test_distillation_optimizer_writes_final_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark, inference = _build_layout(tmp_path)
    source_native = tmp_path / "initial-native"
    source_native.mkdir()
    for name in (
        "photometric_cam.pth",
        "photometric_s_model_native_add3.pth",
        "photometric_d_model_native_add3.pth",
    ):
        (source_native / name).touch()
    mosca_root = tmp_path / "mosca"
    mosca_root.mkdir()
    modules = _WorldTreeModules(
        MonocularCameras=None,
        StaticGaussian=None,
        DynSCFGaussian=None,
        render=_fake_render,
        ssim=lambda left, right: 1.0 - torch.abs(left - right).mean(),
        gs_backend="native_add3",
    )
    monkeypatch.setattr("wfm.distillation._load_worldtree_modules", lambda _: modules)
    monkeypatch.setattr(
        "wfm.distillation._load_worldtree_state",
        lambda **_: (_FakeCameras(), _FakeStatic(), _FakeDynamic()),
    )

    result = distill_generated_views(
        DistillationConfig(
            benchmark_root=benchmark,
            prediction_root=inference,
            output_root=tmp_path / "distilled",
            scene_name="scene",
            source_native=source_native,
            mosca_root=mosca_root,
            target_cameras=("cam_virtual",),
            device="cpu",
            steps=2,
            ssim_lambda=0.0,
            debug_every=1,
        )
    )

    assert result["num_steps"] == 2
    assert result["num_frames"] == 2
    assert len(json.loads((tmp_path / "distilled/distillation_history.json").read_text())) == 2
    for name in (
        "photometric_cam.pth",
        "photometric_s_model_native_add3.pth",
        "photometric_d_model_native_add3.pth",
    ):
        assert (tmp_path / "distilled/native" / name).stat().st_size > 0
