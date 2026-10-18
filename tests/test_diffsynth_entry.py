from __future__ import annotations

import json

from wfm.diffsynth_entry import _model_paths


def test_model_paths_keep_diffusion_shards_grouped(tmp_path):
    shards = []
    for index in range(1, 8):
        shard = tmp_path / f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
        shard.touch()
        shards.append(str(shard.resolve()))
    t5 = tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = tmp_path / "Wan2.1_VAE.pth"
    t5.touch()
    vae.touch()

    paths = json.loads(_model_paths(tmp_path))

    assert paths == [shards, str(t5.resolve()), str(vae.resolve())]
