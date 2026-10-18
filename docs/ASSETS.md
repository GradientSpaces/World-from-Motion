# Model bundle

The core video pipeline expects a versioned local model bundle described by
[`config/artifacts.lock.json`](../config/artifacts.lock.json). The public
Hugging Face repository and download instructions will be added when the assets
are published. Do not replace a configured checkpoint with a similarly named
model from a different recipe.

The current video profile uses a step-16000 DiffSynth checkpoint, four generated
camera views and 200 distillation steps. This is the arbitrary-video profile,
not the paper's eight-view DyCheck experiment configuration.
For Table 1, use the separate step-12500 asset lock and workflow in
[REPRODUCTION.md](../REPRODUCTION.md).

Set `WFM_CACHE_DIR` to the cache root. Under it, the bundle belongs in
`models/0.1.0/`, with its `manifest.json` and the relative paths in the lock file.
The manifest supplies file sizes and SHA-256 fingerprints. To verify a prepared
bundle:

```bash
WFM_CACHE_DIR=/path/to/cache wfm artifacts verify
```

Weights and datasets remain outside Git. Dependency/model license terms remain
applicable; see [third-party notices](../THIRD_PARTY_NOTICES.md).

## Checkpoint license

The Table 1 checkpoint `wfm-step-12500.safetensors` is licensed under
[Apache-2.0](../LICENSE). It is a modified, fine-tuned Wan2.1-VACE-14B model;
[NOTICE](../NOTICE) records the base revision and checkpoint SHA-256.
Distribute `LICENSE` and `NOTICE` alongside the checkpoint, retain applicable
upstream notices, and identify the WfM modifications in the model card.

The separately downloaded Wan components and other pretrained models retain
their respective licenses. The checkpoint download location is still pending.
