"""Resolve fitting presets before backend imports; evaluation uses the reference."""
import os


VARIABLES_V1 = ("WFM_TABLE1_RENDER_CACHE", "WFM_TABLE1_KEEP_ALLOCATOR",
             "WFM_TABLE1_RESIDENT_TOPOLOGY", "WFM_TABLE1_MATRIX_MATH",
             "WFM_TABLE1_DEVICE_MASKS",
             "WFM_TABLE1_FUSED_GAUSSIAN", "WFM_TABLE1_BATCH_LPIPS", "WFM_TABLE1_SSIM")
VARIABLES = VARIABLES_V1[:-1] + ("WFM_TABLE1_GRAPH_LPIPS", VARIABLES_V1[-1])

# Shared defaults; configure_fit_environment applies the method's choices.
FIT_PRESETS = {
    "reference": {name: "reference" if name.endswith("SSIM") else "0" for name in VARIABLES},
    "efficient": {name: "cached" if name.endswith("SSIM") else "1" for name in VARIABLES},
}
FIT_PRESETS["efficient"]["WFM_TABLE1_GRAPH_LPIPS"] = "0"
DEFAULT_FIT_PRESET = "efficient"


def configure_evaluation():
    """Disable fitting switches before evaluation imports any backend modules."""
    return configure_fit_environment("reference")


def configure_fit_environment(preset=None, *, method=None):
    """Resolve switches before importing modules that capture them as constants.

    An explicit preset overrides inherited experimental environment switches.
    Without a preset, use the efficient defaults with per-switch environment
    overrides for benchmark plans. An explicit reference preset always disables
    every optimization, even if the caller exported efficient switches.
    """
    if preset is not None and preset not in FIT_PRESETS:
        raise ValueError(f"Unknown fitting preset: {preset}")
    defaults = dict(FIT_PRESETS[preset or DEFAULT_FIT_PRESET])
    if (preset or DEFAULT_FIT_PRESET) == "efficient" and method == "ms":
        # Retain MoSca's original Gaussian preparation for reconstruction quality.
        defaults["WFM_TABLE1_FUSED_GAUSSIAN"] = "0"
    settings = (defaults if preset is not None else
                {name: os.environ.get(name, default) for name, default in defaults.items()})
    for name in VARIABLES[:-1]:
        if settings[name] not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
    mode = settings["WFM_TABLE1_SSIM"]
    if mode not in ("reference", "cached", "separable", "compiled"):
        raise ValueError("WFM_TABLE1_SSIM must be reference, cached, separable, or compiled")
    if settings["WFM_TABLE1_GRAPH_LPIPS"] == "1":
        if settings["WFM_TABLE1_BATCH_LPIPS"] == "1":
            raise ValueError("Graph LPIPS and batched LPIPS cannot be enabled together")
        if method is not None and method != "ms":
            raise ValueError("Graph LPIPS is currently supported only for MoSca fitting")
    os.environ.update(settings)
    return settings


def configure_fit():
    settings = configure_fit_environment()
    mode = settings["WFM_TABLE1_SSIM"]
    if mode != "reference":
        from lib_mosca import photo_recon
        from efficiency.losses import cached_ssim, separable_ssim
        function = cached_ssim if mode == "cached" else separable_ssim
        if mode == "compiled":
            import torch
            function = torch.compile(function, fullgraph=True)
        # Both static warmup and multiview fitting use this function object.
        # Evaluation runs in a separate worker with the reference loss helper.
        photo_recon.compute_rgb_loss.__globals__["ssim"] = function
    return settings
